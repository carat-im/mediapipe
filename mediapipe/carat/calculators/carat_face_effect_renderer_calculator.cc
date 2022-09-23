// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/carat/formats/carat_face_effect.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"       // NOTYPO
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"  // NOTYPO
#include "mediapipe/framework/port/opencv_imgproc_inc.h"    // NOTYPO
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/modules/face_geometry/libs/effect_renderer.h"
#include "mediapipe/modules/face_geometry/libs/validation_utils.h"
#include "mediapipe/modules/face_geometry/protos/environment.pb.h"
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"
#include "mediapipe/modules/face_geometry/protos/mesh_3d.pb.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {
namespace {

static constexpr char kEnvironmentTag[] = "ENVIRONMENT";
static constexpr char kImageGpuTag[] = "IMAGE_GPU";
static constexpr char kMultiFaceGeometryTag[] = "MULTI_FACE_GEOMETRY";
static constexpr char kCaratFaceEffectListTag[] = "CARAT_FACE_EFFECT_LIST";

class EffectRendererCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc))
        << "Failed to update contract for the GPU helper!";

    cc->InputSidePackets()
        .Tag(kEnvironmentTag)
        .Set<face_geometry::Environment>();

    cc->Inputs().Tag(kImageGpuTag).Set<GpuBuffer>();
    cc->Inputs()
        .Tag(kMultiFaceGeometryTag)
        .Set<std::vector<face_geometry::FaceGeometry>>();
    cc->Inputs().Tag(kCaratFaceEffectListTag).Set<CaratFaceEffectList>();
    
    cc->Outputs().Tag(kImageGpuTag).Set<GpuBuffer>();

    return mediapipe::GlCalculatorHelper::UpdateContract(cc);
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(mediapipe::TimestampDiff(0));

    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc))
        << "Failed to open the GPU helper!";

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // The `IMAGE_GPU` stream is required to have a non-empty packet. In case
    // this requirement is not met, there's nothing to be processed at the
    // current timestamp.
    if (cc->Inputs().Tag(kImageGpuTag).IsEmpty()) {
      return absl::OkStatus();
    }

    return gpu_helper_.RunInGlContext([this, cc]() -> absl::Status {
      const CaratFaceEffectList& effect_list = cc->Inputs().Tag(kCaratFaceEffectListTag).Get<CaratFaceEffectList>();
      int hash = -1;
      int multiplier = 1;
      for (const auto& effect : effect_list.effect()) {
        hash = hash + effect.id() * multiplier;
        multiplier = multiplier * 10;
      }

      if (current_effect_list_hash_ != hash) {
        current_effect_list_hash_ = hash;
        for (auto& effect_renderer : effect_renderers_) {
          effect_renderer.reset();
        }
        effect_renderers_.clear();

        const auto& environment = cc->InputSidePackets()
                                      .Tag(kEnvironmentTag)
                                      .Get<face_geometry::Environment>();

        MP_RETURN_IF_ERROR(face_geometry::ValidateEnvironment(environment))
            << "Invalid environment!";

        for (const auto& effect : effect_list.effect()) {
          std::unique_ptr<face_geometry::EffectRenderer> effect_renderer;

          absl::optional<face_geometry::Mesh3d> effect_mesh_3d;
          if (effect.mesh_3d_path().size() > 0) {
            ASSIGN_OR_RETURN(effect_mesh_3d,
                              ReadMesh3dFromFile(effect.mesh_3d_path()),
                              _ << "Failed to read the effect 3D mesh from file!");

            MP_RETURN_IF_ERROR(face_geometry::ValidateMesh3d(*effect_mesh_3d))
                << "Invalid effect 3D mesh!";
          }

          ASSIGN_OR_RETURN(ImageFrame effect_texture,
                           ReadTextureFromFile(effect.texture_path()),
                           _ << "Failed to read the effect texture from file!");

          ASSIGN_OR_RETURN(effect_renderer,
                           CreateEffectRenderer(environment, effect_mesh_3d,
                                                std::move(effect_texture)),
                           _ << "Failed to create the effect renderer!");
          effect_renderers_.push_back(std::move(effect_renderer));
        }
      }

      const auto& input_gpu_buffer =
          cc->Inputs().Tag(kImageGpuTag).Get<GpuBuffer>();

      GlTexture input_gl_texture =
          gpu_helper_.CreateSourceTexture(input_gpu_buffer);

      if (effect_renderers_.size() == 0) {
        std::unique_ptr<GpuBuffer> output_gpu_buffer =
            input_gl_texture.GetFrame<GpuBuffer>();

        cc->Outputs()
            .Tag(kImageGpuTag)
            .AddPacket(mediapipe::Adopt<GpuBuffer>(output_gpu_buffer.release())
                          .At(cc->InputTimestamp()));

        input_gl_texture.Release();

        return absl::OkStatus();
      }

      std::vector<face_geometry::FaceGeometry> empty_multi_face_geometry;
      const auto& multi_face_geometry =
          cc->Inputs().Tag(kMultiFaceGeometryTag).IsEmpty()
              ? empty_multi_face_geometry
              : cc->Inputs()
                    .Tag(kMultiFaceGeometryTag)
                    .Get<std::vector<face_geometry::FaceGeometry>>();

      // Validate input multi face geometry data.
      for (const face_geometry::FaceGeometry& face_geometry :
           multi_face_geometry) {
        MP_RETURN_IF_ERROR(face_geometry::ValidateFaceGeometry(face_geometry))
            << "Invalid face geometry!";
      }

      GlTexture output_gl_texture = gpu_helper_.CreateDestinationTexture(
          input_gl_texture.width(), input_gl_texture.height());

      bool is_first_renderer = true;
      for (auto& effect_renderer : effect_renderers_) {
        if (!is_first_renderer) {
          input_gl_texture.Release();
          input_gl_texture = output_gl_texture;
          output_gl_texture = gpu_helper_.CreateDestinationTexture(
            input_gl_texture.width(), input_gl_texture.height());
        }

        MP_RETURN_IF_ERROR(effect_renderer->RenderEffect(
            multi_face_geometry, input_gl_texture.width(),
            input_gl_texture.height(), input_gl_texture.target(),
            input_gl_texture.name(), output_gl_texture.target(),
            output_gl_texture.name()))
            << "Failed to render the effect!";

        is_first_renderer = false;
      }

      std::unique_ptr<GpuBuffer> output_gpu_buffer =
          output_gl_texture.GetFrame<GpuBuffer>();

      cc->Outputs()
          .Tag(kImageGpuTag)
          .AddPacket(mediapipe::Adopt<GpuBuffer>(output_gpu_buffer.release())
                         .At(cc->InputTimestamp()));

      output_gl_texture.Release();
      input_gl_texture.Release();

      return absl::OkStatus();
    });
  }

  ~EffectRendererCalculator() {
    gpu_helper_.RunInGlContext([this]() {
      for (auto& effect_renderer : effect_renderers_) {
        effect_renderer.reset();
      }
    });
  }

 private:
  static absl::StatusOr<ImageFrame> ReadTextureFromFile(
      const std::string& texture_path) {
    ASSIGN_OR_RETURN(std::string texture_blob,
                     ReadContentBlobFromFile(texture_path),
                     _ << "Failed to read texture blob from file!");

    // Use OpenCV image decoding functionality to finish reading the texture.
    std::vector<char> texture_blob_vector(texture_blob.begin(),
                                          texture_blob.end());
    cv::Mat decoded_mat =
        cv::imdecode(texture_blob_vector, cv::IMREAD_UNCHANGED);

    RET_CHECK(decoded_mat.type() == CV_8UC3 || decoded_mat.type() == CV_8UC4)
        << "Texture must have `char` as the underlying type and "
           "must have either 3 or 4 channels!";

    ImageFormat::Format image_format = ImageFormat::UNKNOWN;
    cv::Mat output_mat;
    switch (decoded_mat.channels()) {
      case 3:
        image_format = ImageFormat::SRGB;
        cv::cvtColor(decoded_mat, output_mat, cv::COLOR_BGR2RGB);
        break;

      case 4:
        image_format = ImageFormat::SRGBA;
        cv::cvtColor(decoded_mat, output_mat, cv::COLOR_BGRA2RGBA);
        break;

      default:
        RET_CHECK_FAIL()
            << "Unexpected number of channels; expected 3 or 4, got "
            << decoded_mat.channels() << "!";
    }

    ImageFrame output_image_frame(image_format, output_mat.size().width,
                                  output_mat.size().height,
                                  ImageFrame::kGlDefaultAlignmentBoundary);

    output_mat.copyTo(formats::MatView(&output_image_frame));

    return output_image_frame;
  }

  static absl::StatusOr<face_geometry::Mesh3d> ReadMesh3dFromFile(
      const std::string& mesh_3d_path) {
    ASSIGN_OR_RETURN(std::string mesh_3d_blob,
                     ReadContentBlobFromFile(mesh_3d_path),
                     _ << "Failed to read mesh 3D blob from file!");

    face_geometry::Mesh3d mesh_3d;
    RET_CHECK(mesh_3d.ParseFromString(mesh_3d_blob))
        << "Failed to parse a mesh 3D proto from a binary blob!";

    return mesh_3d;
  }

  static absl::StatusOr<std::string> ReadContentBlobFromFile(
      const std::string& unresolved_path) {
    ASSIGN_OR_RETURN(std::string resolved_path,
                     mediapipe::PathToResourceAsFile(unresolved_path),
                     _ << "Failed to resolve path! Path = " << unresolved_path);

    std::string content_blob;
    MP_RETURN_IF_ERROR(
        mediapipe::GetResourceContents(resolved_path, &content_blob))
        << "Failed to read content blob! Resolved path = " << resolved_path;

    return content_blob;
  }

  mediapipe::GlCalculatorHelper gpu_helper_;
  std::vector<std::unique_ptr<face_geometry::EffectRenderer>> effect_renderers_;
  int current_effect_list_hash_ = -1;
};

}  // namespace

using CaratFaceEffectRendererCalculator = EffectRendererCalculator;

REGISTER_CALCULATOR(CaratFaceEffectRendererCalculator);

}  // namespace mediapipe
