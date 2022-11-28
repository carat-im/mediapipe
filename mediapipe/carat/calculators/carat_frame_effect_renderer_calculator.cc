// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#include "mediapipe/carat/formats/carat_frame_effect.pb.h"
#include "mediapipe/carat/libs/frame_effect_renderer.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/framework/port/opencv_core_inc.h"       // NOTYPO
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"  // NOTYPO
#include "mediapipe/framework/port/opencv_imgproc_inc.h"    // NOTYPO

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

namespace mediapipe {

static constexpr char kImageGpuTag[] = "IMAGE_GPU";
static constexpr char kCaratFrameEffectListTag[] = "CARAT_FRAME_EFFECT_LIST";

class CaratFrameEffectRendererCalculator : public CalculatorBase {
 public:
  CaratFrameEffectRendererCalculator() {}
  ~CaratFrameEffectRendererCalculator();

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

  absl::Status GlSetup();
  absl::Status GlRender();

 private:
  static absl::StatusOr<ImageFrame> ReadTextureFromFile(const std::string& texture_path);
  static absl::StatusOr<std::string> ReadContentBlobFromFile(const std::string& unresolved_path);

  GlCalculatorHelper helper_;
  bool initialized_ = false;
  GLuint program_ = 0;

  std::vector<std::unique_ptr<FrameEffectRenderer>> effect_renderers_;
  int current_effect_list_hash_ = -1;
};
REGISTER_CALCULATOR(CaratFrameEffectRendererCalculator);

// static
absl::Status CaratFrameEffectRendererCalculator::GetContract(CalculatorContract* cc) {
  MP_RETURN_IF_ERROR(GlCalculatorHelper::UpdateContract(cc));

  cc->Inputs().Tag(kImageGpuTag).Set<GpuBuffer>();
  cc->Inputs().Tag(kCaratFrameEffectListTag).Set<CaratFrameEffectList>();

  cc->Outputs().Tag(kImageGpuTag).Set<GpuBuffer>();

  return absl::OkStatus();
}

absl::Status CaratFrameEffectRendererCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  return helper_.Open(cc);
}

absl::Status CaratFrameEffectRendererCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kImageGpuTag).IsEmpty()) {
    return absl::OkStatus();
  }

  return helper_.RunInGlContext([this, &cc]() -> absl::Status {
    if (!initialized_) {
      MP_RETURN_IF_ERROR(GlSetup());
      initialized_ = true;
    }

    const CaratFrameEffectList& effect_list = cc->Inputs().Tag(kCaratFrameEffectListTag).Get<CaratFrameEffectList>();
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

      for (const auto& effect : effect_list.effect()) {
        std::unique_ptr<FrameEffectRenderer> effect_renderer;

        ASSIGN_OR_RETURN(ImageFrame effect_texture,
            ReadTextureFromFile(effect.texture_path()),
            _ << "Failed to read the effect texture from file!");

        ASSIGN_OR_RETURN(effect_renderer,
            CreateFrameEffectRenderer(std::move(effect_texture)),
            _ << "Failed to create the effect renderer!");
        effect_renderers_.push_back(std::move(effect_renderer));
      }
    }

    glDisable(GL_BLEND);

    const auto& input_gpu_buffer =
        cc->Inputs().Tag(kImageGpuTag).Get<GpuBuffer>();

    GlTexture input_gl_texture =
        helper_.CreateSourceTexture(input_gpu_buffer);

    GlTexture output_gl_texture = helper_.CreateDestinationTexture(
        input_gl_texture.width(), input_gl_texture.height());

    helper_.BindFramebuffer(output_gl_texture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(input_gl_texture.target(), input_gl_texture.name());
    glUniform1i(glGetUniformLocation(program_, "video_frame"), 1);

    MP_RETURN_IF_ERROR(GlRender());

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(input_gl_texture.target(), 0);

    glFlush();

    for (auto& effect_renderer : effect_renderers_) {
      effect_renderer->RenderEffect();
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

absl::Status CaratFrameEffectRendererCalculator::GlSetup() {
  // Load vertex and fragment shaders
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  // shader program
  GlhCreateProgram(kBasicVertexShader,
                   kBasicTexturedFragmentShader,
                   NUM_ATTRIBUTES, &attr_name[0], attr_location, &program_);
  RET_CHECK(program_) << "Problem initializing the program.";

  glUseProgram(program_);
  return absl::OkStatus();
}

absl::Status CaratFrameEffectRendererCalculator::GlRender() {
  glUseProgram(program_);
  glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, kBasicSquareVertices);
  glEnableVertexAttribArray(ATTRIB_VERTEX);
  glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0,
                        kBasicTextureVertices);
  glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  return absl::OkStatus();
}

// static
absl::StatusOr<ImageFrame> CaratFrameEffectRendererCalculator::ReadTextureFromFile(const std::string& texture_path) {
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

// static
absl::StatusOr<std::string> CaratFrameEffectRendererCalculator::ReadContentBlobFromFile(const std::string& unresolved_path) {
  ASSIGN_OR_RETURN(std::string resolved_path,
      mediapipe::PathToResourceAsFile(unresolved_path),
      _ << "Failed to resolve path! Path = " << unresolved_path);

  std::string content_blob;
  MP_RETURN_IF_ERROR(
      mediapipe::GetResourceContents(resolved_path, &content_blob))
      << "Failed to read content blob! Resolved path = " << resolved_path;

  return content_blob;
}

CaratFrameEffectRendererCalculator::~CaratFrameEffectRendererCalculator() {
  helper_.RunInGlContext([this] {
    if (program_) {
      glDeleteProgram(program_);
      program_ = 0;
    }

    for (auto& effect_renderer : effect_renderers_) {
      effect_renderer.reset();
    }
  });
}

}  // namespace mediapipe
