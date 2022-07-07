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

#include <memory>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/vector.h"

#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/shader_util.h"

namespace mediapipe {

namespace {

constexpr char kGpuBufferTag[] = "IMAGE_GPU";
constexpr char kMultiFaceLandmarksTag[] = "MULTI_FACE_LANDMARKS";

constexpr char kForeheadSizeTag[] = "FOREHEAD_SIZE";
constexpr char kCheekboneSizeTag[] = "CHEEKBONE_SIZE";
constexpr char kTempleSizeTag[] = "TEMPLE_SIZE";
constexpr char kChinSizeTag[] = "CHIN_SIZE";
constexpr char kChinHeightTag[] = "CHIN_HEIGHT";
constexpr char kChinSharpnessTag[] = "CHIN_SHARPNESS";
constexpr char kEyeSizeTag[] = "EYE_SIZE";
constexpr char kEyeHeightTag[] = "EYE_HEIGHT";
constexpr char kEyeSpacingTag[] = "EYE_SPACING";
constexpr char kFrontEyeSizeTag[] = "FRONT_EYE_SIZE";
constexpr char kUnderEyeSizeTag[] = "UNDER_EYE_SIZE";
constexpr char kPupilSizeTag[] = "PUPIL_SIZE";
constexpr char kNoseHeightTag[] = "NOSE_HEIGHT";
constexpr char kNoseWidthTag[] = "NOSE_WIDTH";
constexpr char kNoseBridgeSizeTag[] = "NOSE_BRIDGE_SIZE";
constexpr char kNoseBaseSizeTag[] = "NOSE_BASE_SIZE";
constexpr char kNoseEndSizeTag[] = "NOSE_END_SIZE";
constexpr char kPhiltrumHeightTag[] = "PHILTRUM_HEIGHT";
constexpr char kLipSizeTag[] = "LIP_SIZE";
constexpr char kLipEndUpTag[] = "LIP_END_UP";

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

// Round up n to next multiple of m.
size_t RoundUp(size_t n, size_t m) { return ((n + m - 1) / m) * m; }  // NOLINT
}

class CaratFaceRenderCalculator : public CalculatorBase {
 public:
  CaratFaceRenderCalculator() = default;
  ~CaratFaceRenderCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  // From Calculator.
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  template <typename Type, const char* Tag>
  absl::Status RenderToGpu(CalculatorContext* cc);
  absl::Status GlRender(CalculatorContext* cc);
  template <typename Type, const char* Tag>
  absl::Status GlSetup(CalculatorContext* cc);

  // Indicates if image frame is available as input.
  bool image_frame_available_ = false;

  bool gpu_initialized_ = false;
  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint program_ = 0;
  int width_ = 0;
  int height_ = 0;
};
REGISTER_CALCULATOR(CaratFaceRenderCalculator);

absl::Status CaratFaceRenderCalculator::GetContract(CalculatorContract* cc) {
  CHECK_GE(cc->Inputs().NumEntries(), 1);

  if (cc->Inputs().HasTag(kGpuBufferTag) !=
      cc->Outputs().HasTag(kGpuBufferTag)) {
    return absl::InternalError("GPU output must have GPU input.");
  }

  if (cc->Inputs().HasTag(kGpuBufferTag)) {
    cc->Inputs().Tag(kGpuBufferTag).Set<mediapipe::GpuBuffer>();
    CHECK(cc->Outputs().HasTag(kGpuBufferTag));
  }

  if (cc->Outputs().HasTag(kGpuBufferTag)) {
    cc->Outputs().Tag(kGpuBufferTag).Set<mediapipe::GpuBuffer>();
  }

  MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));

  if (cc->Inputs().HasTag(kMultiFaceLandmarksTag)) {
    cc->Inputs().Tag(kMultiFaceLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();
  }

  if (cc->InputSidePackets().HasTag(kForeheadSizeTag)) {
    cc->InputSidePackets().Tag(kForeheadSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kCheekboneSizeTag)) {
    cc->InputSidePackets().Tag(kCheekboneSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kTempleSizeTag)) {
    cc->InputSidePackets().Tag(kTempleSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kChinSizeTag)) {
    cc->InputSidePackets().Tag(kChinSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kChinHeightTag)) {
    cc->InputSidePackets().Tag(kChinHeightTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kChinSharpnessTag)) {
    cc->InputSidePackets().Tag(kChinSharpnessTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kEyeSizeTag)) {
    cc->InputSidePackets().Tag(kEyeSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kEyeHeightTag)) {
    cc->InputSidePackets().Tag(kEyeHeightTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kEyeSpacingTag)) {
    cc->InputSidePackets().Tag(kEyeSpacingTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kFrontEyeSizeTag)) {
    cc->InputSidePackets().Tag(kFrontEyeSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kUnderEyeSizeTag)) {
    cc->InputSidePackets().Tag(kUnderEyeSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kPupilSizeTag)) {
    cc->InputSidePackets().Tag(kPupilSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kNoseHeightTag)) {
    cc->InputSidePackets().Tag(kNoseHeightTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kNoseWidthTag)) {
    cc->InputSidePackets().Tag(kNoseWidthTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kNoseBridgeSizeTag)) {
    cc->InputSidePackets().Tag(kNoseBridgeSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kNoseBaseSizeTag)) {
    cc->InputSidePackets().Tag(kNoseBaseSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kNoseEndSizeTag)) {
    cc->InputSidePackets().Tag(kNoseEndSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kPhiltrumHeightTag)) {
    cc->InputSidePackets().Tag(kPhiltrumHeightTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kLipSizeTag)) {
    cc->InputSidePackets().Tag(kLipSizeTag).Set<std::unique_ptr<float>>();
  }
  if (cc->InputSidePackets().HasTag(kLipEndUpTag)) {
    cc->InputSidePackets().Tag(kLipEndUpTag).Set<std::unique_ptr<float>>();
  }

  return absl::OkStatus();
}

absl::Status CaratFaceRenderCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  if (cc->Inputs().HasTag(kGpuBufferTag)) {
    image_frame_available_ = true;
  }

  const char* tag = kGpuBufferTag;
  if (image_frame_available_ && !cc->Inputs().Tag(tag).Header().IsEmpty()) {
    const auto& input_header =
        cc->Inputs().Tag(tag).Header().Get<VideoHeader>();
    auto* output_video_header = new VideoHeader(input_header);
    cc->Outputs().Tag(tag).SetHeader(Adopt(output_video_header));
  }

  MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));

  return absl::OkStatus();
}

absl::Status CaratFaceRenderCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kGpuBufferTag) &&
      cc->Inputs().Tag(kGpuBufferTag).IsEmpty()) {
    return absl::OkStatus();
  }

  if (!gpu_initialized_) {
    MP_RETURN_IF_ERROR(
        gpu_helper_.RunInGlContext([this, cc]() -> absl::Status {
          return GlSetup<mediapipe::GpuBuffer, kGpuBufferTag>(cc);
        }));
    gpu_initialized_ = true;
  }

  MP_RETURN_IF_ERROR(
      gpu_helper_.RunInGlContext([this, cc]() -> absl::Status {
        return RenderToGpu<mediapipe::GpuBuffer, kGpuBufferTag>(cc);
      }));

  return absl::OkStatus();
}

absl::Status CaratFaceRenderCalculator::Close(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  gpu_helper_.RunInGlContext([this] {
    if (program_) glDeleteProgram(program_);
    program_ = 0;
  });
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

template <typename Type, const char* Tag>
absl::Status CaratFaceRenderCalculator::RenderToGpu(CalculatorContext* cc) {
  // Source and destination textures.
  const auto& input_frame = cc->Inputs().Tag(Tag).Get<Type>();
  auto input_texture = gpu_helper_.CreateSourceTexture(input_frame);

  auto output_texture = gpu_helper_.CreateDestinationTexture(
      width_, height_, mediapipe::GpuBufferFormat::kBGRA32);

  {
    gpu_helper_.BindFramebuffer(output_texture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, input_texture.name());

    MP_RETURN_IF_ERROR(GlRender(cc));

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
  }

  // Send out image as GPU packet.
  auto output_frame = output_texture.GetFrame<Type>();
  cc->Outputs().Tag(Tag).Add(output_frame.release(), cc->InputTimestamp());

  // Cleanup
  input_texture.Release();
  output_texture.Release();

  return absl::OkStatus();
}

absl::Status CaratFaceRenderCalculator::GlRender(CalculatorContext* cc) {
  static const GLfloat square_vertices[] = {
      -1.0f, -1.0f,  // bottom left
      1.0f,  -1.0f,  // bottom right
      -1.0f, 1.0f,   // top left
      1.0f,  1.0f,   // top right
  };
  static const GLfloat texture_vertices[] = {
      0.0f, 0.0f,  // bottom left
      1.0f, 0.0f,  // bottom right
      0.0f, 1.0f,  // top left
      1.0f, 1.0f,  // top right
  };

  // program
  glUseProgram(program_);

  std::vector<NormalizedLandmarkList> empty_multi_face_landmarks;
  const auto& multi_face_landmarks =
      cc->Inputs().Tag(kMultiFaceLandmarksTag).IsEmpty()
          ? empty_multi_face_landmarks
          : cc->Inputs().Tag(kMultiFaceLandmarksTag).Get<std::vector<NormalizedLandmarkList>>();

  glUniform1i(glGetUniformLocation(program_, "faceCount"), multi_face_landmarks.size());

  glUniform1f(glGetUniformLocation(program_, "foreheadSize"), *cc->InputSidePackets().Tag(kForeheadSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "cheekboneSize"), *cc->InputSidePackets().Tag(kCheekboneSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "templeSize"), *cc->InputSidePackets().Tag(kTempleSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "chinSize"), *cc->InputSidePackets().Tag(kChinSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "chinHeight"), *cc->InputSidePackets().Tag(kChinHeightTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "chinSharpness"), *cc->InputSidePackets().Tag(kChinSharpnessTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "eyeSize"), *cc->InputSidePackets().Tag(kEyeSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "eyeHeight"), *cc->InputSidePackets().Tag(kEyeHeightTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "eyeSpacing"), *cc->InputSidePackets().Tag(kEyeSpacingTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "frontEyeSize"), *cc->InputSidePackets().Tag(kFrontEyeSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "underEyeSize"), *cc->InputSidePackets().Tag(kUnderEyeSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "pupilSize"), *cc->InputSidePackets().Tag(kPupilSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "noseHeight"), *cc->InputSidePackets().Tag(kNoseHeightTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "noseWidth"), *cc->InputSidePackets().Tag(kNoseWidthTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "noseBridgeSize"), *cc->InputSidePackets().Tag(kNoseBridgeSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "noseBaseSize"), *cc->InputSidePackets().Tag(kNoseBaseSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "noseEndSize"), *cc->InputSidePackets().Tag(kNoseEndSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "philtrumHeight"), *cc->InputSidePackets().Tag(kPhiltrumHeightTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "lipSize"), *cc->InputSidePackets().Tag(kLipSizeTag).Get<std::unique_ptr<float>>());
  glUniform1f(glGetUniformLocation(program_, "lipEndUp"), *cc->InputSidePackets().Tag(kLipEndUpTag).Get<std::unique_ptr<float>>());


  for (int i = 0; i < multi_face_landmarks.size(); ++i) {
    const NormalizedLandmarkList& landmarks = multi_face_landmarks[i];

    const NormalizedLandmark& left_eye_left = landmarks.landmark(130);
    const NormalizedLandmark& left_eye_right = landmarks.landmark(243);
    const NormalizedLandmark& left_eye_far_right = landmarks.landmark(244);
    const NormalizedLandmark& left_eye_top = landmarks.landmark(27);
    const NormalizedLandmark& left_eye_far_top = landmarks.landmark(29);
    const NormalizedLandmark& left_eye_iris_left = landmarks.landmark(474);
    const NormalizedLandmark& left_eye_iris_right = landmarks.landmark(476);
    const NormalizedLandmark& left_eye_iris_top = landmarks.landmark(475);

    glUniform2f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].center").c_str()),
      (left_eye_left.x() + left_eye_right.x()) / 2,
      (left_eye_left.y() + left_eye_right.y()) / 2);
    glUniform2f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].back").c_str()),
      left_eye_left.x(),
      left_eye_left.y());
    glUniform2f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].front").c_str()),
      left_eye_right.x(),
      left_eye_right.y());
    glUniform2f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].top").c_str()),
      left_eye_top.x(),
      left_eye_top.y());
    glUniform2f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].farFront").c_str()),
      left_eye_far_right.x(),
      left_eye_far_right.y());
    glUniform2f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].farTop").c_str()),
      left_eye_far_top.x(),
      left_eye_far_top.y());
    glUniform2f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].irisCenter").c_str()),
      (left_eye_iris_left.x() + left_eye_iris_right.x()) / 2,
      (left_eye_iris_left.y() + left_eye_iris_right.y()) / 2);
    glUniform2f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].irisRight").c_str()),
      left_eye_iris_right.x(),
      left_eye_iris_right.y());
    glUniform2f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].irisTop").c_str()),
      left_eye_iris_top.x(),
      left_eye_iris_top.y());

    const NormalizedLandmark& right_eye_left = landmarks.landmark(463);
    const NormalizedLandmark& right_eye_far_left = landmarks.landmark(464);
    const NormalizedLandmark& right_eye_right = landmarks.landmark(359);
    const NormalizedLandmark& right_eye_top = landmarks.landmark(257);
    const NormalizedLandmark& right_eye_far_top = landmarks.landmark(260);
    const NormalizedLandmark& right_eye_iris_left = landmarks.landmark(469);
    const NormalizedLandmark& right_eye_iris_right = landmarks.landmark(471);
    const NormalizedLandmark& right_eye_iris_top = landmarks.landmark(470);

    glUniform2f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].center").c_str()),
      (right_eye_left.x() + right_eye_right.x()) / 2,
      (right_eye_left.y() + right_eye_right.y()) / 2);
    glUniform2f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].back").c_str()),
      right_eye_right.x(),
      right_eye_right.y());
    glUniform2f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].front").c_str()),
      right_eye_left.x(),
      right_eye_left.y());
    glUniform2f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].top").c_str()),
      right_eye_top.x(),
      right_eye_top.y());
    glUniform2f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].farFront").c_str()),
      right_eye_far_left.x(),
      right_eye_far_left.y());
    glUniform2f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].farTop").c_str()),
      right_eye_far_top.x(),
      right_eye_far_top.y());
    glUniform2f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].irisCenter").c_str()),
      (right_eye_iris_left.x() + right_eye_iris_right.x()) / 2,
      (right_eye_iris_left.y() + right_eye_iris_right.y()) / 2);
    glUniform2f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].irisRight").c_str()),
      right_eye_iris_right.x(),
      right_eye_iris_right.y());
    glUniform2f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].irisTop").c_str()),
      right_eye_iris_top.x(),
      right_eye_iris_top.y());

    const NormalizedLandmark& nose_highest_center = landmarks.landmark(6);
    const NormalizedLandmark& nose_high_center = landmarks.landmark(197);
    const NormalizedLandmark& nose_center = landmarks.landmark(195);
    const NormalizedLandmark& nose_low_center = landmarks.landmark(5);
    const NormalizedLandmark& nose_lowest_center = landmarks.landmark(4);
    const NormalizedLandmark& nose_left = landmarks.landmark(203);
    const NormalizedLandmark& nose_right = landmarks.landmark(423);

    glUniform2f(
      glGetUniformLocation(program_, ("noses[" + std::to_string(i) + "].highestCenter").c_str()),
      nose_highest_center.x(),
      nose_highest_center.y());
    glUniform2f(
      glGetUniformLocation(program_, ("noses[" + std::to_string(i) + "].highCenter").c_str()),
      nose_high_center.x(),
      nose_high_center.y());
    glUniform2f(
      glGetUniformLocation(program_, ("noses[" + std::to_string(i) + "].center").c_str()),
      nose_center.x(),
      nose_center.y());
    glUniform2f(
      glGetUniformLocation(program_, ("noses[" + std::to_string(i) + "].lowCenter").c_str()),
      nose_low_center.x(),
      nose_low_center.y());
    glUniform2f(
      glGetUniformLocation(program_, ("noses[" + std::to_string(i) + "].lowestCenter").c_str()),
      nose_lowest_center.x(),
      nose_lowest_center.y());
    glUniform2f(
      glGetUniformLocation(program_, ("noses[" + std::to_string(i) + "].left").c_str()),
      nose_left.x(),
      nose_left.y());
    glUniform2f(
      glGetUniformLocation(program_, ("noses[" + std::to_string(i) + "].right").c_str()),
      nose_right.x(),
      nose_right.y());

    const NormalizedLandmark& top_lip_bottom = landmarks.landmark(13);
    const NormalizedLandmark& bottom_lip_top = landmarks.landmark(14);
    const NormalizedLandmark& philtrum = landmarks.landmark(164);
    const NormalizedLandmark& mouth_left = landmarks.landmark(57);
    const NormalizedLandmark& mouth_far_left = landmarks.landmark(207);
    const NormalizedLandmark& mouth_right = landmarks.landmark(287);
    const NormalizedLandmark& mouth_far_right = landmarks.landmark(427);
    const NormalizedLandmark& mouth_left_tip = landmarks.landmark(92);
    const NormalizedLandmark& mouth_right_tip = landmarks.landmark(322);

    glUniform2f(
      glGetUniformLocation(program_, ("mouthes[" + std::to_string(i) + "].center").c_str()),
      (top_lip_bottom.x() + bottom_lip_top.x()) / 2,
      (top_lip_bottom.y() + bottom_lip_top.y()) / 2);
    glUniform2f(
      glGetUniformLocation(program_, ("mouthes[" + std::to_string(i) + "].philtrum").c_str()),
      philtrum.x(),
      philtrum.y());
    glUniform2f(
      glGetUniformLocation(program_, ("mouthes[" + std::to_string(i) + "].left").c_str()),
      mouth_left.x(),
      mouth_left.y());
    glUniform2f(
      glGetUniformLocation(program_, ("mouthes[" + std::to_string(i) + "].farLeft").c_str()),
      mouth_far_left.x(),
      mouth_far_left.y());
    glUniform2f(
      glGetUniformLocation(program_, ("mouthes[" + std::to_string(i) + "].right").c_str()),
      mouth_right.x(),
      mouth_right.y());
    glUniform2f(
      glGetUniformLocation(program_, ("mouthes[" + std::to_string(i) + "].farRight").c_str()),
      mouth_far_right.x(),
      mouth_far_right.y());
    glUniform2f(
      glGetUniformLocation(program_, ("mouthes[" + std::to_string(i) + "].leftTip").c_str()),
      mouth_left_tip.x(),
      mouth_left_tip.y());
    glUniform2f(
      glGetUniformLocation(program_, ("mouthes[" + std::to_string(i) + "].rightTip").c_str()),
      mouth_right_tip.x(),
      mouth_right_tip.y());

    const NormalizedLandmark& forehead_center = landmarks.landmark(10);
    const NormalizedLandmark& forehead_left = landmarks.landmark(54);
    const NormalizedLandmark& forehead_right = landmarks.landmark(284);
    const NormalizedLandmark& forehead_bottom = landmarks.landmark(9);
    const NormalizedLandmark& temple_left_center = landmarks.landmark(124);
    const NormalizedLandmark& cheekbone_left_center = landmarks.landmark(111);
    const NormalizedLandmark& temple_right_center = landmarks.landmark(353);
    const NormalizedLandmark& cheekbone_right_center = landmarks.landmark(340);
    const NormalizedLandmark& chin_top = landmarks.landmark(18);
    const NormalizedLandmark& chin_bottom = landmarks.landmark(152);
    const NormalizedLandmark& chin_left = landmarks.landmark(136);
    const NormalizedLandmark& chin_right = landmarks.landmark(365);

    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].foreheadCenter").c_str()),
      forehead_center.x(),
      forehead_center.y());
    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].foreheadLeft").c_str()),
      forehead_left.x(),
      forehead_left.y());
    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].foreheadRight").c_str()),
      forehead_right.x(),
      forehead_right.y());
    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].foreheadBottom").c_str()),
      forehead_bottom.x(),
      forehead_bottom.y());
    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].templeLeftCenter").c_str()),
      temple_left_center.x(),
      temple_left_center.y());
    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].cheekboneLeftCenter").c_str()),
      cheekbone_left_center.x(),
      cheekbone_left_center.y());
    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].templeRightCenter").c_str()),
      temple_right_center.x(),
      temple_right_center.y());
    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].cheekboneRightCenter").c_str()),
      cheekbone_right_center.x(),
      cheekbone_right_center.y());
    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].chinTop").c_str()),
      chin_top.x(),
      chin_top.y());
    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].chinBottom").c_str()),
      chin_bottom.x(),
      chin_bottom.y());
    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].chinLeft").c_str()),
      chin_left.x(),
      chin_left.y());
    glUniform2f(
      glGetUniformLocation(program_, ("heads[" + std::to_string(i) + "].chinRight").c_str()),
      chin_right.x(),
      chin_right.y());
  }

  // vertex storage
  GLuint vbo[2];
  glGenBuffers(2, vbo);
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // vbo 0
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), square_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_VERTEX);
  glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, nullptr);

  // vbo 1
  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), texture_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0, nullptr);

  // draw
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // cleanup
  glDisableVertexAttribArray(ATTRIB_VERTEX);
  glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(2, vbo);

  return absl::OkStatus();
}

template <typename Type, const char* Tag>
absl::Status CaratFaceRenderCalculator::GlSetup(CalculatorContext* cc) {
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  constexpr char kFragSrcBody[] = R"(
  DEFAULT_PRECISION(highp, float)
  #ifdef GL_ES
    #define fragColor gl_FragColor
  #else
    out vec4 fragColor;
  #endif  // GL_ES

  #define PI 3.1415926535897932384626433832795

  in vec2 sample_coordinate;
  uniform sampler2D input_frame;

  uniform int faceCount;

  struct Eye {
    vec2 center;
    vec2 back;
    vec2 front;
    vec2 top;
    vec2 farFront;
    vec2 farTop;
    vec2 irisCenter;
    vec2 irisRight;
    vec2 irisTop;
  };

  struct Nose {
    vec2 highestCenter;
    vec2 highCenter;
    vec2 center;
    vec2 lowCenter;
    vec2 lowestCenter;
    vec2 left;
    vec2 right;
  };

  struct Mouth {
    vec2 center;
    vec2 philtrum;
    vec2 left;
    vec2 farLeft;
    vec2 right;
    vec2 farRight;
    vec2 leftTip;
    vec2 rightTip;
  };

  struct Head {
    vec2 foreheadCenter;
    vec2 foreheadLeft;
    vec2 foreheadRight;
    vec2 foreheadBottom;
    vec2 templeLeftCenter;
    vec2 cheekboneLeftCenter;
    vec2 templeRightCenter;
    vec2 cheekboneRightCenter;
    vec2 chinTop;
    vec2 chinBottom;
    vec2 chinLeft;
    vec2 chinRight;
  };

  // 우리는 우선 최대 4명만 인식한다고 가정함.
  uniform Eye leftEyes[4];
  uniform Eye rightEyes[4];
  uniform Nose noses[4];
  uniform Mouth mouthes[4];
  uniform Head heads[4];

  uniform float foreheadSize;
  uniform float cheekboneSize;
  uniform float templeSize;
  uniform float chinSize;
  uniform float chinHeight;
  uniform float chinSharpness;
  uniform float eyeSize;
  uniform float eyeHeight;
  uniform float eyeSpacing;
  uniform float frontEyeSize;
  uniform float underEyeSize;
  uniform float pupilSize;
  uniform float noseHeight;
  uniform float noseWidth;
  uniform float noseBridgeSize;
  uniform float noseBaseSize;
  uniform float noseEndSize;
  uniform float philtrumHeight;
  uniform float lipSize;
  uniform float lipEndUp;

  bool isInEllipse(vec2 coord, vec2 center, float r1, float r2) {
    return pow(coord.x - center.x, 2.0) / pow(r1, 2.0) +
        pow(coord.y - center.y, 2.0) / pow(r2, 2.0) <= 1.0;
  }

  bool isInCircle(vec2 coord, vec2 center, float r) {
    return pow(coord.x - center.x, 2.0) + pow(coord.y - center.y, 2.0) < pow(r, 2.0);
  }

  float dist(vec2 v1, vec2 v2) {
    return sqrt(pow(v2.x - v1.x, 2.0) + pow(v2.y - v1.y, 2.0));
  }

  float rectIntersectionDist(vec2 center, float r1, float r2, float theta) {
    float twoPI = PI * 2.0;

    while (theta < -PI) {
      theta = theta + twoPI;
    }

    while (theta > PI) {
      theta = theta - twoPI;
    }

    float rectAtan = atan(r2, r1);
    float tanTheta = tan(theta);
    int region;
    if ((theta > -rectAtan) && (theta <= rectAtan)) {
        region = 1;
    } else if ((theta > rectAtan) && (theta <= (PI - rectAtan))) {
        region = 2;
    } else if ((theta > (PI - rectAtan)) || (theta <= -(PI - rectAtan))) {
        region = 3;
    } else {
        region = 4;
    }
  
    vec2 edgePoint = center;
    int xFactor = 1;
    int yFactor = 1;
    if (region == 1 || region == 2) {
      yFactor = -1;
    } else {
      xFactor = -1;
    }

    if (region == 1 || region == 3) {
      edgePoint.x = edgePoint.x + float(xFactor) * r1;
      edgePoint.y = edgePoint.y + float(yFactor) * r1 * tanTheta;
    } else {
      edgePoint.x = edgePoint.x + float(xFactor) * r2 * tanTheta;
      edgePoint.y = edgePoint.y + float(yFactor) * r2;
    }

    return dist(center, edgePoint);
  }

  vec2 applyEyeSize(vec2 coord, Eye eye) {
    float r1 = dist(eye.center, eye.front);
    float r2 = dist(eye.center, eye.top);

    if (!isInEllipse(coord, eye.center, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - eye.center;
    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = (1.0 / eyeSize) * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return newRcoord - rcoord;
  }

  vec2 applyEyeSpacing(vec2 coord, Eye eye, bool isLeft) {
    float biggerR1 = dist(eye.center, eye.farFront);
    float biggerR2 = dist(eye.center, eye.farTop);

    if (!isInEllipse(coord, eye.center, biggerR1, biggerR2)) {
      return vec2(0.0, 0.0);
    }

    float r1 = dist(eye.center, eye.front);
    float r2 = dist(eye.center, eye.top);

    float maxMoveDist = (eyeSpacing - 1.0) * r1;
    if (!isLeft) {
      maxMoveDist = maxMoveDist * -1.0;
    }

    if (isInEllipse(coord, eye.center, r1, r2)) {
      return vec2(maxMoveDist, 0.0);
    } else {
      vec2 rcoord = coord - eye.center;
      float theta = atan(rcoord.y, rcoord.x);

      float smallCircleDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
      float bigCircleDist = (biggerR1 * biggerR2) / sqrt(pow(biggerR1, 2.0) * pow(sin(theta), 2.0) + pow(biggerR2, 2.0) * pow(cos(theta), 2.0));
      float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));

      return vec2((bigCircleDist - dist) / (bigCircleDist - smallCircleDist) * maxMoveDist, 0.0);
    }
  }

  vec2 applyEyeHeight(vec2 coord, Eye eye) {
    float r1 = dist(eye.center, eye.front);
    float r2 = dist(eye.center, eye.top);

    if (!isInEllipse(coord, eye.center, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - eye.center;
    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = (1.0 / eyeHeight) * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return vec2(0.0, newRcoord.y - rcoord.y);
  }

  vec2 applyFrontEyeSize(vec2 coord, Eye eye, bool isLeft) {
    vec2 center = eye.farFront;
    float r1 = dist(eye.center, center) / 2.0;
    float r2 = dist(center, eye.top) / 2.0;

    vec2 rcoord = coord - center;

    if (!isInEllipse(coord, center, r1, r2) || (isLeft && rcoord.x > 0.0) || (!isLeft && rcoord.x < 0.0)) {
      return vec2(0.0, 0.0);
    }

    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = frontEyeSize * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return vec2(newRcoord.x - rcoord.x, 0.0);
  }

  vec2 applyUnderEyeSize(vec2 coord, Eye eye) {
    vec2 center = (eye.center + eye.back) / 2.0;
    float r1 = dist(center, eye.center);
    float r2 = dist(center, eye.top);

    vec2 rcoord = coord - center;

    if (!isInEllipse(coord, center, r1, r2) || rcoord.y < 0.0) {
      return vec2(0.0, 0.0);
    }

    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = 1.0 / underEyeSize * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return vec2(0.0, newRcoord.y - rcoord.y);
  }

  vec2 applyPupilSize(vec2 coord, Eye eye) {
    float r1 = dist(eye.irisCenter, eye.irisRight);
    float r2 = dist(eye.irisCenter, eye.irisTop);
    float adder = r1 * 0.2;
    r1 = r1 + adder;
    r2 = r2 + adder;

    if (!isInEllipse(coord, eye.irisCenter, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - eye.irisCenter;
    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = 1.0 / pupilSize * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return newRcoord - rcoord;
  }

  vec2 applyEyeTransforms(vec2 coord, Eye eye, bool isLeft) {
    vec2 ret = coord;
    ret = ret + applyEyeSpacing(ret, eye, isLeft);
    ret = ret + applyEyeSize(ret, eye);
    ret = ret + applyEyeHeight(ret, eye);
    ret = ret + applyFrontEyeSize(ret, eye, isLeft);
    ret = ret + applyUnderEyeSize(ret, eye);
    ret = ret + applyPupilSize(ret, eye);

    return ret;
  }

  vec2 applyNoseHeight(vec2 coord, Nose nose) {
    float r1 = min(dist(nose.lowestCenter, nose.left), dist(nose.lowestCenter, nose.right));
    float r2 = dist(nose.lowestCenter, nose.highCenter);
    float biggerR1 = r1 * 1.8;
    float biggerR2 = dist(nose.lowestCenter, nose.highestCenter);

    if (!isInEllipse(coord, nose.lowestCenter, biggerR1, biggerR2)) {
      return vec2(0.0, 0.0);
    }

    float maxMoveDist = (1.0 - noseHeight) * r2;

    if (isInEllipse(coord, nose.lowestCenter, r1, r2)) {
      return vec2(0.0, maxMoveDist);
    } else {
      vec2 rcoord = coord - nose.lowestCenter;
      float theta = atan(rcoord.y, rcoord.x);

      float smallCircleDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
      float bigCircleDist = (biggerR1 * biggerR2) / sqrt(pow(biggerR1, 2.0) * pow(sin(theta), 2.0) + pow(biggerR2, 2.0) * pow(cos(theta), 2.0));
      float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));

      return vec2(0.0, (bigCircleDist - dist) / (bigCircleDist - smallCircleDist) * maxMoveDist);
    }
  }

  vec2 applyNoseWidth(vec2 coord, Nose nose) {
    float r1 = min(dist(nose.lowestCenter, nose.left), dist(nose.lowestCenter, nose.right));
    float r2 = dist(nose.lowestCenter, nose.highCenter);

    if (!isInEllipse(coord, nose.lowestCenter, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - nose.lowestCenter;
    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = (1.0 / noseWidth) * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return vec2(newRcoord.x - rcoord.x, 0.0);
  }

  vec2 applyNoseBridgeSize(vec2 coord, Nose nose) {
    float r1 = min(nose.lowCenter.x - nose.left.x, nose.right.x - nose.lowCenter.x);
    float r2 = dist(nose.lowCenter, nose.highestCenter);

    if (!isInEllipse(coord, nose.lowCenter, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - nose.lowCenter;
    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = 1.0 / noseBridgeSize * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return vec2(newRcoord.x - rcoord.x, 0.0);
  }

  vec2 applyNoseBaseSize(vec2 coord, Nose nose) {
    float r1 = min(dist(nose.lowestCenter, nose.left), dist(nose.lowestCenter, nose.right)) * 1.5;
    float r2 = dist(nose.lowestCenter, nose.center);

    if (!isInEllipse(coord, nose.lowestCenter, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - nose.lowestCenter;
    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = 1.0 / noseBaseSize * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return vec2(newRcoord.x - rcoord.x, 0.0);
  }

  vec2 applyNoseEndSize(vec2 coord, Nose nose) {
    float r1 = min(dist(nose.lowestCenter, nose.left), dist(nose.lowestCenter, nose.right));
    float r2 = dist(nose.lowestCenter, nose.center);
    r1 = r1 / 1.5;

    if (!isInEllipse(coord, nose.lowestCenter, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - nose.lowestCenter;
    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = 1.0 / noseEndSize * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return newRcoord - rcoord;
  }

  vec2 applyNoseTransforms(vec2 coord, Nose nose) {
    vec2 ret = coord;
    ret = ret + applyNoseHeight(ret, nose);
    ret = ret + applyNoseWidth(ret, nose);
    ret = ret + applyNoseBridgeSize(ret, nose);
    ret = ret + applyNoseBaseSize(ret, nose);
    ret = ret + applyNoseEndSize(ret, nose);

    return ret;
  }

  vec2 applyPhiltrumHeight(vec2 coord, Mouth mouth) {
    float r1 = min(dist(mouth.center, mouth.left), dist(mouth.center, mouth.right));
    float r2 = dist(mouth.center, mouth.philtrum) / 2.0;
    float biggerR1 = min(dist(mouth.center, mouth.farLeft), dist(mouth.center, mouth.farRight));
    float biggerR2 = dist(mouth.center, mouth.philtrum);

    if (!isInEllipse(coord, mouth.center, biggerR1, biggerR2)) {
      return vec2(0.0, 0.0);
    }

    float maxMoveDist = (1.0 - philtrumHeight) * r2;

    if (isInEllipse(coord, mouth.center, r1, r2)) {
      return vec2(0.0, maxMoveDist);
    } else {
      vec2 rcoord = coord - mouth.center;
      float theta = atan(rcoord.y, rcoord.x);

      float smallCircleDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
      float bigCircleDist = (biggerR1 * biggerR2) / sqrt(pow(biggerR1, 2.0) * pow(sin(theta), 2.0) + pow(biggerR2, 2.0) * pow(cos(theta), 2.0));
      float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));

      return vec2(0.0, (bigCircleDist - dist) / (bigCircleDist - smallCircleDist) * maxMoveDist);
    }
  }

  vec2 applyMouthSize(vec2 coord, Mouth mouth) {
    float r1 = min(dist(mouth.center, mouth.left), dist(mouth.center, mouth.right));
    float r2 = dist(mouth.center, mouth.philtrum);

    if (!isInEllipse(coord, mouth.center, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - mouth.center;
    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = (1.0 / lipSize) * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return newRcoord - rcoord;
  }

  vec2 applyMouthEndUp(vec2 coord, Mouth mouth) {
    vec2 center = mouth.leftTip;
    float r1 = dist(center, mouth.left);
    float r2 = r1 * 1.5;

    if (!isInEllipse(coord, center, r1, r2)) {
      center = mouth.rightTip;
      r1 = dist(center, mouth.right);
      r2 = r1 * 1.5;
    }

    if (!isInEllipse(coord, center, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - center;
    if (rcoord.y < 0.0) {
      return vec2(0.0, 0.0);
    }

    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = lipEndUp * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return vec2(0.0, newRcoord.y - rcoord.y);
  }

  vec2 applyMouthTransforms(vec2 coord, Mouth mouth) {
    vec2 ret = coord;
    ret = ret + applyPhiltrumHeight(ret, mouth);
    ret = ret + applyMouthSize(ret, mouth);
    ret = ret + applyMouthEndUp(ret, mouth);

    return ret;
  }

  vec2 applyForeheadSize(vec2 coord, Head head) {
    vec2 center = head.foreheadCenter;
    float verticalDist = dist(center, head.foreheadBottom);
    center.y = center.y - verticalDist;
    float r1 = min(dist(head.foreheadCenter, head.foreheadLeft), dist(head.foreheadCenter, head.foreheadRight)) * 1.5;
    float r2 = verticalDist * 2.0;

    vec2 rcoord = coord - center;

    if (!isInEllipse(coord, center, r1, r2) || rcoord.y < 0.0) {
      return vec2(0.0, 0.0);
    }

    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = foreheadSize * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return vec2(0.0, newRcoord.y - rcoord.y);
  }

  vec2 applyCheekboneSize(vec2 coord, Head head) {
    bool isLeft = true;
    vec2 center = head.cheekboneLeftCenter;
    float verticalDist = dist(center, head.templeLeftCenter);
    center.y = center.y + verticalDist / 2.0;
    float r1 = verticalDist;
    float r2 = verticalDist * 1.2;

    if (!isInEllipse(coord, center, r1, r2)) {
      isLeft = false;
      center = head.cheekboneRightCenter;
      verticalDist = dist(center, head.templeRightCenter);
      center.y = center.y + verticalDist / 2.0;
      r1 = verticalDist;
      r2 = verticalDist * 1.2;
    }

    if (!isInEllipse(coord, center, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - center;
    if ((isLeft && rcoord.x > 0.0) || (!isLeft && rcoord.x < 0.0)) {
      return vec2(0.0, 0.0);
    }

    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = 1.0 / cheekboneSize * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return vec2(newRcoord.x - rcoord.x, 0.0);
  }

  vec2 applyTempleSize(vec2 coord, Head head) {
    bool isLeft = true;
    vec2 center = head.templeLeftCenter;
    float verticalDist = dist(center, head.cheekboneLeftCenter);
    float r1 = verticalDist;
    float r2 = verticalDist * 1.2;

    if (!isInEllipse(coord, center, r1, r2)) {
      isLeft = false;
      center = head.templeRightCenter;
      verticalDist = dist(center, head.cheekboneRightCenter);
      r1 = verticalDist;
      r2 = verticalDist * 1.2;
    }

    if (!isInEllipse(coord, center, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - center;
    if ((isLeft && rcoord.x > 0.0) || (!isLeft && rcoord.x < 0.0)) {
      return vec2(0.0, 0.0);
    }

    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = 1.0 / templeSize * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return vec2(newRcoord.x - rcoord.x, 0.0);
  }

  vec2 applyChinHeight(vec2 coord, Head head) {
    vec2 center = head.chinTop;
    float r1 = min(dist(center, head.chinLeft), dist(center, head.chinRight)) * 1.2;
    float r2 = dist(center, head.chinBottom) * 1.5;

    if (!isInEllipse(coord, center, r1, r2)) {
      return vec2(0.0, 0.0);
    }

    vec2 rcoord = coord - center;
    if (rcoord.y < 0.0) {
      return vec2(0.0, 0.0);
    }

    float theta = atan(rcoord.y, rcoord.x);

    float totalDist = (r1 * r2) / sqrt(pow(r1, 2.0) * pow(sin(theta), 2.0) + pow(r2, 2.0) * pow(cos(theta), 2.0));
    float dist = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float appliedDist = 1.0 / chinHeight * dist;

    float factor = dist / totalDist;
    float newDist = factor * dist + (1.0 - factor) * appliedDist;

    vec2 newRcoord = vec2(newDist * cos(theta), newDist * sin(theta));
    return vec2(0.0, newRcoord.y - rcoord.y);
  }

  vec2 applyHeadTransforms(vec2 coord, Head head) {
    vec2 ret = coord;
    ret = ret + applyForeheadSize(ret, head);
    ret = ret + applyCheekboneSize(ret, head);
    ret = ret + applyTempleSize(ret, head);
    ret = ret + applyChinHeight(ret, head);

    return ret;
  }

  void main() {
    vec2 coord = sample_coordinate;
    for (int i = 0; i < faceCount; i++) {
      Eye leftEye = leftEyes[i];
      Eye rightEye = rightEyes[i];
      coord = applyEyeTransforms(coord, leftEye, true);
      coord = applyEyeTransforms(coord, rightEye, false);

      Nose nose = noses[i];
      coord = applyNoseTransforms(coord, nose);

      Mouth mouth = mouthes[i];
      coord = applyMouthTransforms(coord, mouth);

      Head head = heads[i];
      coord = applyHeadTransforms(coord, head);
    }

    vec3 out_pix = texture2D(input_frame, coord).rgb;
    fragColor.rgb = out_pix;
    fragColor.a = 1.0;
  }
  )";

  const std::string frag_src = absl::StrCat(
      mediapipe::kMediaPipeFragmentShaderPreamble, kFragSrcBody);

  // Create shader program and set parameters
  mediapipe::GlhCreateProgram(mediapipe::kBasicVertexShader, frag_src.c_str(),
                              NUM_ATTRIBUTES, (const GLchar**)&attr_name[0],
                              attr_location, &program_);
  RET_CHECK(program_) << "Problem initializing the program.";
  glUseProgram(program_);
  glUniform1i(glGetUniformLocation(program_, "input_frame"), 1);

  // Ensure GPU texture is divisible by 4. See b/138751944 for more info.
  const float alignment = ImageFrame::kGlDefaultAlignmentBoundary;
  if (image_frame_available_) {
    const auto& input_frame = cc->Inputs().Tag(Tag).Get<Type>();
    width_ = RoundUp(input_frame.width(), alignment);
    height_ = RoundUp(input_frame.height(), alignment);
  }

  return absl::OkStatus();
}

}  // namespace mediapipe
