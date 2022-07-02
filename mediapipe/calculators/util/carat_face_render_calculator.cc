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

constexpr char kEyeSizeTag[] = "EYE_SIZE";

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

  if (cc->InputSidePackets().HasTag(kEyeSizeTag)) {
    cc->InputSidePackets().Tag(kEyeSizeTag).Set<std::unique_ptr<float>>();
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

  glUniform1f(glGetUniformLocation(program_, "eyeSize"), *cc->InputSidePackets().Tag(kEyeSizeTag).Get<std::unique_ptr<float>>());

  for (int i = 0; i < multi_face_landmarks.size(); ++i) {
    const NormalizedLandmarkList& landmarks = multi_face_landmarks[i];

    const NormalizedLandmark& left_eye_left = landmarks.landmark(226);
    const NormalizedLandmark& left_eye_right = landmarks.landmark(243);
    const NormalizedLandmark& left_eye_top = landmarks.landmark(27);

    cv::Point2f left = cv::Point2f(left_eye_left.x(), left_eye_left.y());
    cv::Point2f right = cv::Point2f(left_eye_right.x(), left_eye_right.y());
    cv::Point2f top = cv::Point2f(left_eye_top.x(), left_eye_top.y());
    cv::Point2f center = (left + right) / 2;
    float width = std::sqrt(std::pow(right.x - center.x, 2) + std::pow(right.y - center.y, 2));
    float height = std::sqrt(std::pow(top.x - center.x, 2) + std::pow(top.y - center.y, 2));

    glUniform2f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].center").c_str()),
      center.x,
      center.y);
    glUniform1f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].width").c_str()),
      width);
    glUniform1f(
      glGetUniformLocation(program_, ("leftEyes[" + std::to_string(i) + "].height").c_str()),
      height);

    const NormalizedLandmark& right_eye_left = landmarks.landmark(463);
    const NormalizedLandmark& right_eye_right = landmarks.landmark(446);
    const NormalizedLandmark& right_eye_top = landmarks.landmark(257);

    left = cv::Point2f(right_eye_left.x(), right_eye_left.y());
    right = cv::Point2f(right_eye_right.x(), right_eye_right.y());
    top = cv::Point2f(right_eye_top.x(), right_eye_top.y());
    center = (left + right) / 2;
    width = std::sqrt(std::pow(left.x - center.x, 2) + std::pow(left.y - center.y, 2));
    height = std::sqrt(std::pow(top.x - center.x, 2) + std::pow(top.y - center.y, 2));

    glUniform2f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].center").c_str()),
      center.x,
      center.y);
    glUniform1f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].width").c_str()),
      width);
    glUniform1f(
      glGetUniformLocation(program_, ("rightEyes[" + std::to_string(i) + "].height").c_str()),
      height);
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
  DEFAULT_PRECISION(mediump, float)
  #ifdef GL_ES
    #define fragColor gl_FragColor
  #else
    out vec4 fragColor;
  #endif  // GL_ES

  #define PI 3.1415926535897932384626433832795

  in vec2 sample_coordinate;
  uniform sampler2D input_frame;

  struct Eye {
    vec2 center;
    float width;
    float height;
  };

  // 우리는 우선 최대 4명만 인식한다고 가정함.
  uniform Eye leftEyes[4];
  uniform Eye rightEyes[4];
  uniform float eyeSize;

  uniform int faceCount;

  bool isInEye(vec2 coord, Eye eye) {
    return pow(coord.x - eye.center.x, 2.0) / pow(eye.width, 2.0) +
        pow(coord.y - eye.center.y, 2.0) / pow(eye.height, 2.0) <= 1.0;
  }

  vec2 applyEyeTransform(vec2 coord, Eye eye, bool isLeft) {
    vec2 rcoord = coord - eye.center;
    float r = sqrt(pow(rcoord.x, 2.0) + pow(rcoord.y, 2.0));
    float theta = atan(rcoord.y, rcoord.x);

    float radius = (eye.width * eye.height) / sqrt(pow(eye.width, 2.0) * pow(sin(theta), 2.0) + pow(eye.height, 2.0) * pow(cos(theta), 2.0));
    float factor = r / radius;
    r = factor * r + (1.0 - factor) * eyeSize  * r;
    vec2 newRcoord = vec2(r * cos(theta), r * sin(theta));
    return newRcoord + eye.center;
  }

  void main() {
    vec2 coord = sample_coordinate;
    for (int i = 0; i < faceCount; i++) {
      Eye leftEye = leftEyes[i];
      Eye rightEye = rightEyes[i];

      if (isInEye(coord, leftEye)) {
        coord = applyEyeTransform(coord, leftEye, true);
        break;
      } else if (isInEye(coord, rightEye)) {
        coord = applyEyeTransform(coord, rightEye, false);
        break;
      }
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
