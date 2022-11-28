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
  GlCalculatorHelper helper_;
  bool initialized_ = false;
  GLuint program_ = 0;
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

    MP_RETURN_IF_ERROR(GlRender());

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(input_gl_texture.target(), 0);

    glFlush();

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
  glUniform1i(glGetUniformLocation(program_, "video_frame"), 1);
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

CaratFrameEffectRendererCalculator::~CaratFrameEffectRendererCalculator() {
  helper_.RunInGlContext([this] {
    if (program_) {
      glDeleteProgram(program_);
      program_ = 0;
    }
  });
}

}  // namespace mediapipe
