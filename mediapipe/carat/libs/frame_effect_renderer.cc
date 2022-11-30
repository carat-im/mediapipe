#include "mediapipe/carat/libs/frame_effect_renderer.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/shader_util.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/gl_calculator_helper.h"

namespace {

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

}

namespace mediapipe {

class FrameEffectRendererImpl : public FrameEffectRenderer {
 public:
  FrameEffectRendererImpl(
      std::unique_ptr<GpuBuffer> texture_gpu_buffer,
      std::shared_ptr<GlCalculatorHelper> gpu_helper,
      int width, int height)
      : texture_gpu_buffer_(std::move(texture_gpu_buffer)),
        gpu_helper_(gpu_helper),
        transform_matrix_(Create4x4IdentityMatrix()),
        width_(width),
        height_(height) {
    const GLint attr_location[NUM_ATTRIBUTES] = {
        ATTRIB_VERTEX,
        ATTRIB_TEXTURE_POSITION,
    };
    const GLchar* attr_name[NUM_ATTRIBUTES] = {
        "position",
        "tex_coord",
    };

    const std::string vert_src = std::string(kMediaPipeVertexShaderPreamble) + R"(
      in vec4 position;
      in mediump vec4 tex_coord;
      out mediump vec2 sample_coordinate;
      uniform mat4 matrix;

      void main() {
        sample_coordinate = tex_coord.xy;
        gl_Position = matrix * position;
      }
    )";

    const std::string frag_src = std::string(kMediaPipeFragmentShaderPreamble) + R"(
      DEFAULT_PRECISION(highp, float)

      in vec2 sample_coordinate;
      uniform sampler2D effect_texture;

      void main() {
        gl_FragColor = texture2D(effect_texture, sample_coordinate);
      }
    )";

    GlhCreateProgram(vert_src.c_str(), frag_src.c_str(), NUM_ATTRIBUTES,
        &attr_name[0], attr_location,
        &program_);

    glUseProgram(program_);
    glUniform1i(glGetUniformLocation(program_, "effect_texture"), 1);
    matrix_uniform_ = glGetUniformLocation(program_, "matrix");

    // vertex storage
    glGenBuffers(2, vbo_);
    glGenVertexArrays(1, &vao_);

    // vbo 0
    glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kBasicSquareVertices),
        kBasicSquareVertices, GL_STATIC_DRAW);

    // vbo 1
    glBindBuffer(GL_ARRAY_BUFFER, vbo_[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kBasicTextureVertices),
        kBasicTextureVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

  }

  ~FrameEffectRendererImpl() {
    if (program_) glDeleteProgram(program_);
    if (vao_ != 0) glDeleteVertexArrays(1, &vao_);
    glDeleteBuffers(2, vbo_);

    texture_gpu_buffer_.reset();
  }

  absl::Status RenderEffect() {
    GlTexture frame_texture = gpu_helper_->CreateSourceTexture(*texture_gpu_buffer_.get());

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(frame_texture.target(), frame_texture.name());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glUseProgram(program_);

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto target_width = height_ * 9.0 / 16.0;
    transform_matrix_[0] = target_width / width_;
    glUniformMatrix4fv(matrix_uniform_, 1, GL_FALSE, transform_matrix_.data());

    glBindVertexArray(vao_);

    // vbo 0
    glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
    glEnableVertexAttribArray(ATTRIB_VERTEX);
    glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, nullptr);

    // vbo 1
    glBindBuffer(GL_ARRAY_BUFFER, vbo_[1]);
    glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
    glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0, nullptr);

    // draw
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // cleanup
    glDisableVertexAttribArray(ATTRIB_VERTEX);
    glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(frame_texture.target(), 0);

    glDisable(GL_BLEND);

    glUseProgram(0);
    glFlush();

    frame_texture.Release();

    return absl::OkStatus();
  }

 private:
  static std::array<float, 16> Create4x4IdentityMatrix() {
    return {1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f};
  }

  GLuint program_ = 0;
  GLuint vao_ = 0;
  GLuint vbo_[2] = {0, 0};
  GLint matrix_uniform_;

  std::array<float, 16> transform_matrix_;

  std::unique_ptr<GpuBuffer> texture_gpu_buffer_;
  std::shared_ptr<GlCalculatorHelper> gpu_helper_;

  int width_ = 0;
  int height_ = 0;
};

absl::StatusOr<std::unique_ptr<FrameEffectRenderer>> CreateFrameEffectRenderer(
    std::unique_ptr<GpuBuffer> texture_gpu_buffer,
    std::shared_ptr<GlCalculatorHelper> gpu_helper,
    int width, int height) {
  std::unique_ptr<FrameEffectRenderer> result =
      absl::make_unique<FrameEffectRendererImpl>(std::move(texture_gpu_buffer), gpu_helper, width, height);

  return result;
}

}  // namespace mediapipe
