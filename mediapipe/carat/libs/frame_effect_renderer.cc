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

namespace mediapipe {
namespace {

class Texture {
public:
  static absl::StatusOr<std::unique_ptr<Texture>> CreateFromImageFrame(
      const ImageFrame& image_frame) {
    RET_CHECK(image_frame.IsAligned(ImageFrame::kGlDefaultAlignmentBoundary))
        << "Image frame memory must be aligned for GL usage!";

    RET_CHECK(image_frame.Width() > 0 && image_frame.Height() > 0)
        << "Image frame must have positive dimensions!";

    RET_CHECK(image_frame.Format() == ImageFormat::SRGB ||
        image_frame.Format() == ImageFormat::SRGBA)
        << "Image frame format must be either SRGB or SRGBA!";

    GLint image_format;
    switch (image_frame.NumberOfChannels()) {
      case 3:
        image_format = GL_RGB;
        break;
      case 4:
        image_format = GL_RGBA;
        break;
      default:
        RET_CHECK_FAIL()
            << "Unexpected number of channels; expected 3 or 4, got "
            << image_frame.NumberOfChannels() << "!";
    }

    GLuint handle;
    glGenTextures(1, &handle);
    RET_CHECK(handle) << "Failed to initialize an OpenGL texture!";

    glBindTexture(GL_TEXTURE_2D, handle);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, image_format, image_frame.Width(),
        image_frame.Height(), 0, image_format, GL_UNSIGNED_BYTE,
        image_frame.PixelData());
    glBindTexture(GL_TEXTURE_2D, 0);

    return absl::WrapUnique(new Texture(
        handle, GL_TEXTURE_2D, image_frame.Width(), image_frame.Height(),
        /*is_owned*/ true));
  }

  ~Texture() {
    if (is_owned_) {
      glDeleteProgram(handle_);
    }
  }

  GLuint handle() const { return handle_; }
  GLenum target() const { return target_; }
  int width() const { return width_; }
  int height() const { return height_; }

private:
  Texture(GLuint handle, GLenum target, int width, int height, bool is_owned)
      : handle_(handle),
        target_(target),
        width_(width),
        height_(height),
        is_owned_(is_owned) {}

  GLuint handle_;
  GLenum target_;
  int width_;
  int height_;
  bool is_owned_;
};


class FrameEffectRendererImpl : public FrameEffectRenderer {
 public:
  FrameEffectRendererImpl(std::unique_ptr<Texture> effect_texture)
      : effect_texture_(std::move(effect_texture)),
        identity_matrix_(Create4x4IdentityMatrix()) {
    static const GLint kAttrLocation[NUM_ATTRIBUTES] = {
        ATTRIB_VERTEX,
        ATTRIB_TEXTURE_POSITION,
    };
    static const GLchar* kAttrName[NUM_ATTRIBUTES] = {
        "position",
        "tex_coord",
    };

    static const GLchar* kVertSrc = R"(
      uniform mat4 u_matrix;

      attribute vec4 position;
      attribute vec4 tex_coord;

      varying vec2 v_tex_coord;

      void main() {
        v_tex_coord = tex_coord.xy;
        gl_Position = u_matrix * position;
      }
    )";

    static const GLchar* kFragSrc = R"(
      precision mediump float;

      varying vec2 v_tex_coord;
      uniform sampler2D texture;

      void main() {
        gl_FragColor = texture2D(texture, v_tex_coord);
      }
    )";

    program_handle_ = 0;
    GlhCreateProgram(kVertSrc, kFragSrc, NUM_ATTRIBUTES,
        (const GLchar**)&kAttrName[0], kAttrLocation,
        &program_handle_);

    glUseProgram(program_handle_);
    matrix_uniform_ =
        glGetUniformLocation(program_handle_, "u_matrix");
    texture_uniform_ = glGetUniformLocation(program_handle_, "texture");
  }

  ~FrameEffectRendererImpl() {
    effect_texture_.reset();
    // todo: glHelper를 받아서 runinglcontext로 program등 없애야함.
  }

  absl::Status RenderEffect() {
    glUseProgram(program_handle_);

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    glUniformMatrix4fv(matrix_uniform_, 1, GL_FALSE, identity_matrix_.data());

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(effect_texture_->target(), effect_texture_->handle());
    glUniform1i(texture_uniform_, 1);

    glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, kBasicSquareVertices);
    glEnableVertexAttribArray(ATTRIB_VERTEX);
    glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0,
        kBasicTextureVertices);
    glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(effect_texture_->target(), 0);

    glDisable(GL_BLEND);

    glFlush();

    return absl::OkStatus();
  }

 private:
  enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

  static std::array<float, 16> Create4x4IdentityMatrix() {
    return {1.f, 0.f, 0.f, 0.f,  //
            0.f, 1.f, 0.f, 0.f,  //
            0.f, 0.f, 1.f, 0.f,  //
            0.f, 0.f, 0.f, 1.f};
  }

  GLuint program_handle_;
  GLint matrix_uniform_;
  GLint texture_uniform_;

  std::unique_ptr<Texture> effect_texture_;
  std::array<float, 16> identity_matrix_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<FrameEffectRenderer>> CreateFrameEffectRenderer(ImageFrame&& effect_texture) {
  ASSIGN_OR_RETURN(std::unique_ptr<Texture> effect_gl_texture,
                   Texture::CreateFromImageFrame(effect_texture),
                   _ << "Failed to create an effect texture!");

  std::unique_ptr<FrameEffectRenderer> result =
      absl::make_unique<FrameEffectRendererImpl>(std::move(effect_gl_texture));

  return result;
}

}  // namespace mediapipe
