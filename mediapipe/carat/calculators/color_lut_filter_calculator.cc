#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/carat/formats/color_lut.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"       // NOTYPO
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"  // NOTYPO
#include "mediapipe/framework/port/opencv_imgproc_inc.h"    // NOTYPO
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#include "mediapipe/util/resource_util.h"

namespace {

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

constexpr char kImageGpuTag[] = "IMAGE_GPU";
constexpr char kColorLutTag[] = "COLOR_LUT";

} // namespace

namespace mediapipe {

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

class ColorLutFilterCalculator : public CalculatorBase {
 public:
  ColorLutFilterCalculator() = default;
  ~ColorLutFilterCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status InitGpu(CalculatorContext* cc);
  absl::Status RenderGpu(CalculatorContext* cc);
  void GlRender();

  static absl::StatusOr<ImageFrame> ReadTextureFromFile(const std::string& texture_path);
  static absl::StatusOr<std::string> ReadContentBlobFromFile(const std::string& unresolved_path);

  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint program_ = 0;
  bool initialized_ = false;

  std::string current_lut_path_;
  std::unique_ptr<Texture> lut_texture_;
  float intensity_;
  float grain_;
  float vignette_;
};

REGISTER_CALCULATOR(ColorLutFilterCalculator);

// static
absl::Status ColorLutFilterCalculator::GetContract(CalculatorContract* cc) {
  MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc))
      << "Failed to update contract for the GPU helper!";

  cc->Inputs().Tag(kImageGpuTag).Set<GpuBuffer>();
  cc->Inputs().Tag(kColorLutTag).Set<ColorLut>();

  cc->Outputs().Tag(kImageGpuTag).Set<GpuBuffer>();

  return absl::OkStatus();
}

absl::Status ColorLutFilterCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(mediapipe::TimestampDiff(0));

  MP_RETURN_IF_ERROR(gpu_helper_.Open(cc))
      << "Failed to open the GPU helper!";

  return absl::OkStatus();
}

absl::Status ColorLutFilterCalculator::Process(CalculatorContext* cc) {
  // The `IMAGE_GPU` stream is required to have a non-empty packet. In case
  // this requirement is not met, there's nothing to be processed at the
  // current timestamp.
  if (cc->Inputs().Tag(kImageGpuTag).IsEmpty()) {
    return absl::OkStatus();
  }

  MP_RETURN_IF_ERROR(
      gpu_helper_.RunInGlContext([this, &cc]() -> absl::Status {
          if (!initialized_) {
            MP_RETURN_IF_ERROR(InitGpu(cc));
            initialized_ = true;
          }
          MP_RETURN_IF_ERROR(RenderGpu(cc));
          return absl::OkStatus();
      }));

  return absl::OkStatus();
}

absl::Status ColorLutFilterCalculator::Close(CalculatorContext* cc) {
  gpu_helper_.RunInGlContext([this] {
      if (program_) glDeleteProgram(program_);
      program_ = 0;

      lut_texture_.reset();
  });
  return absl::OkStatus();
}

absl::Status ColorLutFilterCalculator::InitGpu(CalculatorContext *cc) {
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  const std::string frag_src = R"(
  #if __VERSION__ < 130
    #define in varying
  #endif  // __VERSION__ < 130

  #ifdef GL_ES
    #define fragColor gl_FragColor
    precision highp float;
  #else
    #define lowp
    #define mediump
    #define highp
    #define texture2D texture
    out vec4 fragColor;
  #endif  // defined(GL_ES)

    in vec2 sample_coordinate;
    uniform sampler2D frame;
    uniform sampler2D lut_texture;

    uniform float intensity;
    uniform float grain;
    uniform float vignette;

    vec4 lookup_table(vec4 color) {
      float blueColor = color.b * 63.0;

      vec2 quad1;
      quad1.y = floor(floor(blueColor) / 8.0);
      quad1.x = floor(blueColor) - (quad1.y * 8.0);
      vec2 quad2;
      quad2.y = floor(ceil(blueColor) / 8.0);
      quad2.x = ceil(blueColor) - (quad2.y * 8.0);

      vec2 texPos1;
      texPos1.x = (quad1.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.r);
      texPos1.y = (quad1.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.g);
      vec2 texPos2;
      texPos2.x = (quad2.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.r);
      texPos2.y = (quad2.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.g);
      vec4 newColor1 = texture2D(lut_texture, texPos1);
      vec4 newColor2 = texture2D(lut_texture, texPos2);
      vec4 newColor = mix(newColor1, newColor2, fract(blueColor));
      vec4 finalColor = (newColor - color) * intensity + color;
      return vec4(finalColor.rgb, color.w);
    }

    vec4 grain_filter(vec4 color, vec2 v_texCoord, float opacity) {
      highp float a = 12.9898;
      highp float b = 78.233;
      highp float c = 43758.5453;
      highp float dt = dot(v_texCoord.xy, vec2(a,b));
      highp float sn = mod(dt, 3.14);
      highp float noise = fract(sin(sn) * c);
      return color - noise * opacity;

      // lowp version.
      // float noise = (fract(sin(dot(v_texCoord, vec2(12.9898,78.233)*2.0)) * 43758.5453));
      // return color - noise * opacity;
    }

    float vignette_filter(vec2 v_texCoord, float radius) {
      float diff = radius - distance(v_texCoord, vec2(0.5, 0.5));
      return smoothstep(-0.5, 0.5, diff);
    }

    void main() {
      vec4 color = texture2D(frame, sample_coordinate);
      fragColor = lookup_table(color);

      if (grain != 0.0) {
        fragColor = grain_filter(fragColor, sample_coordinate, grain * intensity);
      }

      if (vignette != 0.0) {
        fragColor = fragColor * vignette_filter(sample_coordinate, (1.0 - vignette * intensity));
      }
    }
  )";

  // shader program and params
  mediapipe::GlhCreateProgram(mediapipe::kBasicVertexShader, frag_src.c_str(),
      NUM_ATTRIBUTES, &attr_name[0], attr_location,
      &program_);
  RET_CHECK(program_) << "Problem initializing the program.";

  glUseProgram(program_);
  glUniform1i(glGetUniformLocation(program_, "frame"), 1);
  glUniform1i(glGetUniformLocation(program_, "lut_texture"), 2);

  return absl::OkStatus();
}

absl::Status ColorLutFilterCalculator::RenderGpu(CalculatorContext *cc) {
  const ColorLut& color_lut = cc->Inputs().Tag(kColorLutTag).Get<ColorLut>();

  if (color_lut.has_lut_path()) {
    const auto& new_lut_path = color_lut.lut_path();
    if (current_lut_path_ != new_lut_path) {
      current_lut_path_ = new_lut_path;
      lut_texture_.reset();

      ASSIGN_OR_RETURN(ImageFrame lut_texture_frame,
          ReadTextureFromFile(current_lut_path_),
          _ << "Failed to read the lut texture from file!");
      ASSIGN_OR_RETURN(lut_texture_,
          Texture::CreateFromImageFrame(lut_texture_frame),
          _ << "Failed to create an lut gl texture!");
    }
  } else {
    current_lut_path_ = "";
    lut_texture_.reset();
  }

  const auto& input_gpu_buffer =
      cc->Inputs().Tag(kImageGpuTag).Get<GpuBuffer>();

  GlTexture input_gl_texture =
      gpu_helper_.CreateSourceTexture(input_gpu_buffer);

  if (lut_texture_ == nullptr) {
    std::unique_ptr<GpuBuffer> output_gpu_buffer =
        input_gl_texture.GetFrame<GpuBuffer>();

    cc->Outputs()
        .Tag(kImageGpuTag)
        .AddPacket(mediapipe::Adopt<GpuBuffer>(output_gpu_buffer.release())
            .At(cc->InputTimestamp()));

    input_gl_texture.Release();

    return absl::OkStatus();
  }

  intensity_ = color_lut.intensity();
  grain_ = color_lut.grain();
  vignette_ = color_lut.vignette();

  GlTexture output_gl_texture = gpu_helper_.CreateDestinationTexture(
      input_gl_texture.width(), input_gl_texture.height());

  // Run shader on GPU.
  {
    gpu_helper_.BindFramebuffer(output_gl_texture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(input_gl_texture.target(), input_gl_texture.name());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(lut_texture_->target(), lut_texture_->handle());

    GlRender();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
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
}

void ColorLutFilterCalculator::GlRender() {
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

  glUniform1f(glGetUniformLocation(program_, "intensity"), intensity_);
  glUniform1f(glGetUniformLocation(program_, "grain"), grain_);
  glUniform1f(glGetUniformLocation(program_, "vignette"), vignette_);

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
}

// static
absl::StatusOr<ImageFrame> ColorLutFilterCalculator::ReadTextureFromFile(
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

// static
absl::StatusOr<std::string> ColorLutFilterCalculator::ReadContentBlobFromFile(
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


}  // namespace mediapipe
