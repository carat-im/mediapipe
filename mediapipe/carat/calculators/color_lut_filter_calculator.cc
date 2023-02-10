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

  static absl::StatusOr<std::shared_ptr<ImageFrame>> ReadTextureFromFileAsPng(const std::string& texture_path);
  static absl::StatusOr<std::string> ReadContentBlobFromFile(const std::string& unresolved_path);

  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint program_ = 0;
  GLuint vao_ = 0;
  GLuint vbo_[2] = {0, 0};
  bool initialized_ = false;

  std::string current_lut_path_;
  std::string current_blend_image_path_1_;
  std::string current_blend_image_path_2_;

  // gl_calculator_helper.h 의 CreateSourceTexture에 따르면,
  // 특정 frame을 유지하면서 재사용하고 싶을 땐 GpuBuffer를 전역 변수로 유지시키고,
  // GlTexture가 필요할때마다 CreateSourceTexture를 사용하는게 좋다고 한다.
  std::unique_ptr<GpuBuffer> lut_gpu_buffer_;
  std::unique_ptr<GpuBuffer> blend_image_1_gpu_buffer_;
  std::unique_ptr<GpuBuffer> blend_image_2_gpu_buffer_;
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
  return gpu_helper_.Open(cc);
}

absl::Status ColorLutFilterCalculator::Process(CalculatorContext* cc) {
  // The `IMAGE_GPU` stream is required to have a non-empty packet. In case
  // this requirement is not met, there's nothing to be processed at the
  // current timestamp.
  if (cc->Inputs().Tag(kImageGpuTag).IsEmpty()) {
    return absl::OkStatus();
  }

  return gpu_helper_.RunInGlContext([this, &cc]() -> absl::Status {
      if (!initialized_) {
        MP_RETURN_IF_ERROR(InitGpu(cc));
        initialized_ = true;
      }

      MP_RETURN_IF_ERROR(RenderGpu(cc));

      return absl::OkStatus();
  });
}

absl::Status ColorLutFilterCalculator::Close(CalculatorContext* cc) {
  return gpu_helper_.RunInGlContext([this]() -> absl::Status {
      if (program_) glDeleteProgram(program_);
      if (vao_ != 0) glDeleteVertexArrays(1, &vao_);
      glDeleteBuffers(2, vbo_);

      lut_gpu_buffer_.reset();
      blend_image_1_gpu_buffer_.reset();
      blend_image_2_gpu_buffer_.reset();

      return absl::OkStatus();
  });
}

absl::Status ColorLutFilterCalculator::InitGpu(CalculatorContext *cc) {
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  // kBasicVertexShader 에 선언되어있는 이름들. gl_simple_shaders.cc
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  const std::string frag_src = std::string(kMediaPipeFragmentShaderPreamble) + R"(
    DEFAULT_PRECISION(highp, float)

    in vec2 sample_coordinate;
    uniform sampler2D frame;

    uniform vec2 size;

    uniform sampler2D lut_texture;
    uniform int has_lut_texture;

    uniform float intensity;
    uniform float grain;
    uniform float vignette;

    uniform float radial_blur;

    uniform float rgb_split;

    uniform sampler2D blend_image_texture_1;
    uniform int blend_mode_1;
    uniform int has_blend_image_texture_1;

    uniform sampler2D blend_image_texture_2;
    uniform int blend_mode_2;
    uniform int has_blend_image_texture_2;

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

    vec4 blur_radial(sampler2D tex, vec2 uv, float radius) {
      vec4 total = vec4(0);
      
      float dist = 1.0/50.0; // 50.0을 기준으로 더 높이면 느려지지만 더 blur가 잘됨.
      for(float i = 0.0; i<=1.0; i+=dist) {
        vec2 coord = (uv-0.5) / (1.0+radius*i)+0.5;
        total += texture2D(tex,coord);
      }
      
      return total * dist;
    }

    vec3 screen(vec3 s, vec3 d) {
      return s + d - s * d;
    }

    float overlay(float s, float d) {
      return (d < 0.5) ? 2.0 * s * d : 1.0 - 2.0 * (1.0 - s) * (1.0 - d);
    }

    vec3 overlay(vec3 s, vec3 d) {
      vec3 c;
      c.x = overlay(s.x,d.x);
      c.y = overlay(s.y,d.y);
      c.z = overlay(s.z,d.z);
      return c;
    }

    vec3 multiply(vec3 s, vec3 d) {
      return s*d;
    }

    void main() {
      if (radial_blur != 0.0) {
        gl_FragColor = blur_radial(frame, sample_coordinate, radial_blur);
      } else {
        gl_FragColor = texture2D(frame, sample_coordinate);
      }

      if (rgb_split != 0.0) {
        float split_amount = rgb_split * fract(sin(dot(sample_coordinate, vec2(12.9898, 78.233))) * 43758.5453);
        vec4 splitted = vec4(
          texture2D(frame, vec2(sample_coordinate.x + split_amount, sample_coordinate.y)).r,
          texture2D(frame, sample_coordinate).g,
          texture2D(frame, vec2(sample_coordinate.x - split_amount, sample_coordinate.y)).b,
          1.0
        );
        gl_FragColor = mix(gl_FragColor, splitted, 0.5);
      }

      if (has_lut_texture == 1) {
        gl_FragColor = lookup_table(gl_FragColor);
      }

      if (has_blend_image_texture_1 == 1) {
        vec4 blend_image_color = texture2D(blend_image_texture_1, sample_coordinate);
        if (blend_mode_1 == 0) {
          gl_FragColor = vec4(mix(gl_FragColor.rgb, screen(blend_image_color.rgb, gl_FragColor.rgb), blend_image_color.a), 1.0);
        } else if (blend_mode_1 == 1) {
          gl_FragColor = vec4(mix(gl_FragColor.rgb, overlay(blend_image_color.rgb, gl_FragColor.rgb), blend_image_color.a), 1.0);
        } else if (blend_mode_1 == 2) {
          gl_FragColor = vec4(mix(gl_FragColor.rgb, multiply(blend_image_color.rgb, gl_FragColor.rgb), blend_image_color.a), 1.0);
        }
      }

      if (has_blend_image_texture_2 == 1) {
        vec4 blend_image_color = texture2D(blend_image_texture_2, sample_coordinate);
        if (blend_mode_2 == 0) {
          gl_FragColor = vec4(mix(gl_FragColor.rgb, screen(blend_image_color.rgb, gl_FragColor.rgb), blend_image_color.a), 1.0);
        } else if (blend_mode_2 == 1) {
          gl_FragColor = vec4(mix(gl_FragColor.rgb, overlay(blend_image_color.rgb, gl_FragColor.rgb), blend_image_color.a), 1.0);
        } else if (blend_mode_2 == 2) {
          gl_FragColor = vec4(mix(gl_FragColor.rgb, multiply(blend_image_color.rgb, gl_FragColor.rgb), blend_image_color.a), 1.0);
        }
      }

      if (grain != 0.0) {
        gl_FragColor = grain_filter(gl_FragColor, sample_coordinate, grain * intensity);
      }

      if (vignette != 0.0) {
        gl_FragColor = vec4(vec3(gl_FragColor * vignette_filter(sample_coordinate, (1.0 - vignette * intensity))), 1.0);
      }
    }
  )";

  // shader program and params
  GlhCreateProgram(kBasicVertexShader, frag_src.c_str(),
      NUM_ATTRIBUTES, &attr_name[0], attr_location,
      &program_);
  RET_CHECK(program_) << "Problem initializing the program.";

  glUseProgram(program_);
  // frame 은 GL_TEXTURE1 에, lut_texture 는 GL_TEXTURE2 에 바인딩 될것임.
  // 내 기억으로, GL_TEXTURE0 은 outputTexture로, 그 외에는 1, 2씩 증가하는 식으로 하는게 practice 임.
  // 참고: gl_simple_calculator.h 의 GlRender.
  glUniform1i(glGetUniformLocation(program_, "frame"), 1);
  glUniform1i(glGetUniformLocation(program_, "lut_texture"), 2);
  glUniform1i(glGetUniformLocation(program_, "blend_image_texture_1"), 3);
  glUniform1i(glGetUniformLocation(program_, "blend_image_texture_2"), 4);

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

  return absl::OkStatus();
}

absl::Status ColorLutFilterCalculator::RenderGpu(CalculatorContext *cc) {
  const ColorLut& color_lut = cc->Inputs().Tag(kColorLutTag).Get<ColorLut>();

  if (color_lut.has_lut_path()) {
    const auto& new_lut_path = color_lut.lut_path();
    if (current_lut_path_ != new_lut_path) {
      current_lut_path_ = new_lut_path;
      lut_gpu_buffer_.reset();

      ASSIGN_OR_RETURN(std::shared_ptr<ImageFrame> lut_image_frame,
          ReadTextureFromFileAsPng(current_lut_path_),
          _ << "Failed to read the lut texture from file!");

      lut_gpu_buffer_ = absl::make_unique<GpuBuffer>(gpu_helper_.GpuBufferWithImageFrame(lut_image_frame));
    }
  } else {
    current_lut_path_ = "";
    lut_gpu_buffer_.reset();
  }

  std::unique_ptr<GlTexture> lut_gl_texture;
  if (lut_gpu_buffer_ != nullptr) {
    lut_gl_texture = absl::make_unique<GlTexture>(gpu_helper_.CreateSourceTexture(*lut_gpu_buffer_.get()));
  }

  if (color_lut.has_blend_image_path_1()) {
    const auto& new_path = color_lut.blend_image_path_1();
    if (current_blend_image_path_1_ != new_path) {
      current_blend_image_path_1_ = new_path;
      blend_image_1_gpu_buffer_.reset();

      ASSIGN_OR_RETURN(std::shared_ptr<ImageFrame> image_frame,
          ReadTextureFromFileAsPng(current_blend_image_path_1_),
          _ << "Failed to read the texture from file!");

      blend_image_1_gpu_buffer_ = absl::make_unique<GpuBuffer>(gpu_helper_.GpuBufferWithImageFrame(image_frame));
    }
  } else {
    current_blend_image_path_1_ = "";
    blend_image_1_gpu_buffer_.reset();
  }

  std::unique_ptr<GlTexture> blend_image_1_gl_texture;
  if (blend_image_1_gpu_buffer_ != nullptr) {
    blend_image_1_gl_texture = absl::make_unique<GlTexture>(gpu_helper_.CreateSourceTexture(*blend_image_1_gpu_buffer_.get()));
  }

  if (color_lut.has_blend_image_path_2()) {
    const auto& new_path = color_lut.blend_image_path_2();
    if (current_blend_image_path_2_ != new_path) {
      current_blend_image_path_2_ = new_path;
      blend_image_2_gpu_buffer_.reset();

      ASSIGN_OR_RETURN(std::shared_ptr<ImageFrame> image_frame,
          ReadTextureFromFileAsPng(current_blend_image_path_2_),
          _ << "Failed to read the texture from file!");

      blend_image_2_gpu_buffer_ = absl::make_unique<GpuBuffer>(gpu_helper_.GpuBufferWithImageFrame(image_frame));
    }
  } else {
    current_blend_image_path_2_ = "";
    blend_image_2_gpu_buffer_.reset();
  }

  std::unique_ptr<GlTexture> blend_image_2_gl_texture;
  if (blend_image_2_gpu_buffer_ != nullptr) {
    blend_image_2_gl_texture = absl::make_unique<GlTexture>(gpu_helper_.CreateSourceTexture(*blend_image_2_gpu_buffer_.get()));
  }

  const auto& input_gpu_buffer =
      cc->Inputs().Tag(kImageGpuTag).Get<GpuBuffer>();
  GlTexture input_gl_texture =
      gpu_helper_.CreateSourceTexture(input_gpu_buffer);

  GlTexture output_gl_texture = gpu_helper_.CreateDestinationTexture(
      input_gl_texture.width(), input_gl_texture.height(), input_gpu_buffer.format());

  // Run shader on GPU.
  {
    // 이 함수 내에서 glViewport 까지 처리해줌.
    gpu_helper_.BindFramebuffer(output_gl_texture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(input_gl_texture.target(), input_gl_texture.name());

    glActiveTexture(GL_TEXTURE2);
    if (lut_gl_texture != nullptr) {
      glBindTexture(lut_gl_texture->target(), lut_gl_texture->name());

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
      glBindTexture(GL_TEXTURE_2D, 0);
    }

    glActiveTexture(GL_TEXTURE3);
    if (blend_image_1_gl_texture != nullptr) {
      glBindTexture(blend_image_1_gl_texture->target(), blend_image_1_gl_texture->name());

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
      glBindTexture(GL_TEXTURE_2D, 0);
    }

    glActiveTexture(GL_TEXTURE4);
    if (blend_image_2_gl_texture != nullptr) {
      glBindTexture(blend_image_2_gl_texture->target(), blend_image_2_gl_texture->name());

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
      glBindTexture(GL_TEXTURE_2D, 0);
    }

    glUseProgram(program_);

    glUniform2f(glGetUniformLocation(program_, "size"), input_gl_texture.width(), input_gl_texture.height());

    glUniform1i(glGetUniformLocation(program_, "has_lut_texture"), lut_gl_texture != nullptr ? 1 : 0);
    glUniform1i(glGetUniformLocation(program_, "has_blend_image_texture_1"), blend_image_1_gl_texture != nullptr ? 1 : 0);
    glUniform1i(glGetUniformLocation(program_, "has_blend_image_texture_2"), blend_image_2_gl_texture != nullptr ? 1 : 0);

    glUniform1f(glGetUniformLocation(program_, "intensity"), color_lut.intensity());
    glUniform1f(glGetUniformLocation(program_, "grain"), color_lut.grain());
    glUniform1f(glGetUniformLocation(program_, "vignette"), color_lut.vignette());
    glUniform1f(glGetUniformLocation(program_, "radial_blur"), color_lut.radial_blur());
    glUniform1f(glGetUniformLocation(program_, "rgb_split"), color_lut.rgb_split());
    glUniform1i(glGetUniformLocation(program_, "blend_mode_1"), color_lut.blend_mode_1());
    glUniform1i(glGetUniformLocation(program_, "blend_mode_2"), color_lut.blend_mode_2());

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

  if (lut_gl_texture != nullptr) {
    lut_gl_texture->Release();
    lut_gl_texture.reset();
  }

  if (blend_image_1_gl_texture != nullptr) {
    blend_image_1_gl_texture->Release();
    blend_image_1_gl_texture.reset();
  }

  if (blend_image_2_gl_texture != nullptr) {
    blend_image_2_gl_texture->Release();
    blend_image_2_gl_texture.reset();
  }

  return absl::OkStatus();
}

// 3 channel (jpg)가 아닌 4 channel (png)로 변환해서 가져와야함.
// 그래야 CVPixelBuffer로 전환할 때 문제가 생기지 않음.
// static
absl::StatusOr<std::shared_ptr<ImageFrame>> ColorLutFilterCalculator::ReadTextureFromFileAsPng(
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
      image_format = ImageFormat::SRGBA;
      cv::cvtColor(decoded_mat, output_mat, cv::COLOR_BGR2RGBA);
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

  std::shared_ptr<ImageFrame> result = std::make_shared<ImageFrame>(image_format,
      output_mat.size().width,
      output_mat.size().height,
      ImageFrame::kGlDefaultAlignmentBoundary);

  output_mat.copyTo(formats::MatView(result.get()));

  return result;
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
