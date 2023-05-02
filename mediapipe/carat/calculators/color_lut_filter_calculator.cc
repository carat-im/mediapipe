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
constexpr char kApplyGammaTag[] = "APPLY_GAMMA";

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

  cc->InputSidePackets().Tag(kApplyGammaTag).Set<bool>();

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

    uniform int apply_gamma;

    uniform float exposure;
    uniform float contrast;
    uniform float temperature;
    uniform float tint;
    uniform float saturation;
    uniform float highlight;
    uniform float shadow;
    uniform float sharpen;
    uniform float vibrance;

    uniform vec3 red_mix;
    uniform vec3 orange_mix;
    uniform vec3 yellow_mix;
    uniform vec3 green_mix;
    uniform vec3 blue_mix;
    uniform vec3 purple_mix;

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

    vec3 exposure_filter(vec3 color, float exposure) {
      return color * pow(2.0, exposure);
    }

    vec3 contrast_filter(vec3 color, float contrast) {
      return (color - 0.5) * (contrast + 1.0) + 0.5;
    }

    float rgb2h(vec3 rgb) {
      float r = rgb[0];
      float g = rgb[1];
      float b = rgb[2];

      float M = max(r, max(g, b));
      float m = min(r, min(g, b));

      float h;
      if (M == m) {
          h = 0.0;
      } else if (m == b) {
          h = 60.0 * (g - r) / (M - m) + 60.0;
      } else if (m == r) {
          h = 60.0 * (b - g) / (M - m) + 180.0;
      } else if (m == g) {
          h = 60.0 * (r - b) / (M - m) + 300.0;
      } else {
          h = 0.0;
      }
      h /= 360.0;
      if (h < 0.0) {
          h = h + 1.0;
      } else if (h > 1.0) {
          h = h - 1.0;
      }
      return h;
    }

    float rgb2s4hsv(vec3 rgb) {
      float r = rgb[0];
      float g = rgb[1];
      float b = rgb[2];
      float M = max(r, max(g, b));
      float m = min(r, min(g, b));

      if (M < 1e-10) return 0.0;
      return (M - m) / M;
    }

    vec3 rgb2hsv(vec3 rgb) {
      float h = rgb2h(rgb);
      float s = rgb2s4hsv(rgb);
      float v = max(rgb.x, max(rgb.y, rgb.z));
      return vec3(h, s, v);
    }

    vec3 hsv2rgb(vec3 hsv) {
      vec3 rgb;

      float r, g, b, h, s, v;

      h = hsv.x;
      s = hsv.y;
      v = hsv.z;

      if (s <= 0.001) {
          r = g = b = v;
      } else {
          float f, p, q, t;
          int i;
          h *= 6.0;
          i = int(floor(h));
          f = h - float(i);
          p = v * (1.0 - s);
          q = v * (1.0 - (s * f));
          t = v * (1.0 - (s * (1.0 - f)));
          if (i == 0 || i == 6) {
              r = v; g = t; b = p;
          } else if (i == 1) {
              r = q; g = v; b = p;
          } else if (i == 2) {
              r = p; g = v; b = t;
          } else if (i == 3) {
              r = p; g = q; b = v;
          } else if (i == 4) {
              r = t; g = p; b = v;
          } else if (i == 5) {
              r = v; g = p; b = q;
          }
      }
      rgb.x = r;
      rgb.y = g;
      rgb.z = b;
      return rgb;
    }

    vec3 saturation_filter(vec3 color, float saturation) {
      vec3 hsv = rgb2hsv(clamp(color, 0.0, 1.0));
      float s = clamp(hsv[1] * (saturation + 1.0), 0.0, 1.0);
      return hsv2rgb(vec3(hsv[0], s, hsv[2]));
    }

    // Y'UV (BT.709) to linear RGB
    // Values are from https://en.wikipedia.org/wiki/YUV
    vec3 yuv2rgb(vec3 yuv) {
        const mat3 m = mat3(+1.00000, +1.00000, +1.00000,  // 1st column
                            +0.00000, -0.21482, +2.12798,  // 2nd column
                            +1.28033, -0.38059, +0.00000); // 3rd column
        return m * yuv;
    }

    // Linear RGB to Y'UV (BT.709)
    // Values are from https://en.wikipedia.org/wiki/YUV
    vec3 rgb2yuv(vec3 rgb) {
        const mat3 m = mat3(+0.21260, -0.09991, +0.61500,  // 1st column
                            +0.71520, -0.33609, -0.55861,  // 2nd column
                            +0.07220, +0.43600, -0.05639); // 3rd column
        return m * rgb;
    }

    vec3 temperature_tint_filter(vec3 color, float temperature, float tint) {
      const float scale = 0.10;
      return clamp(yuv2rgb(rgb2yuv(color) + temperature * scale * vec3(0.0, -1.0, 1.0) + tint * scale * vec3(0.0, 1.0, 1.0)), 0.0, 1.0);
    }

    vec3 highlight_filter(vec3 color, float highlight) {
      vec3 weights = vec3(0.2125, 0.7154, 0.0721); // sums to 1
      float luminance = dot(color, weights);
      luminance = smoothstep(0.5, 1.0, luminance);
      return exposure_filter(color, luminance * highlight);
    }

    vec3 shadow_filter(vec3 color, float shadow) {
      vec3 weights = vec3(0.2125, 0.7154, 0.0721); // sums to 1
      float luminance = dot(color, weights);
      luminance = smoothstep(0.0, 0.5, luminance);
      return exposure_filter(color, (1.0 - luminance) * shadow);
    }

    vec3 sharpen_filter(sampler2D tex, vec2 uv, vec2 size, float sharpen) {
      float neighbor = sharpen * -1.0;
      float c = sharpen * 4.0 + 1.0;
      vec2 unit = 1.0 / size;

      vec3 up = texture2D(tex, vec2(uv.x, uv.y - unit.y)).rgb * neighbor;
      vec3 right = texture2D(tex, vec2(uv.x + unit.x, uv.y)).rgb * neighbor;
      vec3 down = texture2D(tex, vec2(uv.x, uv.y + unit.y)).rgb * neighbor;
      vec3 left = texture2D(tex, vec2(uv.x - unit.x, uv.y)).rgb * neighbor;
      vec3 center = texture2D(tex, uv).rgb * c;
      return center + up + right + down + left;
    }

    int modi(int x, int y) {
      return x - y * (x / y);
    }

    int and(int a, int b) {
      int result = 0;
      int n = 1;
      const int BIT_COUNT = 32;

      for(int i = 0; i < BIT_COUNT; i++) {
        if ((modi(a, 2) == 1) && (modi(b, 2) == 1)) {
          result += n;
        }

        a = a / 2;
        b = b / 2;
        n = n * 2;

        if (!(a > 0 && b > 0)) break;
      }
      return result;
    }

    //r,g,b 0.0 to 1.0,  vibrance 1.0 no change, 0.0 image B&W.  
    vec3 vibrance_filter(vec3 inCol, float v) {
      float vibrance = v + 1.0;
      vec3 outCol;
      if (vibrance <= 1.0) {
        float avg = dot(inCol, vec3(0.3, 0.6, 0.1));
        outCol = mix(vec3(avg), inCol, vibrance); 
      } else {
        float hue_a, a, f, p1, p2, p3, i, h, s, v, amt, _max, _min, dlt;
        float br1, br2, br3, br4, br5, br2_or_br1, br3_or_br1, br4_or_br1, br5_or_br1;
        int use;

        _min = min(min(inCol.r, inCol.g), inCol.b);
        _max = max(max(inCol.r, inCol.g), inCol.b);
        dlt = _max - _min + 0.00001 /*Hack to fix divide zero infinities*/;
        h = 0.0;
        v = _max;

        br1 = step(_max, 0.0);
        s = (dlt / _max) * (1.0 - br1);
        h = -1.0 * br1;

        br2 = 1.0 - step(_max - inCol.r, 0.0); 
        br2_or_br1 = max(br2, br1);
        h = ((inCol.g - inCol.b) / dlt) * (1.0 - br2_or_br1) + (h*br2_or_br1);

        br3 = 1.0 - step(_max - inCol.g, 0.0); 
          
        br3_or_br1 = max(br3, br1);
        h = (2.0 + (inCol.b - inCol.r) / dlt) * (1.0 - br3_or_br1) + (h*br3_or_br1);

        br4 = 1.0 - br2*br3;
        br4_or_br1 = max(br4, br1);
        h = (4.0 + (inCol.r - inCol.g) / dlt) * (1.0 - br4_or_br1) + (h*br4_or_br1);

        h = h*(1.0 - br1);

        hue_a = abs(h); // between h of -1 and 1 are skin tones
        a = dlt;      // Reducing enhancements on small rgb differences

        // Reduce the enhancements on skin tones.    
        a = step(1.0, hue_a) * a * (hue_a * 0.67 + 0.33) + step(hue_a, 1.0) * a;                                    
        a *= (vibrance - 1.0);
        s = (1.0 - a) * s + a * pow(s, 0.25);

        i = floor(h);
        f = h - i;

        p1 = v * (1.0 - s);
        p2 = v * (1.0 - (s * f));
        p3 = v * (1.0 - (s * (1.0 - f)));

        inCol = vec3(0.0); 
        i += 6.0;
        use = int(pow(2.0,mod(i,6.0)));
        a = float(and(use , 1)); // i == 0;
        use /= 2;
        inCol += a * vec3(v, p3, p1);

        a = float(and(use , 1)); // i == 1;
        use /= 2;
        inCol += a * vec3(p2, v, p1); 

        a = float( and(use,1)); // i == 2;
        use /= 2;
        inCol += a * vec3(p1, v, p3);

        a = float(and(use, 1)); // i == 3;
        use /= 2;
        inCol += a * vec3(p1, p2, v);

        a = float(and(use, 1)); // i == 4;
        use /= 2;
        inCol += a * vec3(p3, p1, v);

        a = float(and(use, 1)); // i == 5;
        use /= 2;
        inCol += a * vec3(v, p1, p2);

        outCol = inCol;
      }
      return outCol;
    }

    float rgb2s4hsl(vec3 rgb) {
      float r = rgb[0];
      float g = rgb[1];
      float b = rgb[2];
      float M = max(r, max(g, b));
      float m = min(r, min(g, b));

      if (M - m < 1e-10) return 0.0;
      return (M - m) / (1.0 - abs(M + m - 1.0));
    }

    float rgb2L(vec3 rgb) {
      float m = min(min(rgb.x, rgb.y), rgb.z);
      float M = max(max(rgb.x, rgb.y), rgb.z);

      return (M + m) * 0.5;
    }

    vec3 rgb2hsl(vec3 rgb) {
      vec3 hsl;

      hsl.x = rgb2h(rgb);
      hsl.y = rgb2s4hsl(rgb);
      hsl.z = rgb2L(rgb);

      return hsl;
    }

    float h2rgb(float f1, float f2, float hue) {
      if (hue < 0.0)
        hue += 1.0;
      else if (hue >= 1.0)
        hue -= 1.0;

      float res;
      if ((6.0 * hue) < 1.0)
        res = f1 + (f2 - f1) * 6.0 * hue;
      else if ((2.0 * hue) < 1.0)
        res = f2;
      else if ((3.0 * hue) < 2.0)
        res = f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
      else
        res = f1;

      return res;
    }

    vec3 hsl2rgb(vec3 hsl) {
      vec3 rgb;

      if (hsl.y == 0.0) {
        rgb = vec3(hsl.z, hsl.z, hsl.z); // Luminance
      } else {
        float f2;

        if (hsl.z < 0.5) {
          f2 = hsl.z * (1.0 + hsl.y);
        } else {
          f2 = (hsl.z + hsl.y) - (hsl.y * hsl.z);
        }

        float f1 = 2.0 * hsl.z - f2;

        rgb.x = h2rgb(f1, f2, hsl.x + (1.0 / 3.0));
        rgb.y = h2rgb(f1, f2, hsl.x);
        rgb.z = h2rgb(f1, f2, hsl.x - (1.0 / 3.0));
      }

      return rgb;
    }

    vec3 colormix_filter(vec3 color, vec3 hsl_ratios, float center_h, float h_range) {
      vec3 hsl = rgb2hsl(color);
      float dist = min(abs(center_h - hsl.x), abs(center_h - (hsl.x - 1.0)));
      float weight = 1.0 - clamp(dist / h_range, 0.0, 1.0);
      float new_h = hsl.x + hsl_ratios.x * h_range;
      float new_s = clamp(hsl.y * (hsl_ratios.y + 1.0), 0.0, 1.0);
      float new_l = clamp(hsl.z * (hsl_ratios.z + 1.0), 0.0, 1.0);
      vec3 new_hsl = mix(hsl, vec3(new_h, new_s, new_l), weight);
      return hsl2rgb(new_hsl);
    }

    void main() {
      if (radial_blur != 0.0) {
        gl_FragColor = blur_radial(frame, sample_coordinate, radial_blur);
      } else if (sharpen != 0.0) {
        gl_FragColor = vec4(sharpen_filter(frame, sample_coordinate, size, sharpen), gl_FragColor.a);
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

      gl_FragColor = vec4(exposure_filter(gl_FragColor.rgb, exposure), gl_FragColor.a);
      gl_FragColor = vec4(contrast_filter(gl_FragColor.rgb, contrast), gl_FragColor.a);
      gl_FragColor = vec4(temperature_tint_filter(gl_FragColor.rgb, temperature, tint), gl_FragColor.a);
      gl_FragColor = vec4(saturation_filter(gl_FragColor.rgb, saturation), gl_FragColor.a);
      gl_FragColor = vec4(highlight_filter(gl_FragColor.rgb, highlight), gl_FragColor.a);
      gl_FragColor = vec4(shadow_filter(gl_FragColor.rgb, shadow), gl_FragColor.a);
      gl_FragColor = vec4(vibrance_filter(gl_FragColor.rgb, vibrance), gl_FragColor.a);
      gl_FragColor = vec4(colormix_filter(gl_FragColor.rgb, red_mix, 0.0, 30.0/360.0), gl_FragColor.a);
      gl_FragColor = vec4(colormix_filter(gl_FragColor.rgb, orange_mix, 30.0/360.0, 30.0/360.0), gl_FragColor.a);
      gl_FragColor = vec4(colormix_filter(gl_FragColor.rgb, yellow_mix, 60.0/360.0, 30.0/360.0), gl_FragColor.a);
      gl_FragColor = vec4(colormix_filter(gl_FragColor.rgb, green_mix, 120.0/360.0, 60.0/360.0), gl_FragColor.a);
      gl_FragColor = vec4(colormix_filter(gl_FragColor.rgb, blue_mix, 210.0/360.0, 60.0/360.0), gl_FragColor.a);
      gl_FragColor = vec4(colormix_filter(gl_FragColor.rgb, purple_mix, 300.0/360.0, 60.0/360.0), gl_FragColor.a);

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

      if (apply_gamma == 1) {
        gl_FragColor = vec4(pow(gl_FragColor.rgb, vec3(1.2)), gl_FragColor.a);
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
    glUniform1f(glGetUniformLocation(program_, "exposure"), color_lut.exposure());
    glUniform1f(glGetUniformLocation(program_, "contrast"), color_lut.contrast());
    glUniform1f(glGetUniformLocation(program_, "temperature"), color_lut.temperature());
    glUniform1f(glGetUniformLocation(program_, "tint"), color_lut.tint());
    glUniform1f(glGetUniformLocation(program_, "saturation"), color_lut.saturation());
    glUniform1f(glGetUniformLocation(program_, "highlight"), color_lut.highlight());
    glUniform1f(glGetUniformLocation(program_, "shadow"), color_lut.shadow());
    glUniform1f(glGetUniformLocation(program_, "sharpen"), color_lut.sharpen());
    glUniform1f(glGetUniformLocation(program_, "vibrance"), color_lut.vibrance());

    glUniform3f(glGetUniformLocation(program_, "red_mix"), color_lut.red_mix_h(), color_lut.red_mix_s(), color_lut.red_mix_l());
    glUniform3f(glGetUniformLocation(program_, "orange_mix"), color_lut.orange_mix_h(), color_lut.orange_mix_s(), color_lut.orange_mix_l());
    glUniform3f(glGetUniformLocation(program_, "yellow_mix"), color_lut.yellow_mix_h(), color_lut.yellow_mix_s(), color_lut.yellow_mix_l());
    glUniform3f(glGetUniformLocation(program_, "green_mix"), color_lut.green_mix_h(), color_lut.green_mix_s(), color_lut.green_mix_l());
    glUniform3f(glGetUniformLocation(program_, "blue_mix"), color_lut.blue_mix_h(), color_lut.blue_mix_s(), color_lut.blue_mix_l());
    glUniform3f(glGetUniformLocation(program_, "purple_mix"), color_lut.purple_mix_h(), color_lut.purple_mix_s(), color_lut.purple_mix_l());

    bool apply_gamma = cc->InputSidePackets().Tag(kApplyGammaTag).Get<bool>();
    glUniform1i(glGetUniformLocation(program_, "apply_gamma"), apply_gamma ? 1 : 0);

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
