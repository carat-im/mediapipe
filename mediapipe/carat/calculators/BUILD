licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "carat_face_effect_renderer_calculator",
    srcs = ["carat_face_effect_renderer_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/carat/formats:carat_face_effect_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/modules/face_geometry/libs:effect_renderer",
        "//mediapipe/modules/face_geometry/libs:validation_utils",
        "//mediapipe/modules/face_geometry/protos:environment_cc_proto",
        "//mediapipe/modules/face_geometry/protos:face_geometry_cc_proto",
        "//mediapipe/modules/face_geometry/protos:mesh_3d_cc_proto",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library(
    name = "color_lut_filter_calculator",
    srcs = ["color_lut_filter_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/carat/formats:color_lut_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:shader_util",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library(
    name = "carat_frame_effect_renderer_calculator",
    srcs = ["carat_frame_effect_renderer_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:shader_util",
        "//mediapipe/carat/formats:carat_frame_effect_cc_proto",
    ],
    alwayslink = 1,
)
