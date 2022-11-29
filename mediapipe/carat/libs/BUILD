licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "frame_effect_renderer",
    srcs = ["frame_effect_renderer.cc"],
    hdrs = ["frame_effect_renderer.h"],
    deps = [
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:matrix_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/gpu:gl_base",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:shader_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "gl_errors",
    srcs = ["gl_errors.cc"],
    hdrs = ["gl_errors.h"],
    deps = [
        "//mediapipe/framework/port:status",
    ],
)
