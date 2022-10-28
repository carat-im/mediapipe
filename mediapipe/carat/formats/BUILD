load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")
load("//mediapipe/framework:mediapipe_register_type.bzl", "mediapipe_register_type")

package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
)

licenses(["notice"])

filegroup(
    name = "protos_src",
    srcs = glob(["*.proto"]),
    visibility = ["//mediapipe:__subpackages__"],
)

mediapipe_proto_library(
    name = "carat_face_effect_proto",
    srcs = ["carat_face_effect.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_register_type(
    base_name = "carat_face_effect",
    include_headers = ["mediapipe/carat/formats/carat_face_effect.pb.h"],
    types = [
        "::mediapipe::CaratFaceEffect",
        "::mediapipe::CaratFaceEffectList",
        "::mediapipe::CaratFaceEffectListCollection",
        "::std::vector<::mediapipe::CaratFaceEffect>",
        "::std::vector<::mediapipe::CaratFaceEffectList>",
    ],
    deps = [":carat_face_effect_cc_proto"],
)

mediapipe_proto_library(
    name = "color_lut_proto",
    srcs = ["color_lut.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_register_type(
    base_name = "color_lut",
    include_headers = ["mediapipe/carat/formats/color_lut.pb.h"],
    types = [
        "::mediapipe::ColorLut",
    ],
    deps = [":color_lut_cc_proto"],
)
