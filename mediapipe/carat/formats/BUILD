load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")
load("//mediapipe/framework:mediapipe_register_type.bzl", "mediapipe_register_type")

package(
    default_visibility = ["//visibility:private"],
    features = ["-layering_check"],
)

licenses(["notice"])

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
        "::mediapipe::CaratFaceEffectSet",
        "::std::vector<::mediapipe::CaratFaceEffect>",
    ],
    deps = [":carat_face_effect_cc_proto"],
)