load("//mediapipe/framework:encode_binary_proto.bzl", "encode_binary_proto")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

[encode_binary_proto(
    name = data.split("/")[-1].rsplit(".", 1)[0],
    input = data,
    message_type = "mediapipe.face_geometry.Mesh3d",
    output = data.rsplit(".", 1)[0] + ".binarypb",
    deps = [
        "//mediapipe/modules/face_geometry/protos:mesh_3d_proto",
    ],
) for data in glob(["*.pbtxt"])]

# `.pngblob` is used instead of `.png` to prevent iOS build from preprocessing the image.
# OpenCV is unable to read a PNG file preprocessed by the iOS build.
exports_files(glob(["*.pngblob"]))
