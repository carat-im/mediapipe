load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")

ios_framework(
    name = "CaratMediapipeGraph",
    hdrs = [
        "CaratMediapipeGraph.h",
    ],
    infoplists = ["Info.plist"],
    bundle_id = "com.carat.camera.CaratMediapipeGraph",
    families = ["iphone", "ipad"],
    minimum_os_version = "10.0",
    deps = [
        ":Library",
        "@ios_opencv//:OpencvFramework",
    ],
)

objc_library(
    name = "Library",
    srcs = [
        "CaratMediapipeGraph.mm",
    ],
    hdrs = [
        "CaratMediapipeGraph.h",
    ],
    copts = ["-std=c++17"],
    data = [
        "//mediapipe/graphs/face_mesh:carat_face_mesh_mobile_gpu.binarypb",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
    ],
    sdk_frameworks = [
        "AVFoundation",
        "CoreGraphics",
        "CoreMedia",
        "UIKit"
    ],
    deps = [
        "//mediapipe/objc:mediapipe_framework_ios",
    ] + select({
        "//conditions:default": [
            "//mediapipe/graphs/face_mesh:mobile_calculators",
            "//mediapipe/framework/formats:landmark_cc_proto",
        ],
    }),
)
