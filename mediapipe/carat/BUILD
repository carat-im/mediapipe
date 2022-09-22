load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_framework",
 )

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
 )

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
        "//mediapipe/carat:carat_mediapipe_graph.binarypb",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
        "//mediapipe/graphs/face_effect/data:axis.binarypb",
        "//mediapipe/graphs/face_effect/data:axis.pngblob",
        "//mediapipe/graphs/face_effect/data:facepaint.pngblob",
        "//mediapipe/graphs/face_effect/data:glasses.binarypb",
        "//mediapipe/graphs/face_effect/data:glasses.pngblob",
        "//mediapipe/carat/data:dino.pngblob",
        "//mediapipe/carat/data:dino.binarypb",
        "//mediapipe/carat/data:luffy.pngblob",
        "//mediapipe/carat/data:microphone.pngblob",
        "//mediapipe/carat/data:microphone.binarypb",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata.binarypb",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata_detection.binarypb",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata_landmarks.binarypb",
    ],
    features = ["-layering_check"],
    sdk_frameworks = [
        "AVFoundation",
        "CoreGraphics",
        "CoreMedia",
        "UIKit"
    ],
    deps = [
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:mediapipe_input_sources_ios",
        "//mediapipe/objc:mediapipe_layer_renderer",
    ] + select({
        "//mediapipe:ios_i386": [],
        "//mediapipe:ios_x86_64": [],
        "//conditions:default": [
            "//mediapipe/graphs/face_mesh:mobile_calculators",
            "//mediapipe/graphs/face_effect:face_effect_gpu_deps",
            "//mediapipe/framework/formats:landmark_cc_proto",
            "//mediapipe/framework/formats:matrix_data_cc_proto",
            "//mediapipe/framework/port:parse_text_proto",
            "//mediapipe/carat/formats:carat_face_effect_cc_proto",
            "//mediapipe/modules/face_geometry/protos:face_geometry_cc_proto",
            "//mediapipe/carat/calculators:carat_face_effect_renderer_calculator",
        ],
    }),
)

mediapipe_binary_graph(
    name = "carat_mediapipe_graph",
    graph = "carat_mediapipe_graph.pbtxt",
    output_name = "carat_mediapipe_graph.binarypb",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:immediate_mux_calculator",
        "//mediapipe/calculators/core:split_proto_list_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
        "//mediapipe/carat/calculators:carat_face_effect_renderer_calculator",
        "//mediapipe/modules/face_geometry:env_generator_calculator",
        "//mediapipe/modules/face_geometry:face_geometry_from_landmarks",
        "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
    ],
)
