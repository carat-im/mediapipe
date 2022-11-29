load("//mediapipe/java/com/google/mediapipe:mediapipe_aar.bzl", "mediapipe_aar")

mediapipe_aar(
    name = "carat_mediapipe",
    calculators = [
        "//mediapipe/graphs/face_mesh:mobile_calculators",
        "//mediapipe/modules/face_geometry:env_generator_calculator",
        "//mediapipe/modules/face_geometry:face_geometry_from_landmarks",
        "//mediapipe/carat/calculators:carat_face_effect_renderer_calculator",
        "//mediapipe/carat/calculators:color_lut_filter_calculator",
        "//mediapipe/carat/calculators:carat_frame_effect_renderer_calculator",
    ],
)
