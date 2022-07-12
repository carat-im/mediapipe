#! /bin/bash

## iOS ##

bazel build -c opt --config=ios_fat mediapipe/carat:CaratMediapipeGraph
unzip bazel-out/applebin_ios-ios_armv7-opt-ST-b90ba0b72457/bin/mediapipe/carat/CaratMediapipeGraph.zip -d mediapipe/carat
cp -rf mediapipe/carat/CaratMediapipeGraph.framework ../plugins/packages/camera/camera/ios/
cp -rf mediapipe/carat/CaratMediapipeGraph.framework ../ios
cd ../ios
pod install
cd ../mediapipe

# 처음이라면 추가적으로 XCode에서 Runner에 add to files로 파일 인식할 수 있게 추가해주기,
# camera.podspec에 vendored_frameworks 추가해주기.

## Android ##

bazel build -c opt --strip=ALWAYS \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --fat_apk_cpu=arm64-v8a,armeabi-v7a \
    --legacy_whole_archive=0 \
    --features=-legacy_whole_archive \
    --copt=-fvisibility=hidden \
    --copt=-ffunction-sections \
    --copt=-fdata-sections \
    --copt=-fstack-protector \
    --copt=-Oz \
    --copt=-fomit-frame-pointer \
    --copt=-DABSL_MIN_LOG_LEVEL=2 \
    --linkopt=-Wl,--gc-sections,--strip-all \
    //mediapipe/carat/android/src/java/com/carat/camera/caratmediapipegraph/apps/carat:carat_mediapipe.aar
bazel build -c opt mediapipe/graphs/face_mesh:carat_face_mesh_mobile_gpu_binary_graph

mkdir -p ../android/app/libs
mkdir -p ../android/app/src/main/assets
cp -rf bazel-bin/mediapipe/carat/android/src/java/com/carat/camera/caratmediapipegraph/apps/carat/carat_mediapipe.aar ../android/app/libs
cp -rf bazel-bin/mediapipe/graphs/face_mesh/carat_face_mesh_mobile_gpu.binarypb ../android/app/src/main/assets
cp -rf mediapipe/modules/face_landmark/face_landmark_with_attention.tflite ../android/app/src/main/assets
cp -rf mediapipe/modules/face_detection/face_detection_short_range.tflite ../android/app/src/main/assets

mkdir -p ../plugins/packages/camera/camera/android/libs
mkdir -p ../plugins/packages/camera/camera/android/src/main/assets
cp -rf bazel-bin/mediapipe/carat/android/src/java/com/carat/camera/caratmediapipegraph/apps/carat/carat_mediapipe.aar ../plugins/packages/camera/camera/android/libs
cp -rf bazel-bin/mediapipe/graphs/face_mesh/carat_face_mesh_mobile_gpu.binarypb ../plugins/packages/camera/camera/android/src/main/assets
cp -rf mediapipe/modules/face_landmark/face_landmark_with_attention.tflite ../plugins/packages/camera/camera/android/src/main/assets
cp -rf mediapipe/modules/face_detection/face_detection_short_range.tflite ../plugins/packages/camera/camera/android/src/main/assets

# 처음이라면 app/build.gradle에 dependency 추가해주기. https://google.github.io/mediapipe/getting_started/android_archive_library.html
# camera의 build.gradle에도 dependency 추가.
