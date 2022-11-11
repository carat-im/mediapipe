#! /bin/bash

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
bazel build -c opt mediapipe/carat:carat_mediapipe_graph

mkdir -p ../android/app/src/main/assets
cp -rf bazel-bin/mediapipe/carat/android/src/java/com/carat/camera/caratmediapipegraph/apps/carat/carat_mediapipe.aar ../android/mediapipe
cp -rf bazel-bin/mediapipe/carat/carat_mediapipe_graph.binarypb ../android/app/src/main/assets

mkdir -p ../plugins/packages/camera/camera/android/src/main/assets
cp -rf bazel-bin/mediapipe/carat/android/src/java/com/carat/camera/caratmediapipegraph/apps/carat/carat_mediapipe.aar ../plugins/packages/camera/camera/android/mediapipe
cp -rf bazel-bin/mediapipe/carat/carat_mediapipe_graph.binarypb ../plugins/packages/camera/camera/android/src/main/assets

# 처음이라면 app/build.gradle에 dependency 추가해주기. https://google.github.io/mediapipe/getting_started/android_archive_library.html
# camera의 build.gradle에도 dependency 추가.

