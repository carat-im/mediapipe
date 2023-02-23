#! /bin/bash

bazel build -c opt --config=ios_arm64 mediapipe/carat:CaratMediapipeGraph
unzip bazel-out/applebin_ios-ios_arm64-opt-ST-dabdae81af22/bin/mediapipe/carat/CaratMediapipeGraph.zip -d mediapipe/carat
cp -rf mediapipe/carat/CaratMediapipeGraph.framework ../ios
#cd ../ios
#pod install
#cd ../mediapipe

# 처음이라면 추가적으로 XCode에서 Runner에 add to files로 파일 인식할 수 있게 추가해주기,
