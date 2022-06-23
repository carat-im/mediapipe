#! /bin/bash

bazel build -c opt --config=ios_fat mediapipe/carat:CaratMediapipeGraph
unzip bazel-out/applebin_ios-ios_armv7-opt-ST-b90ba0b72457/bin/mediapipe/carat/CaratMediapipeGraph.zip -d mediapipe/carat
cp -r mediapipe/carat/CaratMediapipeGraph.framework ../plugins/packages/camera/camera/ios/
cp -r mediapipe/carat/CaratMediapipeGraph.framework ../ios
cd ../ios
pod install
cd ../mediapipe
