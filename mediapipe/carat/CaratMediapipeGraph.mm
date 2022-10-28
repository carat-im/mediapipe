#import "CaratMediapipeGraph.h"
#import "mediapipe/objc/MPPGraph.h"

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/carat/formats/carat_face_effect.pb.h"
#include "mediapipe/carat/formats/color_lut.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"

static NSString* const kGraphName = @"carat_mediapipe_graph";

static const char* kInputStream = "input_video";
static const char* kOutputStream = "output_video";

static const char* kNumFacesInputSidePacket = "num_faces";
static const char* kCaratFaceEffectListInputStream = "carat_face_effect_list";
static const char* kColorLutInputStream = "color_lut";

static const char* kLandmarksOutputStream = "multi_face_landmarks";
static const char* kMultiFaceGeometryStream = "multi_face_geometry";

static const int kNumFaces = 5;

@interface CaratMediapipeGraph() <MPPGraphDelegate>
@property(nonatomic) MPPGraph* mediapipeGraph;
@property(nonatomic) NSString *caratFaceEffectListString;
@property(nonatomic) NSString *colorLutString;
@end

@implementation CaratMediapipeGraph {
  CMTime _lastTimestamp;

  NSString *_lutFilePath;
  float _lutIntensity;
  float _lutGrain;
  float _lutVignette;
}

#pragma mark - Cleanup methods

- (void)dealloc {
    self.mediapipeGraph.delegate = nil;
    [self.mediapipeGraph cancel];
    [self.mediapipeGraph closeAllInputStreamsWithError:nil];
    [self.mediapipeGraph waitUntilDoneWithError:nil];
}

#pragma mark - Mediapipe graph methods

+ (MPPGraph*)loadGraphFromResource:(NSString*)resource {
    NSError* configLoadError = nil;
    NSBundle* bundle = [NSBundle bundleForClass:[self class]];
    if (!resource || resource.length == 0) { return nil; }

    NSURL* graphURL = [bundle URLForResource:resource withExtension:@"binarypb"];
    NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:&configLoadError];
    if (!data) {
        NSLog(@"Failed to load graph config: %@", configLoadError);
        return nil;
    }

    mediapipe::CalculatorGraphConfig config;
    config.ParseFromArray(data.bytes, data.length);

    MPPGraph* newGraph = [[MPPGraph alloc] initWithGraphConfig:config];
    return newGraph;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        self.mediapipeGraph = [[self class] loadGraphFromResource:kGraphName];
        self.mediapipeGraph.delegate = self;
        self.mediapipeGraph.maxFramesInFlight = 2;

        [self.mediapipeGraph addFrameOutputStream:kOutputStream outputPacketType:MPPPacketTypePixelBuffer];
        [self.mediapipeGraph addFrameOutputStream:kLandmarksOutputStream outputPacketType:MPPPacketTypeRaw];
        [self.mediapipeGraph addFrameOutputStream:kMultiFaceGeometryStream outputPacketType:MPPPacketTypeRaw];
        [self.mediapipeGraph setSidePacket:(mediapipe::MakePacket<int>(kNumFaces)) named:kNumFacesInputSidePacket];

        self.caratFaceEffectListString = @"";
        self.colorLutString = @"";
    }
    return self;
}

- (void)startGraph {
    NSError* error;
    if (![self.mediapipeGraph startWithError:&error]) {
        NSLog(@"Failed to start graph: %@", error);
    } else if (![self.mediapipeGraph waitUntilIdleWithError:&error]) {
        NSLog(@"Failed to complete graph initial run: %@", error);
    }
}

- (void)sendPixelBuffer:(CVPixelBufferRef)pixelBuffer timestamp:(CMTime)timestamp {
    if (CMTimeCompare(_lastTimestamp, timestamp) == 0) {
      return;
    }
    _lastTimestamp = timestamp;

    mediapipe::Timestamp graphTimestamp(static_cast<mediapipe::TimestampBaseType>(
        mediapipe::Timestamp::kTimestampUnitsPerSecond * CMTimeGetSeconds(timestamp)));

    [self.mediapipeGraph sendPixelBuffer:pixelBuffer intoStream:kInputStream packetType:MPPPacketTypePixelBuffer timestamp:graphTimestamp];

    const mediapipe::CaratFaceEffectList& caratFaceEffectList = mediapipe::ParseTextProtoOrDie<mediapipe::CaratFaceEffectList>([self.caratFaceEffectListString UTF8String]);
    mediapipe::Packet caratFaceEffectListPacket =
        mediapipe::MakePacket<mediapipe::CaratFaceEffectList>(caratFaceEffectList).At(graphTimestamp);
    [self.mediapipeGraph movePacket:std::move(caratFaceEffectListPacket) intoStream:kCaratFaceEffectListInputStream error:nil];

    const mediapipe::ColorLut& colorLut = mediapipe::ParseTextProtoOrDie<mediapipe::ColorLut>([self.colorLutString UTF8String]);
    mediapipe::Packet colorLutPacket =
        mediapipe::MakePacket<mediapipe::ColorLut>(colorLut).At(graphTimestamp);
    [self.mediapipeGraph movePacket:std::move(colorLutPacket) intoStream:kColorLutInputStream error:nil];
}

- (void)waitUntilIdle {
  [self.mediapipeGraph waitUntilIdleWithError:nil];
}

- (void)setFaceEffects:(NSArray *)effects {
  int size = [effects count];
  if (size == 0) {
    self.caratFaceEffectListString = @"";
    return;
  }

  NSString *res = @"";
  for (int i = 0; i < size; i++) {
    if (i + 1 < size && [effects[i+1] hasSuffix:@"binarypb"]) {
      NSString *row = [NSString stringWithFormat:@"\neffect { id: %d texture_path: \"%@\" mesh_3d_path: \"%@\" }", ((NSString *)effects[i]).hash, effects[i], effects[i+1]];
      res = [res stringByAppendingString:row];
      i = i + 1;
    } else {
      NSString *row = [NSString stringWithFormat:@"\neffect { id: %d texture_path: \"%@\" }", ((NSString *)effects[i]).hash, effects[i]];
      res = [res stringByAppendingString:row];
    }
  }

  self.caratFaceEffectListString = res;
}

- (void)setColorLut:(NSString *)filePath intensity:(NSNumber *)intensity grain:(NSNumber *)grain vignette:(NSNumber *)vignette {
  if (filePath == [NSNull null]) {
    _lutFilePath = nil;
  } else {
    _lutFilePath = filePath;
  }

  _lutIntensity = intensity == [NSNull null] ? 0 : intensity.floatValue;
  _lutGrain = grain == [NSNull null] ? 0 : grain.floatValue;
  _lutVignette = vignette == [NSNull null] ? 0 : vignette.floatValue;

  [self makeColorLutString];
}

- (void)setColorLutIntensity:(NSNumber *)intensity {
  _lutIntensity = intensity == [NSNull null] ? 0 : intensity.floatValue;

  [self makeColorLutString];
}

- (void)makeColorLutString {
  if (_lutFilePath == nil) {
    self.colorLutString = @"";
  } else {
    self.colorLutString = [NSString stringWithFormat:@"lut_path: \"%@\" intensity: %f grain: %f vignette: %f", _lutFilePath, _lutIntensity, _lutGrain, _lutVignette];
  }
}

#pragma mark - MPPGraphDelegate methods

// Invoked on a Mediapipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer fromStream:(const std::string&)streamName {
    if (streamName == kOutputStream) {
        [_delegate graph:self didOutputPixelBuffer:pixelBuffer];
    }
}

// Invoked on a Mediapipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph didOutputPacket:(const ::mediapipe::Packet&)packet fromStream:(const std::string&)streamName {
    if (streamName == kLandmarksOutputStream) {
        // something.
    } else if (streamName == kMultiFaceGeometryStream) {
        // something.
    }
}

@end
