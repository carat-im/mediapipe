#import "CaratMediapipeGraph.h"
#import "mediapipe/objc/MPPGraph.h"

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/carat/formats/carat_face_effect.pb.h"
#include "mediapipe/carat/formats/color_lut.pb.h"
#include "mediapipe/carat/formats/carat_frame_effect.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"

static NSString* const kGraphName = @"carat_mediapipe_graph";

static const char* kInputStream = "input_video";
static const char* kOutputStream = "output_video";

static const char* kNumFacesInputSidePacket = "num_faces";
static const char* kCaratFaceEffectListInputStream = "carat_face_effect_list";
static const char* kColorLutInputStream = "color_lut";
static const char* kCaratFrameEffectListInputStream = "carat_frame_effect_list";

static const char* kLandmarksOutputStream = "multi_face_landmarks";
static const char* kMultiFaceGeometryStream = "multi_face_geometry";

static const char* kApplyGammaInputSidePacket = "apply_gamma";

static const int kNumFaces = 5;

@interface CaratMediapipeGraph() <MPPGraphDelegate>
@property(nonatomic) MPPGraph* mediapipeGraph;
@end

@implementation CaratMediapipeGraph {
  CMTime _lastTimestamp;

  NSString *_lutFilePath;
  float _lutIntensity;
  float _lutGrain;
  float _lutVignette;
  float _radialBlur;
  float _rgbSplit;
  NSString *_blendImagePath1;
  int _blendMode1;
  NSString *_blendImagePath2;
  int _blendMode2;
  float _exposure;
  float _contrast;
  float _temperature;
  float _tint;
  float _saturation;
  float _highlight;
  float _shadow;
  float _sharpen;
  float _vibrance;
  NSArray *_redMix;
  NSArray *_orangeMix;
  NSArray *_yellowMix;
  NSArray *_greenMix;
  NSArray *_blueMix;
  NSArray *_purpleMix;
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

- (instancetype)initWithApplyGamma:(bool)applyGamma {
    self = [super init];
    if (self) {
        self.mediapipeGraph = [[self class] loadGraphFromResource:kGraphName];
        self.mediapipeGraph.delegate = self;
        self.mediapipeGraph.maxFramesInFlight = 2;

        [self.mediapipeGraph addFrameOutputStream:kOutputStream outputPacketType:MPPPacketTypePixelBuffer];
        [self.mediapipeGraph addFrameOutputStream:kLandmarksOutputStream outputPacketType:MPPPacketTypeRaw];
        [self.mediapipeGraph addFrameOutputStream:kMultiFaceGeometryStream outputPacketType:MPPPacketTypeRaw];
        [self.mediapipeGraph setSidePacket:(mediapipe::MakePacket<int>(kNumFaces)) named:kNumFacesInputSidePacket];
        [self.mediapipeGraph setSidePacket:(mediapipe::MakePacket<bool>(applyGamma)) named:kApplyGammaInputSidePacket];

        self.caratFaceEffectListString = @"";
        self.colorLutString = @"";
        self.caratFrameEffectListString = @"";
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

    const mediapipe::CaratFrameEffectList& caratFrameEffectList = mediapipe::ParseTextProtoOrDie<mediapipe::CaratFrameEffectList>([self.caratFrameEffectListString UTF8String]);
    mediapipe::Packet caratFrameEffectListPacket =
        mediapipe::MakePacket<mediapipe::CaratFrameEffectList>(caratFrameEffectList).At(graphTimestamp);
    [self.mediapipeGraph movePacket:std::move(caratFrameEffectListPacket) intoStream:kCaratFrameEffectListInputStream error:nil];
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

- (void)setColorLut:(NSString *)filePath intensity:(float)intensity grain:(float)grain vignette:(float)vignette
  radialBlur:(float)radialBlur 
  rgbSplit:(float)rgbSplit
  blendImagePath1:(NSString *)blendImagePath1 blendMode1:(int)blendMode1
  blendImagePath2:(NSString *)blendImagePath2 blendMode2:(int)blendMode2
  exposure:(float)exposure contrast:(float)contrast
  temperature:(float)temperature tint:(float)tint
  saturation:(float)saturation
  highlight:(float)highlight shadow:(float)shadow
  sharpen:(float)sharpen
  vibrance:(float)vibrance 
  redMix:(NSArray *)redMix orangeMix:(NSArray *)orangeMix yellowMix:(NSArray *)yellowMix
  greenMix:(NSArray *)greenMix blueMix:(NSArray *)blueMix purpleMix:(NSArray *)purpleMix {
  if (!filePath || filePath == [NSNull null]) {
    _lutFilePath = nil;
  } else {
    _lutFilePath = filePath;
  }

  _lutIntensity = intensity;
  _lutGrain = grain;
  _lutVignette = vignette;
  _radialBlur = radialBlur;
  _rgbSplit = rgbSplit;
  if (!blendImagePath1 || blendImagePath1 == [NSNull null]) {
    _blendImagePath1 = nil;
  } else {
    _blendImagePath1 = blendImagePath1;
  }
  _blendMode1 = blendMode1;
  if (!blendImagePath2 || blendImagePath2 == [NSNull null]) {
    _blendImagePath2 = nil;
  } else {
    _blendImagePath2 = blendImagePath2;
  }
  _blendMode2 = blendMode2;

  _exposure = exposure;
  _contrast = contrast;
  _temperature = temperature;
  _tint = tint;
  _saturation = saturation;
  _highlight = highlight;
  _shadow = shadow;
  _sharpen = sharpen;
  _vibrance = vibrance;
  _redMix = [redMix count] == 3 ? redMix : @[@0, @0, @0];
  _orangeMix = [orangeMix count] == 3 ? orangeMix : @[@0, @0, @0];
  _yellowMix = [yellowMix count] == 3 ? yellowMix : @[@0, @0, @0];
  _greenMix = [greenMix count] == 3 ? greenMix : @[@0, @0, @0];
  _blueMix = [blueMix count] == 3 ? blueMix : @[@0, @0, @0];
  _purpleMix = [purpleMix count] == 3 ? purpleMix : @[@0, @0, @0];

  [self makeColorLutString];
}

- (void)setFrameEffects:(NSArray *)effects {
  int size = [effects count];
  if (size == 0) {
    self.caratFrameEffectListString = @"";
    return;
  }

  NSString *res = @"";
  for (int i = 0; i < size; i++) {
    NSString *row = [NSString stringWithFormat:@"\neffect { id: %d texture_path: \"%@\" }", ((NSString *)effects[i]).hash, effects[i]];
    res = [res stringByAppendingString:row];
  }

  self.caratFrameEffectListString = res;
}

- (void)makeColorLutString {
  self.colorLutString = [NSString stringWithFormat:@"intensity: %f grain: %f vignette: %f \
  radial_blur: %f rgb_split: %f \
  exposure: %f contrast: %f temperature: %f tint: %f saturation: %f highlight: %f shadow: %f sharpen: %f vibrance: %f \
  red_mix_h: %f red_mix_s: %f red_mix_l: %f \
  orange_mix_h: %f orange_mix_s: %f orange_mix_l: %f \
  yellow_mix_h: %f yellow_mix_s: %f yellow_mix_l: %f \
  green_mix_h: %f green_mix_s: %f green_mix_l: %f \
  blue_mix_h: %f blue_mix_s: %f blue_mix_l: %f \
  purple_mix_h: %f purple_mix_s: %f purple_mix_l: %f",
    _lutIntensity, _lutGrain, _lutVignette,
    _radialBlur, _rgbSplit,
    _exposure, _contrast, _temperature, _tint, _saturation, _highlight, _shadow, _sharpen, _vibrance,
    ((NSNumber *)_redMix[0]).floatValue, ((NSNumber *)_redMix[1]).floatValue, ((NSNumber *)_redMix[2]).floatValue,
    ((NSNumber *)_orangeMix[0]).floatValue, ((NSNumber *)_orangeMix[1]).floatValue, ((NSNumber *)_orangeMix[2]).floatValue,
    ((NSNumber *)_yellowMix[0]).floatValue, ((NSNumber *)_yellowMix[1]).floatValue, ((NSNumber *)_yellowMix[2]).floatValue,
    ((NSNumber *)_greenMix[0]).floatValue, ((NSNumber *)_greenMix[1]).floatValue, ((NSNumber *)_greenMix[2]).floatValue,
    ((NSNumber *)_blueMix[0]).floatValue, ((NSNumber *)_blueMix[1]).floatValue, ((NSNumber *)_blueMix[2]).floatValue,
    ((NSNumber *)_purpleMix[0]).floatValue, ((NSNumber *)_purpleMix[1]).floatValue, ((NSNumber *)_purpleMix[2]).floatValue];

  if (_lutFilePath != nil) {
    self.colorLutString = [self.colorLutString stringByAppendingString:[NSString stringWithFormat:@" lut_path: \"%@\"", _lutFilePath]];
  }

  if (_blendImagePath1 != nil) {
    self.colorLutString = [self.colorLutString stringByAppendingString:[NSString stringWithFormat:@" blend_image_path_1: \"%@\" blend_mode_1: %d", _blendImagePath1, _blendMode1]];
  }

  if (_blendImagePath2 != nil) {
    self.colorLutString = [self.colorLutString stringByAppendingString:[NSString stringWithFormat:@" blend_image_path_2: \"%@\" blend_mode_2: %d", _blendImagePath2, _blendMode2]];
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
