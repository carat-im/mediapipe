#import "CaratSaveVideoGraph.h"
#import "mediapipe/objc/MPPGraph.h"

#include "mediapipe/carat/formats/carat_face_effect.pb.h"
#include "mediapipe/carat/formats/color_lut.pb.h"
#include "mediapipe/carat/formats/carat_frame_effect.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"

static NSString* const kGraphName = @"carat_save_video_graph";

static const char* kInputStream = "input_video";
static const char* kOutputStream = "output_video";

static const char* kNumFacesInputSidePacket = "num_faces";
static const char* kCaratFaceEffectListInputStream = "carat_face_effect_list";
static const char* kColorLutInputStream = "color_lut";
static const char* kCaratFrameEffectListInputStream = "carat_frame_effect_list";

static const int kNumFaces = 5;

@interface CaratSaveVideoGraph() <MPPGraphDelegate>
@property(nonatomic) MPPGraph* mediapipeGraph;
@property(nonatomic) NSString *caratFaceEffectListString;
@property(nonatomic) NSString *colorLutString;
@property(nonatomic) NSString *caratFrameEffectListString;
@end

@implementation CaratSaveVideoGraph {
  CMTime _lastTimestamp;
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

- (instancetype)initWithCaratFaceEffectListString:(NSString *)caratFaceEffectListString
  colorLutString:(NSString *)colorLutString
  caratFrameEffectListString:(NSString *)caratFrameEffectListString {
    self = [super init];
    if (self) {
        self.mediapipeGraph = [[self class] loadGraphFromResource:kGraphName];
        self.mediapipeGraph.delegate = self;

        [self.mediapipeGraph addFrameOutputStream:kOutputStream outputPacketType:MPPPacketTypePixelBuffer];
        [self.mediapipeGraph setSidePacket:(mediapipe::MakePacket<int>(kNumFaces)) named:kNumFacesInputSidePacket];

        self.caratFaceEffectListString = caratFaceEffectListString;
        self.colorLutString = colorLutString;
        self.caratFrameEffectListString = caratFrameEffectListString;
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

#pragma mark - MPPGraphDelegate methods

// Invoked on a Mediapipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer fromStream:(const std::string&)streamName {
    if (streamName == kOutputStream) {
        [_delegate saveGraph:self didOutputPixelBuffer:pixelBuffer];
    }
}

// Invoked on a Mediapipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph didOutputPacket:(const ::mediapipe::Packet&)packet fromStream:(const std::string&)streamName {}

@end
