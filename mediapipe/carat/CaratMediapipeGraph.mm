#import "CaratMediapipeGraph.h"
#import "mediapipe/objc/MPPGraph.h"

#include "mediapipe/framework/formats/landmark.pb.h"

static NSString* const kGraphName = @"face_mesh_mobile_gpu";
static const char* kInputStream = "input_video";
static const char* kOutputStream = "output_video";
static const char* kNumFacesInputSidePacket = "num_faces";
static const char* kLandmarksOutputStream = "multi_face_landmarks";

static const int kNumFaces = 4;

static const char* kEyeSizeInputSidePacket = "eye_size";

@interface CaratMediapipeGraph() <MPPGraphDelegate>
@property(nonatomic) MPPGraph* mediapipeGraph;
@end

@implementation CaratMediapipeGraph {
    mediapipe::Packet _eyeSizePacket;
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
        [self.mediapipeGraph setSidePacket:(mediapipe::MakePacket<int>(kNumFaces)) named:kNumFacesInputSidePacket];

        mediapipe::Packet packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kEyeSizeInputSidePacket];
        _eyeSizePacket = packet;
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

- (void)setFaceSettingsParamsWithEyeSize:(float)eyeSize {
    float *f = mediapipe::GetFromUniquePtr<float>(_eyeSizePacket);
    *f = eyeSize;
}

- (void)sendPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    [self.mediapipeGraph sendPixelBuffer:pixelBuffer intoStream:kInputStream packetType:MPPPacketTypePixelBuffer];
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
    }
}

@end
