#import "CaratMediapipeGraph.h"
#import "mediapipe/objc/MPPGraph.h"

#include "mediapipe/framework/formats/landmark.pb.h"

static NSString* const kGraphName = @"carat_face_mesh_mobile_gpu";
static const char* kInputStream = "input_video";
static const char* kOutputStream = "output_video";
static const char* kNumFacesInputSidePacket = "num_faces";
static const char* kLandmarksOutputStream = "multi_face_landmarks";

static const int kNumFaces = 4;

static const char* kForeheadSizeInputSidePacket = "forehead_size";
static const char* kCheekboneSizeInputSidePacket = "cheekbone_size";
static const char* kTempleSizeInputSidePacket = "temple_size";
static const char* kChinSizeInputSidePacket = "chin_size";
static const char* kChinHeightInputSidePacket = "chin_height";
static const char* kChinSharpnessInputSidePacket = "chin_sharpness";
static const char* kEyeSizeInputSidePacket = "eye_size";
static const char* kEyeHeightInputSidePacket = "eye_height";
static const char* kEyeSpacingInputSidePacket = "eye_spacing";
static const char* kFrontEyeSizeInputSidePacket = "front_eye_size";
static const char* kUnderEyeSizeInputSidePacket = "under_eye_size";
static const char* kPupilSizeInputSidePacket = "pupil_size";
static const char* kNoseHeightInputSidePacket = "nose_height";
static const char* kNoseWidthInputSidePacket = "nose_width";
static const char* kNoseBridgeSizeInputSidePacket = "nose_bridge_size";
static const char* kNoseBaseSizeInputSidePacket = "nose_base_size";
static const char* kNoseEndSizeInputSidePacket = "nose_end_size";
static const char* kPhiltrumHeightInputSidePacket = "philtrum_height";
static const char* kLipSizeInputSidePacket = "lip_size";
static const char* kLipEndUpInputSidePacket = "lip_end_up";

@interface CaratMediapipeGraph() <MPPGraphDelegate>
@property(nonatomic) MPPGraph* mediapipeGraph;
@end

@implementation CaratMediapipeGraph {
    mediapipe::Packet _foreheadSizePacket;
    mediapipe::Packet _cheekboneSizePacket;
    mediapipe::Packet _templeSizePacket;
    mediapipe::Packet _chinSizePacket;
    mediapipe::Packet _chinHeightPacket;
    mediapipe::Packet _chinSharpnessPacket;
    mediapipe::Packet _eyeSizePacket;
    mediapipe::Packet _eyeHeightPacket;
    mediapipe::Packet _eyeSpacingPacket;
    mediapipe::Packet _frontEyeSizePacket;
    mediapipe::Packet _underEyeSizePacket;
    mediapipe::Packet _pupilSizePacket;
    mediapipe::Packet _noseHeightPacket;
    mediapipe::Packet _noseWidthPacket;
    mediapipe::Packet _noseBridgeSizePacket;
    mediapipe::Packet _noseBaseSizePacket;
    mediapipe::Packet _noseEndSizePacket;
    mediapipe::Packet _philtrumHeightPacket;
    mediapipe::Packet _lipSizePacket;
    mediapipe::Packet _lipEndUpPacket;
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
        [self.mediapipeGraph setSidePacket:packet named:kForeheadSizeInputSidePacket];
        _foreheadSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kCheekboneSizeInputSidePacket];
        _cheekboneSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kTempleSizeInputSidePacket];
        _templeSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kChinSizeInputSidePacket];
        _chinSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kChinHeightInputSidePacket];
        _chinHeightPacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kChinSharpnessInputSidePacket];
        _chinSharpnessPacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kEyeSizeInputSidePacket];
        _eyeSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kEyeHeightInputSidePacket];
        _eyeHeightPacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kEyeSpacingInputSidePacket];
        _eyeSpacingPacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kFrontEyeSizeInputSidePacket];
        _frontEyeSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kUnderEyeSizeInputSidePacket];
        _underEyeSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kPupilSizeInputSidePacket];
        _pupilSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kNoseHeightInputSidePacket];
        _noseHeightPacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kNoseWidthInputSidePacket];
        _noseWidthPacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kNoseBridgeSizeInputSidePacket];
        _noseBridgeSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kNoseBaseSizeInputSidePacket];
        _noseBaseSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kNoseEndSizeInputSidePacket];
        _noseEndSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kPhiltrumHeightInputSidePacket];
        _philtrumHeightPacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kLipSizeInputSidePacket];
        _lipSizePacket = packet;

        packet = mediapipe::AdoptAsUniquePtr<float>(new float(1.f));
        [self.mediapipeGraph setSidePacket:packet named:kLipEndUpInputSidePacket];
        _lipEndUpPacket = packet;
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

- (void)setFaceSettingsParamsWithForeheadSize:(float)foreheadSize
                                 cheekboneSize:(float)cheekboneSize
                                 templeSize:(float)templeSize 
                                 chinSize:(float)chinSize
                                 chinHeight:(float)chinHeight
                                 chinSharpness:(float)chinSharpness
                                 eyeSize:(float)eyeSize
                                 eyeHeight:(float)eyeHeight
                                 eyeSpacing:(float)eyeSpacing
                                 frontEyeSize:(float)frontEyeSize
                                 underEyeSize:(float)underEyeSize
                                 pupilSize:(float)pupilSize
                                 noseHeight:(float)noseHeight
                                 noseWidth:(float)noseWidth
                                 noseBridgeSize:(float)noseBridgeSize
                                 noseBaseSize:(float)noseBaseSize
                                 noseEndSize:(float)noseEndSize
                                 philtrumHeight:(float)philtrumHeight
                                 lipSize:(float)lipSize
                                 lipEndUp:(float)lipEndUp {
    float *f = mediapipe::GetFromUniquePtr<float>(_foreheadSizePacket);
    *f = foreheadSize;

    f = mediapipe::GetFromUniquePtr<float>(_cheekboneSizePacket);
    *f = cheekboneSize;

    f = mediapipe::GetFromUniquePtr<float>(_templeSizePacket);
    *f = templeSize;

    f = mediapipe::GetFromUniquePtr<float>(_chinSizePacket);
    *f = chinSize;

    f = mediapipe::GetFromUniquePtr<float>(_chinHeightPacket);
    *f = chinHeight;

    f = mediapipe::GetFromUniquePtr<float>(_chinSharpnessPacket);
    *f = chinSharpness;

    f = mediapipe::GetFromUniquePtr<float>(_eyeSizePacket);
    *f = eyeSize;

    f = mediapipe::GetFromUniquePtr<float>(_eyeHeightPacket);
    *f = eyeHeight;

    f = mediapipe::GetFromUniquePtr<float>(_eyeSpacingPacket);
    *f = eyeSpacing;

    f = mediapipe::GetFromUniquePtr<float>(_frontEyeSizePacket);
    *f = frontEyeSize;

    f = mediapipe::GetFromUniquePtr<float>(_underEyeSizePacket);
    *f = underEyeSize;

    f = mediapipe::GetFromUniquePtr<float>(_pupilSizePacket);
    *f = pupilSize;

    f = mediapipe::GetFromUniquePtr<float>(_noseHeightPacket);
    *f = noseHeight;

    f = mediapipe::GetFromUniquePtr<float>(_noseWidthPacket);
    *f = noseWidth;

    f = mediapipe::GetFromUniquePtr<float>(_noseBridgeSizePacket);
    *f = noseBridgeSize;

    f = mediapipe::GetFromUniquePtr<float>(_noseBaseSizePacket);
    *f = noseBaseSize;

    f = mediapipe::GetFromUniquePtr<float>(_noseEndSizePacket);
    *f = noseEndSize;

    f = mediapipe::GetFromUniquePtr<float>(_philtrumHeightPacket);
    *f = philtrumHeight;

    f = mediapipe::GetFromUniquePtr<float>(_lipSizePacket);
    *f = lipSize;

    f = mediapipe::GetFromUniquePtr<float>(_lipEndUpPacket);
    *f = lipEndUp;
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
