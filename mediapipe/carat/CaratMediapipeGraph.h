#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>

@class CaratMediapipeGraph;

@protocol CaratMediapipeGraphDelegate <NSObject>
- (void)graph:(CaratMediapipeGraph*)graph didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer;
@end

@interface CaratMediapipeGraph: NSObject
- (instancetype)init;
- (void)startGraph;
- (void)sendPixelBuffer:(CVPixelBufferRef)pixelBuffer;
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
                                lipEndUp:(float)lipEndUp
                                skinSmooth:(float)skinSmooth;
@property(weak, nonatomic) id <CaratMediapipeGraphDelegate> delegate;
@end
