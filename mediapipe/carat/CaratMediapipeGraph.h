#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>

@class CaratMediapipeGraph;

@protocol CaratMediapipeGraphDelegate <NSObject>
- (void)graph:(CaratMediapipeGraph*)graph didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer;
@end

@interface CaratMediapipeGraph: NSObject
- (instancetype)init;
- (void)startGraph;
- (void)sendPixelBuffer:(CVPixelBufferRef)pixelBuffer timestamp:(CMTime)timestamp;
- (void)setFaceEffects:(NSArray *)effects;
- (void)setColorLut:(NSString *)filePath intensity:(float)intensity grain:(float)grain vignette:(float)vignette;
- (void)setFrameEffects:(NSArray *)effects;
- (void)setColorLutIntensity:(NSNumber *)intensity;
- (void)waitUntilIdle;
@property(weak, nonatomic) id <CaratMediapipeGraphDelegate> delegate;
@end
