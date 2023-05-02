#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>

@class CaratMediapipeGraph;

@protocol CaratMediapipeGraphDelegate <NSObject>
- (void)graph:(CaratMediapipeGraph*)graph didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer;
@end

@interface CaratMediapipeGraph: NSObject
- (instancetype)initWithApplyGamma:(bool)applyGamma;
- (void)startGraph;
- (void)sendPixelBuffer:(CVPixelBufferRef)pixelBuffer timestamp:(CMTime)timestamp;
- (void)setFaceEffects:(NSArray *)effects;
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
  greenMix:(NSArray *)greenMix blueMix:(NSArray *)blueMix purpleMix:(NSArray *)purpleMix;
- (void)setFrameEffects:(NSArray *)effects;
- (void)waitUntilIdle;
@property(weak, nonatomic) id <CaratMediapipeGraphDelegate> delegate;
@property(nonatomic) NSString *caratFaceEffectListString;
@property(nonatomic) NSString *colorLutString;
@property(nonatomic) NSString *caratFrameEffectListString;
@end
