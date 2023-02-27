#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>

@class CaratSaveVideoGraph;

@protocol CaratSaveVideoGraphDelegate <NSObject>
- (void)saveGraph:(CaratSaveVideoGraph*)graph didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer;
@end


@interface CaratSaveVideoGraph: NSObject
- (instancetype)initWithCaratFaceEffectListString:(NSString *)caratFaceEffectListString
  colorLutString:(NSString *)colorLutString
  caratFrameEffectListString:(NSString *)caratFrameEffectListString;
- (void)startGraph;
- (void)sendPixelBuffer:(CVPixelBufferRef)pixelBuffer timestamp:(CMTime)timestamp;
@property(weak, nonatomic) id <CaratSaveVideoGraphDelegate> delegate;
@end
