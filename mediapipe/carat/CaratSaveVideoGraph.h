#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>

@class CaratSaveVideoGraph;

@interface CaratSaveVideoGraph: NSObject
- (instancetype)initWithInputVideoPath:(NSString *)inputVideoPath
  outputVideoPath:(NSString *)outputVideoPath
  caratFaceEffectListString:(NSString *)caratFaceEffectListString
  colorLutString:(NSString *)colorLutString
  caratFrameEffectListString:(NSString *)caratFrameEffectListString;
- (void)startGraph;
- (BOOL)waitUntilDone;
@end
