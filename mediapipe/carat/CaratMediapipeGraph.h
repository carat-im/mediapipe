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
@property(weak, nonatomic) id <CaratMediapipeGraphDelegate> delegate;
@end
