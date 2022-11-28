#ifndef MEDIAPIPE_MODULES_FRAME_EFFECT_RENDERER_H_
#define MEDIAPIPE_MODULES_FRAME_EFFECT_RENDERER_H_

#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gl_base.h"

namespace mediapipe {

class FrameEffectRenderer {
 public:
  virtual ~FrameEffectRenderer() = default;

  // Must be called in the same GL context as was used upon initialization.
  virtual absl::Status RenderEffect() = 0;
};

// Must be called in the same GL context as will be used for rendering.
absl::StatusOr<std::unique_ptr<FrameEffectRenderer>> CreateFrameEffectRenderer(ImageFrame&& effect_texture);

}  // namespace mediapipe

#endif  // MEDIAPIPE_MODULES_FRAME_EFFECT_RENDERER_H_
