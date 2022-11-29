#ifndef CARAT_LIBS_GL_ERRORS_H_
#define CARAT_LIBS_GL_ERRORS_H_

#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// @return recent opengl errors and packs them into Status.
absl::Status GetOpenGlErrors();

}  // namespace mediapipe

#endif  // CARAT_LIBS_GL_ERRORS_H_
