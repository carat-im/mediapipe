#include "mediapipe/carat/libs/gl_errors.h"

#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "mediapipe/framework/port/status.h"

namespace {

const char* ErrorToString(GLenum error) {
  switch (error) {
    case GL_INVALID_ENUM:
      return "[GL_INVALID_ENUM]: An unacceptable value is specified for an "
             "enumerated argument.";
    case GL_INVALID_VALUE:
      return "[GL_INVALID_VALUE]: A numeric argument is out of range.";
    case GL_INVALID_OPERATION:
      return "[GL_INVALID_OPERATION]: The specified operation is not allowed "
             "in the current state.";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      return "[GL_INVALID_FRAMEBUFFER_OPERATION]: The framebuffer object is "
             "not complete.";
    case GL_OUT_OF_MEMORY:
      return "[GL_OUT_OF_MEMORY]: There is not enough memory left to execute "
             "the command.";
  }
  return "[UNKNOWN_GL_ERROR]";
}

struct ErrorFormatter {
  void operator()(std::string* out, GLenum error) const {
    absl::StrAppend(out, ErrorToString(error));
  }
};

}  // namespace

namespace mediapipe {

absl::Status GetOpenGlErrors() {
  auto error = glGetError();
  if (error == GL_NO_ERROR) {
    return absl::OkStatus();
  }
  auto error2 = glGetError();
  if (error2 == GL_NO_ERROR) {
    return absl::InternalError(ErrorToString(error));
  }
  std::vector<GLenum> errors = {error, error2};
  for (error = glGetError(); error != GL_NO_ERROR; error = glGetError()) {
    errors.push_back(error);
  }
  return absl::InternalError(absl::StrJoin(errors, ",", ErrorFormatter()));
}

}  // namespace mediapipe
