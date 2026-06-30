#include "rm_model/detail/runtime_loader.h"

#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace rm_model::detail {

bool has_shell_unsafe_chars(const std::string& value) {
  static constexpr char kMetacharacters[] = "\"`$;&|<>";
  if (value.find_first_of(kMetacharacters) != std::string::npos) {
    return true;
  }
  for (char ch : value) {
    if (ch == '\n' || ch == '\r' || ch == '\t') {
      return true;
    }
  }
  return false;
}

std::string shell_quote_checked(const std::string& value, const char* label) {
  if (value.empty()) {
    throw std::runtime_error(std::string("Empty shell argument: ") + label);
  }
  if (has_shell_unsafe_chars(value)) {
    throw std::runtime_error(std::string("Unsafe shell argument for ") + label + ": " + value);
  }
  return "\"" + value + "\"";
}

void run_command(const std::string& cmd, const std::string& failure_prefix) {
  int status = std::system(cmd.c_str());
  if (status != 0) {
    throw std::runtime_error(failure_prefix + cmd);
  }
}

void close_shared_library(void* handle) noexcept {
  if (!handle) {
    return;
  }
#if defined(_WIN32)
  FreeLibrary(static_cast<HMODULE>(handle));
#else
  dlclose(handle);
#endif
}

SharedLibrary::~SharedLibrary() {
  close();
}

SharedLibrary::SharedLibrary(SharedLibrary&& other) noexcept
    : handle_(other.handle_) {
  other.handle_ = nullptr;
}

SharedLibrary& SharedLibrary::operator=(SharedLibrary&& other) noexcept {
  if (this != &other) {
    close();
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

void SharedLibrary::open(const std::filesystem::path& path,
                         SharedLibraryLoadMode mode,
                         const std::string& error_message,
                         bool include_loader_error) {
  close();
#if defined(_WIN32)
  handle_ = LoadLibraryA(path.string().c_str());
  if (!handle_) {
    throw std::runtime_error(error_message);
  }
#else
  int flags = (mode == SharedLibraryLoadMode::Now) ? RTLD_NOW : RTLD_LAZY;
  handle_ = dlopen(path.string().c_str(), flags);
  if (!handle_) {
    const char* err = include_loader_error ? dlerror() : nullptr;
    throw std::runtime_error(error_message + (err ? " (" + std::string(err) + ")" : ""));
  }
#endif
}

void SharedLibrary::close() noexcept {
  close_shared_library(handle_);
  handle_ = nullptr;
}

void* SharedLibrary::symbol(const char* name) const {
  if (!handle_) {
    return nullptr;
  }
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
  return dlsym(handle_, name);
#endif
}

void* SharedLibrary::release() {
  void* out = handle_;
  handle_ = nullptr;
  return out;
}

} // namespace rm_model::detail
