#ifndef RM_MODEL_DETAIL_RUNTIME_LOADER_H
#define RM_MODEL_DETAIL_RUNTIME_LOADER_H

#include <filesystem>
#include <stdexcept>
#include <string>

namespace rm_model::detail {

enum class SharedLibraryLoadMode {
  Lazy,
  Now,
};

bool has_shell_unsafe_chars(const std::string& value);
std::string shell_quote_checked(const std::string& value, const char* label);
void run_command(const std::string& cmd, const std::string& failure_prefix);
void close_shared_library(void* handle) noexcept;

class SharedLibrary {
 public:
  SharedLibrary() = default;
  explicit SharedLibrary(const std::filesystem::path& path,
                         SharedLibraryLoadMode mode = SharedLibraryLoadMode::Lazy,
                         const std::string& error_message = "Failed to load shared library",
                         bool include_loader_error = false) {
    open(path, mode, error_message, include_loader_error);
  }
  ~SharedLibrary();

  SharedLibrary(const SharedLibrary&) = delete;
  SharedLibrary& operator=(const SharedLibrary&) = delete;

  SharedLibrary(SharedLibrary&& other) noexcept;
  SharedLibrary& operator=(SharedLibrary&& other) noexcept;

  void open(const std::filesystem::path& path,
            SharedLibraryLoadMode mode = SharedLibraryLoadMode::Lazy,
            const std::string& error_message = "Failed to load shared library",
            bool include_loader_error = false);
  void close() noexcept;
  void* symbol(const char* name) const;
  void* handle() const { return handle_; }
  void* release();

  template <typename T>
  T required_symbol(const char* name) const {
    void* sym = symbol(name);
    if (!sym) {
      throw std::runtime_error(std::string("Missing symbol: ") + name);
    }
    return reinterpret_cast<T>(sym);
  }

  template <typename T>
  T optional_symbol(const char* name) const {
    void* sym = symbol(name);
    return sym ? reinterpret_cast<T>(sym) : nullptr;
  }

 private:
  void* handle_ = nullptr;
};

} // namespace rm_model::detail

#endif // RM_MODEL_DETAIL_RUNTIME_LOADER_H
