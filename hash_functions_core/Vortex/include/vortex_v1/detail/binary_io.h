#ifndef VORTEX_V1_DETAIL_BINARY_IO_H
#define VORTEX_V1_DETAIL_BINARY_IO_H

#include <cstddef>
#include <ios>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace vortex::detail {

inline void write_bytes(std::ostream& out,
                        const void* data,
                        std::size_t len,
                        const char* error_message) {
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(len));
  if (!out) {
    throw std::runtime_error(error_message);
  }
}

template <typename T>
void write_vector(std::ostream& out,
                  const std::vector<T>& vec,
                  const char* error_message) {
  if (vec.empty()) {
    return;
  }
  write_bytes(out, vec.data(), sizeof(T) * vec.size(), error_message);
}

inline void read_bytes(std::istream& in,
                       void* data,
                       std::size_t len,
                       const char* error_message) {
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(len));
  if (!in) {
    throw std::runtime_error(error_message);
  }
}

template <typename T>
void read_vector(std::istream& in,
                 std::vector<T>& vec,
                 std::size_t count,
                 const char* error_message) {
  vec.resize(count);
  if (count == 0) {
    return;
  }
  read_bytes(in, vec.data(), sizeof(T) * count, error_message);
}

} // namespace vortex::detail

#endif // VORTEX_V1_DETAIL_BINARY_IO_H
