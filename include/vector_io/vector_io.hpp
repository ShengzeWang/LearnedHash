#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace vector_io {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct ReadOptions {
  bool validate_dimensions = true;
};

template <typename T>
struct RowView {
  T* data = nullptr;
  std::size_t size = 0;

  T& operator[](std::size_t index) const {
    return data[index];
  }

  T* begin() const { return data; }
  T* end() const { return data + size; }
};

template <typename T>
struct VectorStorage {
  std::uint32_t dim = 0;
  std::size_t count = 0;
  std::vector<T> values;

  VectorStorage() = default;

  VectorStorage(std::uint32_t dimension, std::size_t vector_count)
      : dim(dimension), count(vector_count), values(vector_count * dimension) {}

  bool empty() const { return dim == 0 || count == 0; }

  T* data() { return values.data(); }
  const T* data() const { return values.data(); }

  RowView<T> row(std::size_t index) {
    return RowView<T>{values.data() + index * dim, dim};
  }

  RowView<const T> row(std::size_t index) const {
    return RowView<const T>{values.data() + index * dim, dim};
  }
};

VectorStorage<float> read_fvecs(const std::string& path,
                                ReadOptions options = {});
VectorStorage<std::int32_t> read_ivecs(const std::string& path,
                                       ReadOptions options = {});

void write_fvecs(const std::string& path,
                 const VectorStorage<float>& storage);
void write_ivecs(const std::string& path,
                 const VectorStorage<std::int32_t>& storage);

}  // namespace vector_io
