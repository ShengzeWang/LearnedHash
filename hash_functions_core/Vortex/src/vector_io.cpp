#include "vector_io/vector_io.hpp"

#include <algorithm>
#include <fstream>
#include <limits>

namespace vector_io {

namespace {

template <typename T, typename SizeT>
void checked_read(std::ifstream& in, T* data, SizeT count) {
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
  if (!in) {
    throw Error("Failed to read vector data");
  }
}

template <typename T>
VectorStorage<T> read_vecs_impl(const std::string& path, ReadOptions options) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw Error("Unable to open vector file: " + path);
  }

  in.seekg(0, std::ios::end);
  std::streamoff size = in.tellg();
  if (size < static_cast<std::streamoff>(sizeof(std::int32_t))) {
    throw Error("Vector file too small: " + path);
  }
  in.seekg(0, std::ios::beg);

  std::int32_t dim = 0;
  checked_read(in, &dim, 1);
  if (dim <= 0) {
    throw Error("Invalid dimension in vector file: " + path);
  }

  std::size_t stride = sizeof(std::int32_t) + sizeof(T) * static_cast<std::size_t>(dim);
  if (stride == 0 || (size % static_cast<std::streamoff>(stride)) != 0) {
    throw Error("Vector file size is not a multiple of stride: " + path);
  }

  std::size_t count = static_cast<std::size_t>(size) / stride;
  VectorStorage<T> storage(static_cast<std::uint32_t>(dim), count);

  in.seekg(0, std::ios::beg);
  for (std::size_t i = 0; i < count; ++i) {
    std::int32_t d = 0;
    checked_read(in, &d, 1);
    if (options.validate_dimensions && d != dim) {
      throw Error("Inconsistent vector dimension in file: " + path);
    }
    checked_read(in, storage.values.data() + i * static_cast<std::size_t>(dim), dim);
  }

  return storage;
}

template <typename T>
void write_vecs_impl(const std::string& path, const VectorStorage<T>& storage) {
  if (storage.empty()) {
    throw Error("Cannot write empty vector storage: " + path);
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw Error("Unable to open vector file for writing: " + path);
  }

  std::int32_t dim = static_cast<std::int32_t>(storage.dim);
  std::size_t count = storage.count;
  for (std::size_t i = 0; i < count; ++i) {
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(storage.values.data() + i * storage.dim),
              sizeof(T) * storage.dim);
    if (!out) {
      throw Error("Failed to write vector data: " + path);
    }
  }
}

} // namespace

VectorStorage<float> read_fvecs(const std::string& path, ReadOptions options) {
  return read_vecs_impl<float>(path, options);
}

VectorStorage<std::int32_t> read_ivecs(const std::string& path, ReadOptions options) {
  return read_vecs_impl<std::int32_t>(path, options);
}

void write_fvecs(const std::string& path, const VectorStorage<float>& storage) {
  write_vecs_impl<float>(path, storage);
}

void write_ivecs(const std::string& path, const VectorStorage<std::int32_t>& storage) {
  write_vecs_impl<std::int32_t>(path, storage);
}

} // namespace vector_io
