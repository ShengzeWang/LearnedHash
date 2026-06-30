#include "rm_model/data_loader.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace rm_model {
namespace {

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define RM_MODEL_BIG_ENDIAN 1
#else
#define RM_MODEL_BIG_ENDIAN 0
#endif

#if RM_MODEL_BIG_ENDIAN
#if defined(_MSC_VER)
#include <intrin.h>
inline uint64_t bswap64(uint64_t value) { return _byteswap_uint64(value); }
inline uint32_t bswap32(uint32_t value) { return _byteswap_ulong(value); }
#else
inline uint64_t bswap64(uint64_t value) { return __builtin_bswap64(value); }
inline uint32_t bswap32(uint32_t value) { return __builtin_bswap32(value); }
#endif
#endif

class MMap {
 public:
  explicit MMap(const std::string& path) {
#if defined(_WIN32)
    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("Unable to open data file: " + path);
    }
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size)) {
      CloseHandle(file);
      throw std::runtime_error("Unable to stat data file: " + path);
    }
    size_ = static_cast<std::size_t>(size.QuadPart);

    HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
      CloseHandle(file);
      throw std::runtime_error("Unable to create file mapping: " + path);
    }
    data_ = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    CloseHandle(file);
    if (!data_) {
      throw std::runtime_error("Unable to map file: " + path);
    }
#else
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("Unable to open data file: " + path);
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
      close(fd);
      throw std::runtime_error("Unable to stat data file: " + path);
    }
    size_ = static_cast<std::size_t>(st.st_size);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      throw std::runtime_error("Unable to map file: " + path);
    }
#endif
  }

  ~MMap() {
#if defined(_WIN32)
    if (data_) {
      UnmapViewOfFile(data_);
    }
#else
    if (data_) {
      munmap(data_, size_);
    }
#endif
  }

  const uint8_t* data() const { return static_cast<const uint8_t*>(data_); }
  std::size_t size() const { return size_; }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0;
};

inline uint64_t read_u64_le(const uint8_t* ptr) {
  uint64_t value = 0;
  std::memcpy(&value, ptr, sizeof(uint64_t));
#if RM_MODEL_BIG_ENDIAN
  value = bswap64(value);
#endif
  return value;
}

inline uint32_t read_u32_le(const uint8_t* ptr) {
  uint32_t value = 0;
  std::memcpy(&value, ptr, sizeof(uint32_t));
#if RM_MODEL_BIG_ENDIAN
  value = bswap32(value);
#endif
  return value;
}

inline double read_f64_le(const uint8_t* ptr) {
  uint64_t bits = read_u64_le(ptr);
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(double));
  return value;
}

template <typename T>
class SliceAdapter : public TrainingDataIteratorProvider<T> {
 public:
  SliceAdapter(std::shared_ptr<MMap> mmap, std::size_t length)
      : mmap_(std::move(mmap)),
        base_(mmap_->data() + 8),
        length_(length),
        stride_(sizeof(T)) {
#if !RM_MODEL_BIG_ENDIAN
    if (reinterpret_cast<std::uintptr_t>(base_) % alignof(T) == 0) {
      raw_keys_ = reinterpret_cast<const T*>(base_);
    }
#endif
  }

  std::size_t len() const override { return length_; }

  KeyType key_type() const override {
    if constexpr (std::is_same_v<T, uint64_t>) return KeyType::U64;
    if constexpr (std::is_same_v<T, uint32_t>) return KeyType::U32;
    return KeyType::F64;
  }

  bool get(std::size_t idx, T& key, std::size_t& offset) const override {
    if (idx >= length_) return false;
    const uint8_t* base = base_;
    if constexpr (std::is_same_v<T, uint64_t>) {
      key = read_u64_le(base + idx * stride_);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      key = read_u32_le(base + idx * stride_);
    } else {
      key = read_f64_le(base + idx * stride_);
    }
    offset = idx;
    return true;
  }

  const T* raw_keys() const override { return raw_keys_; }

 private:
  std::shared_ptr<MMap> mmap_;
  const uint8_t* base_;
  std::size_t length_;
  std::size_t stride_;
  const T* raw_keys_ = nullptr;
};

} // namespace

MappedDataset MappedDataset::soft_copy() const {
  return visit([](const auto& data) -> MappedDataset { return MappedDataset(data.soft_copy()); });
}

std::optional<MappedDataset::U64Data> MappedDataset::into_u64() const {
  if (std::holds_alternative<U64Data>(data_)) {
    return std::get<U64Data>(data_);
  }
  return std::nullopt;
}

std::pair<std::size_t, MappedDataset> load_data(const std::string& filepath, DataType dt) {
  auto mmap = std::make_shared<MMap>(filepath);
  if (mmap->size() < 8) {
    throw std::runtime_error("Data file too small: " + filepath);
  }

  std::size_t num_items = static_cast<std::size_t>(read_u64_le(mmap->data()));
  std::size_t file_size = mmap->size();
  auto ensure_payload = [&](std::size_t elem_size) {
    std::size_t header_size = sizeof(uint64_t);
    if (file_size < header_size) {
      throw std::runtime_error("Data file too small: " + filepath);
    }
    std::size_t available = file_size - header_size;
    if (elem_size == 0 || num_items > available / elem_size) {
      throw std::runtime_error("Data file length does not match header count: " + filepath);
    }
  };

  if (dt == DataType::UINT64) {
    ensure_payload(sizeof(uint64_t));
    auto provider = std::make_shared<SliceAdapter<uint64_t>>(mmap, num_items);
    return {num_items, MappedDataset(TrainingData<uint64_t>(provider))};
  }

  if (dt == DataType::UINT32) {
    ensure_payload(sizeof(uint32_t));
    auto provider = std::make_shared<SliceAdapter<uint32_t>>(mmap, num_items);
    return {num_items, MappedDataset(TrainingData<uint32_t>(provider))};
  }

  ensure_payload(sizeof(double));
  auto provider = std::make_shared<SliceAdapter<double>>(mmap, num_items);
  return {num_items, MappedDataset(TrainingData<double>(provider))};
}

} // namespace rm_model
