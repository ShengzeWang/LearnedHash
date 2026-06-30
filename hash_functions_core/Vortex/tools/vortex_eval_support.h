#ifndef VORTEX_V1_TOOLS_VORTEX_EVAL_SUPPORT_H
#define VORTEX_V1_TOOLS_VORTEX_EVAL_SUPPORT_H

#include "vortex_v1/uint256.h"

#include "vector_io/vector_io.hpp"

#include <faiss/IndexFlat.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace vortex::tools::eval {

struct HashEntry64 {
  uint64_t hash = 0;
  uint32_t index = 0;
};

struct HashEntry256 {
  vortex::UInt256 hash;
  uint32_t index = 0;
};

struct LatencySummary {
  double avg_ns = 0.0;
  double p95_ns = 0.0;
  double p99_ns = 0.0;
};

struct TruthCacheHeader {
  uint64_t magic = 0x564f525458545248ULL; // "VORTXTRH"
  uint32_t version = 1;
  uint32_t dim = 0;
  uint64_t base_count = 0;
  uint64_t query_count = 0;
  uint32_t search_k = 0;
  uint32_t exclude_self = 0;
  uint64_t base_size = 0;
  uint64_t query_size = 0;
  int64_t base_mtime = 0;
  int64_t query_mtime = 0;
};

struct HashIndexCacheHeader {
  uint64_t magic = 0x564f525458484958ULL; // "VORTXHIX"
  uint32_t version = 1;
  uint32_t dim = 0;
  uint32_t hash_bits = 0;
  uint32_t reserved = 0;
  uint64_t base_count = 0;
  uint64_t base_size = 0;
  int64_t base_mtime = 0;
  uint64_t model_bin_size = 0;
  int64_t model_bin_mtime = 0;
  uint64_t model_bin_hash = 0;
};

struct HashIndexCacheEntry64 {
  uint64_t hash = 0;
  uint32_t index = 0;
  uint32_t reserved = 0;
};

inline double elapsed_ms(std::chrono::steady_clock::time_point start) {
  auto end = std::chrono::steady_clock::now();
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) /
         1000.0;
}

inline uint32_t normalized_thread_count(uint32_t requested_threads, std::size_t work_items) {
  if (work_items <= 1) {
    return 1;
  }
  uint32_t threads = requested_threads;
  if (threads == 0) {
    threads = std::thread::hardware_concurrency();
  }
  if (threads == 0) {
    threads = 1;
  }
  return std::max<uint32_t>(
      1, std::min<uint32_t>(threads, static_cast<uint32_t>(
                                         std::min<std::size_t>(
                                             work_items,
                                             std::numeric_limits<uint32_t>::max()))));
}

template <typename Fn>
void parallel_for_indices(std::size_t total, uint32_t threads, Fn&& fn) {
  if (total == 0) {
    return;
  }
  threads = normalized_thread_count(threads, total);
  if (threads <= 1) {
    for (std::size_t i = 0; i < total; ++i) {
      fn(i);
    }
    return;
  }

  std::exception_ptr first_error;
  std::mutex error_mutex;
  std::vector<std::thread> workers;
  workers.reserve(threads);
  std::size_t chunk = (total + static_cast<std::size_t>(threads) - 1) /
                      static_cast<std::size_t>(threads);
  for (uint32_t t = 0; t < threads; ++t) {
    std::size_t begin = static_cast<std::size_t>(t) * chunk;
    if (begin >= total) {
      break;
    }
    std::size_t end = std::min<std::size_t>(total, begin + chunk);
    workers.emplace_back([&, begin, end]() {
      try {
        for (std::size_t i = begin; i < end; ++i) {
          fn(i);
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!first_error) {
          first_error = std::current_exception();
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

inline double percentile_sorted(const std::vector<double>& latencies, double p) {
  if (latencies.empty()) return 0.0;
  double idx = p * static_cast<double>(latencies.size() - 1);
  std::size_t lo = static_cast<std::size_t>(idx);
  std::size_t hi = std::min<std::size_t>(lo + 1, latencies.size() - 1);
  double frac = idx - static_cast<double>(lo);
  return latencies[lo] * (1.0 - frac) + latencies[hi] * frac;
}

inline LatencySummary summarize_latency_ns(std::vector<double> latencies) {
  LatencySummary summary;
  if (latencies.empty()) {
    return summary;
  }
  std::sort(latencies.begin(), latencies.end());
  double sum = 0.0;
  for (double v : latencies) {
    sum += v;
  }
  summary.avg_ns = sum / static_cast<double>(latencies.size());
  summary.p95_ns = percentile_sorted(latencies, 0.95);
  summary.p99_ns = percentile_sorted(latencies, 0.99);
  return summary;
}

inline std::vector<uint32_t> default_node_counts(std::size_t base_count) {
  const std::vector<uint32_t> defaults = {32, 64, 128, 256, 512, 1024};
  std::vector<uint32_t> out;
  for (uint32_t value : defaults) {
    if (value >= 2 && value <= base_count) {
      out.push_back(value);
    }
  }
  if (out.empty() && base_count >= 2) {
    out.push_back(static_cast<uint32_t>(
        std::min<std::size_t>(base_count,
                              static_cast<std::size_t>(
                                  std::numeric_limits<uint32_t>::max()))));
  }
  return out;
}

inline std::vector<uint32_t>
normalize_node_counts(const std::vector<uint32_t>& requested,
                      std::size_t base_count) {
  std::vector<uint32_t> out = requested.empty() ? default_node_counts(base_count)
                                                : requested;
  out.erase(std::remove_if(out.begin(),
                           out.end(),
                           [&](uint32_t value) {
                             return value < 2 || value > base_count;
                           }),
            out.end());
  if (out.empty() && base_count >= 2) {
    out.push_back(static_cast<uint32_t>(
        std::min<std::size_t>(base_count,
                              static_cast<std::size_t>(
                                  std::numeric_limits<uint32_t>::max()))));
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

inline std::string format_node_counts(const std::vector<uint32_t>& values) {
  std::string out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out += ",";
    }
    out += std::to_string(values[i]);
  }
  return out;
}

template <typename Fn>
LatencySummary measure_latency_ns(std::size_t total, Fn&& fn) {
  std::vector<double> latencies;
  latencies.reserve(total);
  for (std::size_t i = 0; i < total; ++i) {
    auto start = std::chrono::steady_clock::now();
    fn(i);
    auto end = std::chrono::steady_clock::now();
    double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    latencies.push_back(ns);
  }
  return summarize_latency_ns(std::move(latencies));
}

inline void print_latency_summary(const std::string& prefix, const LatencySummary& summary) {
  auto to_ms = [](double ns) { return ns / 1.0e6; };
  std::cout << "  " << prefix << "latency_avg_ms: " << to_ms(summary.avg_ns) << "\n";
  std::cout << "  " << prefix << "latency_p95_ms: " << to_ms(summary.p95_ns) << "\n";
  std::cout << "  " << prefix << "latency_p99_ms: " << to_ms(summary.p99_ns) << "\n";
}

inline std::size_t lower_bound_hash(const std::vector<HashEntry64>& entries,
                                    uint64_t hash) {
  auto it = std::lower_bound(entries.begin(), entries.end(), hash,
                             [](const HashEntry64& entry, uint64_t value) {
                               return entry.hash < value;
                             });
  return static_cast<std::size_t>(it - entries.begin());
}

inline std::size_t lower_bound_hash(const std::vector<HashEntry256>& entries,
                                    const vortex::UInt256& hash) {
  std::size_t lo = 0;
  std::size_t hi = entries.size();
  while (lo < hi) {
    std::size_t mid = lo + (hi - lo) / 2;
    int cmp = vortex::compare(entries[mid].hash, hash);
    if (cmp < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

inline int64_t file_time_key(const std::filesystem::path& path) {
  std::error_code ec;
  auto value = std::filesystem::last_write_time(path, ec);
  if (ec) {
    return 0;
  }
  return static_cast<int64_t>(value.time_since_epoch().count());
}

inline uint64_t file_content_hash64(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return 0;
  }
  uint64_t hash = 1469598103934665603ULL;
  char buffer[64 * 1024];
  while (in) {
    in.read(buffer, sizeof(buffer));
    std::streamsize count = in.gcount();
    for (std::streamsize i = 0; i < count; ++i) {
      hash ^= static_cast<unsigned char>(buffer[i]);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

inline TruthCacheHeader make_truth_cache_header(const std::filesystem::path& base_path,
                                                const std::filesystem::path& query_path,
                                                const vector_io::VectorStorage<float>& base,
                                                std::size_t query_count,
                                                uint32_t search_k,
                                                bool exclude_self) {
  TruthCacheHeader header;
  header.dim = base.dim;
  header.base_count = static_cast<uint64_t>(base.count);
  header.query_count = static_cast<uint64_t>(query_count);
  header.search_k = search_k;
  header.exclude_self = exclude_self ? 1U : 0U;
  std::error_code ec;
  header.base_size = std::filesystem::file_size(base_path, ec);
  if (ec) {
    header.base_size = 0;
  }
  header.query_size = std::filesystem::file_size(query_path, ec);
  if (ec) {
    header.query_size = 0;
  }
  header.base_mtime = file_time_key(base_path);
  header.query_mtime = file_time_key(query_path);
  return header;
}

inline bool same_truth_cache_header(const TruthCacheHeader& lhs,
                                    const TruthCacheHeader& rhs) {
  return lhs.magic == rhs.magic && lhs.version == rhs.version &&
         lhs.dim == rhs.dim && lhs.base_count == rhs.base_count &&
         lhs.query_count == rhs.query_count && lhs.search_k == rhs.search_k &&
         lhs.exclude_self == rhs.exclude_self && lhs.base_size == rhs.base_size &&
         lhs.query_size == rhs.query_size && lhs.base_mtime == rhs.base_mtime &&
         lhs.query_mtime == rhs.query_mtime;
}

inline bool read_truth_cache(const std::filesystem::path& path,
                             const TruthCacheHeader& expected,
                             std::vector<faiss::idx_t>* labels) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  TruthCacheHeader header;
  in.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!in || !same_truth_cache_header(header, expected)) {
    return false;
  }
  std::size_t label_count =
      static_cast<std::size_t>(expected.query_count) * expected.search_k;
  labels->assign(label_count, faiss::idx_t{-1});
  if (label_count > 0) {
    in.read(reinterpret_cast<char*>(labels->data()),
            static_cast<std::streamsize>(label_count * sizeof(faiss::idx_t)));
  }
  return static_cast<bool>(in);
}

inline bool write_truth_cache(const std::filesystem::path& path,
                              const TruthCacheHeader& header,
                              const std::vector<faiss::idx_t>& labels) {
  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  out.write(reinterpret_cast<const char*>(&header), sizeof(header));
  if (!labels.empty()) {
    out.write(reinterpret_cast<const char*>(labels.data()),
              static_cast<std::streamsize>(labels.size() * sizeof(faiss::idx_t)));
  }
  return static_cast<bool>(out);
}

inline bool read_groundtruth_ivecs(const std::filesystem::path& path,
                                   std::size_t query_count,
                                   uint32_t search_k,
                                   std::size_t base_count,
                                   std::vector<faiss::idx_t>* labels) {
  auto truth = vector_io::read_ivecs(path.string());
  if (truth.count < query_count) {
    throw std::runtime_error("Ground-truth query count is smaller than evaluated query count");
  }
  if (truth.dim < search_k) {
    throw std::runtime_error("Ground-truth row width is smaller than requested k");
  }
  labels->assign(query_count * search_k, faiss::idx_t{-1});
  for (std::size_t qi = 0; qi < query_count; ++qi) {
    const std::size_t src_offset = qi * truth.dim;
    const std::size_t dst_offset = qi * search_k;
    for (uint32_t j = 0; j < search_k; ++j) {
      int32_t id = truth.values[src_offset + j];
      if (id < 0 || static_cast<std::size_t>(id) >= base_count) {
        throw std::runtime_error("Ground-truth neighbor id is outside the base range");
      }
      (*labels)[dst_offset + j] = static_cast<faiss::idx_t>(id);
    }
  }
  return true;
}

inline HashIndexCacheHeader make_hash_index_cache_header(
    const std::filesystem::path& base_path,
    const std::filesystem::path& codegen_dir,
    const vector_io::VectorStorage<float>& base,
    uint32_t hash_bits) {
  HashIndexCacheHeader header;
  header.dim = base.dim;
  header.hash_bits = hash_bits;
  header.base_count = static_cast<uint64_t>(base.count);
  std::error_code ec;
  header.base_size = std::filesystem::file_size(base_path, ec);
  if (ec) {
    header.base_size = 0;
  }
  header.base_mtime = file_time_key(base_path);

  std::filesystem::path model_bin = codegen_dir / "model.bin";
  header.model_bin_size = std::filesystem::file_size(model_bin, ec);
  if (ec) {
    header.model_bin_size = 0;
  }
  header.model_bin_mtime = file_time_key(model_bin);
  header.model_bin_hash = file_content_hash64(model_bin);
  return header;
}

inline bool same_hash_index_cache_header(const HashIndexCacheHeader& lhs,
                                         const HashIndexCacheHeader& rhs) {
  return lhs.magic == rhs.magic && lhs.version == rhs.version &&
         lhs.dim == rhs.dim && lhs.hash_bits == rhs.hash_bits &&
         lhs.base_count == rhs.base_count && lhs.base_size == rhs.base_size &&
         lhs.base_mtime == rhs.base_mtime &&
         lhs.model_bin_size == rhs.model_bin_size &&
         lhs.model_bin_hash == rhs.model_bin_hash;
}

inline bool read_hash_index_cache64(const std::filesystem::path& path,
                                    const HashIndexCacheHeader& expected,
                                    std::vector<HashEntry64>* entries) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  HashIndexCacheHeader header;
  in.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!in || !same_hash_index_cache_header(header, expected)) {
    return false;
  }
  std::size_t count = static_cast<std::size_t>(expected.base_count);
  std::vector<HashIndexCacheEntry64> raw(count);
  if (count > 0) {
    in.read(reinterpret_cast<char*>(raw.data()),
            static_cast<std::streamsize>(raw.size() * sizeof(HashIndexCacheEntry64)));
  }
  if (!in) {
    return false;
  }
  entries->clear();
  entries->reserve(count);
  for (const auto& item : raw) {
    if (item.index >= count) {
      entries->clear();
      return false;
    }
    entries->push_back(HashEntry64{item.hash, item.index});
  }
  if (!std::is_sorted(entries->begin(), entries->end(), [](const auto& a, const auto& b) {
        if (a.hash != b.hash) return a.hash < b.hash;
        return a.index < b.index;
      })) {
    entries->clear();
    return false;
  }
  return true;
}

inline bool write_hash_index_cache64(const std::filesystem::path& path,
                                     const HashIndexCacheHeader& header,
                                     const std::vector<HashEntry64>& entries) {
  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  out.write(reinterpret_cast<const char*>(&header), sizeof(header));
  std::vector<HashIndexCacheEntry64> raw;
  raw.reserve(entries.size());
  for (const auto& entry : entries) {
    raw.push_back(HashIndexCacheEntry64{entry.hash, entry.index, 0});
  }
  if (!raw.empty()) {
    out.write(reinterpret_cast<const char*>(raw.data()),
              static_cast<std::streamsize>(raw.size() * sizeof(HashIndexCacheEntry64)));
  }
  return static_cast<bool>(out);
}

} // namespace vortex::tools::eval

#endif // VORTEX_V1_TOOLS_VORTEX_EVAL_SUPPORT_H
