#include "vector_io/vector_io.hpp"

#include "cli_common.h"

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

using vortex::tools::ArgList;
using vortex::tools::parse_u32;
using vortex::tools::parse_u64;
using vortex::tools::validate_known_flags;

void print_usage() {
  std::cout << "Usage: hnsw_eval --base <fvecs> --query <fvecs> [options]\n";
  std::cout << "Options: --output-index --k --M --efConstruction --efSearch "
               "--max-queries --warmup\n";
}

std::filesystem::path unique_temp_index_path() {
  auto temp_dir = std::filesystem::temp_directory_path();
  std::random_device rd;
  for (int attempt = 0; attempt < 64; ++attempt) {
    uint64_t suffix = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    auto candidate = temp_dir / ("vortex_hnsw_eval_" + std::to_string(suffix) + ".index");
    if (!std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  throw std::runtime_error("Could not allocate temporary HNSW index path");
}

} // namespace

int main(int argc, char** argv) {
  try {
    ArgList args(argc, argv);
    if (argc < 2 || args.has("--help")) {
      print_usage();
      return 0;
    }
    validate_known_flags(args, {"--base", "--query", "--output-index", "--k", "--M",
                                "--efConstruction", "--efSearch", "--max-queries",
                                "--warmup", "--help"});

    std::filesystem::path base_path = args.require("--base");
    std::filesystem::path query_path = args.require("--query");
    std::optional<std::filesystem::path> output_index;
    if (args.value("--output-index").has_value()) {
      output_index = std::filesystem::path(*args.value("--output-index"));
    }

    uint32_t k = args.value("--k").has_value() ? parse_u32(*args.value("--k"), "--k") : 10;
    uint32_t M = args.value("--M").has_value() ? parse_u32(*args.value("--M"), "--M") : 16;
    uint32_t efc = args.value("--efConstruction").has_value()
        ? parse_u32(*args.value("--efConstruction"), "--efConstruction")
        : 100;
    uint32_t efSearch = args.value("--efSearch").has_value()
        ? parse_u32(*args.value("--efSearch"), "--efSearch")
        : 64;
    uint64_t max_queries = args.value("--max-queries").has_value()
        ? parse_u64(*args.value("--max-queries"), "--max-queries")
        : 0;
    uint64_t warmup = args.value("--warmup").has_value()
        ? parse_u64(*args.value("--warmup"), "--warmup")
        : 100;

    if (k == 0) {
      throw std::runtime_error("--k must be > 0");
    }

    auto base = vector_io::read_fvecs(base_path.string());
    if (base.empty()) {
      throw std::runtime_error("Base dataset is empty");
    }
    auto query = vector_io::read_fvecs(query_path.string());
    if (query.empty()) {
      throw std::runtime_error("Query dataset is empty");
    }
    if (base.dim != query.dim) {
      throw std::runtime_error("Base/query dimension mismatch");
    }

    std::size_t query_count = query.count;
    if (max_queries > 0 && max_queries < query_count) {
      query_count = static_cast<std::size_t>(max_queries);
    }
    if (query_count == 0) {
      throw std::runtime_error("No queries to evaluate");
    }

    faiss::IndexHNSWFlat index(static_cast<int>(base.dim), static_cast<int>(M),
                               faiss::METRIC_L2);
    index.hnsw.efConstruction = static_cast<int>(efc);
    index.add(static_cast<faiss::idx_t>(base.count), base.values.data());
    index.hnsw.efSearch = static_cast<int>(efSearch);

    std::filesystem::path index_path;
    bool keep_index = output_index.has_value();
    if (keep_index) {
      index_path = *output_index;
      if (!index_path.parent_path().empty()) {
        std::filesystem::create_directories(index_path.parent_path());
      }
    } else {
      index_path = unique_temp_index_path();
    }

    std::uintmax_t index_size = 0;
    try {
      faiss::write_index(&index, index_path.string().c_str());
      index_size = std::filesystem::file_size(index_path);
    } catch (...) {
      if (!keep_index) {
        std::error_code ignored;
        std::filesystem::remove(index_path, ignored);
      }
      throw;
    }
    if (!keep_index) {
      std::error_code ignored;
      std::filesystem::remove(index_path, ignored);
    }

    std::size_t warmup_count = std::min<std::size_t>(static_cast<std::size_t>(warmup), query_count);
    std::vector<faiss::idx_t> labels(k);
    std::vector<float> distances(k);

    for (std::size_t i = 0; i < warmup_count; ++i) {
      const float* vec = query.values.data() + i * query.dim;
      index.search(1, vec, static_cast<int>(k), distances.data(), labels.data());
    }

    std::vector<double> latencies;
    latencies.reserve(query_count);
    for (std::size_t i = 0; i < query_count; ++i) {
      const float* vec = query.values.data() + i * query.dim;
      auto start = std::chrono::steady_clock::now();
      index.search(1, vec, static_cast<int>(k), distances.data(), labels.data());
      auto end = std::chrono::steady_clock::now();
      double ns = static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
      latencies.push_back(ns);
    }

    std::sort(latencies.begin(), latencies.end());
    auto percentile = [&](double p) -> double {
      if (latencies.empty()) return 0.0;
      double idx = p * static_cast<double>(latencies.size() - 1);
      std::size_t lo = static_cast<std::size_t>(idx);
      std::size_t hi = std::min<std::size_t>(lo + 1, latencies.size() - 1);
      double frac = idx - static_cast<double>(lo);
      return latencies[lo] * (1.0 - frac) + latencies[hi] * frac;
    };

    double sum = 0.0;
    for (double v : latencies) {
      sum += v;
    }
    double avg_ns = latencies.empty() ? 0.0 : (sum / latencies.size());
    auto to_ms = [](double ns) { return ns / 1.0e6; };

    std::cout << "hnsw_eval\n";
    std::cout << "  base_count: " << base.count << "\n";
    std::cout << "  query_count: " << query_count << "\n";
    std::cout << "  dim: " << base.dim << "\n";
    std::cout << "  k: " << k << "\n";
    std::cout << "  M: " << M << "\n";
    std::cout << "  efConstruction: " << efc << "\n";
    std::cout << "  efSearch: " << efSearch << "\n";
    if (keep_index) {
      std::cout << "  output_index: " << index_path.string() << "\n";
    }
    std::cout << "  index_size_bytes: " << index_size << "\n";
    std::cout << "  index_size_mb: " << (static_cast<double>(index_size) / (1024.0 * 1024.0)) << "\n";
    std::cout << "  latency_avg_ms: " << to_ms(avg_ns) << "\n";
    std::cout << "  latency_p95_ms: " << to_ms(percentile(0.95)) << "\n";
    std::cout << "  latency_p99_ms: " << to_ms(percentile(0.99)) << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
