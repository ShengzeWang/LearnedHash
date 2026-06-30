#include "vortex_v1/codegen_loader.h"
#include "vortex_v1/types.h"
#include "vortex_v1/uint256.h"

#include "cli_common.h"

#include "vector_io/vector_io.hpp"

#include "vortex_eval_support.h"
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using vortex::tools::ArgList;
using vortex::tools::parse_u32;
using vortex::tools::parse_u32_list;
using vortex::tools::parse_u64;
using vortex::tools::validate_known_flags;
using namespace vortex::tools::eval;

void print_usage() {
  std::cout << "Usage: vortex_eval --codegen-dir <dir> (--base <fvecs> | --load-only) "
               "[options]\n";
  std::cout << "Options: --query --groundtruth --k --window --node-count-values --max-queries --include-self "
               "--bench --bench-breakdown --bench-count --bench-warmup "
               "--threads "
               "--truth-cache --hash-index-cache "
               "--load-only "
               "--disable-centroid-index --centroid-index-pivots --disable-query-cache\n";
}

} // namespace

using namespace vortex::tools::eval;

int main(int argc, char** argv) {
  try {
    ArgList args(argc, argv);
    if (argc < 2 || args.has("--help")) {
      print_usage();
      return 0;
    }
    validate_known_flags(args, {"--codegen-dir", "--base", "--query", "--groundtruth", "--k", "--window",
                                "--node-count-values",
                                "--max-queries", "--include-self", "--bench", "--bench-count",
                                "--bench-breakdown", "--bench-warmup", "--truth-cache",
                                "--hash-index-cache", "--load-only",
                                "--threads",
                                "--disable-centroid-index", "--centroid-index-pivots",
                                "--disable-query-cache", "--help"});

    std::filesystem::path codegen_dir = args.require("--codegen-dir");
    bool load_only = args.has("--load-only");
    if (!load_only && !args.value("--base").has_value()) {
      throw std::runtime_error("Missing required flag: --base");
    }
    std::filesystem::path base_path = args.value("--base").has_value()
        ? std::filesystem::path(*args.value("--base"))
        : std::filesystem::path();
    std::filesystem::path query_path = args.value("--query").has_value()
        ? std::filesystem::path(*args.value("--query"))
        : base_path;
    std::filesystem::path groundtruth_path = args.value("--groundtruth").has_value()
        ? std::filesystem::path(*args.value("--groundtruth"))
        : std::filesystem::path();

    uint32_t k = args.value("--k").has_value() ? parse_u32(*args.value("--k"), "--k") : 10;
    uint64_t window = args.value("--window").has_value()
        ? parse_u64(*args.value("--window"), "--window")
        : 0;
    std::vector<uint32_t> requested_node_counts =
        args.value("--node-count-values").has_value()
            ? parse_u32_list(*args.value("--node-count-values"), "--node-count-values")
            : std::vector<uint32_t>{};
    uint64_t max_queries = args.value("--max-queries").has_value()
        ? parse_u64(*args.value("--max-queries"), "--max-queries")
        : 0;
    bool run_bench_breakdown = args.has("--bench-breakdown");
    bool run_bench = args.has("--bench") || run_bench_breakdown;
    std::filesystem::path truth_cache_path = args.value("--truth-cache").has_value()
        ? std::filesystem::path(*args.value("--truth-cache"))
        : std::filesystem::path();
    std::filesystem::path hash_index_cache_path =
        args.value("--hash-index-cache").has_value()
            ? std::filesystem::path(*args.value("--hash-index-cache"))
            : std::filesystem::path();
    uint64_t bench_count = args.value("--bench-count").has_value()
        ? parse_u64(*args.value("--bench-count"), "--bench-count")
        : 0;
    uint64_t bench_warmup = args.value("--bench-warmup").has_value()
        ? parse_u64(*args.value("--bench-warmup"), "--bench-warmup")
        : 100;
    uint32_t requested_threads = args.value("--threads").has_value()
        ? parse_u32(*args.value("--threads"), "--threads")
        : 1;
    bool include_self = args.has("--include-self");
    bool enable_centroid_index = !args.has("--disable-centroid-index");
    uint32_t centroid_index_pivots = args.value("--centroid-index-pivots").has_value()
        ? parse_u32(*args.value("--centroid-index-pivots"), "--centroid-index-pivots")
        : 16;
    bool enable_query_cache = !args.has("--disable-query-cache");

    if (k == 0) {
      throw std::runtime_error("--k must be > 0");
    }

    vortex::CodegenHasher hasher;
    vortex::CodegenLoadOptions load_opts;
    load_opts.enable_centroid_index = enable_centroid_index;
    load_opts.centroid_index_pivots = centroid_index_pivots;
    load_opts.enable_query_cache = enable_query_cache;
    auto load_start = std::chrono::steady_clock::now();
    hasher.load(codegen_dir, load_opts);
    double load_time_ms = elapsed_ms(load_start);

    if (load_only) {
      std::cout << "vortex_eval\n";
      std::cout << "  mode: load_only\n";
      std::cout << "  dim: " << hasher.dim() << "\n";
      std::cout << "  hash_bits: " << hasher.hash_bits() << "\n";
      std::cout << "  requested_threads: " << requested_threads << "\n";
      std::cout << "  centroid_index_enabled: "
                << (enable_centroid_index ? "true" : "false") << "\n";
      std::cout << "  centroid_index_pivots: " << centroid_index_pivots << "\n";
      std::cout << "  query_cache_enabled: " << (enable_query_cache ? "true" : "false") << "\n";
      std::cout << "  load_time_ms: " << load_time_ms << "\n";
      std::cout << "  codegen_library_status: " << hasher.library_status() << "\n";
      std::cout << "  codegen_library_path: " << hasher.library_path().string() << "\n";
      return 0;
    }

    auto read_base_start = std::chrono::steady_clock::now();
    auto base = vector_io::read_fvecs(base_path.string());
    double read_base_time_ms = elapsed_ms(read_base_start);
    if (base.empty()) {
      throw std::runtime_error("Base dataset is empty");
    }

    auto read_query_start = std::chrono::steady_clock::now();
    auto query = vector_io::read_fvecs(query_path.string());
    double read_query_time_ms = elapsed_ms(read_query_start);
    if (query.empty()) {
      throw std::runtime_error("Query dataset is empty");
    }

    if (base.dim != query.dim) {
      throw std::runtime_error("Base/query dimension mismatch");
    }

    if (hasher.dim() != base.dim) {
      throw std::runtime_error("Hasher dim does not match dataset");
    }

    uint32_t hash_bits = hasher.hash_bits();
    std::size_t base_count = base.count;
    std::size_t query_count = query.count;
    if (base_count > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::runtime_error("Base dataset too large: max supported count is uint32_t");
    }
    if (max_queries > 0 && max_queries < query_count) {
      query_count = static_cast<std::size_t>(max_queries);
    }

    if (window == 0) {
      window = static_cast<uint64_t>(std::min<std::size_t>(1000, base_count / 10 + 1));
    }
    if (window >= base_count) {
      window = base_count > 0 ? static_cast<uint64_t>(base_count - 1) : 0;
    }
    uint32_t eval_threads = normalized_thread_count(requested_threads, base_count);
    std::vector<uint32_t> node_counts =
        normalize_node_counts(requested_node_counts, base_count);

    bool exclude_self = (!include_self && base_path == query_path);

    std::vector<HashEntry64> hashes64;
    std::vector<HashEntry256> hashes256;
    std::vector<uint64_t> base_hash64;
    std::vector<vortex::UInt256> base_hash256;
    double hash_base_time_ms = 0.0;
    double sort_base_time_ms = 0.0;
    double rank_index_time_ms = 0.0;
    double hash_index_cache_load_time_ms = 0.0;
    double hash_index_cache_write_time_ms = 0.0;
    std::string hash_index_cache_status =
        hash_index_cache_path.empty() ? "disabled" : "miss";
    HashIndexCacheHeader hash_index_header = make_hash_index_cache_header(
        base_path, codegen_dir, base, hash_bits);

    if (hash_bits <= 64) {
      bool hash_index_loaded_from_cache = false;
      if (!hash_index_cache_path.empty()) {
        auto cache_start = std::chrono::steady_clock::now();
        hash_index_loaded_from_cache =
            read_hash_index_cache64(hash_index_cache_path, hash_index_header, &hashes64);
        hash_index_cache_load_time_ms = elapsed_ms(cache_start);
        hash_index_cache_status = hash_index_loaded_from_cache ? "hit" : "miss";
      }
      base_hash64.resize(base_count);
      if (!hash_index_loaded_from_cache) {
        auto hash_base_start = std::chrono::steady_clock::now();
        hashes64.resize(base_count);
        parallel_for_indices(base_count, eval_threads, [&](std::size_t i) {
          const float* vec = base.values.data() + i * base.dim;
          bool ok = false;
          uint64_t h = hasher.hash64(vec, &ok);
          if (!ok) {
            throw std::runtime_error("Failed to hash base vector");
          }
          hashes64[i] = HashEntry64{h, static_cast<uint32_t>(i)};
          base_hash64[i] = h;
        });
        hash_base_time_ms = elapsed_ms(hash_base_start);
        auto sort_base_start = std::chrono::steady_clock::now();
        std::sort(hashes64.begin(), hashes64.end(), [](const auto& a, const auto& b) {
          if (a.hash != b.hash) return a.hash < b.hash;
          return a.index < b.index;
        });
        sort_base_time_ms = elapsed_ms(sort_base_start);
        if (!hash_index_cache_path.empty()) {
          auto cache_write_start = std::chrono::steady_clock::now();
          bool wrote = write_hash_index_cache64(hash_index_cache_path,
                                                hash_index_header,
                                                hashes64);
          hash_index_cache_write_time_ms = elapsed_ms(cache_write_start);
          if (!wrote) {
            hash_index_cache_status = "write_failed";
          }
        }
      }
    } else {
      if (!hash_index_cache_path.empty()) {
        hash_index_cache_status = "unsupported";
      }
      auto hash_base_start = std::chrono::steady_clock::now();
      hashes256.resize(base_count);
      base_hash256.resize(base_count);
      parallel_for_indices(base_count, eval_threads, [&](std::size_t i) {
        const float* vec = base.values.data() + i * base.dim;
        vortex::UInt256 h{};
        if (!hasher.hash(vec, &h)) {
          throw std::runtime_error("Failed to hash base vector");
        }
        hashes256[i] = HashEntry256{h, static_cast<uint32_t>(i)};
        base_hash256[i] = h;
      });
      hash_base_time_ms = elapsed_ms(hash_base_start);
      auto sort_base_start = std::chrono::steady_clock::now();
      std::sort(hashes256.begin(), hashes256.end(), [](const auto& a, const auto& b) {
        int cmp = vortex::compare(a.hash, b.hash);
        if (cmp != 0) return cmp < 0;
        return a.index < b.index;
      });
      sort_base_time_ms = elapsed_ms(sort_base_start);
    }

    auto rank_index_start = std::chrono::steady_clock::now();
    std::vector<std::size_t> rank_by_index(base_count, 0);
    if (hash_bits <= 64) {
      parallel_for_indices(hashes64.size(), eval_threads, [&](std::size_t i) {
        rank_by_index[hashes64[i].index] = i;
        base_hash64[hashes64[i].index] = hashes64[i].hash;
      });
    } else {
      parallel_for_indices(hashes256.size(), eval_threads, [&](std::size_t i) {
        rank_by_index[hashes256[i].index] = i;
        base_hash256[hashes256[i].index] = hashes256[i].hash;
      });
    }
    rank_index_time_ms = elapsed_ms(rank_index_start);

    uint32_t search_k = k + (exclude_self ? 1U : 0U);
    if (search_k > base.count) {
      search_k = static_cast<uint32_t>(base.count);
    }

    std::vector<faiss::idx_t> labels(query_count * search_k);
    double truth_cache_load_time_ms = 0.0;
    double truth_cache_write_time_ms = 0.0;
    double truth_groundtruth_load_time_ms = 0.0;
    double truth_search_time_ms = 0.0;
    std::string truth_cache_status = truth_cache_path.empty() ? "disabled" : "miss";
    TruthCacheHeader truth_header = make_truth_cache_header(
        base_path, query_path, base, query_count, search_k, exclude_self);
    bool truth_loaded_from_cache = false;
    if (!truth_cache_path.empty()) {
      auto cache_start = std::chrono::steady_clock::now();
      truth_loaded_from_cache = read_truth_cache(truth_cache_path, truth_header, &labels);
      truth_cache_load_time_ms = elapsed_ms(cache_start);
      truth_cache_status = truth_loaded_from_cache ? "hit" : "miss";
    }
    if (!truth_loaded_from_cache && !groundtruth_path.empty()) {
      auto groundtruth_start = std::chrono::steady_clock::now();
      truth_loaded_from_cache = read_groundtruth_ivecs(
          groundtruth_path, query_count, search_k, base_count, &labels);
      truth_groundtruth_load_time_ms = elapsed_ms(groundtruth_start);
      truth_cache_status = "groundtruth";
      if (!truth_cache_path.empty()) {
        auto cache_write_start = std::chrono::steady_clock::now();
        bool wrote = write_truth_cache(truth_cache_path, truth_header, labels);
        truth_cache_write_time_ms = elapsed_ms(cache_write_start);
        if (!wrote) {
          truth_cache_status = "groundtruth_write_failed";
        }
      }
    }
    if (!truth_loaded_from_cache) {
      auto truth_search_start = std::chrono::steady_clock::now();
      faiss::IndexFlatL2 index(static_cast<int>(base.dim));
      index.add(static_cast<faiss::idx_t>(base.count), base.values.data());

      std::vector<float> distances(query_count * search_k);
      index.search(static_cast<faiss::idx_t>(query_count),
                   query.values.data(),
                   static_cast<int>(search_k),
                   distances.data(),
                   labels.data());
      truth_search_time_ms = elapsed_ms(truth_search_start);
      if (!truth_cache_path.empty()) {
        auto cache_write_start = std::chrono::steady_clock::now();
        bool wrote = write_truth_cache(truth_cache_path, truth_header, labels);
        truth_cache_write_time_ms = elapsed_ms(cache_write_start);
        if (!wrote) {
          truth_cache_status = "write_failed";
        }
      }
    }

    std::size_t total_neighbors = 0;
    std::size_t total_in_window = 0;
    long double total_rank_distance = 0.0L;
    long double total_hash_distance = 0.0L;
    std::vector<std::size_t> same_node_hits(node_counts.size(), 0);
    std::vector<std::size_t> near_1_node_hits(node_counts.size(), 0);
    std::vector<std::size_t> near_2_node_hits(node_counts.size(), 0);
    std::vector<std::size_t> near_4_node_hits(node_counts.size(), 0);
    std::vector<long double> total_node_distance_norm(node_counts.size(), 0.0L);
    std::vector<long double> overlay_match_sum_by_count(node_counts.size(), 0.0L);
    std::vector<long double> overlay_match_weight_by_count(node_counts.size(), 0.0L);
    std::vector<double> query_overlay_match_scores;
    query_overlay_match_scores.reserve(query_count);

    const std::size_t window_size = static_cast<std::size_t>(window);

    auto eval_query_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < query_count; ++i) {
      const float* qvec = query.values.data() + i * query.dim;
      std::size_t pos = 0;
      uint64_t qhash64 = 0;
      vortex::UInt256 qhash256{};

      if (hash_bits <= 64) {
        bool ok = false;
        qhash64 = hasher.hash64(qvec, &ok);
        if (!ok) {
          throw std::runtime_error("Failed to hash query vector");
        }
        pos = lower_bound_hash(hashes64, qhash64);
      } else {
        if (!hasher.hash(qvec, &qhash256)) {
          throw std::runtime_error("Failed to hash query vector");
        }
        pos = lower_bound_hash(hashes256, qhash256);
      }

      if (pos >= base_count) {
        pos = base_count - 1;
      }

      std::size_t lo = (pos > window_size) ? (pos - window_size) : 0;
      std::size_t hi = std::min(base_count - 1, pos + window_size);

      std::size_t used = 0;
      long double query_overlay_match_sum = 0.0L;
      long double query_overlay_match_weight = 0.0L;
      for (uint32_t j = 0; j < search_k && used < k; ++j) {
        faiss::idx_t label = labels[i * search_k + j];
        if (label < 0) {
          continue;
        }
        uint32_t idx = static_cast<uint32_t>(label);
        if (exclude_self && idx == i) {
          continue;
        }

        double neighbor_weight = 1.0 / std::log2(static_cast<double>(used) + 2.0);
        std::size_t rank = rank_by_index[idx];
        if (rank >= lo && rank <= hi) {
          total_in_window += 1;
        }
        total_rank_distance += static_cast<long double>(
            (rank > pos) ? (rank - pos) : (pos - rank));
        for (std::size_t ni = 0; ni < node_counts.size(); ++ni) {
          uint32_t node_count = node_counts[ni];
          uint32_t query_node = static_cast<uint32_t>(
              (pos * static_cast<std::size_t>(node_count)) / base_count);
          uint32_t neighbor_node = static_cast<uint32_t>(
              (rank * static_cast<std::size_t>(node_count)) / base_count);
          if (query_node >= node_count) {
            query_node = node_count - 1;
          }
          if (neighbor_node >= node_count) {
            neighbor_node = node_count - 1;
          }
          uint32_t direct_distance = query_node > neighbor_node
              ? query_node - neighbor_node
              : neighbor_node - query_node;
          uint32_t node_distance =
              std::min(direct_distance, node_count - direct_distance);
          if (node_distance == 0) {
            same_node_hits[ni] += 1;
          }
          if (node_distance <= 1) {
            near_1_node_hits[ni] += 1;
          }
          if (node_distance <= 2) {
            near_2_node_hits[ni] += 1;
          }
          if (node_distance <= 4) {
            near_4_node_hits[ni] += 1;
          }
          double max_distance = std::max(1.0, static_cast<double>(node_count) * 0.5);
          total_node_distance_norm[ni] +=
              static_cast<long double>(static_cast<double>(node_distance) / max_distance);
          double match = 1.0 / (1.0 + static_cast<double>(node_distance));
          overlay_match_sum_by_count[ni] +=
              static_cast<long double>(neighbor_weight * match);
          overlay_match_weight_by_count[ni] +=
              static_cast<long double>(neighbor_weight);
          query_overlay_match_sum += static_cast<long double>(neighbor_weight * match);
          query_overlay_match_weight += static_cast<long double>(neighbor_weight);
        }

        if (hash_bits <= 64) {
          uint64_t bh = base_hash64[idx];
          uint64_t diff = (qhash64 >= bh) ? (qhash64 - bh) : (bh - qhash64);
          total_hash_distance += static_cast<long double>(diff);
        }

        used += 1;
      }

      if (query_overlay_match_weight > 0.0L) {
        query_overlay_match_scores.push_back(
            static_cast<double>(query_overlay_match_sum / query_overlay_match_weight));
      }
      total_neighbors += used;
    }
    double eval_query_time_ms = elapsed_ms(eval_query_start);

    if (total_neighbors == 0) {
      throw std::runtime_error("No neighbors evaluated (check inputs)");
    }

    long double mean_recall = static_cast<long double>(total_in_window) /
                              static_cast<long double>(total_neighbors);
    long double mean_rank_dist = total_rank_distance /
                                 static_cast<long double>(total_neighbors);
    long double mean_rank_dist_norm = mean_rank_dist /
                                      static_cast<long double>(base_count);
    std::vector<double> same_node_rates(node_counts.size(), 0.0);
    std::vector<double> near_1_node_rates(node_counts.size(), 0.0);
    std::vector<double> near_2_node_rates(node_counts.size(), 0.0);
    std::vector<double> near_4_node_rates(node_counts.size(), 0.0);
    std::vector<double> mean_node_distance_norm(node_counts.size(), 0.0);
    std::vector<double> overlay_match_by_count(node_counts.size(), 0.0);
    long double node_score_sum = 0.0L;
    long double overlay_match_sum = 0.0L;
    for (std::size_t ni = 0; ni < node_counts.size(); ++ni) {
      double denom = static_cast<double>(total_neighbors);
      same_node_rates[ni] = static_cast<double>(same_node_hits[ni]) / denom;
      near_1_node_rates[ni] = static_cast<double>(near_1_node_hits[ni]) / denom;
      near_2_node_rates[ni] = static_cast<double>(near_2_node_hits[ni]) / denom;
      near_4_node_rates[ni] = static_cast<double>(near_4_node_hits[ni]) / denom;
      mean_node_distance_norm[ni] =
          static_cast<double>(total_node_distance_norm[ni] /
                              static_cast<long double>(total_neighbors));
      if (overlay_match_weight_by_count[ni] > 0.0L) {
        overlay_match_by_count[ni] =
            static_cast<double>(overlay_match_sum_by_count[ni] /
                                overlay_match_weight_by_count[ni]);
      }
      overlay_match_sum += static_cast<long double>(overlay_match_by_count[ni]);
      double count_score =
          0.35 * same_node_rates[ni] +
          0.30 * near_1_node_rates[ni] +
          0.20 * near_2_node_rates[ni] +
          0.10 * near_4_node_rates[ni] +
          0.05 * std::max(0.0, 1.0 - mean_node_distance_norm[ni]);
      node_score_sum += static_cast<long double>(
          std::clamp(count_score, 0.0, 1.0));
    }
    double node_locality_score =
        node_counts.empty()
            ? 0.0
            : static_cast<double>(node_score_sum /
                                  static_cast<long double>(node_counts.size()));
    double overlay_match_score =
        node_counts.empty()
            ? 0.0
            : static_cast<double>(overlay_match_sum /
                                  static_cast<long double>(node_counts.size()));
    std::sort(query_overlay_match_scores.begin(), query_overlay_match_scores.end());
    double overlay_match_p05 = percentile_sorted(query_overlay_match_scores, 0.05);
    double overlay_match_p50 = percentile_sorted(query_overlay_match_scores, 0.50);
    double overlay_match_p95 = percentile_sorted(query_overlay_match_scores, 0.95);

    std::cout << "vortex_eval\n";
    std::cout << "  base_count: " << base_count << "\n";
    std::cout << "  query_count: " << query_count << "\n";
    std::cout << "  hash_bits: " << hash_bits << "\n";
    std::cout << "  eval_threads: " << eval_threads << "\n";
    std::cout << "  load_time_ms: " << load_time_ms << "\n";
    std::cout << "  codegen_library_status: " << hasher.library_status() << "\n";
    std::cout << "  codegen_library_path: " << hasher.library_path().string() << "\n";
    std::cout << "  read_base_time_ms: " << read_base_time_ms << "\n";
    std::cout << "  read_query_time_ms: " << read_query_time_ms << "\n";
    std::cout << "  hash_base_time_ms: " << hash_base_time_ms << "\n";
    std::cout << "  sort_base_time_ms: " << sort_base_time_ms << "\n";
    std::cout << "  rank_index_time_ms: " << rank_index_time_ms << "\n";
    std::cout << "  hash_index_cache: " << hash_index_cache_status << "\n";
    std::cout << "  hash_index_cache_load_time_ms: " << hash_index_cache_load_time_ms << "\n";
    std::cout << "  hash_index_cache_write_time_ms: " << hash_index_cache_write_time_ms << "\n";
    std::cout << "  truth_cache: " << truth_cache_status << "\n";
    std::cout << "  truth_cache_load_time_ms: " << truth_cache_load_time_ms << "\n";
    std::cout << "  truth_cache_write_time_ms: " << truth_cache_write_time_ms << "\n";
    std::cout << "  truth_groundtruth_load_time_ms: " << truth_groundtruth_load_time_ms << "\n";
    std::cout << "  truth_search_time_ms: " << truth_search_time_ms << "\n";
    std::cout << "  eval_query_time_ms: " << eval_query_time_ms << "\n";
    std::cout << "  k: " << k << "\n";
    std::cout << "  window: " << window << " (rank positions)\n";
    std::cout << "  exclude_self: " << (exclude_self ? "true" : "false") << "\n";
    std::cout << "  recall@k_in_window: " << static_cast<double>(mean_recall) << "\n";
    std::cout << "  mean_rank_distance: " << static_cast<double>(mean_rank_dist) << "\n";
    std::cout << "  mean_rank_distance_norm: " << static_cast<double>(mean_rank_dist_norm) << "\n";
    std::cout << "  node_count_values: " << format_node_counts(node_counts) << "\n";
    std::cout << "  overlay_match_score: " << overlay_match_score << "\n";
    std::cout << "  overlay_match_p05: " << overlay_match_p05 << "\n";
    std::cout << "  overlay_match_p50: " << overlay_match_p50 << "\n";
    std::cout << "  overlay_match_p95: " << overlay_match_p95 << "\n";
    std::cout << "  node_locality_score: " << node_locality_score << "\n";
    for (std::size_t ni = 0; ni < node_counts.size(); ++ni) {
      std::cout << "  overlay_match_score_" << node_counts[ni]
                << ": " << overlay_match_by_count[ni] << "\n";
      std::cout << "  node_locality_same_node_hit_rate_" << node_counts[ni]
                << ": " << same_node_rates[ni] << "\n";
      std::cout << "  node_locality_near_1_node_hit_rate_" << node_counts[ni]
                << ": " << near_1_node_rates[ni] << "\n";
      std::cout << "  node_locality_near_2_node_hit_rate_" << node_counts[ni]
                << ": " << near_2_node_rates[ni] << "\n";
      std::cout << "  node_locality_near_4_node_hit_rate_" << node_counts[ni]
                << ": " << near_4_node_rates[ni] << "\n";
      std::cout << "  node_locality_mean_node_distance_norm_" << node_counts[ni]
                << ": " << mean_node_distance_norm[ni] << "\n";
    }

    if (hash_bits <= 64) {
      long double denom = std::ldexp(1.0L, hash_bits);
      long double mean_hash_dist = total_hash_distance /
                                   static_cast<long double>(total_neighbors);
      long double mean_hash_dist_norm = (denom > 0.0L) ? (mean_hash_dist / denom) : 0.0L;
      std::cout << "  mean_hash_distance: " << static_cast<double>(mean_hash_dist) << "\n";
      std::cout << "  mean_hash_distance_norm: " << static_cast<double>(mean_hash_dist_norm) << "\n";
    }

    if (run_bench) {
      std::size_t bench_total = query.count;
      if (bench_count > 0 && bench_count < bench_total) {
        bench_total = static_cast<std::size_t>(bench_count);
      }
      if (bench_total == 0) {
        throw std::runtime_error("bench_count must be > 0");
      }

      std::size_t warmup = std::min<std::size_t>(bench_warmup, bench_total);
      for (std::size_t i = 0; i < warmup; ++i) {
        const float* vec = query.values.data() + i * query.dim;
        if (hash_bits <= 64) {
          bool ok = false;
          (void)hasher.hash64(vec, &ok);
          if (!ok) {
            throw std::runtime_error("Failed to hash query vector during warmup");
          }
        } else {
          vortex::UInt256 tmp{};
          if (!hasher.hash(vec, &tmp)) {
            throw std::runtime_error("Failed to hash query vector during warmup");
          }
        }
      }

      auto hash_once = [&](const float* vec, const char* phase) {
        if (hash_bits <= 64) {
          bool ok = false;
          (void)hasher.hash64(vec, &ok);
          if (!ok) {
            throw std::runtime_error(std::string("Failed to hash query vector during ") + phase);
          }
        } else {
          vortex::UInt256 tmp{};
          if (!hasher.hash(vec, &tmp)) {
            throw std::runtime_error(std::string("Failed to hash query vector during ") + phase);
          }
        }
      };

      LatencySummary hash_summary = measure_latency_ns(bench_total, [&](std::size_t i) {
        const float* vec = query.values.data() + i * query.dim;
        hash_once(vec, "bench");
      });

      std::cout << "  bench_count: " << bench_total << "\n";
      print_latency_summary("", hash_summary);

      if (run_bench_breakdown) {
        const float* first_vec = query.values.data();
        uint32_t centroid = 0;
        float dist2 = 0.0f;
        uint64_t cache_token = 0;
        bool nearest_available =
            hasher.nearest_active_centroid(first_vec, &centroid, &dist2, &cache_token);
        std::cout << "  nearest_latency_available: "
                  << (nearest_available ? "true" : "false") << "\n";
        if (nearest_available) {
          LatencySummary nearest_summary = measure_latency_ns(bench_total, [&](std::size_t i) {
            const float* vec = query.values.data() + i * query.dim;
            uint32_t local_centroid = 0;
            float local_dist2 = 0.0f;
            uint64_t local_cache_token = 0;
            if (!hasher.nearest_active_centroid(
                    vec, &local_centroid, &local_dist2, &local_cache_token)) {
              throw std::runtime_error("Failed nearest_active_centroid during bench");
            }
          });
          print_latency_summary("nearest_", nearest_summary);
        }

        vortex::CodegenHashCentroidResult first_result{};
        bool hash_with_centroid_available = hasher.hash_with_centroid(first_vec, &first_result);
        std::cout << "  hash_with_centroid_latency_available: "
                  << (hash_with_centroid_available ? "true" : "false") << "\n";
        if (hash_with_centroid_available) {
          LatencySummary with_centroid_summary =
              measure_latency_ns(bench_total, [&](std::size_t i) {
                const float* vec = query.values.data() + i * query.dim;
                vortex::CodegenHashCentroidResult result{};
                if (!hasher.hash_with_centroid(vec, &result)) {
                  throw std::runtime_error("Failed hash_with_centroid during bench");
                }
              });
          print_latency_summary("hash_with_centroid_", with_centroid_summary);
        }

        std::vector<vortex::CodegenAnnNeighbor> ann_neighbors;
        bool ann_available = hasher.ann_query(first_vec, k, &ann_neighbors);
        std::cout << "  ann_latency_available: "
                  << (ann_available ? "true" : "false") << "\n";
        if (ann_available) {
          LatencySummary ann_summary = measure_latency_ns(bench_total, [&](std::size_t i) {
            const float* vec = query.values.data() + i * query.dim;
            if (!hasher.ann_query(vec, k, &ann_neighbors)) {
              throw std::runtime_error("Failed ANN query during bench");
            }
          });
          print_latency_summary("ann_", ann_summary);

          LatencySummary hash_ann_summary = measure_latency_ns(bench_total, [&](std::size_t i) {
            const float* vec = query.values.data() + i * query.dim;
            hash_once(vec, "combined ANN bench");
            if (!hasher.ann_query(vec, k, &ann_neighbors)) {
              throw std::runtime_error("Failed ANN query during combined bench");
            }
          });
          print_latency_summary("hash_ann_", hash_ann_summary);
        }

        bool ann_cached_available = false;
        if (hash_with_centroid_available && first_result.cache_token != 0) {
          ann_cached_available =
              hasher.ann_query(first_vec, k, &ann_neighbors, first_result.cache_token);
        }
        std::cout << "  ann_cached_latency_available: "
                  << (ann_cached_available ? "true" : "false") << "\n";
        if (ann_cached_available) {
          std::vector<double> ann_latencies;
          ann_latencies.reserve(bench_total);
          for (std::size_t i = 0; i < bench_total; ++i) {
            const float* vec = query.values.data() + i * query.dim;
            vortex::CodegenHashCentroidResult seed{};
            if (!hasher.hash_with_centroid(vec, &seed) || seed.cache_token == 0) {
              throw std::runtime_error("Failed to seed cached ANN query during bench");
            }
            auto start = std::chrono::steady_clock::now();
            if (!hasher.ann_query(vec, k, &ann_neighbors, seed.cache_token)) {
              throw std::runtime_error("Failed cached ANN query during bench");
            }
            auto end = std::chrono::steady_clock::now();
            double ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            ann_latencies.push_back(ns);
          }
          print_latency_summary("ann_cached_", summarize_latency_ns(std::move(ann_latencies)));

          LatencySummary hash_ann_summary = measure_latency_ns(bench_total, [&](std::size_t i) {
            const float* vec = query.values.data() + i * query.dim;
            vortex::CodegenHashCentroidResult result{};
            if (!hasher.hash_with_centroid(vec, &result) || result.cache_token == 0) {
              throw std::runtime_error("Failed hash_with_centroid during combined bench");
            }
            if (!hasher.ann_query(vec, k, &ann_neighbors, result.cache_token)) {
              throw std::runtime_error("Failed cached ANN query during combined bench");
            }
          });
          print_latency_summary("hash_ann_cached_", hash_ann_summary);
        }
      }
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
