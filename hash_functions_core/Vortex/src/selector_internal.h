#ifndef VORTEX_V1_SELECTOR_INTERNAL_H
#define VORTEX_V1_SELECTOR_INTERNAL_H

#include "vortex_v1/model_selector.h"

#include "model_spec_utils.h"
#include "training_internal.h"

#include "vortex_v1/model.h"
#include "vortex_v1/nsw_csr.h"
#include "vortex_v1/training.h"
#include "vortex_v1/uint256.h"

#include "vector_io/vector_io.hpp"

#include <faiss/Index.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace vortex::selector_internal {

struct PhaseLimits {
  std::string name;
  uint32_t base_limit = 0;
  uint32_t query_limit = 0;
  uint32_t latency_iterations = 0;
  uint32_t latency_warmup = 0;
};

struct EvalDataset {
  vector_io::VectorStorage<float> base;
  vector_io::VectorStorage<float> query;
  std::vector<faiss::idx_t> truth_labels;
  std::vector<faiss::idx_t> self_base_indices;
  uint64_t base_identity = 0;
  uint32_t truth_k = 0;
  uint64_t window = 0;
  std::vector<uint64_t> windows;
  std::vector<uint32_t> node_counts;
  bool exclude_self = false;
};

struct SelectorRecommendationRoles {
  std::optional<SelectorCandidateMetrics> peak_recall;
  std::optional<SelectorCandidateMetrics> knee;
  std::optional<SelectorCandidateMetrics> fast;
  std::optional<SelectorCandidateMetrics> small;
};

struct TrainedCandidate {
  VortexModel model;
  double train_time_ms = 0.0;
  training_internal::VortexTrainingProfile train_profile;
  uint64_t model_memory_bytes = 0;
  uint64_t model_size_bytes = 0;
};

struct HashEntry64 {
  uint64_t hash = 0;
  uint32_t index = 0;
};

struct HashEntry256 {
  UInt256 hash;
  uint32_t index = 0;
};

struct EvalNearestCache {
  uint32_t dim = 0;
  std::size_t count = 0;
  uint64_t memory_bytes = 0;
  std::vector<uint32_t> centroid;
  std::vector<double> dist2;
  uint64_t distance_terms_evaluated = 0;
  uint64_t distance_terms_skipped = 0;
  bool used_early_abandon = false;
};

struct EvalNearestCacheMeta {
  uint64_t memory_bytes = 0;
  uint64_t last_access_tick = 0;
  uint32_t hits = 0;
  bool ready = false;
  bool calibration_tier = false;
};

struct EvalBaseRankCache {
  uint32_t dim = 0;
  std::size_t count = 0;
  bool hash_bits_64 = true;
  uint32_t hash_threads = 1;
  uint32_t rank_threads = 1;
  double nearest_cache_ms = 0.0;
  double hash_base_ms = 0.0;
  double sort_base_ms = 0.0;
  double rank_index_ms = 0.0;
  uint64_t memory_bytes = 0;
  std::vector<HashEntry64> hashed64;
  std::vector<HashEntry256> hashed256;
  std::vector<std::size_t> rank_by_index;
};

struct EvalBaseRankCacheMeta {
  uint64_t memory_bytes = 0;
  uint64_t last_access_tick = 0;
  uint32_t hits = 0;
  bool ready = false;
  bool calibration_tier = false;
  std::string attribution_key;
};

struct TrainCache {
  std::mutex mutex;
  std::mutex training_mutex;
  std::unordered_map<std::string, std::shared_future<TrainedCandidate>> future_by_key;
  std::unordered_map<std::string, std::shared_ptr<const training_internal::VortexTrainingBase>>
      training_base_by_key;
  std::unordered_map<std::string, std::shared_ptr<const std::vector<uint32_t>>>
      training_order_by_key;
  std::unordered_map<std::string, std::shared_future<std::shared_ptr<const training_internal::VortexCdfFit>>>
      training_cdf_by_key;
  std::unordered_map<std::string, std::shared_future<std::shared_ptr<const EvalNearestCache>>>
      nearest_cache_by_key;
  std::unordered_map<std::string, EvalNearestCacheMeta> nearest_cache_meta_by_key;
  std::unordered_map<std::string, std::shared_future<std::shared_ptr<const EvalBaseRankCache>>>
      base_rank_cache_by_key;
  std::unordered_map<std::string, EvalBaseRankCacheMeta> base_rank_cache_meta_by_key;
  std::unordered_map<std::string, SelectorBaseRankCacheAttribution>
      base_rank_cache_attribution_by_key;
  std::unordered_map<std::string, double> best_recall_by_context_k;
  std::unordered_map<std::string, double> best_quality_by_context_k;
  const vector_io::VectorStorage<float>* dataset = nullptr;
  uint64_t shared_dataset_bytes = 0;
  uint64_t memory_budget_bytes = 0;
  uint32_t max_model_trains = 0;
  uint32_t model_trains_started = 0;
  double max_observed_train_seconds = 0.0;
  double max_observed_eval_hash_seconds_per_vector = 0.0;
  uint32_t cache_hits = 0;
  uint32_t cache_misses = 0;
  double cache_wait_ms = 0.0;
  double training_lane_wait_ms = 0.0;
  uint32_t training_base_cache_hits = 0;
  uint32_t training_base_cache_misses = 0;
  uint32_t training_order_cache_hits = 0;
  uint32_t training_order_cache_misses = 0;
  uint32_t training_cdf_cache_hits = 0;
  uint32_t training_cdf_cache_misses = 0;
  uint32_t training_cdf_cache_failures = 0;
  double training_base_cache_wait_ms = 0.0;
  double training_order_cache_wait_ms = 0.0;
  double training_cdf_cache_wait_ms = 0.0;
  double training_base_cache_build_ms = 0.0;
  double training_order_cache_build_ms = 0.0;
  double training_cdf_cache_build_ms = 0.0;
  uint64_t training_base_cache_memory_bytes = 0;
  uint64_t training_order_cache_memory_bytes = 0;
  uint64_t training_cdf_cache_memory_bytes = 0;
  double train_time_ms_sum = 0.0;
  double train_gather_skeleton_ms_sum = 0.0;
  double train_cluster_assign_ms_sum = 0.0;
  double train_assign_centroids_ms_sum = 0.0;
  double train_centroid_order_ms_sum = 0.0;
  double train_range_alloc_ms_sum = 0.0;
  double train_cdf_fit_ms_sum = 0.0;
  double train_model_assembly_ms_sum = 0.0;
  uint32_t scheduler_ready_cached_candidates = 0;
  uint32_t scheduler_in_flight_candidates = 0;
  uint32_t scheduler_missing_candidates = 0;
  uint32_t scheduler_single_admission_batches = 0;
  uint32_t scheduler_reuse_prioritized_candidates = 0;
  uint32_t scheduler_fresh_base_candidates = 0;
  uint32_t eval_nearest_cache_hits = 0;
  uint32_t eval_nearest_cache_misses = 0;
  uint32_t eval_nearest_cache_inflight_bypasses = 0;
  uint32_t eval_nearest_cache_prewarm_requests = 0;
  uint32_t eval_nearest_cache_prewarm_ready_models = 0;
  uint32_t eval_nearest_cache_prewarm_skipped = 0;
  uint32_t eval_nearest_cache_prewarm_failures = 0;
  uint32_t eval_nearest_cache_prewarm_threads = 0;
  double eval_nearest_cache_prewarm_ms = 0.0;
  double eval_nearest_cache_wait_ms = 0.0;
  double eval_nearest_cache_sampled_wait_ms = 0.0;
  double eval_nearest_cache_calibration_wait_ms = 0.0;
  double eval_nearest_cache_build_ms = 0.0;
  double eval_nearest_cache_sampled_build_ms = 0.0;
  double eval_nearest_cache_calibration_build_ms = 0.0;
  uint64_t eval_nearest_cache_distance_terms = 0;
  uint64_t eval_nearest_cache_skipped_distance_terms = 0;
  uint32_t eval_nearest_cache_early_abandon_builds = 0;
  uint32_t eval_nearest_cache_entries = 0;
  uint64_t eval_nearest_cache_memory_bytes = 0;
  uint64_t eval_nearest_cache_sampled_budget_bytes = 0;
  uint64_t eval_nearest_cache_sampled_memory_bytes = 0;
  uint64_t eval_nearest_cache_calibration_memory_bytes = 0;
  uint32_t eval_nearest_cache_evictions = 0;
  uint64_t eval_nearest_cache_evicted_bytes = 0;
  uint64_t eval_nearest_cache_access_tick = 0;
  uint32_t eval_base_rank_cache_hits = 0;
  uint32_t eval_base_rank_cache_misses = 0;
  double eval_base_rank_cache_wait_ms = 0.0;
  double eval_base_rank_cache_build_ms = 0.0;
  double eval_base_rank_cache_nearest_ms = 0.0;
  double eval_base_rank_cache_hash_ms = 0.0;
  double eval_base_rank_cache_sort_ms = 0.0;
  double eval_base_rank_cache_rank_index_ms = 0.0;
  double eval_base_rank_cache_overhead_ms = 0.0;
  uint32_t eval_base_rank_cache_entries = 0;
  uint64_t eval_base_rank_cache_memory_bytes = 0;
  uint64_t eval_base_rank_cache_sampled_budget_bytes = 0;
  uint64_t eval_base_rank_cache_sampled_memory_bytes = 0;
  uint64_t eval_base_rank_cache_calibration_memory_bytes = 0;
  uint32_t eval_base_rank_cache_evictions = 0;
  uint64_t eval_base_rank_cache_evicted_bytes = 0;
  uint64_t eval_base_rank_cache_access_tick = 0;
  uint32_t context_model_train_start = 0;
  uint32_t context_model_train_limit = 0;
  uint32_t peak_parallelism = 1;
  uint32_t peak_threads_per_candidate = 1;
  bool model_train_budget_hit = false;
  bool context_model_train_budget_hit = false;
  bool memory_budget_limited = false;
};

struct SearchController {
  std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
  std::optional<std::chrono::steady_clock::time_point> deadline;
  double final_calibration_reserve_seconds = 0.0;
  std::atomic<bool> time_budget_hit{false};
  std::atomic<bool> remaining_time_guard_hit{false};
  std::atomic<bool> final_calibration_reserve_hit{false};

  bool can_launch_more_work() {
    return can_launch_work_with_min_seconds(0.0);
  }

  bool can_launch_work_with_min_seconds(double min_remaining_seconds,
                                        double reserve_seconds = 0.0) {
    if (!deadline.has_value()) {
      return true;
    }
    auto now = std::chrono::steady_clock::now();
    if (now >= *deadline) {
      time_budget_hit.store(true, std::memory_order_relaxed);
      return false;
    }
    double clamped_seconds =
        std::max(0.0, min_remaining_seconds) + std::max(0.0, reserve_seconds);
    auto required = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(clamped_seconds));
    if (now + required < *deadline) {
      return true;
    }
    remaining_time_guard_hit.store(true, std::memory_order_relaxed);
    if (reserve_seconds > 0.0) {
      final_calibration_reserve_hit.store(true, std::memory_order_relaxed);
    }
    return false;
  }

  double elapsed_seconds() const {
    auto now = std::chrono::steady_clock::now();
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count()) /
           1e6;
  }
};

struct NswSelectorContext {
  std::filesystem::path nsw_path;
  std::string display_nsw_path;
  NswCsr csr;
  uint64_t target_skeleton = 0;
  std::vector<uint64_t> requested_target_skeletons;
  std::string skeleton_identity;
};

struct ScopedTempFiles {
  std::vector<std::filesystem::path> paths;

  ~ScopedTempFiles() {
    for (const auto& path : paths) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  }
};

struct Phase2SearchResult {
  std::vector<SelectorCandidateConfig> candidates;
  uint32_t refine_evaluated = 0;
  uint32_t refine_rounds = 0;
  uint32_t frontier_candidates = 0;
  uint32_t compacted_candidates = 0;
  uint32_t plateau_rounds = 0;
  double best_margin = 0.0;
  std::string stopped_reason;
};

struct ContextSelectionPlan {
  std::vector<std::size_t> indices;
  uint32_t probe_contexts_evaluated = 0;
};

struct EvalResourcePlan {
  uint32_t parallelism = 1;
  uint32_t threads_per_candidate = 1;
  bool memory_limited = false;
};

using SelectorClock = std::chrono::steady_clock;

inline constexpr double kNormEpsilon = 1e-12;

SelectorWeights default_selector_weights();
bool has_default_selector_weights(const SelectorWeights& weights);
SelectorWeights resolve_effective_weights(const SelectorOptions& options,
                                          std::size_t dataset_count,
                                          bool* auto_weighting_applied);
std::string normalize_selector_profile(std::string value);
std::string resolve_effective_selector_profile(const SelectorOptions& options,
                                               std::size_t dataset_count);
uint32_t resolve_context_limit(const SelectorOptions& options,
                               const std::string& effective_profile,
                               std::size_t dataset_count,
                               std::size_t context_count);
uint32_t resolve_model_train_budget(const SelectorOptions& options,
                                    const std::string& effective_profile,
                                    std::size_t dataset_count,
                                    std::size_t context_count);
uint32_t resolve_search_time_budget_seconds(const SelectorOptions& options,
                                            const std::string& effective_profile,
                                            std::size_t dataset_count);
uint32_t resolve_stagnant_context_limit(const SelectorOptions& options,
                                        const std::string& effective_profile,
                                        std::size_t context_count);
uint32_t resolve_post_exploration_candidate_count(const SelectorOptions& options,
                                                  uint32_t max_model_trains);
uint32_t resolve_final_calibration_full_count(const SelectorOptions& options,
                                              std::size_t dataset_count);
uint32_t resolve_final_calibration_screen_candidate_count(const SelectorOptions& options,
                                                          std::size_t dataset_count);
uint32_t resolve_final_calibration_screen_query_limit(const SelectorOptions& options,
                                                      std::size_t query_count);
uint64_t resolve_selector_memory_budget_bytes(const SelectorOptions& options);
using detail::split_model_spec;
using detail::trim_copy;
std::string selector_config_key(const SelectorCandidateConfig& config);
std::string selector_training_cache_key(const NswSelectorContext& context,
                                        const SelectorCandidateConfig& config,
                                        const SelectorOptions& options);
bool selector_training_base_includes_centroid_graph(
    const SelectorOptions& options,
    const SelectorCandidateConfig& config);
std::string selector_training_base_cache_key(const NswSelectorContext& context,
                                             const SelectorCandidateConfig& config,
                                             uint64_t seed,
                                             bool use_full_dataset_for_assignments,
                                             uint64_t assignment_sample_limit,
                                             bool include_centroid_graph);
std::string selector_training_order_cache_key(const std::string& base_key,
                                              const SelectorCandidateConfig& config);
std::string selector_training_cdf_cache_key(const std::string& base_key,
                                            const SelectorCandidateConfig& config);
std::vector<uint64_t> normalize_target_skeleton_values(std::vector<uint64_t> values);
std::vector<double> normalize_target_skeleton_percentages(std::vector<double> values);
bool uses_auto_target_skeleton_policy(const SelectorOptions& options);
std::vector<uint64_t> default_target_skeleton_values(const SelectorOptions& options,
                                                     std::size_t dataset_count);
uint64_t default_assignment_sample_limit(const SelectorOptions& options,
                                         std::size_t dataset_count);
std::vector<uint64_t> resolve_target_skeleton_values(const SelectorOptions& options,
                                                     std::size_t dataset_count);
uint64_t estimate_model_memory_bytes(const VortexModel& model);
std::filesystem::path unique_temp_path(const std::string& stem, const std::string& ext);
uint64_t serialized_model_size_bytes(const VortexModel& model);
std::vector<std::size_t> all_indices(std::size_t count);
std::vector<std::size_t> sample_indices(std::size_t count,
                                        std::size_t limit,
                                        uint64_t seed);
vector_io::VectorStorage<float> subset_vectors(const vector_io::VectorStorage<float>& src,
                                                const std::vector<std::size_t>& indices);
uint32_t nearest_value(const std::vector<uint32_t>& values, uint32_t target);
uint64_t nearest_value(const std::vector<uint64_t>& values, uint64_t target);
void dedup_sort(std::vector<uint32_t>* values);
void dedup_sort(std::vector<uint64_t>* values);
void dedup_strings(std::vector<std::string>* values);
std::vector<uint32_t> default_k_values(std::size_t dataset_count,
                                       std::size_t skeleton_count,
                                       bool optimize_for_recall);
std::vector<uint32_t> default_knn_values();
std::vector<uint32_t> default_knn_values(std::size_t skeleton_count);
std::vector<uint64_t> default_cdf_branching_values(std::size_t dataset_count,
                                                   uint32_t max_k,
                                                   bool optimize_for_recall);
std::vector<std::string> default_cdf_model_specs();
std::vector<std::string> default_cdf_model_specs(bool optimize_for_recall);
std::vector<uint32_t> default_two_opt_iterations();
double skeleton_capacity_scale(std::size_t skeleton_count,
                               std::size_t max_skeleton_count);
std::vector<uint32_t> context_capacity_values(const std::vector<uint32_t>& values,
                                              std::size_t skeleton_count,
                                              double scale,
                                              std::size_t min_keep,
                                              uint32_t anchor);
SelectorOptions apply_context_capacity_scaling(const SelectorOptions& options,
                                               std::size_t skeleton_count,
                                               std::size_t max_skeleton_count);
bool meets_recall_target(const SelectorCandidateMetrics& metrics, double recall_target);
double elapsed_ms(SelectorClock::time_point start);
void append_timing(std::vector<SelectorPhaseTiming>* timings,
                   const std::string& name,
                   SelectorClock::time_point start);
uint32_t count_ok_metrics(const std::vector<SelectorCandidateMetrics>& metrics);
uint32_t count_budget_failures(const std::vector<SelectorCandidateMetrics>& metrics);
uint32_t read_model_trains_started(TrainCache* cache);
bool read_context_budget_hit(TrainCache* cache);
uint32_t requested_context_count(const std::vector<NswSelectorContext>& contexts);
std::vector<SelectorCandidateConfig> fit_candidates_to_remaining_train_budget(
    const std::vector<SelectorCandidateConfig>& candidates,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    TrainCache* cache);
uint32_t fair_context_train_budget(TrainCache* cache,
                                   std::size_t remaining_contexts,
                                   uint32_t phase1_keep,
                                   uint32_t post_exploration_reserve);
const NswSelectorContext* find_context_for_metric(
    const std::vector<NswSelectorContext>& contexts,
    const SelectorCandidateMetrics& metric);
void validate_model_specs(const std::vector<std::string>& model_specs);
SelectorOptions normalize_selector_options(const SelectorOptions& raw,
                                           std::size_t skeleton_count);
SelectorCandidateConfig clamp_candidate_config(const SelectorCandidateConfig& raw,
                                               std::size_t skeleton_count);
std::vector<SelectorCandidateConfig> dedup_candidates(
    const std::vector<SelectorCandidateConfig>& candidates,
    std::size_t skeleton_count);
std::vector<SelectorCandidateConfig> build_phase1_candidates_internal(
    const SelectorOptions& options,
    std::size_t skeleton_count);
SelectorCandidateConfig build_context_probe_candidate(const SelectorOptions& options,
                                                      std::size_t skeleton_count,
                                                      uint64_t target_skeleton);

vector_io::VectorStorage<float> load_vectors_or_throw(const std::filesystem::path& path);
double percentile_ms(std::vector<uint64_t> values, double quantile);
uint32_t eval_worker_count(std::size_t item_count, uint32_t requested_threads);
EvalDataset build_eval_dataset(const vector_io::VectorStorage<float>& full_base,
                               const std::optional<vector_io::VectorStorage<float>>& full_query,
                               uint32_t truth_k,
                               uint64_t window,
                               const std::vector<uint64_t>& window_values,
                               const std::vector<uint32_t>& node_count_values,
                               uint32_t base_limit,
                               uint32_t query_limit,
                               uint64_t seed,
                               bool prefix_query = false);
const TrainedCandidate& get_or_train_candidate(const SelectorCandidateConfig& config,
                                               const SelectorOptions& options,
                                               const NswSelectorContext& context,
                                               TrainCache* cache,
                                               SearchController* search);
SelectorCandidateMetrics evaluate_model_metrics(const SelectorCandidateConfig& config,
                                                const TrainedCandidate& trained,
                                                const EvalDataset& eval,
                                                const PhaseLimits& phase,
                                                uint32_t eval_threads,
                                                const std::optional<double>& recall_target,
                                                TrainCache* cache,
                                                const NswSelectorContext* context = nullptr);
void prepare_eval_base_rank_cache_for_phase(TrainCache* cache,
                                            const PhaseLimits& phase);
uint64_t eval_base_rank_sampled_cache_budget_bytes(TrainCache* cache);
std::vector<SelectorBaseRankCacheAttribution> snapshot_eval_base_rank_cache_attribution(
    TrainCache* cache);
bool is_eval_base_rank_calibration_phase(const PhaseLimits& phase);
rm_model::CentroidRouter::QueryResult nearest_for_eval_cache_exact(
    const VortexModel& model,
    const float* vector,
    uint64_t* distance_terms_evaluated = nullptr);
std::shared_ptr<const EvalNearestCache> build_eval_nearest_cache(const VortexModel& model,
                                                                 const EvalDataset& eval,
                                                                 uint32_t eval_threads);
std::shared_ptr<const EvalNearestCache> get_or_build_eval_nearest_cache(
    const VortexModel& model,
    const EvalDataset& eval,
    uint32_t eval_threads,
    TrainCache* cache,
    bool calibration_tier);
bool can_use_eval_nearest_cache(const VortexModel& model,
                                const PhaseLimits& phase,
                                const EvalDataset& eval);

void assign_objective_scores(std::vector<SelectorCandidateMetrics>* metrics,
                             const SelectorWeights& weights);
double selector_quality_score(const SelectorCandidateMetrics& item);
SelectorRecommendationRoles choose_recommendation_roles(
    const std::vector<SelectorCandidateMetrics>& metrics,
    const std::optional<uint64_t>& size_budget_bytes,
    const std::optional<double>& recall_target);
std::vector<SelectorCandidateMetrics> pareto_front(
    const std::vector<SelectorCandidateMetrics>& metrics);
std::vector<SelectorCandidateMetrics> top_ok_candidates(
    const std::vector<SelectorCandidateMetrics>& metrics,
    std::size_t limit,
    const std::optional<uint64_t>& size_budget_bytes,
    const std::optional<double>& recall_target);

std::vector<SelectorCandidateConfig> collect_seed_configs(
    const std::vector<SelectorCandidateMetrics>& ranked,
    std::size_t keep);
std::vector<SelectorCandidateConfig> local_neighbors(const SelectorCandidateConfig& center,
                                                     const SelectorOptions& options,
                                                     std::size_t skeleton_count,
                                                     uint32_t radius,
                                                     uint32_t neighbor_limit);
std::vector<SelectorCandidateConfig> compact_attribution_guided_candidates(
    const std::vector<SelectorCandidateConfig>& candidates,
    const std::vector<SelectorCandidateMetrics>& phase1_metrics,
    const SelectorOptions& options,
    uint32_t weak_high_k_family_limit,
    uint32_t* pruned_out);
std::vector<SelectorCandidateConfig> expand_phase2_candidates_grid(
    const std::vector<SelectorCandidateMetrics>& ranked,
    const SelectorOptions& options,
    std::size_t skeleton_count);
std::vector<SelectorCandidateConfig> build_post_exploration_candidates(
    const std::vector<SelectorCandidateMetrics>& ranked,
    const SelectorOptions& options,
    std::size_t skeleton_count,
    uint32_t candidate_limit);
std::vector<SelectorCandidateMetrics> choose_final_calibration_seeds(
    const std::vector<SelectorCandidateMetrics>& pool,
    uint32_t limit,
    const std::optional<double>& recall_target);
std::vector<SelectorCandidateMetrics> choose_role_aware_final_calibration_seeds(
    const std::vector<SelectorCandidateMetrics>& screen_metrics,
    const std::vector<SelectorCandidateMetrics>& fallback_seeds,
    uint32_t limit,
    const std::optional<uint64_t>& size_budget_bytes,
    const std::optional<double>& recall_target,
    uint32_t* role_promoted,
    uint32_t* recall_promoted);
std::vector<SelectorCandidateMetrics> run_post_exploration(
    const std::vector<SelectorCandidateMetrics>& scored_pool,
    const std::vector<NswSelectorContext>& contexts,
    const SelectorOptions& eval_options,
    std::size_t max_skeleton_count,
    const EvalDataset& full_eval,
    const PhaseLimits& phase2_limits,
    uint32_t candidate_limit,
    TrainCache* train_cache,
    SearchController* search_controller);
SelectorCandidateMetrics evaluate_candidate(const SelectorCandidateConfig& config,
                                            const SelectorOptions& options,
                                            uint32_t eval_threads,
                                            const NswSelectorContext& context,
                                            const EvalDataset& eval,
                                            const PhaseLimits& phase,
                                            TrainCache* cache,
                                            SearchController* search);
EvalResourcePlan compute_eval_resource_plan(const SelectorOptions& options,
                                            const NswSelectorContext& context,
                                            const EvalDataset& eval,
                                            TrainCache* cache,
                                            std::size_t candidate_count);
void begin_context_train_budget(TrainCache* cache, uint32_t train_limit);
void end_context_train_budget(TrainCache* cache);
std::vector<SelectorCandidateMetrics> evaluate_candidates(
    const std::vector<SelectorCandidateConfig>& candidates,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    const EvalDataset& eval,
    const PhaseLimits& phase,
    TrainCache* cache,
    SearchController* search);
std::vector<std::size_t> evenly_spaced_indices(std::size_t count, std::size_t keep);
ContextSelectionPlan select_context_indices_for_full_search(
    const std::vector<NswSelectorContext>& contexts,
    const SelectorOptions& effective_raw,
    const EvalDataset& coarse_eval,
    const PhaseLimits& phase1_limits,
    std::size_t dataset_count,
    std::size_t max_skeleton_count,
    const std::string& effective_profile,
    TrainCache* train_cache,
    SearchController* search);
Phase2SearchResult search_phase2_candidates_hybrid(
    const std::vector<SelectorCandidateMetrics>& phase1_metrics,
    const std::vector<SelectorCandidateMetrics>& phase1_ranked,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    const EvalDataset& coarse_eval,
    const PhaseLimits& coarse_limits,
    std::size_t skeleton_count,
    TrainCache* cache,
    SearchController* search);
bool has_valid_metrics(const std::vector<SelectorCandidateMetrics>& metrics);
double best_recall_in_metrics(const std::vector<SelectorCandidateMetrics>& metrics);
double best_quality_in_metrics(const std::vector<SelectorCandidateMetrics>& metrics);
double quality_margin_in_ranked(const std::vector<SelectorCandidateMetrics>& ranked);
uint32_t adaptive_phase2_candidate_limit(const SelectorOptions& options,
                                         std::size_t skeleton_count,
                                         std::size_t max_skeleton_count,
                                         double context_phase1_best_recall,
                                         double global_best_recall,
                                         bool recall_target_met_so_far);
SelectorOptions tune_phase2_search_options(const SelectorOptions& options,
                                           uint32_t phase2_limit);
SelectorCandidateConfig choose_recall_anchor(
    const std::vector<SelectorCandidateMetrics>& metrics,
    const SelectorCandidateConfig& fallback);
SelectorCandidateConfig choose_quality_anchor(
    const std::vector<SelectorCandidateMetrics>& metrics,
    const SelectorCandidateConfig& fallback);
std::vector<SelectorCandidateMetrics> refine_phase2_for_recall(
    const std::vector<SelectorCandidateMetrics>& seed_metrics,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    const EvalDataset& full_eval,
    const PhaseLimits& phase2_limits,
    std::size_t skeleton_count,
    TrainCache* cache,
    SearchController* search,
    uint32_t* rounds_out);

rm_model::json::Value selector_config_to_json(const SelectorCandidateConfig& config);
rm_model::json::Value selector_candidate_to_json(const SelectorCandidateMetrics& metrics,
                                                 const SelectorOptions& options);

std::vector<NswSelectorContext> build_nsw_contexts(const SelectorOptions& raw,
                                                   const vector_io::VectorStorage<float>& full_base,
                                                   ScopedTempFiles* temp_files);

} // namespace vortex::selector_internal

#endif // VORTEX_V1_SELECTOR_INTERNAL_H
