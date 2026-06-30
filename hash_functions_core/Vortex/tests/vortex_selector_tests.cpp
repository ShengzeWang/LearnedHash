#include "vortex_selector_test_common.h"

using namespace vortex_test;

namespace {

enum class JsonFieldKind {
  kAny,
  kObject,
  kArray,
  kNumber,
  kString,
  kBool,
};

struct JsonFieldExpectation {
  const char* name;
  JsonFieldKind kind;
};

bool json_value_matches_kind(const rm_model::json::Value& value,
                             JsonFieldKind kind) {
  switch (kind) {
    case JsonFieldKind::kAny:
      return true;
    case JsonFieldKind::kObject:
      return value.is_object();
    case JsonFieldKind::kArray:
      return value.is_array();
    case JsonFieldKind::kNumber:
      return value.is_number();
    case JsonFieldKind::kString:
      return value.is_string();
    case JsonFieldKind::kBool:
      return value.is_bool();
  }
  return false;
}

const char* json_kind_name(JsonFieldKind kind) {
  switch (kind) {
    case JsonFieldKind::kAny:
      return "present";
    case JsonFieldKind::kObject:
      return "object";
    case JsonFieldKind::kArray:
      return "array";
    case JsonFieldKind::kNumber:
      return "number";
    case JsonFieldKind::kString:
      return "string";
    case JsonFieldKind::kBool:
      return "bool";
  }
  return "unknown";
}

void check_json_fields(TestContext& ctx,
                       const rm_model::json::Value& object,
                       const std::vector<JsonFieldExpectation>& fields,
                       const std::string& scope) {
  ctx.check(object.is_object(), scope + " is an object");
  if (!object.is_object()) {
    return;
  }
  for (const auto& field : fields) {
    const auto* value = object.find(field.name);
    ctx.check(value != nullptr,
              scope + " includes `" + field.name + "`");
    if (value) {
      ctx.check(json_value_matches_kind(*value, field.kind),
                scope + " field `" + field.name + "` is " +
                    json_kind_name(field.kind));
    }
  }
}

void check_selector_json_stable_surface(TestContext& ctx,
                                        const rm_model::json::Value& json,
                                        const std::string& scope) {
  check_json_fields(
      ctx,
      json,
      {
          {"selector_json_schema_version", JsonFieldKind::kNumber},
          {"selector_json_mode", JsonFieldKind::kString},
          {"dataset", JsonFieldKind::kString},
          {"nsw", JsonFieldKind::kString},
          {"query", JsonFieldKind::kString},
          {"hash_bits", JsonFieldKind::kNumber},
          {"threads", JsonFieldKind::kNumber},
          {"seed", JsonFieldKind::kNumber},
          {"selection", JsonFieldKind::kObject},
          {"candidates", JsonFieldKind::kArray},
          {"pareto_front", JsonFieldKind::kArray},
          {"recommended", JsonFieldKind::kArray},
          {"recommendation_roles", JsonFieldKind::kObject},
          {"configs", JsonFieldKind::kArray},
          {"best", JsonFieldKind::kAny},
          {"selected_target_skeleton", JsonFieldKind::kAny},
      },
      scope);
}

void check_selector_diagnostics_stable_surface(
    TestContext& ctx,
    const rm_model::json::Value& diagnostics,
    const std::string& scope) {
  check_json_fields(
      ctx,
      diagnostics,
      {
          {"schema_version", JsonFieldKind::kNumber},
          {"search", JsonFieldKind::kObject},
          {"training", JsonFieldKind::kObject},
          {"scheduler", JsonFieldKind::kObject},
          {"evaluation", JsonFieldKind::kObject},
          {"strategy", JsonFieldKind::kObject},
          {"final_calibration", JsonFieldKind::kObject},
          {"memory", JsonFieldKind::kObject},
          {"errors", JsonFieldKind::kObject},
      },
      scope);
  if (!diagnostics.is_object()) {
    return;
  }
  if (const auto* search = diagnostics.find("search")) {
    check_json_fields(ctx,
                      *search,
                      {
                          {"contexts", JsonFieldKind::kObject},
                          {"selector_profile", JsonFieldKind::kString},
                          {"effective_selector_profile", JsonFieldKind::kString},
                          {"stopped_reason", JsonFieldKind::kString},
                          {"best_quality", JsonFieldKind::kNumber},
                      },
                      scope + ".search");
  }
  if (const auto* training = diagnostics.find("training")) {
    check_json_fields(ctx,
                      *training,
                      {
                          {"model_cache", JsonFieldKind::kObject},
                          {"base_cache", JsonFieldKind::kObject},
                          {"order_cache", JsonFieldKind::kObject},
                          {"cdf_cache", JsonFieldKind::kObject},
                          {"stage_ms_sum", JsonFieldKind::kObject},
                      },
                      scope + ".training");
  }
  if (const auto* evaluation = diagnostics.find("evaluation")) {
    check_json_fields(ctx,
                      *evaluation,
                      {
                          {"nearest_cache", JsonFieldKind::kObject},
                          {"base_rank_cache", JsonFieldKind::kObject},
                      },
                      scope + ".evaluation");
  }
  if (const auto* strategy = diagnostics.find("strategy")) {
    check_json_fields(ctx,
                      *strategy,
                      {
                          {"primary_metric", JsonFieldKind::kString},
                          {"deployment_metric", JsonFieldKind::kString},
                          {"compatibility_locality_metric", JsonFieldKind::kString},
                          {"quality_score_formula", JsonFieldKind::kString},
                      },
                      scope + ".strategy");
  }
}

void check_selector_selection_stable_surface(
    TestContext& ctx,
    const rm_model::json::Value& selection,
    const std::string& scope) {
  check_json_fields(
      ctx,
      selection,
      {
          {"schema_version", JsonFieldKind::kNumber},
          {"flat_compatibility_fields", JsonFieldKind::kBool},
          {"phase1_keep", JsonFieldKind::kNumber},
          {"max_phase2_candidates", JsonFieldKind::kNumber},
          {"attribution_guided_compaction", JsonFieldKind::kBool},
          {"use_full_dataset_for_assignments", JsonFieldKind::kBool},
          {"auto_assignment_sample_limit", JsonFieldKind::kBool},
          {"assignment_sample_limit", JsonFieldKind::kNumber},
          {"assignment_sample_policy", JsonFieldKind::kString},
          {"graph_centroid_order", JsonFieldKind::kBool},
          {"graph_centroid_order_evidence", JsonFieldKind::kObject},
          {"recommend_count", JsonFieldKind::kNumber},
          {"selector_parallelism", JsonFieldKind::kNumber},
          {"selector_profile", JsonFieldKind::kString},
          {"effective_selector_profile", JsonFieldKind::kString},
          {"objective_score_scope", JsonFieldKind::kString},
          {"objective_score_compare_rule", JsonFieldKind::kString},
          {"diagnostics_schema_version", JsonFieldKind::kNumber},
          {"diagnostics", JsonFieldKind::kObject},
          {"eval_node_counts", JsonFieldKind::kArray},
          {"dataset_count", JsonFieldKind::kNumber},
          {"eval_k", JsonFieldKind::kNumber},
          {"eval_window", JsonFieldKind::kNumber},
          {"eval_windows", JsonFieldKind::kArray},
          {"coarse_base_limit", JsonFieldKind::kNumber},
          {"coarse_query_limit", JsonFieldKind::kNumber},
          {"eval_base_limit", JsonFieldKind::kNumber},
          {"eval_query_limit", JsonFieldKind::kNumber},
          {"coarse_latency_iterations", JsonFieldKind::kNumber},
          {"latency_iterations", JsonFieldKind::kNumber},
          {"weights", JsonFieldKind::kObject},
          {"requested_weights", JsonFieldKind::kObject},
          {"auto_dataset_weighting", JsonFieldKind::kBool},
          {"auto_dataset_weighting_applied", JsonFieldKind::kBool},
      },
      scope);
  if (selection.is_object()) {
    if (const auto* resolved = selection.find("resolved_target_skeleton_values")) {
      ctx.check(resolved->is_array(),
                scope + " optional `resolved_target_skeleton_values` is array");
    }
    if (const auto* diagnostics = selection.find("diagnostics")) {
      check_selector_diagnostics_stable_surface(ctx,
                                                *diagnostics,
                                                scope + ".diagnostics");
    }
  }
}

void test_shared_model_spec_helpers(TestContext& ctx) {
  auto parts = vortex::selector_internal::split_model_spec(" linear , , cubic \t,");
  ctx.check(parts.size() == 2 &&
                parts[0] == "linear" &&
                parts[1] == "cubic",
            "shared model-spec splitter trims and skips empty layers");
  ctx.check(vortex::selector_internal::trim_copy("\n linear,linear \r") == "linear,linear",
            "shared trim_copy removes surrounding ASCII whitespace");
}

void test_model_selector_small_dataset(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(4, 64);
  storage.values.resize(storage.count * storage.dim);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& value : storage.values) {
    value = dist(rng);
  }
  auto dataset_path = unique_temp_path("vortex_selector_dataset.fvecs");
  vector_io::write_fvecs(dataset_path.string(), storage);

  vortex::NswCsr csr;
  csr.dim = storage.dim;
  csr.metric = vortex::Metric::L2;
  csr.layer = 0;
  csr.node_ids.resize(storage.count);
  for (std::size_t i = 0; i < storage.count; ++i) {
    csr.node_ids[i] = static_cast<uint64_t>(i);
  }
  csr.offsets.assign(storage.count + 1, 0);
  auto nsw_path = unique_temp_path("vortex_selector_dataset.csr");
  csr.write(nsw_path);

  vortex::SelectorOptions opts;
  opts.dataset_path = dataset_path;
  opts.nsw_path = nsw_path;
  opts.hash_bits = 32;
  opts.threads = 1;
  opts.selector_parallelism = 0;
  opts.seed = 7;
  opts.K_values = {4, 8};
  opts.centroid_knn_values = {2, 4};
  opts.cdf_branching_values = {4, 8};
  opts.cdf_model_specs = {"linear,linear"};
  opts.two_opt_iterations_values = {2};
  opts.include_disable_2opt = true;
  opts.phase1_keep = 3;
  opts.max_phase2_candidates = 6;
  opts.recommend_count = 2;
  opts.eval_k = 5;
  opts.eval_window = 12;
  opts.eval_window_values = {6, 12, 24};
  opts.coarse_base_limit = 32;
  opts.coarse_query_limit = 16;
  opts.eval_base_limit = 48;
  opts.eval_query_limit = 16;
  opts.coarse_latency_iterations = 32;
  opts.coarse_latency_warmup = 8;
  opts.latency_iterations = 64;
  opts.latency_warmup = 16;
  opts.final_calibration_count = 2;
  opts.final_calibration_full_count = 1;
  opts.final_calibration_screen_query_limit = 4;
  opts.final_calibration_query_limit = 16;
  opts.post_exploration_candidates = 2;
  opts.selector_memory_budget_bytes = 1024 * 1024;
  opts.max_search_seconds = 120;

  auto result = vortex::select_vortex_models(opts);
  ctx.check(result.best.has_value(), "selector returns a best candidate");
  ctx.check(!result.recommended.empty(), "selector returns recommended candidates");
  ctx.check(result.dataset_count == storage.count,
            "selector reports dataset_count in result metadata");
  ctx.check(result.elapsed_search_seconds >= 0.0,
            "selector reports non-negative elapsed search time");
  ctx.check(result.target_skeleton_contexts_evaluated >= 1,
            "selector reports evaluated context count");
  ctx.check(result.target_skeleton_contexts_total == result.target_skeleton_contexts_unique,
            "selector preserves total as unique context count");
  ctx.check(result.target_skeleton_contexts_requested >= result.target_skeleton_contexts_unique,
            "selector reports requested and unique context counts");
  ctx.check(!result.timings.empty(), "selector reports phase timings");
  ctx.check(!result.context_stats.empty(), "selector reports context telemetry");
  ctx.check(!result.stopped_reason.empty(),
            "selector reports a stopped reason");
  ctx.check(result.best_quality >= 0.0 && result.best_quality <= 1.0,
            "selector reports bounded best quality");
  ctx.check(result.best_margin >= 0.0,
            "selector reports non-negative best quality margin");
  ctx.check(result.strategy_frontier_candidates >= result.strategy_compacted_candidates,
            "selector reports bounded attribution-guided compaction telemetry");
  ctx.check(!result.effective_eval_windows.empty(),
            "selector reports effective recall-window ladder");
  ctx.check(result.recommended_peak_recall.has_value(),
            "selector reports peak-recall recommendation role");
  if (result.recommended_peak_recall.has_value()) {
    const auto& peak = *result.recommended_peak_recall;
    for (const auto& item : result.candidates) {
      if (!item.ok || item.phase != peak.phase) {
        continue;
      }
      ctx.check(peak.recall_at_k_in_window + 1e-9 >= item.recall_at_k_in_window,
                "peak-recall recommendation ranks by measured recall first");
    }
  }
  ctx.check(result.recommended_knee.has_value(),
            "selector reports knee recommendation role");
  ctx.check(result.recommended_fast.has_value(),
            "selector reports fast recommendation role");
  ctx.check(result.recommended_small.has_value(),
            "selector reports small recommendation role");
  ctx.check(!result.context_stats.front().requested_target_skeletons.empty(),
            "selector context telemetry records requested target aliases");
  ctx.check(!result.context_stats.front().effective_K_values.empty(),
            "selector context telemetry records effective K ladder");
  ctx.check(!result.context_stats.front().promotion_decision.empty(),
            "selector context telemetry records promotion decision");
  ctx.check(!result.context_stats.front().phase2_stopped_reason.empty(),
            "selector context telemetry records phase2 stop reason");
  ctx.check(!result.context_stats.front().effective_centroid_knn_values.empty(),
            "selector context telemetry records effective centroid-knn ladder");
  ctx.check(!result.context_stats.front().effective_cdf_branching_values.empty(),
            "selector context telemetry records effective CDF branch ladder");
  ctx.check(result.final_calibration_seed_count >= result.final_calibration_evaluated,
            "selector reports final calibration seed count");
  ctx.check(result.effective_final_calibration_screen_candidate_count <=
                opts.final_calibration_count,
            "selector reports bounded calibration screen seed limit");
  ctx.check(result.final_calibration_screen_evaluated > 0,
            "selector evaluates final calibration screen when promotion is capped");
  ctx.check(result.final_calibration_screen_promoted <=
                result.effective_final_calibration_full_count,
            "selector caps screen-promoted final calibration candidates");
  ctx.check(result.final_calibration_role_promoted <=
                result.final_calibration_screen_promoted,
            "selector reports bounded role-promoted calibration candidates");
  ctx.check(result.final_calibration_recall_promoted <=
                result.final_calibration_screen_promoted,
            "selector reports bounded recall-promoted calibration candidates");
  ctx.check(result.final_calibration_role_promoted +
                result.final_calibration_recall_promoted <=
            result.final_calibration_screen_promoted,
            "selector reports screen promotion source counts");
  ctx.check(!result.final_calibration_screen_status.empty(),
            "selector reports calibration screen activation status");
  ctx.check(result.effective_final_calibration_screen_query_limit ==
                opts.final_calibration_screen_query_limit,
            "selector reports effective explicit calibration screen query limit");
  ctx.check(result.final_calibration_evaluated <= result.effective_final_calibration_full_count,
            "selector respects effective full calibration count");
  ctx.check(result.final_calibration_evaluated > 0,
            "selector evaluates final calibration candidates when requested");
  ctx.check(result.final_calibration_reserve_seconds > 0.0,
            "selector reports an active final calibration time reserve");
  ctx.check(result.final_calibration_reused <= result.final_calibration_evaluated,
            "selector reports bounded final calibration reuse count");
  ctx.check(!result.final_calibration_progressive_probes.empty(),
            "selector reports final calibration stability probes");
  if (!result.final_calibration_progressive_probes.empty()) {
    const auto& probe = result.final_calibration_progressive_probes.front();
    ctx.check(probe.evaluated > 0,
              "selector calibration stability probe reports evaluated candidates");
    ctx.check(probe.final_base_count >= probe.probe_base_count,
              "selector calibration stability probe compares against at least as much base data");
    ctx.check(probe.final_peak_recall >= 0.0 && probe.final_peak_recall <= 1.0,
              "selector calibration stability probe reports bounded final peak recall");
    ctx.check(probe.matched_role_count <= 4,
              "selector calibration stability probe reports bounded matched role count");
  }
  ctx.check(result.post_exploration_evaluated <= opts.post_exploration_candidates,
            "selector caps post-exploration candidate count");
  ctx.check(result.selector_peak_parallelism >= 1,
            "selector reports peak candidate parallelism");
  ctx.check(result.selector_peak_threads_per_candidate >= 1,
            "selector reports peak candidate thread width");
  ctx.check(result.train_cache_hits + result.train_cache_misses >= result.model_trains_started,
            "selector reports train cache hit/miss telemetry");
  ctx.check(result.train_cache_wait_ms >= 0.0 && result.training_lane_wait_ms >= 0.0,
            "selector reports non-negative training wait telemetry");
  ctx.check(result.training_base_cache_hits > 0 &&
                result.training_base_cache_misses > 0,
            "selector reuses cached centroid training bases");
  ctx.check(result.training_order_cache_hits > 0 &&
                result.training_order_cache_misses > 0,
            "selector reuses cached centroid orderings");
  ctx.check(result.training_cdf_cache_hits > 0 &&
                result.training_cdf_cache_misses > 0,
            "selector reuses cached CDF fits");
  ctx.check(result.training_cdf_cache_build_ms >= 0.0 &&
                result.training_cdf_cache_memory_bytes > 0,
            "selector reports CDF cache cost and memory");
  ctx.check(result.training_base_cache_build_ms >= 0.0 &&
                result.training_base_cache_memory_bytes > 0,
            "selector reports training-base cache cost and memory");
  ctx.check(result.train_time_ms_sum >= result.train_cdf_fit_ms_sum &&
                result.train_cluster_assign_ms_sum >= 0.0,
            "selector reports aggregate training-stage telemetry");
  ctx.check(result.scheduler_missing_candidates >= result.model_trains_started,
            "selector reports missing-candidate scheduler telemetry");
  ctx.check(result.scheduler_single_admission_batches > 0,
            "selector uses single-admission batches for fresh training work");
  ctx.check(result.scheduler_reuse_prioritized_candidates <=
                result.scheduler_missing_candidates,
            "selector reports bounded reuse-prioritized scheduler telemetry");
  ctx.check(result.scheduler_fresh_base_candidates <=
                result.scheduler_missing_candidates,
            "selector reports bounded fresh-base scheduler telemetry");
  ctx.check(result.eval_nearest_cache_hits + result.eval_nearest_cache_misses > 0,
            "selector uses nearest-centroid cache during final calibration");
  ctx.check(result.eval_nearest_cache_wait_ms >= 0.0 &&
                result.eval_nearest_cache_build_ms >= 0.0,
            "selector reports non-negative nearest-centroid cache telemetry");
  ctx.check(result.eval_base_rank_cache_hits > 0,
            "selector reuses base-rank cache between screen and final calibration");
  ctx.check(result.eval_base_rank_cache_misses > 0,
            "selector builds base-rank cache during calibration");
  ctx.check(result.eval_base_rank_cache_wait_ms >= 0.0 &&
                result.eval_base_rank_cache_build_ms >= 0.0,
            "selector reports non-negative base-rank cache telemetry");
  ctx.check(result.eval_base_rank_cache_nearest_ms >= 0.0 &&
                result.eval_base_rank_cache_hash_ms >= 0.0 &&
                result.eval_base_rank_cache_sort_ms >= 0.0 &&
                result.eval_base_rank_cache_rank_index_ms >= 0.0 &&
                result.eval_base_rank_cache_overhead_ms >= 0.0,
            "selector reports non-negative base-rank cache substage telemetry");
  ctx.check(result.eval_base_rank_cache_build_ms + 5.0 >=
                result.eval_base_rank_cache_nearest_ms +
                    result.eval_base_rank_cache_hash_ms +
                    result.eval_base_rank_cache_sort_ms +
                    result.eval_base_rank_cache_rank_index_ms,
            "selector base-rank cache total covers measured substages");
  ctx.check(result.eval_base_rank_cache_entries > 0 &&
                result.eval_base_rank_cache_memory_bytes > 0,
            "selector reports base-rank cache retained memory telemetry");
  ctx.check(!result.eval_base_rank_cache_attribution.empty(),
            "selector reports base-rank cache attribution rows");
  uint32_t attributed_hits = 0;
  uint32_t attributed_misses = 0;
  bool saw_role_attribution = false;
  for (const auto& row : result.eval_base_rank_cache_attribution) {
    attributed_hits += row.hits;
    attributed_misses += row.misses;
    ctx.check(!row.phase.empty(), "base-rank attribution includes phase");
    ctx.check(row.K > 0, "base-rank attribution includes K family");
    if (!row.roles.empty()) {
      saw_role_attribution = true;
    }
  }
  ctx.check(attributed_hits == result.eval_base_rank_cache_hits,
            "base-rank attribution hit count matches aggregate telemetry");
  ctx.check(attributed_misses == result.eval_base_rank_cache_misses,
            "base-rank attribution miss count matches aggregate telemetry");
  ctx.check(saw_role_attribution,
            "base-rank attribution marks final recommendation roles");
  ctx.check(result.eval_nearest_cache_early_abandon_builds <=
                result.eval_nearest_cache_misses,
            "selector reports bounded early-abandon cache build telemetry");
  ctx.check(result.effective_selector_memory_budget_bytes == opts.selector_memory_budget_bytes,
            "selector reports effective explicit memory budget");
  ctx.check(std::abs(result.effective_weights.recall - 0.55) < 1e-9,
            "selector default effective recall weight for small dataset");
  ctx.check(!result.auto_dataset_weighting_applied,
            "selector auto dataset weighting is not applied for small dataset");
  bool saw_scored_phase2_candidate = false;
  bool saw_final_calibration_screen_candidate = false;
  bool saw_final_calibration_candidate = false;
  for (const auto& item : result.candidates) {
    if (!item.ok) {
      continue;
    }
    ctx.check(item.objective_score >= 0.0,
              "selector stores non-negative objective scores on reported candidates");
    ctx.check(item.eval_total_ms >= 0.0 &&
                  item.eval_hash_base_ms >= 0.0 &&
                  item.eval_sort_base_ms >= 0.0 &&
                  item.eval_rank_index_ms >= 0.0 &&
                  item.eval_query_ms >= 0.0 &&
                  item.eval_latency_ms >= 0.0,
              "selector stores non-negative evaluation substage timings");
    ctx.check(item.eval_hash_base_threads >= 1 && item.eval_rank_index_threads >= 1,
              "selector stores evaluation thread counts");
    ctx.check(!item.recall_windows.empty(),
              "selector stores recall-window ladder on candidates");
    ctx.check(item.recall_windows.size() == item.recall_at_k_by_window.size(),
              "selector stores recall curve values for every window");
    ctx.check(item.recall_auc_log_window >= 0.0 &&
                  item.recall_auc_log_window <= 1.0,
              "selector stores bounded recall AUC");
    ctx.check(item.query_recall_p05 >= 0.0 && item.query_recall_p05 <= 1.0,
              "selector stores bounded query tail recall");
    ctx.check(!item.node_counts.empty(),
              "selector stores overlay node-count ladder on candidates");
    ctx.check(item.node_counts.size() == item.same_node_hit_by_count.size() &&
                  item.node_counts.size() == item.near_1_node_hit_by_count.size() &&
                  item.node_counts.size() == item.near_2_node_hit_by_count.size() &&
                  item.node_counts.size() == item.near_4_node_hit_by_count.size() &&
                  item.node_counts.size() == item.mean_node_distance_norm_by_count.size(),
              "selector stores node-locality curve values for every node count");
    ctx.check(item.node_locality_score >= 0.0 &&
                  item.node_locality_score <= 1.0,
              "selector stores bounded node locality score");
    ctx.check(item.overlay_match_score >= 0.0 &&
                  item.overlay_match_score <= 1.0,
              "selector stores bounded overlay match score");
    ctx.check(item.query_overlay_match_p05 >= 0.0 &&
                  item.query_overlay_match_p05 <= 1.0,
              "selector stores bounded query-tail overlay match");
    ctx.check(item.node_counts.size() == item.overlay_match_by_count.size(),
              "selector stores overlay-match curve values for every node count");
    ctx.check(item.locality_quality_score >= 0.0 &&
                  item.locality_quality_score <= 1.0,
              "selector stores bounded locality quality score");
    if (item.phase == "phase2") {
      saw_scored_phase2_candidate = true;
    }
    if (item.phase == "final_calibration_screen") {
      saw_final_calibration_screen_candidate = true;
    }
    if (item.phase == "final_calibration") {
      saw_final_calibration_candidate = true;
    }
  }
  ctx.check(saw_scored_phase2_candidate,
            "selector reports at least one scored phase2 candidate");
  ctx.check(saw_final_calibration_screen_candidate,
            "selector reports final calibration screen candidates when used");
  ctx.check(saw_final_calibration_candidate,
            "selector reports final calibration candidates when requested");
  if (result.best.has_value()) {
    ctx.check(result.best->ok, "selector best candidate is valid");
    ctx.check(result.best->recall_at_k_in_window >= 0.0 &&
                  result.best->recall_at_k_in_window <= 1.0,
              "selector best recall is in [0,1]");
  }

  auto json = vortex::selector_result_to_json(result, opts);
  ctx.check(json.is_object(), "selector json output is an object");
  if (json.is_object()) {
    check_selector_json_stable_surface(ctx, json, "compat selector json");
    const auto* schema_version = json.find("selector_json_schema_version");
    ctx.check(schema_version && schema_version->is_number() &&
                  schema_version->as_number().as_uint64() == 1,
              "default selector json uses compatibility schema version");
    const auto* json_mode = json.find("selector_json_mode");
    ctx.check(json_mode && json_mode->is_string() &&
                  json_mode->as_string() == "compat",
              "default selector json uses compatibility mode");
    const auto* rec = json.find("recommended");
    ctx.check(rec && rec->is_array(), "selector json contains recommended array");
    const auto* selection = json.find("selection");
    ctx.check(selection && selection->is_object(), "selector json includes selection object");
    if (selection && selection->is_object()) {
      check_selector_selection_stable_surface(ctx,
                                              *selection,
                                              "compat selector selection");
      const auto* selection_schema_version = selection->find("schema_version");
      ctx.check(selection_schema_version && selection_schema_version->is_number() &&
                    selection_schema_version->as_number().as_uint64() == 1,
                "default selector selection uses compatibility schema version");
      const auto* flat_compatibility_fields = selection->find("flat_compatibility_fields");
      ctx.check(flat_compatibility_fields && flat_compatibility_fields->is_bool() &&
                    flat_compatibility_fields->as_bool(),
                "default selector json keeps flat compatibility fields");
      const auto* weights = selection->find("weights");
      ctx.check(weights && weights->is_object(), "selector json includes effective weights");
      const auto* requested_weights = selection->find("requested_weights");
      ctx.check(requested_weights && requested_weights->is_object(),
                "selector json includes requested weights");
      const auto* dataset_count = selection->find("dataset_count");
      ctx.check(dataset_count && dataset_count->is_number(),
                "selector json includes dataset_count");
      const auto* auto_weighting = selection->find("auto_dataset_weighting");
      ctx.check(auto_weighting && auto_weighting->is_bool(),
                "selector json includes auto_dataset_weighting");
      const auto* auto_applied = selection->find("auto_dataset_weighting_applied");
      ctx.check(auto_applied && auto_applied->is_bool(),
                "selector json includes auto_dataset_weighting_applied");
      const auto* advanced_search = selection->find("advanced_search");
      ctx.check(advanced_search && advanced_search->is_bool(),
                "selector json includes advanced_search");
      const auto* train_cache_hits = selection->find("train_cache_hits");
      ctx.check(train_cache_hits && train_cache_hits->is_number(),
                "selector json includes train cache hit telemetry");
      const auto* peak_threads = selection->find("selector_peak_threads_per_candidate");
      ctx.check(peak_threads && peak_threads->is_number(),
                "selector json includes peak candidate thread width");
      const auto* training_lane_wait_ms = selection->find("training_lane_wait_ms");
      ctx.check(training_lane_wait_ms && training_lane_wait_ms->is_number(),
                "selector json includes training lane wait telemetry");
      const auto* training_base_cache_hits = selection->find("training_base_cache_hits");
      ctx.check(training_base_cache_hits && training_base_cache_hits->is_number(),
                "selector json includes training-base cache hit telemetry");
      const auto* training_base_cache_misses = selection->find("training_base_cache_misses");
      ctx.check(training_base_cache_misses && training_base_cache_misses->is_number(),
                "selector json includes training-base cache miss telemetry");
      const auto* training_order_cache_hits = selection->find("training_order_cache_hits");
      ctx.check(training_order_cache_hits && training_order_cache_hits->is_number(),
                "selector json includes training-order cache hit telemetry");
      const auto* training_order_cache_misses = selection->find("training_order_cache_misses");
      ctx.check(training_order_cache_misses && training_order_cache_misses->is_number(),
                "selector json includes training-order cache miss telemetry");
      const auto* training_cdf_cache_hits = selection->find("training_cdf_cache_hits");
      ctx.check(training_cdf_cache_hits && training_cdf_cache_hits->is_number(),
                "selector json includes training-CDF cache hit telemetry");
      const auto* training_cdf_cache_misses = selection->find("training_cdf_cache_misses");
      ctx.check(training_cdf_cache_misses && training_cdf_cache_misses->is_number(),
                "selector json includes training-CDF cache miss telemetry");
      const auto* training_cdf_cache_failures = selection->find("training_cdf_cache_failures");
      ctx.check(training_cdf_cache_failures && training_cdf_cache_failures->is_number(),
                "selector json includes training-CDF cache failure telemetry");
      const auto* training_cdf_cache_build_ms =
          selection->find("training_cdf_cache_build_ms");
      ctx.check(training_cdf_cache_build_ms && training_cdf_cache_build_ms->is_number(),
                "selector json includes training-CDF cache build timing");
      const auto* candidate_error_cdf_fit_count =
          selection->find("candidate_error_cdf_fit_count");
      ctx.check(candidate_error_cdf_fit_count &&
                    candidate_error_cdf_fit_count->is_number(),
                "selector json includes CDF-fit error count telemetry");
      const auto* train_cluster_assign_ms_sum =
          selection->find("train_cluster_assign_ms_sum");
      ctx.check(train_cluster_assign_ms_sum && train_cluster_assign_ms_sum->is_number(),
                "selector json includes aggregate train cluster timing");
      const auto* train_cdf_fit_ms_sum = selection->find("train_cdf_fit_ms_sum");
      ctx.check(train_cdf_fit_ms_sum && train_cdf_fit_ms_sum->is_number(),
                "selector json includes aggregate train CDF timing");
      const auto* scheduler_ready_cached_candidates =
          selection->find("scheduler_ready_cached_candidates");
      ctx.check(scheduler_ready_cached_candidates &&
                    scheduler_ready_cached_candidates->is_number(),
                "selector json includes ready cached scheduler telemetry");
      const auto* scheduler_in_flight_candidates =
          selection->find("scheduler_in_flight_candidates");
      ctx.check(scheduler_in_flight_candidates &&
                    scheduler_in_flight_candidates->is_number(),
                "selector json includes in-flight scheduler telemetry");
      const auto* scheduler_missing_candidates =
          selection->find("scheduler_missing_candidates");
      ctx.check(scheduler_missing_candidates &&
                    scheduler_missing_candidates->is_number(),
                "selector json includes missing scheduler telemetry");
      const auto* scheduler_single_admission_batches =
          selection->find("scheduler_single_admission_batches");
      ctx.check(scheduler_single_admission_batches &&
                    scheduler_single_admission_batches->is_number(),
                "selector json includes single-admission batch telemetry");
      const auto* scheduler_reuse_prioritized_candidates =
          selection->find("scheduler_reuse_prioritized_candidates");
      ctx.check(scheduler_reuse_prioritized_candidates &&
                    scheduler_reuse_prioritized_candidates->is_number(),
                "selector json includes reuse-prioritized scheduler telemetry");
      const auto* scheduler_fresh_base_candidates =
          selection->find("scheduler_fresh_base_candidates");
      ctx.check(scheduler_fresh_base_candidates &&
                    scheduler_fresh_base_candidates->is_number(),
                "selector json includes fresh-base scheduler telemetry");
      const auto* eval_nearest_cache_hits = selection->find("eval_nearest_cache_hits");
      ctx.check(eval_nearest_cache_hits && eval_nearest_cache_hits->is_number(),
                "selector json includes nearest-centroid cache hit telemetry");
      const auto* eval_nearest_cache_misses = selection->find("eval_nearest_cache_misses");
      ctx.check(eval_nearest_cache_misses && eval_nearest_cache_misses->is_number(),
                "selector json includes nearest-centroid cache miss telemetry");
      const auto* eval_nearest_cache_build_ms = selection->find("eval_nearest_cache_build_ms");
      ctx.check(eval_nearest_cache_build_ms && eval_nearest_cache_build_ms->is_number(),
                "selector json includes nearest-centroid cache build telemetry");
      const auto* eval_nearest_cache_early_abandon_builds =
          selection->find("eval_nearest_cache_early_abandon_builds");
      ctx.check(eval_nearest_cache_early_abandon_builds &&
                    eval_nearest_cache_early_abandon_builds->is_number(),
                "selector json includes nearest-centroid early-abandon telemetry");
      const auto* eval_nearest_cache_memory_bytes =
          selection->find("eval_nearest_cache_memory_bytes");
      ctx.check(eval_nearest_cache_memory_bytes &&
                    eval_nearest_cache_memory_bytes->is_number(),
                "selector json includes nearest-centroid cache memory telemetry");
      const auto* eval_nearest_cache_evictions =
          selection->find("eval_nearest_cache_evictions");
      ctx.check(eval_nearest_cache_evictions &&
                    eval_nearest_cache_evictions->is_number(),
                "selector json includes nearest-centroid cache eviction telemetry");
      const auto* eval_base_rank_cache_hits =
          selection->find("eval_base_rank_cache_hits");
      ctx.check(eval_base_rank_cache_hits && eval_base_rank_cache_hits->is_number(),
                "selector json includes base-rank cache hit telemetry");
      const auto* eval_base_rank_cache_misses =
          selection->find("eval_base_rank_cache_misses");
      ctx.check(eval_base_rank_cache_misses &&
                    eval_base_rank_cache_misses->is_number(),
                "selector json includes base-rank cache miss telemetry");
      const auto* eval_base_rank_cache_build_ms =
          selection->find("eval_base_rank_cache_build_ms");
      ctx.check(eval_base_rank_cache_build_ms &&
                    eval_base_rank_cache_build_ms->is_number(),
                "selector json includes base-rank cache build telemetry");
      const char* base_rank_substage_fields[] = {
          "eval_base_rank_cache_nearest_ms",
          "eval_base_rank_cache_hash_ms",
          "eval_base_rank_cache_sort_ms",
          "eval_base_rank_cache_rank_index_ms",
          "eval_base_rank_cache_overhead_ms",
      };
      for (const char* field : base_rank_substage_fields) {
        const auto* value = selection->find(field);
        ctx.check(value && value->is_number(),
                  "selector json includes base-rank cache substage telemetry");
      }
      const auto* eval_base_rank_cache_memory_bytes =
          selection->find("eval_base_rank_cache_memory_bytes");
      ctx.check(eval_base_rank_cache_memory_bytes &&
                    eval_base_rank_cache_memory_bytes->is_number(),
                "selector json includes base-rank cache memory telemetry");
      const auto* eval_base_rank_cache_sampled_budget_bytes =
          selection->find("eval_base_rank_cache_sampled_budget_bytes");
      ctx.check(eval_base_rank_cache_sampled_budget_bytes &&
                    eval_base_rank_cache_sampled_budget_bytes->is_number(),
                "selector json includes base-rank sampled cache budget telemetry");
      const auto* eval_base_rank_cache_evictions =
          selection->find("eval_base_rank_cache_evictions");
      ctx.check(eval_base_rank_cache_evictions &&
                    eval_base_rank_cache_evictions->is_number(),
                "selector json includes base-rank cache eviction telemetry");
      const auto* eval_base_rank_cache_attribution =
          selection->find("eval_base_rank_cache_attribution");
      ctx.check(eval_base_rank_cache_attribution &&
                    eval_base_rank_cache_attribution->is_array(),
                "selector json includes base-rank cache attribution rows");
      if (eval_base_rank_cache_attribution &&
          eval_base_rank_cache_attribution->is_array() &&
          !eval_base_rank_cache_attribution->as_array().empty()) {
        const auto& row = eval_base_rank_cache_attribution->as_array().front();
        ctx.check(row.is_object(), "base-rank attribution row is an object");
        if (row.is_object()) {
          const auto* phase = row.find("phase");
          const auto* k = row.find("K");
          const auto* misses = row.find("misses");
          const auto* hash_ms = row.find("hash_ms");
          const auto* retained = row.find("retained_memory_bytes");
          const auto* roles = row.find("roles");
          ctx.check(phase && phase->is_string(),
                    "base-rank attribution row includes phase");
          ctx.check(k && k->is_number(),
                    "base-rank attribution row includes K");
          ctx.check(misses && misses->is_number(),
                    "base-rank attribution row includes misses");
          ctx.check(hash_ms && hash_ms->is_number(),
                    "base-rank attribution row includes hash_ms");
          ctx.check(retained && retained->is_number(),
                    "base-rank attribution row includes retained bytes");
          ctx.check(roles && roles->is_array(),
                    "base-rank attribution row includes roles");
        }
      }
      const auto* beam_width = selection->find("beam_width");
      ctx.check(beam_width && beam_width->is_number(),
                "selector json includes beam_width");
      const auto* beam_rounds = selection->find("beam_rounds");
      ctx.check(beam_rounds && beam_rounds->is_number(),
                "selector json includes beam_rounds");
      const auto* hill_climb_steps = selection->find("hill_climb_steps");
      ctx.check(hill_climb_steps && hill_climb_steps->is_number(),
                "selector json includes hill_climb_steps");
      const auto* beam_neighbor_limit = selection->find("beam_neighbor_limit");
      ctx.check(beam_neighbor_limit && beam_neighbor_limit->is_number(),
                "selector json includes beam_neighbor_limit");
      const auto* strategy_refine_evaluated = selection->find("strategy_refine_evaluated");
      ctx.check(strategy_refine_evaluated && strategy_refine_evaluated->is_number(),
                "selector json includes strategy_refine_evaluated");
      const auto* strategy_refine_rounds = selection->find("strategy_refine_rounds");
      ctx.check(strategy_refine_rounds && strategy_refine_rounds->is_number(),
                "selector json includes strategy_refine_rounds");
      const auto* attribution_guided_compaction =
          selection->find("attribution_guided_compaction");
      ctx.check(attribution_guided_compaction &&
                    attribution_guided_compaction->is_bool(),
                "selector json includes attribution-guided compaction option");
      const auto* strategy_frontier_candidates =
          selection->find("strategy_frontier_candidates");
      ctx.check(strategy_frontier_candidates &&
                    strategy_frontier_candidates->is_number(),
                "selector json includes strategy_frontier_candidates");
      const auto* strategy_compacted_candidates =
          selection->find("strategy_compacted_candidates");
      ctx.check(strategy_compacted_candidates &&
                    strategy_compacted_candidates->is_number(),
                "selector json includes strategy_compacted_candidates");
      const auto* selector_profile = selection->find("selector_profile");
      ctx.check(selector_profile && selector_profile->is_string(),
                "selector json includes selector_profile");
      const auto* effective_selector_profile = selection->find("effective_selector_profile");
      ctx.check(effective_selector_profile && effective_selector_profile->is_string(),
                "selector json includes effective_selector_profile");
      const auto* objective_score_scope = selection->find("objective_score_scope");
      ctx.check(objective_score_scope && objective_score_scope->is_string() &&
                    objective_score_scope->as_string() == "per-phase",
                "selector json marks objective scores as per-phase");
      const auto* objective_score_compare_rule =
          selection->find("objective_score_compare_rule");
      ctx.check(objective_score_compare_rule && objective_score_compare_rule->is_string(),
                "selector json includes objective score compare rule");
      const auto* diagnostics_schema_version =
          selection->find("diagnostics_schema_version");
      ctx.check(diagnostics_schema_version &&
                    diagnostics_schema_version->is_number(),
                "selector json includes grouped diagnostics schema version");
      const auto* diagnostics = selection->find("diagnostics");
      ctx.check(diagnostics && diagnostics->is_object(),
                "selector json includes grouped diagnostics");
      if (diagnostics && diagnostics->is_object()) {
        const auto* diagnostics_version = diagnostics->find("schema_version");
        ctx.check(diagnostics_version && diagnostics_version->is_number(),
                  "grouped diagnostics includes schema_version");
        const char* diagnostic_groups[] = {
            "search",
            "training",
            "scheduler",
            "evaluation",
            "strategy",
            "final_calibration",
            "memory",
            "errors",
        };
        for (const char* group : diagnostic_groups) {
          const auto* value = diagnostics->find(group);
          ctx.check(value && value->is_object(),
                    "grouped diagnostics includes expected object");
        }
        const auto* search = diagnostics->find("search");
        if (search && search->is_object()) {
          const auto* contexts = search->find("contexts");
          ctx.check(contexts && contexts->is_object(),
                    "grouped search diagnostics include context summary");
          const auto* stopped = search->find("stopped_reason");
          ctx.check(stopped && stopped->is_string(),
                    "grouped search diagnostics include stop reason");
        }
        const auto* training = diagnostics->find("training");
        if (training && training->is_object()) {
          const auto* stage_ms = training->find("stage_ms_sum");
          ctx.check(stage_ms && stage_ms->is_object(),
                    "grouped training diagnostics include stage timings");
          const auto* cdf_cache = training->find("cdf_cache");
          ctx.check(cdf_cache && cdf_cache->is_object(),
                    "grouped training diagnostics include CDF cache summary");
        }
        const auto* evaluation = diagnostics->find("evaluation");
        if (evaluation && evaluation->is_object()) {
          const auto* nearest_cache = evaluation->find("nearest_cache");
          ctx.check(nearest_cache && nearest_cache->is_object(),
                    "grouped evaluation diagnostics include nearest cache");
          const auto* base_rank_cache = evaluation->find("base_rank_cache");
          ctx.check(base_rank_cache && base_rank_cache->is_object(),
                    "grouped evaluation diagnostics include base-rank cache");
        }
      }
      const auto* contexts_total = selection->find("target_skeleton_contexts_total");
      ctx.check(contexts_total && contexts_total->is_number(),
                "selector json includes target_skeleton_contexts_total");
      const auto* contexts_requested = selection->find("target_skeleton_contexts_requested");
      ctx.check(contexts_requested && contexts_requested->is_number(),
                "selector json includes target_skeleton_contexts_requested");
      const auto* contexts_unique = selection->find("target_skeleton_contexts_unique");
      ctx.check(contexts_unique && contexts_unique->is_number(),
                "selector json includes target_skeleton_contexts_unique");
      const auto* contexts_deduped = selection->find("target_skeleton_contexts_deduplicated");
      ctx.check(contexts_deduped && contexts_deduped->is_number(),
                "selector json includes target_skeleton_contexts_deduplicated");
      const auto* contexts_selected = selection->find("target_skeleton_contexts_selected");
      ctx.check(contexts_selected && contexts_selected->is_number(),
                "selector json includes target_skeleton_contexts_selected");
      const auto* max_model_trains = selection->find("max_model_trains");
      ctx.check(max_model_trains && max_model_trains->is_number(),
                "selector json includes max_model_trains");
      const auto* model_trains_started = selection->find("model_trains_started");
      ctx.check(model_trains_started && model_trains_started->is_number(),
                "selector json includes model_trains_started");
      const auto* model_train_budget_hit = selection->find("model_train_budget_hit");
      ctx.check(model_train_budget_hit && model_train_budget_hit->is_bool(),
                "selector json includes model_train_budget_hit");
      const auto* max_search_seconds = selection->find("max_search_seconds");
      ctx.check(max_search_seconds && max_search_seconds->is_number(),
                "selector json includes max_search_seconds");
      const auto* effective_max_search_seconds = selection->find("effective_max_search_seconds");
      ctx.check(effective_max_search_seconds && effective_max_search_seconds->is_number(),
                "selector json includes effective_max_search_seconds");
      const auto* elapsed_search_seconds = selection->find("elapsed_search_seconds");
      ctx.check(elapsed_search_seconds && elapsed_search_seconds->is_number(),
                "selector json includes elapsed_search_seconds");
      const auto* search_time_budget_hit = selection->find("search_time_budget_hit");
      ctx.check(search_time_budget_hit && search_time_budget_hit->is_bool(),
                "selector json includes search_time_budget_hit");
      const auto* remaining_time_guard_hit = selection->find("remaining_time_guard_hit");
      ctx.check(remaining_time_guard_hit && remaining_time_guard_hit->is_bool(),
                "selector json includes remaining_time_guard_hit");
      const auto* final_calibration_reserve_seconds =
          selection->find("final_calibration_reserve_seconds");
      ctx.check(final_calibration_reserve_seconds &&
                    final_calibration_reserve_seconds->is_number(),
                "selector json includes final calibration reserve seconds");
      const auto* final_calibration_reserve_hit =
          selection->find("final_calibration_reserve_hit");
      ctx.check(final_calibration_reserve_hit &&
                    final_calibration_reserve_hit->is_bool(),
                "selector json includes final calibration reserve hit");
      const auto* contexts_evaluated = selection->find("target_skeleton_contexts_evaluated");
      ctx.check(contexts_evaluated && contexts_evaluated->is_number(),
                "selector json includes target_skeleton_contexts_evaluated");
      const auto* contexts_skipped = selection->find("contexts_skipped_low_potential");
      ctx.check(contexts_skipped && contexts_skipped->is_number(),
                "selector json includes contexts_skipped_low_potential");
      const auto* max_stagnant_contexts = selection->find("max_stagnant_contexts");
      ctx.check(max_stagnant_contexts && max_stagnant_contexts->is_number(),
                "selector json includes max_stagnant_contexts");
      const auto* min_recall_improvement = selection->find("min_recall_improvement");
      ctx.check(min_recall_improvement && min_recall_improvement->is_number(),
                "selector json includes min_recall_improvement");
      const auto* min_quality_improvement = selection->find("min_quality_improvement");
      ctx.check(min_quality_improvement && min_quality_improvement->is_number(),
                "selector json includes min_quality_improvement");
      const auto* optimize_for_recall = selection->find("optimize_for_recall");
      ctx.check(optimize_for_recall && optimize_for_recall->is_bool(),
                "selector json includes optimize_for_recall");
      const auto* max_recall_refine_rounds = selection->find("max_recall_refine_rounds");
      ctx.check(max_recall_refine_rounds && max_recall_refine_rounds->is_number(),
                "selector json includes max_recall_refine_rounds");
      const auto* recall_refine_evaluated = selection->find("recall_refine_evaluated");
      ctx.check(recall_refine_evaluated && recall_refine_evaluated->is_number(),
                "selector json includes recall_refine_evaluated");
      const auto* recall_refine_rounds = selection->find("recall_refine_rounds");
      ctx.check(recall_refine_rounds && recall_refine_rounds->is_number(),
                "selector json includes recall_refine_rounds");
      const auto* post_exploration = selection->find("post_exploration_exploitation");
      ctx.check(post_exploration && post_exploration->is_bool(),
                "selector json includes post_exploration_exploitation");
      const auto* post_exploration_candidates =
          selection->find("post_exploration_candidates");
      ctx.check(post_exploration_candidates &&
                    post_exploration_candidates->is_number(),
                "selector json includes post_exploration_candidates");
      const auto* post_exploration_evaluated =
          selection->find("post_exploration_evaluated");
      ctx.check(post_exploration_evaluated &&
                    post_exploration_evaluated->is_number(),
                "selector json includes post_exploration_evaluated");
      const auto* post_exploration_rounds =
          selection->find("post_exploration_rounds");
      ctx.check(post_exploration_rounds && post_exploration_rounds->is_number(),
                "selector json includes post_exploration_rounds");
      const auto* final_calibration_count = selection->find("final_calibration_count");
      ctx.check(final_calibration_count && final_calibration_count->is_number(),
                "selector json includes final_calibration_count");
      const auto* final_calibration_full_count =
          selection->find("final_calibration_full_count");
      ctx.check(final_calibration_full_count &&
                    final_calibration_full_count->is_number(),
                "selector json includes final_calibration_full_count");
      const auto* effective_final_calibration_full_count =
          selection->find("effective_final_calibration_full_count");
      ctx.check(effective_final_calibration_full_count &&
                    effective_final_calibration_full_count->is_number(),
                "selector json includes effective_final_calibration_full_count");
      const auto* final_calibration_seed_count =
          selection->find("final_calibration_seed_count");
      ctx.check(final_calibration_seed_count &&
                    final_calibration_seed_count->is_number(),
                "selector json includes final_calibration_seed_count");
      const auto* final_calibration_screening =
          selection->find("final_calibration_screening");
      ctx.check(final_calibration_screening &&
                    final_calibration_screening->is_bool(),
                "selector json includes final_calibration_screening");
      const auto* final_calibration_screen_candidate_count =
          selection->find("final_calibration_screen_candidate_count");
      ctx.check(final_calibration_screen_candidate_count &&
                    final_calibration_screen_candidate_count->is_number(),
                "selector json includes final_calibration_screen_candidate_count");
      const auto* effective_final_calibration_screen_candidate_count =
          selection->find("effective_final_calibration_screen_candidate_count");
      ctx.check(effective_final_calibration_screen_candidate_count &&
                    effective_final_calibration_screen_candidate_count->is_number(),
                "selector json includes effective_final_calibration_screen_candidate_count");
      const auto* final_calibration_screen_query_limit =
          selection->find("final_calibration_screen_query_limit");
      ctx.check(final_calibration_screen_query_limit &&
                    final_calibration_screen_query_limit->is_number(),
                "selector json includes final_calibration_screen_query_limit");
      const auto* effective_final_calibration_screen_query_limit =
          selection->find("effective_final_calibration_screen_query_limit");
      ctx.check(effective_final_calibration_screen_query_limit &&
                    effective_final_calibration_screen_query_limit->is_number(),
                "selector json includes effective_final_calibration_screen_query_limit");
      const auto* final_calibration_screen_evaluated =
          selection->find("final_calibration_screen_evaluated");
      ctx.check(final_calibration_screen_evaluated &&
                    final_calibration_screen_evaluated->is_number(),
                "selector json includes final_calibration_screen_evaluated");
      const auto* final_calibration_screen_promoted =
          selection->find("final_calibration_screen_promoted");
      ctx.check(final_calibration_screen_promoted &&
                    final_calibration_screen_promoted->is_number(),
                "selector json includes final_calibration_screen_promoted");
      const auto* final_calibration_role_promoted =
          selection->find("final_calibration_role_promoted");
      ctx.check(final_calibration_role_promoted &&
                    final_calibration_role_promoted->is_number(),
                "selector json includes final_calibration_role_promoted");
      const auto* final_calibration_recall_promoted =
          selection->find("final_calibration_recall_promoted");
      ctx.check(final_calibration_recall_promoted &&
                    final_calibration_recall_promoted->is_number(),
                "selector json includes final_calibration_recall_promoted");
      const auto* final_calibration_screen_status =
          selection->find("final_calibration_screen_status");
      ctx.check(final_calibration_screen_status &&
                    final_calibration_screen_status->is_string(),
                "selector json includes final_calibration_screen_status");
      const auto* final_calibration_query_limit =
          selection->find("final_calibration_query_limit");
      ctx.check(final_calibration_query_limit &&
                    final_calibration_query_limit->is_number(),
                "selector json includes final_calibration_query_limit");
      const auto* final_calibration_evaluated =
          selection->find("final_calibration_evaluated");
      ctx.check(final_calibration_evaluated &&
                    final_calibration_evaluated->is_number(),
                "selector json includes final_calibration_evaluated");
      const auto* final_calibration_reused =
          selection->find("final_calibration_reused");
      ctx.check(final_calibration_reused &&
                    final_calibration_reused->is_number(),
                "selector json includes final_calibration_reused");
      const auto* calibration_probes =
          selection->find("final_calibration_progressive_probes");
      ctx.check(calibration_probes && calibration_probes->is_array(),
                "selector json includes final calibration stability probes");
      if (calibration_probes && calibration_probes->is_array() &&
          !calibration_probes->as_array().empty() &&
          calibration_probes->as_array().front().is_object()) {
        const auto& probe = calibration_probes->as_array().front();
        ctx.check(probe.find("probe_base_count") &&
                      probe.find("probe_base_count")->is_number(),
                  "selector calibration probe includes probe_base_count");
        ctx.check(probe.find("final_base_count") &&
                      probe.find("final_base_count")->is_number(),
                  "selector calibration probe includes final_base_count");
        ctx.check(probe.find("peak_recall_match") &&
                      probe.find("peak_recall_match")->is_bool(),
                  "selector calibration probe includes peak_recall_match");
        ctx.check(probe.find("all_roles_match") &&
                      probe.find("all_roles_match")->is_bool(),
                  "selector calibration probe includes all_roles_match");
        ctx.check(probe.find("matched_role_count") &&
                      probe.find("matched_role_count")->is_number(),
                  "selector calibration probe includes matched_role_count");
      }
      const auto* timings = selection->find("timings");
      ctx.check(timings && timings->is_array(),
                "selector json includes timing telemetry");
      const auto* stopped_reason = selection->find("stopped_reason");
      ctx.check(stopped_reason && stopped_reason->is_string(),
                "selector json includes stopped_reason");
      const auto* plateau_rounds = selection->find("plateau_rounds");
      ctx.check(plateau_rounds && plateau_rounds->is_number(),
                "selector json includes plateau_rounds");
      const auto* best_margin = selection->find("best_margin");
      ctx.check(best_margin && best_margin->is_number(),
                "selector json includes best_margin");
      const auto* eval_windows = selection->find("eval_windows");
      ctx.check(eval_windows && eval_windows->is_array(),
                "selector json includes effective recall-window ladder");
      const auto* context_stats = selection->find("context_stats");
      ctx.check(context_stats && context_stats->is_array(),
                "selector json includes context telemetry");
      const auto* promotion_history = selection->find("promotion_history");
      ctx.check(promotion_history && promotion_history->is_array(),
                "selector json includes promotion history");
      if (context_stats && context_stats->is_array() &&
          !context_stats->as_array().empty() &&
          context_stats->as_array().front().is_object()) {
        const auto* requested_targets =
            context_stats->as_array().front().find("requested_target_skeletons");
        ctx.check(requested_targets && requested_targets->is_array(),
                  "selector context telemetry includes requested target aliases");
        const auto* effective_k_values =
            context_stats->as_array().front().find("effective_K_values");
        ctx.check(effective_k_values && effective_k_values->is_array(),
                  "selector context telemetry includes effective K values");
        const auto* pruned_k_values =
            context_stats->as_array().front().find("pruned_K_values");
        ctx.check(pruned_k_values && pruned_k_values->is_array(),
                  "selector context telemetry includes pruned K values");
        const auto* phase2_compacted_candidates =
            context_stats->as_array().front().find("phase2_compacted_candidates");
        ctx.check(phase2_compacted_candidates &&
                      phase2_compacted_candidates->is_number(),
                  "selector context telemetry includes compacted candidate count");
        const auto* promotion_decision =
            context_stats->as_array().front().find("promotion_decision");
        ctx.check(promotion_decision && promotion_decision->is_string(),
                  "selector context telemetry includes promotion decision");
        const auto* phase2_quality_gain =
            context_stats->as_array().front().find("phase2_quality_gain");
        ctx.check(phase2_quality_gain && phase2_quality_gain->is_number(),
                  "selector context telemetry includes phase2 quality gain");
        const auto* vectors_per_centroid =
            context_stats->as_array().front().find("min_vectors_per_centroid");
        ctx.check(vectors_per_centroid && vectors_per_centroid->is_number(),
                  "selector context telemetry includes vectors-per-centroid");
        const auto* skeleton_nodes_per_centroid =
            context_stats->as_array().front().find("min_skeleton_nodes_per_centroid");
        ctx.check(skeleton_nodes_per_centroid &&
                      skeleton_nodes_per_centroid->is_number(),
                  "selector context telemetry includes skeleton nodes per centroid");
      }
      const auto* memory_budget = selection->find("selector_memory_budget_bytes");
      ctx.check(memory_budget && memory_budget->is_number(),
                "selector json includes selector_memory_budget_bytes");
      const auto* effective_memory_budget =
          selection->find("effective_selector_memory_budget_bytes");
      ctx.check(effective_memory_budget && effective_memory_budget->is_number(),
                "selector json includes effective_selector_memory_budget_bytes");
      const auto* peak_parallelism = selection->find("selector_peak_parallelism");
      ctx.check(peak_parallelism && peak_parallelism->is_number(),
                "selector json includes selector_peak_parallelism");
      const auto* memory_limited = selection->find("selector_memory_budget_limited");
      ctx.check(memory_limited && memory_limited->is_bool(),
                "selector json includes selector_memory_budget_limited");
      const auto* eval_node_counts = selection->find("eval_node_counts");
      ctx.check(eval_node_counts && eval_node_counts->is_array(),
                "selector json includes eval node counts");
    }
    const auto* candidates = json.find("candidates");
    ctx.check(candidates && candidates->is_array() && !candidates->as_array().empty(),
              "selector json includes candidates array");
    if (candidates && candidates->is_array() && !candidates->as_array().empty()) {
      const auto& first = candidates->as_array().front();
      ctx.check(first.is_object(), "selector candidate json entry is an object");
      if (first.is_object()) {
        const auto* score_scope = first.find("objective_score_scope");
        ctx.check(score_scope && score_scope->is_string(),
                  "selector candidate json includes objective_score_scope");
        const char* timing_fields[] = {
            "eval_total_ms",
            "eval_hash_base_ms",
            "eval_sort_base_ms",
            "eval_rank_index_ms",
            "eval_query_ms",
            "eval_latency_ms",
        };
        for (const char* field : timing_fields) {
          const auto* timing = first.find(field);
          ctx.check(timing && timing->is_number(),
                    "selector candidate json includes evaluation substage timing");
        }
        const auto* recall_curve = first.find("recall_curve");
        ctx.check(recall_curve && recall_curve->is_array(),
                  "selector candidate json includes recall curve");
        const auto* recall_auc = first.find("recall_auc_log_window");
        ctx.check(recall_auc && recall_auc->is_number(),
                  "selector candidate json includes recall AUC");
        const auto* query_tail = first.find("query_recall_p05");
        ctx.check(query_tail && query_tail->is_number(),
                  "selector candidate json includes tail query recall");
        const auto* rank_p95 = first.find("rank_distance_norm_p95");
        ctx.check(rank_p95 && rank_p95->is_number(),
                  "selector candidate json includes rank-distance percentile");
        const auto* node_score = first.find("node_locality_score");
        ctx.check(node_score && node_score->is_number(),
                  "selector candidate json includes node locality score");
        const auto* node_curve = first.find("node_locality_curve");
        ctx.check(node_curve && node_curve->is_array(),
                  "selector candidate json includes node locality curve");
        const auto* match_score = first.find("overlay_match_score");
        ctx.check(match_score && match_score->is_number(),
                  "selector candidate json includes overlay match score");
        const auto* match_curve = first.find("overlay_match_curve");
        ctx.check(match_curve && match_curve->is_array(),
                  "selector candidate json includes overlay match curve");
        const auto* match_tail = first.find("query_overlay_match_p05");
        ctx.check(match_tail && match_tail->is_number(),
                  "selector candidate json includes tail query overlay match");
        const auto* quality = first.find("locality_quality_score");
        ctx.check(quality && quality->is_number(),
                  "selector candidate json includes locality quality score");
      }
    }
    const auto* roles = json.find("recommendation_roles");
    ctx.check(roles && roles->is_object(),
              "selector json includes recommendation roles");
    if (roles && roles->is_object()) {
      ctx.check(roles->find("peak_recall") != nullptr,
                "selector recommendation roles include peak_recall");
      ctx.check(roles->find("knee") != nullptr,
                "selector recommendation roles include knee");
      ctx.check(roles->find("fast") != nullptr,
                "selector recommendation roles include fast");
      ctx.check(roles->find("small") != nullptr,
                "selector recommendation roles include small");
    }
  }

  vortex::SelectorJsonOptions compact_json_options;
  compact_json_options.include_flat_selection_compatibility_fields = false;
  auto compact_json = vortex::selector_result_to_json(result, opts, compact_json_options);
  ctx.check(compact_json.is_object(), "compact selector json output is an object");
  if (compact_json.is_object()) {
    check_selector_json_stable_surface(ctx, compact_json, "compact selector json");
    const auto* schema_version = compact_json.find("selector_json_schema_version");
    ctx.check(schema_version && schema_version->is_number() &&
                  schema_version->as_number().as_uint64() == 2,
              "compact selector json uses schema version 2");
    const auto* json_mode = compact_json.find("selector_json_mode");
    ctx.check(json_mode && json_mode->is_string() &&
                  json_mode->as_string() == "compact",
              "compact selector json reports compact mode");
    const auto* selection = compact_json.find("selection");
    ctx.check(selection && selection->is_object(),
              "compact selector json includes selection object");
    if (selection && selection->is_object()) {
      check_selector_selection_stable_surface(ctx,
                                              *selection,
                                              "compact selector selection");
      const auto* selection_schema_version = selection->find("schema_version");
      ctx.check(selection_schema_version && selection_schema_version->is_number() &&
                    selection_schema_version->as_number().as_uint64() == 2,
                "compact selector selection uses schema version 2");
      const auto* flat_compatibility_fields = selection->find("flat_compatibility_fields");
      ctx.check(flat_compatibility_fields && flat_compatibility_fields->is_bool() &&
                    !flat_compatibility_fields->as_bool(),
                "compact selector json disables flat compatibility fields");
      ctx.check(selection->find("diagnostics") &&
                    selection->find("diagnostics")->is_object(),
                "compact selector json retains grouped diagnostics");
      ctx.check(selection->find("target_skeleton_contexts_total") == nullptr,
                "compact selector json omits flat context telemetry");
      ctx.check(selection->find("eval_nearest_cache_hits") == nullptr,
                "compact selector json omits flat nearest-cache telemetry");
      ctx.check(selection->find("eval_base_rank_cache_attribution") == nullptr,
                "compact selector json omits detailed base-rank attribution rows");
      ctx.check(selection->find("context_stats") == nullptr,
                "compact selector json omits detailed context stats");
      ctx.check(selection->find("final_calibration_progressive_probes") == nullptr,
                "compact selector json omits detailed calibration probe rows");
      ctx.check(selection->find("selector_memory_budget_bytes") == nullptr,
                "compact selector json omits flat memory telemetry");
      ctx.check(selection->find("weights") && selection->find("weights")->is_object(),
                "compact selector json keeps objective weights");
      ctx.check(selection->find("dataset_count") && selection->find("dataset_count")->is_number(),
                "compact selector json keeps dataset count");
      ctx.check(selection->find("eval_windows") && selection->find("eval_windows")->is_array(),
                "compact selector json keeps evaluation window ladder");
    }
    ctx.check(compact_json.find("candidates") &&
                  compact_json.find("candidates")->is_array(),
              "compact selector json keeps candidate rows");
    ctx.check(compact_json.find("recommendation_roles") &&
                  compact_json.find("recommendation_roles")->is_object(),
              "compact selector json keeps recommendation roles");
  }

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(nsw_path);
}

} // namespace

int main() {
  TestContext ctx;

  test_shared_model_spec_helpers(ctx);
  test_model_selector_small_dataset(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All vortex_v1 selector smoke/schema tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " tests failed\n";
  return 1;
}
