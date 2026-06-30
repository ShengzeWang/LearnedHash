#include "vortex_v1/model_selector.h"

#include "cli_common.h"
#include "vortex_model_selector_report.h"

#include "rm_model/logging.h"

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace selector_report = vortex::tools::model_selector_report;

namespace {

using vortex::tools::ArgList;
using vortex::tools::parse_double;
using vortex::tools::parse_double_list;
using vortex::tools::parse_i32;
using vortex::tools::parse_u32;
using vortex::tools::parse_u32_list;
using vortex::tools::parse_u64;
using vortex::tools::parse_u64_list;
using vortex::tools::split_list;
using vortex::tools::trim_copy;
using vortex::tools::validate_known_flags;
using vortex::tools::validate_required_flag_values;

std::vector<std::string> parse_cdf_specs(const std::string& value) {
  std::vector<std::string> out;
  if (value.find(';') == std::string::npos) {
    std::string trimmed = trim_copy(value);
    if (trimmed.empty()) {
      throw std::runtime_error("--cdf-models must contain at least one model spec");
    }
    out.push_back(trimmed);
    return out;
  }
  for (const auto& token : split_list(value, ';')) {
    out.push_back(token);
  }
  if (out.empty()) {
    throw std::runtime_error("--cdf-models must contain at least one model spec");
  }
  return out;
}

void print_usage() {
  std::cout << "Usage:\n";
  std::cout << "  vortex_model_selector --dataset <base.fvecs> --nsw <skeleton.csr> [options]\n";
  std::cout << "  vortex_model_selector --dataset <base.fvecs> --index <hnsw.index> "
               "[--target-skeleton-values <list>] [options]\n\n";

  std::cout << "Required input:\n";
  std::cout << "  --dataset <fvecs>              Base vectors used for training/evaluation.\n";
  std::cout << "  --nsw <csr>                    Existing NSW skeleton, or use --index for skeleton search.\n";
  std::cout << "  --index <hnsw.index>           HNSW index used when extracting skeleton sizes.\n";
  std::cout << "  --query <fvecs>                Optional query set; defaults to sampled base vectors.\n\n";

  std::cout << "Skeleton search:\n";
  std::cout << "  --target-skeleton-values <csv>       Candidate skeleton sizes; otherwise auto-scaled.\n";
  std::cout << "  --target-skeleton-percentages <csv>  Candidate dataset percentages in (0,100].\n";
  std::cout << "  --max-target-skeleton-contexts <n>   Cap selected skeleton contexts.\n\n";

  std::cout << "Search space:\n";
  std::cout << "  --selector-profile <auto|fast|balanced|exhaustive>  Preset budget policy.\n";
  std::cout << "  --K-values <csv>                  Centroid counts to search.\n";
  std::cout << "  --knn-values <csv>                Centroid-neighbor values to search.\n";
  std::cout << "  --cdf-branches <csv>              CDF branch factors.\n";
  std::cout << "  --cdf-models <spec[;spec...]>     CDF model pairs, e.g. linear,linear.\n";
  std::cout << "  --two-opt-iters <csv>             2-opt iteration counts.\n\n";

  std::cout << "Objective and budget:\n";
  std::cout << "  --recall-target <0..1>            Prefer configs meeting this recall target.\n";
  std::cout << "  --optimize-for-recall             Increase recall weight and recall-first search effort.\n";
  std::cout << "  --recommend <n>                   Number of recommended configs to print/report.\n";
  std::cout << "  --size-budget-bytes <bytes>       Optional serialized model size budget.\n";
  std::cout << "  --max-model-trains <n>            Training budget guard.\n";
  std::cout << "  --max-search-seconds <n>          Wall-time budget guard.\n\n";

  std::cout << "Evaluation fidelity:\n";
  std::cout << "  --eval-k <n>                      Recall@k target.\n";
  std::cout << "  --window <n>                      Rank-window for locality recall.\n";
  std::cout << "  --window-values <csv>             Recall curve windows.\n";
  std::cout << "  --node-count-values <csv>         Overlay node counts for match/locality scoring.\n";
  std::cout << "  --eval-base-limit <n>             Full-eval base sample size; 0 means all.\n";
  std::cout << "  --eval-query-limit <n>            Full-eval query sample size; 0 means all.\n\n";

  std::cout << "Execution and output:\n";
  std::cout << "  --threads <n>                     Threads per candidate.\n";
  std::cout << "  --selector-parallelism <n>        Candidates to evaluate concurrently.\n";
  std::cout << "  --seed <n>                        Deterministic sampling/training seed.\n";
  std::cout << "  --output <path>                   JSON report path.\n";
  std::cout << "  --selector-json-mode <compat|compact>  JSON telemetry shape; default compat.\n";
  std::cout << "  --html-report <path>              HTML report path; defaults next to JSON output.\n\n";

  std::cout << "Advanced diagnostics:\n";
  std::cout << "  Run with --help-advanced for internal tuning, disable flags, weights,\n";
  std::cout << "  calibration controls, coarse-fidelity limits, and debug switches.\n";
}

void print_advanced_usage() {
  std::cout << "Usage:\n";
  std::cout << "  vortex_model_selector --dataset <base.fvecs> "
               "(--nsw <skeleton.csr> | --index <hnsw.index>) [options]\n\n";

  std::cout << "This advanced help lists compatibility and diagnostics flags. Normal\n";
  std::cout << "workflow help is available with --help.\n\n";

  std::cout << "Core and reporting controls:\n";
  std::cout << "  --hash_bits --layer --output --selector-json-mode\n";
  std::cout << "  --html-report --disable-html-report\n";
  std::cout << "  --threads --selector-parallelism --seed --selector-profile\n\n";

  std::cout << "Search-space controls:\n";
  std::cout << "  --K-values --knn-values --cdf-branches --cdf-models --two-opt-iters\n";
  std::cout << "  --disable-2opt-search --disable-skeleton-capacity-scaling\n\n";

  std::cout << "Coarse/full evaluation limits:\n";
  std::cout << "  --eval-k --window --window-values\n";
  std::cout << "  --node-count-values\n";
  std::cout << "  --coarse-base-limit --coarse-query-limit\n";
  std::cout << "  --eval-base-limit --eval-query-limit\n";
  std::cout << "  --coarse-latency-iters --latency-iters\n";
  std::cout << "  --coarse-latency-warmup --latency-warmup\n\n";

  std::cout << "Candidate ranking and budget controls:\n";
  std::cout << "  --phase1-keep --max-candidates --recommend --size-budget-bytes\n";
  std::cout << "  --max-target-skeleton-contexts --max-model-trains --max-search-seconds\n";
  std::cout << "  --selector-memory-budget-bytes\n\n";

  std::cout << "Adaptive-search diagnostics:\n";
  std::cout << "  --disable-advanced-search --beam-width --beam-rounds --hill-climb-steps\n";
  std::cout << "  --beam-neighbor-limit --max-stagnant-contexts\n";
  std::cout << "  --min-quality-improvement --min-recall-improvement\n";
  std::cout << "  --max-recall-refine-rounds --disable-post-exploration\n";
  std::cout << "  --post-exploration-candidates --disable-attribution-guided-compaction\n";
  std::cout << "  --disable-full-dataset-assignments --assignment-sample-limit\n";
  std::cout << "  --disable-graph-centroid-order\n\n";

  std::cout << "Final-calibration diagnostics:\n";
  std::cout << "  --final-calibration-count --final-calibration-full-count\n";
  std::cout << "  --disable-final-calibration-screen\n";
  std::cout << "  --final-calibration-screen-candidate-count\n";
  std::cout << "  --final-calibration-screen-query-limit --final-calibration-query-limit\n\n";

  std::cout << "Objective weights:\n";
  std::cout << "  --weight-recall --weight-rank-distance --weight-latency\n";
  std::cout << "  --weight-model-size --weight-train-time --disable-auto-dataset-weighting\n\n";

  std::cout << "Recall-oriented public controls:\n";
  std::cout << "  --recall-target --optimize-for-recall\n";
}

std::filesystem::path default_output_path(const std::filesystem::path& dataset_path) {
  std::string stem = dataset_path.stem().string();
  if (stem.empty()) {
    stem = "dataset";
  }
  return std::filesystem::path("vortex_v1_output") / ("selector_" + stem + ".json");
}

} // namespace

int main(int argc, char** argv) {
  try {
    ArgList args(argc, argv);
    if (args.has("--help-advanced")) {
      print_advanced_usage();
      return 0;
    }
    if (argc < 2 || args.has("--help")) {
      print_usage();
      return 0;
    }

    validate_known_flags(
        args,
        {"--dataset", "--nsw", "--index", "--target-skeleton-values",
         "--target-skeleton-percentages", "--layer", "--query",
         "--hash_bits", "--threads", "--selector-parallelism", "--seed",
         "--selector-profile",
         "--output", "--selector-json-mode", "--html-report", "--disable-html-report",
         "--K-values", "--knn-values", "--cdf-branches", "--cdf-models", "--two-opt-iters",
         "--disable-2opt-search", "--phase1-keep", "--max-candidates", "--recommend",
         "--eval-k", "--window", "--window-values", "--node-count-values",
         "--coarse-base-limit", "--coarse-query-limit",
         "--eval-base-limit", "--eval-query-limit", "--coarse-latency-iters", "--latency-iters",
         "--coarse-latency-warmup", "--latency-warmup", "--size-budget-bytes", "--recall-target",
         "--max-target-skeleton-contexts", "--max-model-trains", "--max-search-seconds",
         "--disable-skeleton-capacity-scaling", "--disable-advanced-search",
         "--disable-attribution-guided-compaction", "--disable-full-dataset-assignments",
         "--assignment-sample-limit", "--disable-graph-centroid-order",
         "--beam-width", "--beam-rounds", "--hill-climb-steps", "--beam-neighbor-limit",
         "--max-stagnant-contexts", "--min-quality-improvement", "--min-recall-improvement",
         "--optimize-for-recall", "--max-recall-refine-rounds",
         "--disable-post-exploration", "--post-exploration-candidates",
         "--selector-memory-budget-bytes",
         "--final-calibration-count", "--final-calibration-full-count",
         "--disable-final-calibration-screen", "--final-calibration-screen-candidate-count",
         "--final-calibration-screen-query-limit", "--final-calibration-query-limit",
         "--weight-recall", "--weight-rank-distance", "--weight-latency", "--weight-model-size",
         "--weight-train-time", "--disable-auto-dataset-weighting", "--help",
         "--help-advanced"});
    validate_required_flag_values(
        args,
        {"--dataset", "--nsw", "--index", "--target-skeleton-values",
         "--target-skeleton-percentages", "--layer", "--query",
         "--hash_bits", "--threads", "--selector-parallelism", "--seed", "--selector-profile",
         "--output",
         "--selector-json-mode",
         "--html-report",
         "--K-values", "--knn-values", "--cdf-branches", "--cdf-models", "--two-opt-iters",
         "--phase1-keep", "--max-candidates", "--recommend", "--eval-k", "--window",
         "--window-values", "--node-count-values",
         "--coarse-base-limit", "--coarse-query-limit", "--eval-base-limit", "--eval-query-limit",
         "--coarse-latency-iters", "--latency-iters", "--coarse-latency-warmup",
         "--latency-warmup", "--size-budget-bytes", "--max-target-skeleton-contexts",
         "--max-model-trains", "--max-search-seconds",
         "--weight-recall", "--weight-rank-distance",
         "--weight-latency", "--weight-model-size", "--weight-train-time",
         "--recall-target", "--beam-width", "--beam-rounds", "--hill-climb-steps",
         "--beam-neighbor-limit", "--max-stagnant-contexts",
         "--min-quality-improvement", "--min-recall-improvement",
         "--max-recall-refine-rounds", "--post-exploration-candidates",
         "--selector-memory-budget-bytes", "--assignment-sample-limit",
         "--final-calibration-count",
         "--final-calibration-full-count",
         "--final-calibration-screen-candidate-count",
         "--final-calibration-screen-query-limit",
         "--final-calibration-query-limit"});

    vortex::SelectorOptions options;
    options.dataset_path = args.require("--dataset");
    if (args.value("--nsw").has_value()) {
      options.nsw_path = std::filesystem::path(*args.value("--nsw"));
    }
    if (args.value("--index").has_value()) {
      options.nsw_index_path = std::filesystem::path(*args.value("--index"));
    }
    if (args.value("--target-skeleton-values").has_value()) {
      options.target_skeleton_values =
          parse_u64_list(*args.value("--target-skeleton-values"), "--target-skeleton-values");
    }
    if (args.value("--target-skeleton-percentages").has_value()) {
      options.target_skeleton_percentages = parse_double_list(
          *args.value("--target-skeleton-percentages"), "--target-skeleton-percentages");
    }
    if (args.value("--layer").has_value()) {
      const std::string layer = *args.value("--layer");
      if (layer != "auto") {
        options.nsw_layer = parse_i32(layer, "--layer");
      }
    }
    if (options.target_skeleton_values.empty() && options.target_skeleton_percentages.empty()) {
      if (!args.value("--nsw").has_value() && !options.nsw_index_path.has_value()) {
        throw std::runtime_error("Missing required flag: --nsw or --index");
      }
    } else if (!options.nsw_index_path.has_value()) {
      throw std::runtime_error(
          "--index is required when --target-skeleton-values or --target-skeleton-percentages is set");
    }
    if (args.value("--query").has_value()) {
      options.query_path = std::filesystem::path(*args.value("--query"));
    }
    if (args.value("--hash_bits").has_value()) {
      options.hash_bits = parse_u32(*args.value("--hash_bits"), "--hash_bits");
    }
    if (args.value("--threads").has_value()) {
      options.threads = parse_u32(*args.value("--threads"), "--threads");
    }
    if (args.value("--selector-parallelism").has_value()) {
      options.selector_parallelism =
          parse_u32(*args.value("--selector-parallelism"), "--selector-parallelism");
    }
    if (args.value("--seed").has_value()) {
      options.seed = parse_u64(*args.value("--seed"), "--seed");
    }
    if (args.value("--selector-profile").has_value()) {
      options.selector_profile = trim_copy(*args.value("--selector-profile"));
    }
    vortex::SelectorJsonOptions json_options;
    if (args.value("--selector-json-mode").has_value()) {
      const std::string mode = trim_copy(*args.value("--selector-json-mode"));
      if (mode == "compat") {
        json_options.include_flat_selection_compatibility_fields = true;
      } else if (mode == "compact") {
        json_options.include_flat_selection_compatibility_fields = false;
      } else {
        throw std::runtime_error(
            "--selector-json-mode must be 'compat' or 'compact'");
      }
    }

    if (args.value("--K-values").has_value()) {
      options.K_values = parse_u32_list(*args.value("--K-values"), "--K-values");
    }
    if (args.value("--knn-values").has_value()) {
      options.centroid_knn_values = parse_u32_list(*args.value("--knn-values"), "--knn-values");
    }
    if (args.value("--cdf-branches").has_value()) {
      options.cdf_branching_values =
          parse_u64_list(*args.value("--cdf-branches"), "--cdf-branches");
    }
    if (args.value("--cdf-models").has_value()) {
      options.cdf_model_specs = parse_cdf_specs(*args.value("--cdf-models"));
    }
    if (args.value("--two-opt-iters").has_value()) {
      options.two_opt_iterations_values =
          parse_u32_list(*args.value("--two-opt-iters"), "--two-opt-iters");
    }
    if (args.has("--disable-2opt-search")) {
      options.include_disable_2opt = false;
    }

    if (args.value("--phase1-keep").has_value()) {
      options.phase1_keep = parse_u32(*args.value("--phase1-keep"), "--phase1-keep");
    }
    if (args.value("--max-candidates").has_value()) {
      options.max_phase2_candidates =
          parse_u32(*args.value("--max-candidates"), "--max-candidates");
    }
    if (args.value("--recommend").has_value()) {
      options.recommend_count = parse_u32(*args.value("--recommend"), "--recommend");
    }
    if (args.value("--size-budget-bytes").has_value()) {
      options.size_budget_bytes = parse_u64(*args.value("--size-budget-bytes"), "--size-budget-bytes");
    }
    if (args.value("--max-target-skeleton-contexts").has_value()) {
      options.max_target_skeleton_contexts = parse_u32(
          *args.value("--max-target-skeleton-contexts"), "--max-target-skeleton-contexts");
    }
    if (args.value("--max-model-trains").has_value()) {
      options.max_model_trains =
          parse_u32(*args.value("--max-model-trains"), "--max-model-trains");
    }
    if (args.value("--max-search-seconds").has_value()) {
      options.max_search_seconds =
          parse_u32(*args.value("--max-search-seconds"), "--max-search-seconds");
    }
    if (args.value("--recall-target").has_value()) {
      options.recall_target = parse_double(*args.value("--recall-target"), "--recall-target");
    }
    if (args.has("--disable-skeleton-capacity-scaling")) {
      options.skeleton_capacity_scaling = false;
    }
    if (args.has("--disable-advanced-search")) {
      options.advanced_search = false;
    }
    if (args.has("--disable-attribution-guided-compaction")) {
      options.attribution_guided_compaction = false;
    }
    if (args.has("--disable-full-dataset-assignments")) {
      options.use_full_dataset_for_assignments = false;
    }
    if (args.has("--disable-graph-centroid-order")) {
      options.enable_graph_centroid_order = false;
    }
    if (args.value("--assignment-sample-limit").has_value()) {
      options.assignment_sample_limit =
          parse_u64(*args.value("--assignment-sample-limit"), "--assignment-sample-limit");
      options.auto_assignment_sample_limit = false;
    }
    if (args.value("--beam-width").has_value()) {
      options.beam_width = parse_u32(*args.value("--beam-width"), "--beam-width");
    }
    if (args.value("--beam-rounds").has_value()) {
      options.beam_rounds = parse_u32(*args.value("--beam-rounds"), "--beam-rounds");
    }
    if (args.value("--hill-climb-steps").has_value()) {
      options.hill_climb_steps =
          parse_u32(*args.value("--hill-climb-steps"), "--hill-climb-steps");
    }
    if (args.value("--beam-neighbor-limit").has_value()) {
      options.beam_neighbor_limit =
          parse_u32(*args.value("--beam-neighbor-limit"), "--beam-neighbor-limit");
    }
    if (args.value("--max-stagnant-contexts").has_value()) {
      options.max_stagnant_contexts =
          parse_u32(*args.value("--max-stagnant-contexts"), "--max-stagnant-contexts");
    }
    if (args.value("--min-recall-improvement").has_value()) {
      options.min_recall_improvement =
          parse_double(*args.value("--min-recall-improvement"), "--min-recall-improvement");
      options.min_quality_improvement = options.min_recall_improvement;
    }
    if (args.value("--min-quality-improvement").has_value()) {
      options.min_quality_improvement =
          parse_double(*args.value("--min-quality-improvement"), "--min-quality-improvement");
    }
    if (args.has("--optimize-for-recall")) {
      options.optimize_for_recall = true;
    }
    if (args.has("--disable-post-exploration")) {
      options.post_exploration_exploitation = false;
    }
    if (args.value("--max-recall-refine-rounds").has_value()) {
      options.max_recall_refine_rounds =
          parse_u32(*args.value("--max-recall-refine-rounds"), "--max-recall-refine-rounds");
    }
    if (args.value("--post-exploration-candidates").has_value()) {
      options.post_exploration_candidates =
          parse_u32(*args.value("--post-exploration-candidates"),
                    "--post-exploration-candidates");
    }
    if (args.value("--final-calibration-count").has_value()) {
      options.final_calibration_count =
          parse_u32(*args.value("--final-calibration-count"), "--final-calibration-count");
    }
    if (args.value("--final-calibration-full-count").has_value()) {
      options.final_calibration_full_count =
          parse_u32(*args.value("--final-calibration-full-count"),
                    "--final-calibration-full-count");
    }
    if (args.has("--disable-final-calibration-screen")) {
      options.final_calibration_screening = false;
    }
    if (args.value("--final-calibration-screen-candidate-count").has_value()) {
      options.final_calibration_screen_candidate_count =
          parse_u32(*args.value("--final-calibration-screen-candidate-count"),
                    "--final-calibration-screen-candidate-count");
    }
    if (args.value("--final-calibration-screen-query-limit").has_value()) {
      options.final_calibration_screen_query_limit =
          parse_u32(*args.value("--final-calibration-screen-query-limit"),
                    "--final-calibration-screen-query-limit");
    }
    if (args.value("--final-calibration-query-limit").has_value()) {
      options.final_calibration_query_limit =
          parse_u32(*args.value("--final-calibration-query-limit"),
                    "--final-calibration-query-limit");
    }
    if (args.value("--selector-memory-budget-bytes").has_value()) {
      options.selector_memory_budget_bytes =
          parse_u64(*args.value("--selector-memory-budget-bytes"),
                    "--selector-memory-budget-bytes");
    }

    if (args.value("--eval-k").has_value()) {
      options.eval_k = parse_u32(*args.value("--eval-k"), "--eval-k");
    }
    if (args.value("--window").has_value()) {
      options.eval_window = parse_u64(*args.value("--window"), "--window");
    }
    if (args.value("--window-values").has_value()) {
      options.eval_window_values =
          parse_u64_list(*args.value("--window-values"), "--window-values");
    }
    if (args.value("--node-count-values").has_value()) {
      options.eval_node_count_values =
          parse_u32_list(*args.value("--node-count-values"), "--node-count-values");
    }
    if (args.value("--coarse-base-limit").has_value()) {
      options.coarse_base_limit =
          parse_u32(*args.value("--coarse-base-limit"), "--coarse-base-limit");
    }
    if (args.value("--coarse-query-limit").has_value()) {
      options.coarse_query_limit =
          parse_u32(*args.value("--coarse-query-limit"), "--coarse-query-limit");
    }
    if (args.value("--eval-base-limit").has_value()) {
      options.eval_base_limit =
          parse_u32(*args.value("--eval-base-limit"), "--eval-base-limit");
    }
    if (args.value("--eval-query-limit").has_value()) {
      options.eval_query_limit =
          parse_u32(*args.value("--eval-query-limit"), "--eval-query-limit");
    }

    if (args.value("--coarse-latency-iters").has_value()) {
      options.coarse_latency_iterations =
          parse_u32(*args.value("--coarse-latency-iters"), "--coarse-latency-iters");
    }
    if (args.value("--latency-iters").has_value()) {
      options.latency_iterations = parse_u32(*args.value("--latency-iters"), "--latency-iters");
    }
    if (args.value("--coarse-latency-warmup").has_value()) {
      options.coarse_latency_warmup =
          parse_u32(*args.value("--coarse-latency-warmup"), "--coarse-latency-warmup");
    }
    if (args.value("--latency-warmup").has_value()) {
      options.latency_warmup = parse_u32(*args.value("--latency-warmup"), "--latency-warmup");
    }

    bool has_manual_weight_override = false;
    if (args.value("--weight-recall").has_value()) {
      options.weights.recall = parse_double(*args.value("--weight-recall"), "--weight-recall");
      has_manual_weight_override = true;
    }
    if (args.value("--weight-rank-distance").has_value()) {
      options.weights.rank_distance =
          parse_double(*args.value("--weight-rank-distance"), "--weight-rank-distance");
      has_manual_weight_override = true;
    }
    if (args.value("--weight-latency").has_value()) {
      options.weights.latency = parse_double(*args.value("--weight-latency"), "--weight-latency");
      has_manual_weight_override = true;
    }
    if (args.value("--weight-model-size").has_value()) {
      options.weights.model_size =
          parse_double(*args.value("--weight-model-size"), "--weight-model-size");
      has_manual_weight_override = true;
    }
    if (args.value("--weight-train-time").has_value()) {
      options.weights.train_time =
          parse_double(*args.value("--weight-train-time"), "--weight-train-time");
      has_manual_weight_override = true;
    }
    if (has_manual_weight_override) {
      options.auto_dataset_weighting = false;
    }
    if (args.has("--disable-auto-dataset-weighting")) {
      options.auto_dataset_weighting = false;
    }
    bool disable_html_report = args.has("--disable-html-report");
    std::optional<std::filesystem::path> html_report_output;
    if (args.value("--html-report").has_value()) {
      html_report_output = std::filesystem::path(*args.value("--html-report"));
    }
    if (disable_html_report && html_report_output.has_value()) {
      throw std::runtime_error("--disable-html-report cannot be combined with --html-report");
    }

    rm_model::init_logging();
    auto result = vortex::select_vortex_models(options);
    selector_report::print_summary(result, options);

    std::filesystem::path output = args.value("--output").has_value()
        ? std::filesystem::path(*args.value("--output"))
        : default_output_path(options.dataset_path);
    if (!output.parent_path().empty()) {
      std::filesystem::create_directories(output.parent_path());
    }

    auto json = vortex::selector_result_to_json(result, options, json_options);
    std::ofstream out(output);
    if (!out) {
      throw std::runtime_error("Failed to open selector output path: " + output.string());
    }
    rm_model::json::write(out, json);
    std::cout << "  output: " << output.string() << "\n";

    if (!disable_html_report) {
      std::filesystem::path html_output =
          html_report_output.has_value()
              ? *html_report_output
              : selector_report::default_html_report_path(output);
      selector_report::write_html_report(html_output, result, options, output);
      std::cout << "  html_report: " << html_output.string() << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[vortex_model_selector] error: " << ex.what() << "\n";
    return 1;
  }
}
