#include "vortex_test_common.h"

#include "vortex_v1/training.h"

#include "rm_model/json.h"
#include "vector_io/vector_io.hpp"

using namespace vortex_test;

namespace {

std::string run_command_capture(const std::string& command,
                                const std::filesystem::path& output_path,
                                int* status) {
  std::ostringstream wrapped;
  wrapped << command << " > " << shell_quote(output_path.string()) << " 2>&1";
  *status = std::system(wrapped.str().c_str());
  return std::filesystem::exists(output_path) ? read_text_file(output_path) : std::string();
}

void test_model_selector_help_contract(TestContext& ctx) {
  std::filesystem::path selector_path = model_selector_tool_path();
  if (!std::filesystem::exists(selector_path)) {
    if (model_selector_tool_is_required()) {
      ctx.check(false, "vortex_model_selector test binary exists for help contract test");
    } else {
      std::cerr << "[SKIP] vortex_model_selector binary not built for help contract test\n";
    }
    return;
  }

  auto help_path = unique_temp_path("vortex_selector_help.txt");
  int status = 0;
  std::string normal_help = run_command_capture(
      shell_quote(selector_path.string()) + " --help", help_path, &status);
  ctx.check(status == 0, "vortex_model_selector --help succeeds");
  ctx.check(normal_help.find("Required input:") != std::string::npos,
            "normal selector help is workflow-oriented");
  ctx.check(normal_help.find("--target-skeleton-values") != std::string::npos,
            "normal selector help includes skeleton-search flags");
  ctx.check(normal_help.find("--recall-target") != std::string::npos,
            "normal selector help includes recall target");
  ctx.check(normal_help.find("--help-advanced") != std::string::npos,
            "normal selector help points to advanced help");
  ctx.check(normal_help.find("--selector-json-mode") != std::string::npos,
            "normal selector help includes JSON schema mode");
  ctx.check(normal_help.find("--disable-advanced-search") == std::string::npos,
            "normal selector help hides internal advanced-search disable switch");
  ctx.check(normal_help.find("--disable-final-calibration-screen") == std::string::npos,
            "normal selector help hides final-calibration diagnostics switch");
  ctx.check(normal_help.find("--weight-recall") == std::string::npos,
            "normal selector help hides manual objective weights");

  auto advanced_help_path = unique_temp_path("vortex_selector_help_advanced.txt");
  std::string advanced_help = run_command_capture(
      shell_quote(selector_path.string()) + " --help-advanced",
      advanced_help_path,
      &status);
  ctx.check(status == 0, "vortex_model_selector --help-advanced succeeds");
  ctx.check(advanced_help.find("compatibility and diagnostics flags") != std::string::npos,
            "advanced selector help explains diagnostic scope");
  ctx.check(advanced_help.find("--disable-advanced-search") != std::string::npos,
            "advanced selector help includes advanced-search disable switch");
  ctx.check(advanced_help.find("--disable-final-calibration-screen") != std::string::npos,
            "advanced selector help includes final-calibration diagnostics switch");
  ctx.check(advanced_help.find("--weight-recall") != std::string::npos,
            "advanced selector help includes manual objective weights");
  ctx.check(advanced_help.find("--disable-html-report") != std::string::npos,
            "advanced selector help includes report disable switch");
  ctx.check(advanced_help.find("--selector-json-mode") != std::string::npos,
            "advanced selector help includes JSON schema mode");

  auto compat_path = unique_temp_path("vortex_selector_advanced_flag_compat.txt");
  std::string compat_output = run_command_capture(
      shell_quote(selector_path.string()) + " --disable-advanced-search",
      compat_path,
      &status);
  ctx.check(status != 0, "vortex_model_selector still parses advanced flags before required-input validation");
  ctx.check(compat_output.find("Missing required flag: --dataset") != std::string::npos,
            "advanced flag compatibility reaches required-input validation");
  ctx.check(compat_output.find("Unknown flag") == std::string::npos,
            "advanced flag remains accepted by strict flag validation");

  std::filesystem::remove(help_path);
  std::filesystem::remove(advanced_help_path);
  std::filesystem::remove(compat_path);
}

void test_model_selector_html_report_contract(TestContext& ctx) {
  std::filesystem::path selector_path = model_selector_tool_path();
  if (!std::filesystem::exists(selector_path)) {
    if (model_selector_tool_is_required()) {
      ctx.check(false, "vortex_model_selector test binary exists");
    } else {
      std::cerr << "[SKIP] vortex_model_selector binary not built\n";
    }
    return;
  }

  vector_io::VectorStorage<float> storage(24, 64);
  storage.values.resize(storage.count * storage.dim);
  std::mt19937 rng(777);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& value : storage.values) {
    value = dist(rng);
  }

  auto dataset_path = unique_temp_path("vortex_selector_report.fvecs");
  auto index_path = unique_temp_path("vortex_selector_report.index");
  auto json_path = unique_temp_path("vortex_selector_report.json");
  auto html_path = unique_temp_path("vortex_selector_report.html");

  vector_io::write_fvecs(dataset_path.string(), storage);

  vortex::BuildHnswOptions build_opts;
  build_opts.dataset_path = dataset_path;
  build_opts.M = 8;
  build_opts.ef_construction = 80;
  vortex::build_hnsw_index(build_opts, index_path);

  {
    std::ostringstream cmd;
    cmd << shell_quote(selector_path.string())
        << " --dataset " << shell_quote(dataset_path.string())
        << " --query " << shell_quote(dataset_path.string())
        << " --index " << shell_quote(index_path.string())
        << " --target-skeleton-values 10,20"
        << " --selector-profile fast"
        << " --threads 1"
        << " --selector-parallelism 1"
        << " --K-values 4"
        << " --knn-values 4"
        << " --cdf-branches 4"
        << " --cdf-models linear,linear"
        << " --two-opt-iters 2"
        << " --phase1-keep 1"
        << " --max-candidates 2"
        << " --recommend 1"
        << " --eval-k 5"
        << " --window 20"
        << " --coarse-base-limit 16"
        << " --coarse-query-limit 8"
        << " --eval-base-limit 24"
        << " --eval-query-limit 8"
        << " --coarse-latency-iters 4"
        << " --latency-iters 8"
        << " --coarse-latency-warmup 1"
        << " --latency-warmup 2"
        << " --max-model-trains 4"
        << " --final-calibration-count 2"
        << " --final-calibration-full-count 1"
        << " --final-calibration-screen-candidate-count 2"
        << " --final-calibration-screen-query-limit 4"
        << " --final-calibration-query-limit 8"
        << " --output " << shell_quote(json_path.string())
        << " --selector-json-mode compact"
        << " --html-report " << shell_quote(html_path.string());
    int status = std::system(cmd.str().c_str());
    ctx.check(status == 0, "vortex_model_selector HTML report smoke command succeeds");
  }

  if (std::filesystem::exists(json_path)) {
    auto json = rm_model::json::parse_file(json_path.string());
    ctx.check(json.is_object(), "selector report smoke json is an object");
    if (json.is_object()) {
      const auto* schema_version = json.find("selector_json_schema_version");
      ctx.check(schema_version && schema_version->is_number() &&
                    schema_version->as_number().as_uint64() == 2,
                "selector report smoke json uses compact schema version");
      const auto* json_mode = json.find("selector_json_mode");
      ctx.check(json_mode && json_mode->is_string() &&
                    json_mode->as_string() == "compact",
                "selector report smoke json records compact mode");
      const auto* selection = json.find("selection");
      ctx.check(selection && selection->is_object(),
                "selector report smoke json includes selection object");
      if (selection && selection->is_object()) {
        const auto* flat_compatibility_fields = selection->find("flat_compatibility_fields");
        ctx.check(flat_compatibility_fields && flat_compatibility_fields->is_bool() &&
                      !flat_compatibility_fields->as_bool(),
                  "selector report smoke json disables flat compatibility fields");
        const auto* score_scope = selection->find("objective_score_scope");
        ctx.check(score_scope && score_scope->is_string() &&
                      score_scope->as_string() == "per-phase",
                  "selector report smoke json exposes objective score scope");
        ctx.check(selection->find("target_skeleton_contexts_total") == nullptr,
                  "selector report smoke compact json omits flat context telemetry");
        const auto* diagnostics = selection->find("diagnostics");
        ctx.check(diagnostics && diagnostics->is_object(),
                  "selector report smoke json includes grouped diagnostics");
        if (diagnostics && diagnostics->is_object()) {
          const auto* search = diagnostics->find("search");
          const auto* training = diagnostics->find("training");
          const auto* evaluation = diagnostics->find("evaluation");
          ctx.check(search && search->is_object(), "selector diagnostics include search group");
          ctx.check(training && training->is_object(),
                    "selector diagnostics include training group");
          ctx.check(evaluation && evaluation->is_object(),
                    "selector diagnostics include evaluation group");
        }
      }
      const auto* candidates = json.find("candidates");
      ctx.check(candidates && candidates->is_array() &&
                    !candidates->as_array().empty(),
                "selector report smoke json includes candidate rows");
      if (candidates && candidates->is_array() && !candidates->as_array().empty()) {
        const auto& first = candidates->as_array().front();
        ctx.check(first.is_object(), "selector candidate row is an object");
        if (first.is_object()) {
          ctx.check(first.find("config") && first.find("config")->is_object(),
                    "selector candidate row includes config object");
          ctx.check(first.find("train_command") && first.find("train_command")->is_string(),
                    "selector candidate row includes train command");
        }
      }
      const auto* roles = json.find("recommendation_roles");
      ctx.check(roles && roles->is_object(),
                "selector report smoke json includes recommendation roles object");
      const auto* configs = json.find("configs");
      ctx.check(configs && configs->is_array(),
                "selector report smoke json includes recommended configs array");
    }
  } else {
    ctx.check(false, "selector report smoke json output exists");
  }

  if (std::filesystem::exists(html_path)) {
    std::string html = read_text_file(html_path);
    ctx.check(html.find("Skeleton Sweep Screening Benchmarks") != std::string::npos,
              "selector HTML report labels skeleton benchmarks as screening metrics");
    ctx.check(html.find("How to read this report") != std::string::npos,
              "selector HTML report includes metric guide");
    ctx.check(html.find("Search health and budget state") != std::string::npos,
              "selector HTML report foregrounds search health");
    ctx.check(html.find("<details class=\"panel\"><summary>Selector Context Telemetry") !=
                  std::string::npos,
              "selector HTML report collapses detailed context telemetry");
    ctx.check(html.find("Figure 1") != std::string::npos &&
                  html.find("candidate</text>") != std::string::npos &&
                  html.find("recommended</text>") != std::string::npos,
              "selector HTML report charts include figure labels and legend");
    ctx.check(html.find("Dataset %") != std::string::npos,
              "selector HTML report includes dataset percentage column");
    ctx.check(html.find("Objective scores are phase-local") != std::string::npos,
              "selector HTML report explains objective score scope");
    ctx.check(html.find("Selected-best and recommendation-role rows above are the calibrated") !=
                  std::string::npos,
              "selector HTML report distinguishes screening rows from calibrated rows");
    ctx.check(html.find("-1.000000") == std::string::npos,
              "selector HTML report does not contain placeholder objective scores");
    ctx.check(html.find("radial-gradient") == std::string::npos,
              "selector HTML report avoids decorative gradient noise");
    ctx.check(html.find("Skeleton Benchmark (10%-100%)") == std::string::npos,
              "selector HTML report no longer uses legacy synthetic benchmark title");
  } else {
    ctx.check(false, "selector HTML report output exists");
  }

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(index_path);
  std::filesystem::remove(json_path);
  std::filesystem::remove(html_path);
}

void test_model_selector_rejects_malformed_numeric_flags(TestContext& ctx) {
  std::filesystem::path selector_path = model_selector_tool_path();
  if (!std::filesystem::exists(selector_path)) {
    if (model_selector_tool_is_required()) {
      ctx.check(false, "vortex_model_selector test binary exists for malformed numeric flag test");
    } else {
      std::cerr << "[SKIP] vortex_model_selector binary not built for malformed flag test\n";
    }
    return;
  }

  auto json_path = unique_temp_path("vortex_selector_bad_numeric.json");
  std::ostringstream cmd;
  cmd << shell_quote(selector_path.string())
      << " --dataset " << shell_quote("missing.fvecs")
      << " --query " << shell_quote("missing.fvecs")
      << " --target-skeleton-values 10junk"
      << " --output " << shell_quote(json_path.string());
  int status = std::system(cmd.str().c_str());
  ctx.check(status != 0, "vortex_model_selector rejects malformed numeric flag values");
  std::filesystem::remove(json_path);

  json_path = unique_temp_path("vortex_selector_negative_numeric.json");
  std::ostringstream negative_cmd;
  negative_cmd << shell_quote(selector_path.string())
               << " --dataset " << shell_quote("missing.fvecs")
               << " --query " << shell_quote("missing.fvecs")
               << " --target-skeleton-values -1"
               << " --output " << shell_quote(json_path.string());
  status = std::system(negative_cmd.str().c_str());
  ctx.check(status != 0, "vortex_model_selector rejects negative unsigned flag values");
  std::filesystem::remove(json_path);
}

void test_vortex_cli_rejects_malformed_numeric_flags(TestContext& ctx) {
  std::filesystem::path cli_path = vortex_cli_tool_path();
  if (!std::filesystem::exists(cli_path)) {
    if (vortex_cli_tool_is_required()) {
      ctx.check(false, "vortex_v1_cli test binary exists for malformed numeric flag test");
    } else {
      std::cerr << "[SKIP] vortex_v1_cli binary not built for malformed flag test\n";
    }
    return;
  }

  std::ostringstream malformed_cmd;
  malformed_cmd << shell_quote(cli_path.string())
                << " train"
                << " --dataset " << shell_quote("missing.fvecs")
                << " --nsw " << shell_quote("missing.csr")
                << " --K 8junk"
                << " --hash_bits 32"
                << " --cdf_models linear,linear"
                << " --cdf_branch 4"
                << " --centroid_knn 4";
  int status = std::system(malformed_cmd.str().c_str());
  ctx.check(status != 0, "vortex_v1_cli rejects malformed numeric flag values");

  std::ostringstream negative_cmd;
  negative_cmd << shell_quote(cli_path.string())
               << " train"
               << " --dataset " << shell_quote("missing.fvecs")
               << " --nsw " << shell_quote("missing.csr")
               << " --K -1"
               << " --hash_bits 32"
               << " --cdf_models linear,linear"
               << " --cdf_branch 4"
               << " --centroid_knn 4";
  status = std::system(negative_cmd.str().c_str());
  ctx.check(status != 0, "vortex_v1_cli rejects negative unsigned flag values");
}

} // namespace

int main() {
  TestContext ctx;

  test_model_selector_help_contract(ctx);
  test_model_selector_html_report_contract(ctx);
  test_model_selector_rejects_malformed_numeric_flags(ctx);
  test_vortex_cli_rejects_malformed_numeric_flags(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All vortex_v1 CLI tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " tests failed\n";
  return 1;
}
