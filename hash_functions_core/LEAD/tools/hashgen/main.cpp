#include "lead/trainer.h"

#include "rm_model/detail/cli_args.h"
#include "rm_model/logging.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using rm_model::detail::has_help_flag;
using rm_model::detail::is_flag;
using rm_model::detail::parse_finite_double_arg;
using rm_model::detail::parse_size_arg;
using rm_model::detail::parse_u32_arg;
using rm_model::detail::parse_u64_arg;
using rm_model::detail::require_value;

struct Options {
  std::string input;
  std::optional<std::string> namespace_name;
  std::optional<std::string> models;
  std::optional<uint64_t> branching_factor;
  bool no_code = false;
  std::optional<uint64_t> max_size;
  double size_weight = 0.5;
  double error_weight = 0.5;
  std::size_t selector_limit = 10;
  std::optional<std::string> output_dir;
  std::string data_path;
  bool data_path_explicit = false;
  bool no_errors = false;
  std::size_t threads = 4;
  bool zero_build_time = false;
  uint32_t space_bits = 64;
  double scale_factor = 1.0;
  bool no_calibration = false;
};

void print_usage(std::ostream& out) {
  out << "Usage: lead_hashgen <input> [namespace] [models] [branching factor] [options]\n";
  out << "Options:\n";
  out << "  --max-size <bytes>             Train under size budget\n";
  out << "  --selector-limit <n>           Candidate limit in size-constrained selection\n";
  out << "  --space-bits <1..256>          Hash width\n";
  out << "  --scale-factor <v>             Hash-space scale multiplier\n";
  out << "  --output-dir, -o <dir>         Output directory\n";
  out << "  --data-path, -d <dir>          Model data directory\n";
  out << "  --threads, -t <n>              Thread count\n";
  out << "  --no-code                      Skip code generation\n";
  out << "  --no-calibration               Disable calibration stage\n";
  out << "  --help, -h                     Show this help\n";
}

Options parse_args(int argc, char** argv) {
  Options opts;
  std::vector<std::string> positional;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (!is_flag(arg)) {
      positional.push_back(arg);
      continue;
    }

    if (arg == "--no-code") {
      opts.no_code = true;
    } else if (arg == "--max-size") {
      opts.max_size = parse_u64_arg(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--output-dir" || arg == "-o") {
      opts.output_dir = require_value(i, argc, argv, arg);
    } else if (arg == "--data-path" || arg == "-d") {
      opts.data_path = require_value(i, argc, argv, arg);
      opts.data_path_explicit = true;
    } else if (arg == "--no-errors") {
      opts.no_errors = true;
    } else if (arg == "--threads" || arg == "-t") {
      opts.threads = parse_size_arg(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--zero-build-time") {
      opts.zero_build_time = true;
    } else if (arg == "--no-calibration") {
      opts.no_calibration = true;
    } else if (arg == "--size-weight") {
      opts.size_weight = parse_finite_double_arg(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--error-weight") {
      opts.error_weight = parse_finite_double_arg(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--selector-limit") {
      opts.selector_limit = parse_size_arg(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--space-bits") {
      opts.space_bits = parse_u32_arg(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--scale-factor") {
      opts.scale_factor = parse_finite_double_arg(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--help" || arg == "-h") {
      throw std::runtime_error("Usage: lead_hashgen <input> [namespace] [models] [branching factor] [options]");
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  if (positional.empty()) {
    throw std::runtime_error("Usage: lead_hashgen <input> [namespace] [models] [branching factor] [options]");
  }

  opts.input = positional[0];
  if (positional.size() > 1) {
    opts.namespace_name = positional[1];
  }
  if (positional.size() > 2) {
    opts.models = positional[2];
  }
  if (positional.size() > 3) {
    opts.branching_factor = parse_u64_arg(positional[3], "branching factor");
  }

  return opts;
}

} // namespace

int main(int argc, char** argv) {
  if (has_help_flag(argc, argv)) {
    print_usage(std::cout);
    return 0;
  }

  try {
    rm_model::init_logging();

    Options opts = parse_args(argc, argv);
    if (opts.models.has_value() != opts.branching_factor.has_value() && !opts.max_size.has_value()) {
      throw std::runtime_error("Both models and branching factor are required when not using --max-size");
    }
    if (opts.max_size.has_value() && (opts.models.has_value() || opts.branching_factor.has_value())) {
      throw std::runtime_error("Do not combine --max-size with explicit models/branching factor");
    }
    if (opts.space_bits == 0 || opts.space_bits > 256) {
      throw std::runtime_error("--space-bits must be in [1, 256]");
    }

    lead::TrainOptions train_opts;
    train_opts.input = opts.input;
    train_opts.namespace_name = opts.namespace_name;
    train_opts.model_spec = opts.models;
    train_opts.branching_factor = opts.branching_factor;
    train_opts.max_size_bytes = opts.max_size;
    train_opts.size_weight = opts.size_weight;
    train_opts.error_weight = opts.error_weight;
    train_opts.selector_limit = opts.selector_limit;
    if (train_opts.selector_limit < 2) {
      throw std::runtime_error("--selector-limit must be >= 2");
    }
    train_opts.space_bits = opts.space_bits;
    train_opts.scale_factor = opts.scale_factor;
    if (opts.output_dir.has_value()) {
      train_opts.output_dir = std::filesystem::path(*opts.output_dir);
    }
    train_opts.data_path = opts.data_path;
    train_opts.data_path_explicit = opts.data_path_explicit;
    train_opts.emit_code = !opts.no_code;
    train_opts.include_errors = !opts.no_errors;
    train_opts.threads = opts.threads;
    train_opts.zero_build_time = opts.zero_build_time;
    train_opts.calibrate = !opts.no_calibration;

    auto result = lead::train_and_generate(train_opts);

    RM_MODEL_LOG_INFO("LEAD output dir: " << result.lead_dir.string());
    RM_MODEL_LOG_INFO("RM_MODEL output dir: " << result.rm_model_dir.string());
  } catch (const std::exception& ex) {
    RM_MODEL_LOG_ERROR(ex.what());
    return 1;
  }

  return 0;
}
