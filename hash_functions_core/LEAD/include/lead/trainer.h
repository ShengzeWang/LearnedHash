#ifndef LEAD_TRAINER_H
#define LEAD_TRAINER_H

#include "lead/manifest.h"

#include "rm_model/train.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace lead {

struct TrainOptions {
  std::string input;
  std::optional<std::string> namespace_name;
  std::optional<std::string> model_spec;
  std::optional<uint64_t> branching_factor;
  std::optional<uint64_t> max_size_bytes;
  double size_weight = 0.5;
  double error_weight = 0.5;
  uint32_t space_bits = 64;
  double scale_factor = 1.0;
  bool calibrate = true;
  std::optional<std::filesystem::path> output_dir;
  std::filesystem::path data_path;
  bool data_path_explicit = false;
  bool emit_code = true;
  bool include_errors = true;
  bool zero_build_time = false;
  std::size_t threads = 4;
  std::size_t selector_limit = 10;
};

struct TrainResult {
  LeadManifest manifest;
  std::filesystem::path lead_dir;
  std::filesystem::path rm_model_dir;
};

TrainResult train_and_generate(const TrainOptions& options);

} // namespace lead

#endif // LEAD_TRAINER_H
