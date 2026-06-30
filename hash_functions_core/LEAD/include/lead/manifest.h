#ifndef LEAD_MANIFEST_H
#define LEAD_MANIFEST_H

#include "lead/mapping.h"

#include "rm_model/models/model.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

namespace lead {

struct PredictionCalibration {
  bool enabled = false;
  double min_pred = 0.0;
  double max_pred = 0.0;
};

struct LeadManifest {
  uint32_t schema_version = 2;
  std::string name;
  rm_model::KeyType key_type = rm_model::KeyType::U64;
  MappingConfig mapping;
  PredictionCalibration calibration;
  std::string model_spec;
  uint64_t branching_factor = 0;
  uint64_t model_size_bytes = 0;
  uint64_t build_time_ns = 0;
  bool loadable = true;

  std::filesystem::path rm_model_dir;
  std::filesystem::path rm_model_manifest;

  static LeadManifest load(const std::filesystem::path& path);
  void write(const std::filesystem::path& path) const;
};

const char* key_type_name(rm_model::KeyType key_type);
rm_model::KeyType parse_key_type(const std::string& value);

} // namespace lead

#endif // LEAD_MANIFEST_H
