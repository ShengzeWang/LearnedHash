#include "lead/manifest.h"

#include "rm_model/json.h"

#include <cmath>
#include <fstream>
#include <optional>
#include <stdexcept>

namespace lead {

namespace {

std::string maybe_relative(const std::filesystem::path& base,
                           const std::filesystem::path& target) {
  if (!base.is_absolute() || !target.is_absolute()) {
    return target.string();
  }
  std::error_code ec;
  auto rel = std::filesystem::relative(target, base, ec);
  if (!ec && !rel.empty() && rel.native()[0] != '.') {
    return rel.string();
  }
  return target.string();
}

std::optional<std::filesystem::path> resolve_path(const std::filesystem::path& base,
                                                  const rm_model::json::Value* value) {
  if (!value || !value->is_string()) return std::nullopt;
  std::filesystem::path path(value->as_string());
  if (path.is_relative()) {
    path = base / path;
  }
  return path;
}

} // namespace

const char* key_type_name(rm_model::KeyType key_type) {
  switch (key_type) {
    case rm_model::KeyType::U64:
      return "u64";
    case rm_model::KeyType::U32:
      return "u32";
    case rm_model::KeyType::F64:
      return "f64";
    case rm_model::KeyType::U128:
      return "u128";
  }
  return "u64";
}

rm_model::KeyType parse_key_type(const std::string& value) {
  if (value == "u64") return rm_model::KeyType::U64;
  if (value == "u32") return rm_model::KeyType::U32;
  if (value == "f64") return rm_model::KeyType::F64;
  if (value == "u128") return rm_model::KeyType::U128;
  throw std::runtime_error("Unknown key type: " + value);
}

LeadManifest LeadManifest::load(const std::filesystem::path& path) {
  auto root = rm_model::json::parse_file(path.string());
  if (!root.is_object()) {
    throw std::runtime_error("lead.json must contain an object");
  }

  LeadManifest manifest;
  if (const auto* schema = root.find("schema_version"); schema && schema->is_number()) {
    manifest.schema_version = static_cast<uint32_t>(schema->as_number().as_uint64());
  }
  if (const auto* name = root.find("name"); name && name->is_string()) {
    manifest.name = name->as_string();
  }
  if (const auto* key_type = root.find("key_type"); key_type && key_type->is_string()) {
    manifest.key_type = parse_key_type(key_type->as_string());
  }
  if (const auto* model_spec = root.find("model_spec"); model_spec && model_spec->is_string()) {
    manifest.model_spec = model_spec->as_string();
  }
  if (const auto* bf = root.find("branching_factor"); bf && bf->is_number()) {
    manifest.branching_factor = bf->as_number().as_uint64();
  }
  if (const auto* size = root.find("model_size_bytes"); size && size->is_number()) {
    manifest.model_size_bytes = size->as_number().as_uint64();
  }
  if (const auto* bt = root.find("build_time_ns"); bt && bt->is_number()) {
    manifest.build_time_ns = bt->as_number().as_uint64();
  }
  if (const auto* loadable = root.find("loadable"); loadable && loadable->is_bool()) {
    manifest.loadable = loadable->as_bool();
  }

  if (const auto* mapping = root.find("mapping"); mapping && mapping->is_object()) {
    bool has_bits = false;
    bool has_max_index = false;
    bool has_scale_factor = false;
    if (const auto* bits = mapping->find("space_bits"); bits && bits->is_number()) {
      manifest.mapping.space_bits = static_cast<uint32_t>(bits->as_number().as_uint64());
      has_bits = true;
    }
    if (const auto* max_index = mapping->find("max_index"); max_index && max_index->is_number()) {
      manifest.mapping.max_index = max_index->as_number().as_uint64();
      has_max_index = true;
    }
    if (const auto* sf = mapping->find("scale_factor"); sf && sf->is_number()) {
      manifest.mapping.scale_factor = sf->as_number().as_double();
      has_scale_factor = true;
    }
    if (const auto* words = mapping->find("scale_words"); words && words->is_array()) {
      const auto& arr = words->as_array();
      if (arr.size() != 4) {
        throw std::runtime_error("scale_words must contain 4 entries");
      }
      for (size_t i = 0; i < 4; ++i) {
        if (!arr[i].is_number()) {
          throw std::runtime_error("scale_words entries must be numbers");
        }
        manifest.mapping.scale_words[i] = arr[i].as_number().as_uint64();
      }
    } else {
      throw std::runtime_error("mapping.scale_words is required");
    }
    if (!has_bits) {
      throw std::runtime_error("mapping.space_bits is required");
    }
    if (!has_max_index) {
      throw std::runtime_error("mapping.max_index is required");
    }
    if (!has_scale_factor) {
      throw std::runtime_error("mapping.scale_factor is required");
    }
    if (manifest.mapping.space_bits == 0 || manifest.mapping.space_bits > 256) {
      throw std::runtime_error("mapping.space_bits must be in [1, 256]");
    }
    if (!(manifest.mapping.scale_factor > 0.0) || manifest.mapping.scale_factor > 1.0) {
      throw std::runtime_error("mapping.scale_factor must be in (0, 1]");
    }
  } else {
    throw std::runtime_error("mapping section is required");
  }

  if (const auto* calibration = root.find("calibration"); calibration && calibration->is_object()) {
    if (const auto* enabled = calibration->find("enabled"); enabled && enabled->is_bool()) {
      manifest.calibration.enabled = enabled->as_bool();
    }
    if (manifest.calibration.enabled) {
      bool has_min = false;
      bool has_max = false;
      if (const auto* min_pred = calibration->find("min_pred");
          min_pred && min_pred->is_number()) {
        manifest.calibration.min_pred = min_pred->as_number().as_double();
        has_min = true;
      }
      if (const auto* max_pred = calibration->find("max_pred");
          max_pred && max_pred->is_number()) {
        manifest.calibration.max_pred = max_pred->as_number().as_double();
        has_max = true;
      }
      if (!has_min || !has_max) {
        throw std::runtime_error("calibration.min_pred and calibration.max_pred are required");
      }
      if (!std::isfinite(manifest.calibration.min_pred) ||
          !std::isfinite(manifest.calibration.max_pred) ||
          !(manifest.calibration.max_pred > manifest.calibration.min_pred)) {
        throw std::runtime_error("calibration range must be finite with max_pred > min_pred");
      }
    }
  }

  std::filesystem::path base = path.parent_path();
  if (const auto* rm_model = root.find("rm_model"); rm_model && rm_model->is_object()) {
    if (const auto* dir_val = rm_model->find("dir")) {
      auto resolved = resolve_path(base, dir_val);
      if (resolved.has_value()) {
        manifest.rm_model_dir = *resolved;
      }
    }
    if (const auto* manifest_val = rm_model->find("manifest")) {
      auto resolved = resolve_path(base, manifest_val);
      if (resolved.has_value()) {
        manifest.rm_model_manifest = *resolved;
      }
    }
  }

  if (manifest.rm_model_dir.empty()) {
    throw std::runtime_error("rm_model.dir is required in lead.json");
  }

  return manifest;
}

void LeadManifest::write(const std::filesystem::path& path) const {
  std::filesystem::create_directories(path.parent_path());

  std::filesystem::path base = std::filesystem::absolute(path.parent_path());
  std::filesystem::path abs_rm_dir = rm_model_dir.is_absolute()
      ? rm_model_dir
      : std::filesystem::absolute(rm_model_dir);
  std::filesystem::path manifest_path = rm_model_manifest.empty()
      ? (rm_model_dir / "model.json")
      : rm_model_manifest;
  std::filesystem::path abs_rm_manifest = manifest_path.is_absolute()
      ? manifest_path
      : std::filesystem::absolute(manifest_path);

  rm_model::json::Value::Object root;
  root.emplace_back("schema_version", rm_model::json::Value(static_cast<uint64_t>(schema_version)));
  root.emplace_back("name", rm_model::json::Value(name));
  root.emplace_back("key_type", rm_model::json::Value(key_type_name(key_type)));
  root.emplace_back("model_spec", rm_model::json::Value(model_spec));
  root.emplace_back("branching_factor", rm_model::json::Value(branching_factor));
  root.emplace_back("model_size_bytes", rm_model::json::Value(model_size_bytes));
  root.emplace_back("build_time_ns", rm_model::json::Value(build_time_ns));
  root.emplace_back("loadable", rm_model::json::Value(loadable));

  rm_model::json::Value::Object mapping_obj;
  mapping_obj.emplace_back("space_bits", rm_model::json::Value(static_cast<uint64_t>(mapping.space_bits)));
  mapping_obj.emplace_back("max_index", rm_model::json::Value(mapping.max_index));
  mapping_obj.emplace_back("scale_factor", rm_model::json::Value(mapping.scale_factor));
  rm_model::json::Value::Array scale_words;
  for (auto word : mapping.scale_words) {
    scale_words.emplace_back(word);
  }
  mapping_obj.emplace_back("scale_words", rm_model::json::Value(std::move(scale_words)));
  root.emplace_back("mapping", rm_model::json::Value(std::move(mapping_obj)));

  rm_model::json::Value::Object calibration_obj;
  calibration_obj.emplace_back("enabled", rm_model::json::Value(calibration.enabled));
  calibration_obj.emplace_back("min_pred", rm_model::json::Value(calibration.min_pred));
  calibration_obj.emplace_back("max_pred", rm_model::json::Value(calibration.max_pred));
  root.emplace_back("calibration", rm_model::json::Value(std::move(calibration_obj)));

  rm_model::json::Value::Object rm_obj;
  rm_obj.emplace_back("dir", rm_model::json::Value(maybe_relative(base, abs_rm_dir)));
  rm_obj.emplace_back("manifest", rm_model::json::Value(maybe_relative(base, abs_rm_manifest)));
  root.emplace_back("rm_model", rm_model::json::Value(std::move(rm_obj)));

  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not write lead.json");
  }
  rm_model::json::write(out, rm_model::json::Value(std::move(root)));
}

} // namespace lead
