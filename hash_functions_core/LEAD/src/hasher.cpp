#include "lead/hasher.h"

#include <cmath>
#include <stdexcept>

namespace lead {

Hasher::Hasher() = default;

bool Hasher::load(const std::filesystem::path& model_dir) {
  std::filesystem::path manifest_path = model_dir / "lead.json";
  manifest_ = LeadManifest::load(manifest_path);
  if (!manifest_.loadable) {
    throw std::runtime_error("LEAD artifacts are not loadable (generated with --no-code)");
  }

  mapping_ = manifest_.mapping;
  mapper_ = IndexMapper(mapping_);
  if (!model_.load(manifest_.rm_model_dir)) {
    return false;
  }

  key_type_ = model_.key_type();
  if (manifest_.key_type == rm_model::KeyType::U32 && key_type_ != RmKeyType::U32) {
    throw std::runtime_error("Model key type mismatch (expected u32)");
  }
  if (manifest_.key_type == rm_model::KeyType::U64 && key_type_ != RmKeyType::U64) {
    throw std::runtime_error("Model key type mismatch (expected u64)");
  }
  if (manifest_.key_type == rm_model::KeyType::F64 && key_type_ != RmKeyType::F64) {
    throw std::runtime_error("Model key type mismatch (expected f64)");
  }

  calibration_ = manifest_.calibration;
  calibration_enabled_ = false;
  calibration_scale_ = 1.0;
  raw_predict_available_ = model_.has_raw_predict(key_type_);
  if (calibration_.enabled) {
    if (!std::isfinite(calibration_.min_pred) || !std::isfinite(calibration_.max_pred) ||
        !(calibration_.max_pred > calibration_.min_pred)) {
      throw std::runtime_error("Invalid calibration range in manifest");
    }
    if (raw_predict_available_ && mapping_.max_index > 0) {
      double denom = calibration_.max_pred - calibration_.min_pred;
      calibration_scale_ = static_cast<double>(mapping_.max_index) / denom;
      calibration_enabled_ = std::isfinite(calibration_scale_) && calibration_scale_ > 0.0;
    }
  }

  loaded_ = true;
  return true;
}

double Hasher::apply_calibration(double pred) const {
  if (!calibration_enabled_) {
    return pred;
  }
  return (pred - calibration_.min_pred) * calibration_scale_;
}

HashOutput Hasher::hash(uint64_t key) const {
  if (!loaded_) {
    throw std::runtime_error("Hasher is not loaded");
  }
  if (key_type_ != RmKeyType::U64) {
    throw std::runtime_error("Hasher key type mismatch (expected u64)");
  }
  double pred = raw_predict_available_
      ? model_.predict_raw_u64(key, nullptr)
      : model_.predict_u64(key, nullptr);
  double calibrated = apply_calibration(pred);
  return mapper_.map_double(calibrated);
}

HashOutput Hasher::hash(uint32_t key) const {
  if (!loaded_) {
    throw std::runtime_error("Hasher is not loaded");
  }
  if (key_type_ != RmKeyType::U32) {
    throw std::runtime_error("Hasher key type mismatch (expected u32)");
  }
  double pred = raw_predict_available_
      ? model_.predict_raw_u32(key, nullptr)
      : model_.predict_u32(key, nullptr);
  double calibrated = apply_calibration(pred);
  return mapper_.map_double(calibrated);
}

HashOutput Hasher::hash(double key) const {
  if (!loaded_) {
    throw std::runtime_error("Hasher is not loaded");
  }
  if (key_type_ != RmKeyType::F64) {
    throw std::runtime_error("Hasher key type mismatch (expected f64)");
  }
  double pred = raw_predict_available_
      ? model_.predict_raw_f64(key, nullptr)
      : model_.predict_f64(key, nullptr);
  double calibrated = apply_calibration(pred);
  return mapper_.map_double(calibrated);
}

HashOutput Hasher::hash_with_pred(uint64_t key, double* prediction) const {
  if (!loaded_) {
    throw std::runtime_error("Hasher is not loaded");
  }
  if (key_type_ != RmKeyType::U64) {
    throw std::runtime_error("Hasher key type mismatch (expected u64)");
  }
  double pred = raw_predict_available_
      ? model_.predict_raw_u64(key, nullptr)
      : model_.predict_u64(key, nullptr);
  if (prediction) {
    *prediction = pred;
  }
  double calibrated = apply_calibration(pred);
  return mapper_.map_double(calibrated);
}

HashOutput Hasher::hash_with_pred(uint32_t key, double* prediction) const {
  if (!loaded_) {
    throw std::runtime_error("Hasher is not loaded");
  }
  if (key_type_ != RmKeyType::U32) {
    throw std::runtime_error("Hasher key type mismatch (expected u32)");
  }
  double pred = raw_predict_available_
      ? model_.predict_raw_u32(key, nullptr)
      : model_.predict_u32(key, nullptr);
  if (prediction) {
    *prediction = pred;
  }
  double calibrated = apply_calibration(pred);
  return mapper_.map_double(calibrated);
}

HashOutput Hasher::hash_with_pred(double key, double* prediction) const {
  if (!loaded_) {
    throw std::runtime_error("Hasher is not loaded");
  }
  if (key_type_ != RmKeyType::F64) {
    throw std::runtime_error("Hasher key type mismatch (expected f64)");
  }
  double pred = raw_predict_available_
      ? model_.predict_raw_f64(key, nullptr)
      : model_.predict_f64(key, nullptr);
  if (prediction) {
    *prediction = pred;
  }
  double calibrated = apply_calibration(pred);
  return mapper_.map_double(calibrated);
}

uint64_t Hasher::hash64(uint64_t key) const {
  return hash(key).to_u64_msb();
}

uint64_t Hasher::hash64(uint32_t key) const {
  return hash(key).to_u64_msb();
}

uint64_t Hasher::hash64(double key) const {
  return hash(key).to_u64_msb();
}

std::array<uint64_t, 2> Hasher::hash128(uint64_t key) const {
  return hash(key).to_u128_msb();
}

std::array<uint64_t, 2> Hasher::hash128(uint32_t key) const {
  return hash(key).to_u128_msb();
}

std::array<uint64_t, 2> Hasher::hash128(double key) const {
  return hash(key).to_u128_msb();
}

UInt256 Hasher::hash256(uint64_t key) const {
  return hash(key).value;
}

UInt256 Hasher::hash256(uint32_t key) const {
  return hash(key).value;
}

UInt256 Hasher::hash256(double key) const {
  return hash(key).value;
}

} // namespace lead
