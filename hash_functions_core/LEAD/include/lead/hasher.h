#ifndef LEAD_HASHER_H
#define LEAD_HASHER_H

#include "lead/hash.h"
#include "lead/mapping.h"
#include "lead/manifest.h"
#include "lead/rm_loader.h"

#include <filesystem>

namespace lead {

class Hasher {
 public:
  Hasher();

  bool load(const std::filesystem::path& model_dir);

  HashOutput hash(uint64_t key) const;
  HashOutput hash(uint32_t key) const;
  HashOutput hash(double key) const;

  HashOutput hash_with_pred(uint64_t key, double* prediction) const;
  HashOutput hash_with_pred(uint32_t key, double* prediction) const;
  HashOutput hash_with_pred(double key, double* prediction) const;

  uint64_t hash64(uint64_t key) const;
  uint64_t hash64(uint32_t key) const;
  uint64_t hash64(double key) const;

  std::array<uint64_t, 2> hash128(uint64_t key) const;
  std::array<uint64_t, 2> hash128(uint32_t key) const;
  std::array<uint64_t, 2> hash128(double key) const;

  UInt256 hash256(uint64_t key) const;
  UInt256 hash256(uint32_t key) const;
  UInt256 hash256(double key) const;

  RmKeyType key_type() const { return key_type_; }
  uint32_t space_bits() const { return mapper_.space_bits(); }
  uint64_t max_index() const { return mapper_.max_index(); }
  const MappingConfig& mapping() const { return mapping_; }
  const PredictionCalibration& calibration() const { return calibration_; }
  bool calibration_enabled() const { return calibration_enabled_; }
  double calibration_scale() const { return calibration_scale_; }
  bool raw_predict_available() const { return raw_predict_available_; }
  const LeadManifest& manifest() const { return manifest_; }

 private:
  double apply_calibration(double pred) const;

  LeadManifest manifest_{};
  MappingConfig mapping_{};
  IndexMapper mapper_{mapping_};
  RmModel model_;
  RmKeyType key_type_ = RmKeyType::U64;
  PredictionCalibration calibration_{};
  double calibration_scale_ = 1.0;
  bool calibration_enabled_ = false;
  bool raw_predict_available_ = false;
  bool loaded_ = false;
};

} // namespace lead

#endif // LEAD_HASHER_H
