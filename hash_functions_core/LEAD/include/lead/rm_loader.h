#ifndef LEAD_RM_LOADER_H
#define LEAD_RM_LOADER_H

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace lead {

enum class RmKeyType {
  U64 = 0,
  U32 = 1,
  F64 = 2,
};

class RmModel {
 public:
  RmModel();
  ~RmModel();
  RmModel(const RmModel&) = delete;
  RmModel& operator=(const RmModel&) = delete;

  bool load(const std::filesystem::path& model_dir,
            const std::optional<std::filesystem::path>& model_lib = std::nullopt,
            const std::optional<std::filesystem::path>& data_dir = std::nullopt);

  double predict_u64(uint64_t key, size_t* err = nullptr) const;
  double predict_u32(uint32_t key, size_t* err = nullptr) const;
  double predict_f64(double key, size_t* err = nullptr) const;
  double predict_raw_u64(uint64_t key, size_t* err = nullptr) const;
  double predict_raw_u32(uint32_t key, size_t* err = nullptr) const;
  double predict_raw_f64(double key, size_t* err = nullptr) const;

  bool has_raw_predict(RmKeyType key_type) const;

  RmKeyType key_type() const { return key_type_; }
  void cleanup();

 private:
  using InferLoadFn = bool (*)(const char*);
  using InferCleanupFn = void (*)();
  using InferKeyTypeFn = int (*)();
  using InferPredictU64Fn = double (*)(uint64_t, size_t*);
  using InferPredictU32Fn = double (*)(uint32_t, size_t*);
  using InferPredictF64Fn = double (*)(double, size_t*);
  using InferPredictRawU64Fn = double (*)(uint64_t, size_t*);
  using InferPredictRawU32Fn = double (*)(uint32_t, size_t*);
  using InferPredictRawF64Fn = double (*)(double, size_t*);

  void* handle_ = nullptr;
  InferCleanupFn infer_cleanup_ = nullptr;
  InferKeyTypeFn infer_key_type_ = nullptr;
  InferPredictU64Fn predict_u64_ = nullptr;
  InferPredictU32Fn predict_u32_ = nullptr;
  InferPredictF64Fn predict_f64_ = nullptr;
  InferPredictRawU64Fn predict_raw_u64_ = nullptr;
  InferPredictRawU32Fn predict_raw_u32_ = nullptr;
  InferPredictRawF64Fn predict_raw_f64_ = nullptr;
  RmKeyType key_type_ = RmKeyType::U64;
  bool loaded_ = false;
};

} // namespace lead

#endif // LEAD_RM_LOADER_H
