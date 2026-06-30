#include "training_common_internal.h"

#include <limits>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace vortex::training_internal {

double elapsed_ms(TrainingClock::time_point start) {
  auto end = TrainingClock::now();
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) /
         1000.0;
}

uint64_t byte_count(std::size_t count, std::size_t item_size) {
  if (item_size != 0 && count > std::numeric_limits<uint64_t>::max() / item_size) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(count) * static_cast<uint64_t>(item_size);
}

uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs + rhs;
}

void set_threads(uint32_t threads) {
#if defined(_OPENMP)
  if (threads > 0) {
    omp_set_num_threads(static_cast<int>(threads));
  }
#else
  (void)threads;
#endif
}

vector_io::VectorStorage<float> load_vectors_or_throw(const std::filesystem::path& path) {
  auto ext = path.extension().string();
  if (ext == ".fvecs") {
    return vector_io::read_fvecs(path.string());
  }
  throw std::runtime_error("Unsupported vector dataset (expected .fvecs): " + path.string());
}

void validate_train_options(const TrainOptions& options) {
  if (options.K == 0) {
    throw std::runtime_error("K must be > 0");
  }
  if (options.hash_bits == 0 || options.hash_bits > 256) {
    throw std::runtime_error("hash_bits must be in [1, 256]");
  }
  if (options.cdf_model_spec.empty()) {
    throw std::runtime_error("cdf_model_spec must not be empty");
  }
  if (options.cdf_branching_factor < 2) {
    throw std::runtime_error("cdf_branching_factor must be >= 2");
  }
  if (options.cdf_branching_factor > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("cdf_branching_factor too large");
  }
  if (options.centroid_knn == 0) {
    throw std::runtime_error("centroid_knn must be > 0");
  }
  if (options.enable_2opt && options.two_opt_iterations == 0) {
    throw std::runtime_error("two_opt_iterations must be > 0 when enable_2opt is set");
  }
}

void validate_training_base(const TrainOptions& options, const VortexTrainingBase& base) {
  if (base.dim == 0 ||
      base.K != options.K ||
      base.seed != options.seed ||
      base.centroids.size() != static_cast<std::size_t>(base.K) * base.dim ||
      base.mass.size() != base.K ||
      base.distances.size() != base.K) {
    throw std::runtime_error("Training base does not match requested Vortex config");
  }
}

} // namespace vortex::training_internal
