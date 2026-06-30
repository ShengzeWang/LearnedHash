#ifndef VORTEX_V1_TRAINING_COMMON_INTERNAL_H
#define VORTEX_V1_TRAINING_COMMON_INTERNAL_H

#include "training_internal.h"

#include "vector_io/vector_io.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace vortex::training_internal {

using TrainingClock = std::chrono::steady_clock;

double elapsed_ms(TrainingClock::time_point start);
uint64_t byte_count(std::size_t count, std::size_t item_size);
uint64_t saturating_add(uint64_t lhs, uint64_t rhs);

void set_threads(uint32_t threads);
vector_io::VectorStorage<float> load_vectors_or_throw(const std::filesystem::path& path);

void validate_train_options(const TrainOptions& options);
void validate_training_base(const TrainOptions& options, const VortexTrainingBase& base);

} // namespace vortex::training_internal

#endif // VORTEX_V1_TRAINING_COMMON_INTERNAL_H
