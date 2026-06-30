#include "selector_internal.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vortex::selector_internal {

namespace {

void hash_combine_u64(uint64_t value, uint64_t* hash) {
  *hash ^= value;
  *hash *= 1099511628211ULL;
}

std::string nsw_identity_key(const NswCsr& csr) {
  uint64_t hash = 1469598103934665603ULL;
  hash_combine_u64(csr.dim, &hash);
  hash_combine_u64(static_cast<uint32_t>(csr.metric), &hash);
  hash_combine_u64(static_cast<uint64_t>(static_cast<int64_t>(csr.layer)), &hash);
  hash_combine_u64(static_cast<uint64_t>(csr.node_ids.size()), &hash);
  hash_combine_u64(static_cast<uint64_t>(csr.offsets.size()), &hash);
  hash_combine_u64(static_cast<uint64_t>(csr.neighbors.size()), &hash);
  for (uint64_t value : csr.node_ids) {
    hash_combine_u64(value, &hash);
  }
  for (uint64_t value : csr.offsets) {
    hash_combine_u64(value, &hash);
  }
  for (uint32_t value : csr.neighbors) {
    hash_combine_u64(value, &hash);
  }
  return std::to_string(csr.dim) + ":" + std::to_string(static_cast<uint32_t>(csr.metric)) +
         ":" + std::to_string(csr.layer) + ":" + std::to_string(csr.node_ids.size()) +
         ":" + std::to_string(csr.neighbors.size()) + ":" + std::to_string(hash);
}

uint64_t choose_representative_target(const std::vector<uint64_t>& targets,
                                      std::size_t skeleton_count) {
  if (targets.empty()) {
    return 0;
  }
  uint64_t best = targets.front();
  uint64_t actual = static_cast<uint64_t>(
      std::min<std::size_t>(skeleton_count, std::numeric_limits<uint64_t>::max()));
  auto distance = [actual](uint64_t value) {
    return value > actual ? (value - actual) : (actual - value);
  };
  for (uint64_t value : targets) {
    uint64_t best_distance = distance(best);
    uint64_t value_distance = distance(value);
    if (value_distance < best_distance ||
        (value_distance == best_distance && value > best)) {
      best = value;
    }
  }
  return best;
}

std::string display_path_for_target(const SelectorOptions& raw, uint64_t target) {
  std::string stem = raw.dataset_path.stem().string();
  if (stem.empty()) {
    stem = "dataset";
  }
  return (std::filesystem::path("vortex_v1_output") /
          (stem + "_nsw_ts" + std::to_string(target) + ".csr"))
      .string();
}

void refresh_representative_target(NswSelectorContext* context,
                                   const SelectorOptions& raw) {
  uint64_t target =
      choose_representative_target(context->requested_target_skeletons,
                                   context->csr.node_ids.size());
  if (target == 0) {
    return;
  }
  context->target_skeleton = target;
  context->display_nsw_path = display_path_for_target(raw, target);
}

} // namespace

std::vector<NswSelectorContext> build_nsw_contexts(const SelectorOptions& raw,
                                                   const vector_io::VectorStorage<float>& full_base,
                                                   ScopedTempFiles* temp_files) {
  std::vector<NswSelectorContext> contexts;
  std::vector<uint64_t> target_values = resolve_target_skeleton_values(raw, full_base.count);
  if (!target_values.empty()) {
    if (!raw.nsw_index_path.has_value()) {
      throw std::runtime_error("target_skeleton_values/percentages require nsw_index_path");
    }
    contexts.reserve(target_values.size());
    std::unordered_map<std::string, std::size_t> context_by_identity;
    context_by_identity.reserve(target_values.size() * 2 + 1);
    for (uint64_t target : target_values) {
      ExtractNswOptions extract_opts;
      extract_opts.index_path = raw.nsw_index_path;
      extract_opts.target_skeleton = target;
      extract_opts.layer = raw.nsw_layer;
      std::filesystem::path temp_nsw = unique_temp_path("vortex_selector_nsw", ".csr");
      NswCsr csr;
      try {
        csr = extract_nsw(extract_opts, temp_nsw);
      } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temp_nsw, ignored);
        throw;
      }

      NswSelectorContext context;
      context.nsw_path = temp_nsw;
      context.csr = std::move(csr);
      context.skeleton_identity = nsw_identity_key(context.csr);
      context.target_skeleton = target;
      context.requested_target_skeletons.push_back(target);
      context.display_nsw_path = display_path_for_target(raw, target);
      auto duplicate = context_by_identity.find(context.skeleton_identity);
      if (duplicate != context_by_identity.end()) {
        auto& existing = contexts[duplicate->second];
        existing.requested_target_skeletons.push_back(target);
        refresh_representative_target(&existing, raw);
        temp_files->paths.push_back(temp_nsw);
        continue;
      }
      context_by_identity.emplace(context.skeleton_identity, contexts.size());
      contexts.push_back(std::move(context));
      temp_files->paths.push_back(temp_nsw);
    }
  } else {
    if (raw.nsw_path.empty()) {
      throw std::runtime_error(
          "nsw_path is required when target_skeleton_values/percentages are not provided");
    }
    NswSelectorContext context;
    context.nsw_path = raw.nsw_path;
    context.display_nsw_path = raw.nsw_path.string();
    context.csr = NswCsr::read(raw.nsw_path);
    context.target_skeleton = 0;
    context.requested_target_skeletons.push_back(0);
    context.skeleton_identity = nsw_identity_key(context.csr);
    contexts.push_back(std::move(context));
  }

  for (const auto& context : contexts) {
    if (context.csr.dim != full_base.dim) {
      throw std::runtime_error("NSW dim does not match dataset dim");
    }
    if (context.csr.node_ids.empty()) {
      throw std::runtime_error("NSW skeleton has no nodes");
    }
  }
  return contexts;
}

} // namespace vortex::selector_internal
