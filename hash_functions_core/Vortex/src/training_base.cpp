#include "training_common_internal.h"
#include "training_internal.h"

#include "rm_model/parallel.h"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace vortex {
namespace {

struct ClusteringResult {
  std::vector<float> centroids;
  std::vector<faiss::idx_t> labels;
  std::vector<float> distances;
};

struct AssignmentResult {
  std::vector<uint64_t> mass;
  std::vector<std::vector<float>> distances;
};

struct CentroidGraphAffinity {
  std::vector<uint64_t> offsets;
  std::vector<uint32_t> neighbors;
  std::vector<float> weights;
};

ClusteringResult cluster_and_assign(const std::vector<float>& points,
                                    uint32_t dim,
                                    uint32_t k,
                                    uint64_t seed) {
  if (points.empty() || dim == 0 || k == 0) {
    throw std::runtime_error("Invalid clustering inputs");
  }
  if (points.size() % dim != 0) {
    throw std::runtime_error("Point buffer size is not divisible by dimension");
  }
  std::size_t n = points.size() / dim;
  if (k > n) {
    throw std::runtime_error("Cluster count K must be <= number of points");
  }

  faiss::Clustering clus(static_cast<int>(dim), static_cast<int>(k));
  clus.seed = static_cast<int>(seed);
  clus.nredo = 1;
  clus.niter = 20;
  clus.verbose = false;

  faiss::IndexFlatL2 quantizer(static_cast<int>(dim));
  clus.train(static_cast<faiss::idx_t>(n), points.data(), quantizer);

  if (clus.centroids.size() != static_cast<std::size_t>(k) * dim) {
    throw std::runtime_error("Unexpected centroid count from clustering");
  }

  faiss::IndexFlatL2 assigner(static_cast<int>(dim));
  assigner.add(static_cast<faiss::idx_t>(k), clus.centroids.data());

  ClusteringResult result;
  result.centroids = clus.centroids;
  result.labels.resize(n);
  result.distances.resize(n);
  assigner.search(static_cast<faiss::idx_t>(n), points.data(), 1,
                  result.distances.data(), result.labels.data());
  return result;
}

AssignmentResult assign_centroids(const std::vector<faiss::idx_t>& labels,
                                  const std::vector<float>& distances,
                                  uint32_t k) {
  if (labels.size() != distances.size()) {
    throw std::runtime_error("Assignment label and distance sizes differ");
  }

  AssignmentResult result;
  result.mass.assign(k, 0);
  result.distances.assign(k, {});

  for (std::size_t i = 0; i < labels.size(); ++i) {
    faiss::idx_t label = labels[i];
    if (label < 0 || static_cast<uint32_t>(label) >= k) {
      throw std::runtime_error("Invalid centroid assignment");
    }
    result.mass[static_cast<uint32_t>(label)] += 1;
  }

  for (uint32_t cid = 0; cid < k; ++cid) {
    if (result.mass[cid] > 0) {
      result.distances[cid].reserve(static_cast<std::size_t>(result.mass[cid]));
    }
  }

  for (std::size_t i = 0; i < labels.size(); ++i) {
    uint32_t cid = static_cast<uint32_t>(labels[i]);
    result.distances[cid].push_back(distances[i]);
  }

  return result;
}

AssignmentResult assign_points_to_centroids(const float* points,
                                            std::size_t count,
                                            uint32_t dim,
                                            const std::vector<float>& centroids,
                                            uint32_t k,
                                            uint64_t sample_limit) {
  if (points == nullptr || count == 0 || dim == 0 || k == 0) {
    throw std::runtime_error("Invalid full-dataset assignment inputs");
  }
  if (centroids.size() != static_cast<std::size_t>(k) * dim) {
    throw std::runtime_error("Centroid buffer size mismatch during assignment");
  }
  if (sample_limit > 0 && sample_limit < k) {
    throw std::runtime_error("assignment_sample_limit must be 0 or >= K");
  }

  std::size_t assignment_count = count;
  if (sample_limit > 0) {
    uint64_t bounded_limit = std::min<uint64_t>(static_cast<uint64_t>(count), sample_limit);
    assignment_count = static_cast<std::size_t>(bounded_limit);
  }
  bool use_sample = assignment_count < count;

  faiss::IndexFlatL2 assigner(static_cast<int>(dim));
  assigner.add(static_cast<faiss::idx_t>(k), centroids.data());

  AssignmentResult result;
  result.mass.assign(k, 0);
  result.distances.assign(k, {});
  std::size_t expected_per_centroid = std::max<std::size_t>(1, assignment_count / k);
  for (auto& bucket : result.distances) {
    bucket.reserve(expected_per_centroid);
  }

  constexpr std::size_t kBatchSize = 65536;
  std::vector<faiss::idx_t> labels(kBatchSize);
  std::vector<float> distances(kBatchSize);
  std::vector<float> sampled_points;
  if (use_sample) {
    sampled_points.resize(kBatchSize * static_cast<std::size_t>(dim));
  }
  auto sample_index = [&](std::size_t ordinal) -> std::size_t {
    if (!use_sample) {
      return ordinal;
    }
    if (assignment_count <= 1) {
      return count / 2;
    }
    long double numerator =
        static_cast<long double>(ordinal) * static_cast<long double>(count - 1);
    long double denominator = static_cast<long double>(assignment_count - 1);
    std::size_t index = static_cast<std::size_t>(numerator / denominator);
    return std::min<std::size_t>(index, count - 1);
  };

  for (std::size_t offset = 0; offset < assignment_count; offset += kBatchSize) {
    std::size_t batch = std::min<std::size_t>(kBatchSize, assignment_count - offset);
    const float* batch_points = points + offset * static_cast<std::size_t>(dim);
    if (use_sample) {
      for (std::size_t i = 0; i < batch; ++i) {
        std::size_t source = sample_index(offset + i);
        const float* src = points + source * static_cast<std::size_t>(dim);
        std::copy(src, src + dim, sampled_points.data() + i * static_cast<std::size_t>(dim));
      }
      batch_points = sampled_points.data();
    }
    assigner.search(static_cast<faiss::idx_t>(batch),
                    batch_points,
                    1,
                    distances.data(),
                    labels.data());
    for (std::size_t i = 0; i < batch; ++i) {
      faiss::idx_t label = labels[i];
      if (label < 0 || static_cast<uint32_t>(label) >= k) {
        throw std::runtime_error("Invalid centroid assignment");
      }
      uint32_t cid = static_cast<uint32_t>(label);
      result.mass[cid] += 1;
      result.distances[cid].push_back(distances[i]);
    }
  }

  return result;
}

void prune_affinity_row(std::unordered_map<uint32_t, uint64_t>* row,
                        std::size_t max_neighbors) {
  if (row == nullptr || row->size() <= max_neighbors) {
    return;
  }

  std::vector<std::pair<uint32_t, uint64_t>> entries;
  entries.reserve(row->size());
  for (const auto& item : *row) {
    entries.emplace_back(item.first, item.second);
  }
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second) {
      return a.second > b.second;
    }
    return a.first < b.first;
  });
  entries.resize(max_neighbors);

  row->clear();
  for (const auto& entry : entries) {
    row->emplace(entry.first, entry.second);
  }
}

CentroidGraphAffinity build_centroid_graph_affinity(const NswCsr& csr,
                                                    const std::vector<faiss::idx_t>& labels,
                                                    uint32_t k) {
  if (k == 0) {
    return {};
  }
  if (labels.size() != csr.node_ids.size()) {
    throw std::runtime_error("Skeleton label count does not match CSR node count");
  }
  if (csr.offsets.size() != csr.node_ids.size() + 1) {
    throw std::runtime_error("CSR offsets size mismatch while building centroid graph");
  }

  constexpr std::size_t kMaxGraphNeighborsPerCentroid = 128;
  constexpr std::size_t kPruneTrigger =
      kMaxGraphNeighborsPerCentroid * static_cast<std::size_t>(4);
  std::vector<std::unordered_map<uint32_t, uint64_t>> rows(k);

  auto label_at = [&](std::size_t local) -> uint32_t {
    faiss::idx_t label = labels[local];
    if (label < 0 || static_cast<uint32_t>(label) >= k) {
      throw std::runtime_error("Invalid skeleton centroid assignment");
    }
    return static_cast<uint32_t>(label);
  };

  auto add_affinity = [&](uint32_t from, uint32_t to) {
    if (from == to) {
      return;
    }
    auto& row = rows[from];
    row[to] += 1;
    if (row.size() > kPruneTrigger) {
      prune_affinity_row(&row, kMaxGraphNeighborsPerCentroid);
    }
  };

  const std::size_t n = csr.node_ids.size();
  for (std::size_t local = 0; local < n; ++local) {
    uint32_t from_centroid = label_at(local);
    uint64_t begin = csr.offsets[local];
    uint64_t end = csr.offsets[local + 1];
    if (end < begin || end > csr.neighbors.size()) {
      throw std::runtime_error("CSR neighbor offsets are invalid while building centroid graph");
    }
    for (uint64_t pos = begin; pos < end; ++pos) {
      uint32_t neighbor = csr.neighbors[static_cast<std::size_t>(pos)];
      if (neighbor >= n) {
        continue;
      }
      uint32_t to_centroid = label_at(neighbor);
      add_affinity(from_centroid, to_centroid);
      add_affinity(to_centroid, from_centroid);
    }
  }

  CentroidGraphAffinity graph;
  graph.offsets.assign(static_cast<std::size_t>(k) + 1, 0);
  for (uint32_t cid = 0; cid < k; ++cid) {
    auto& row = rows[cid];
    prune_affinity_row(&row, kMaxGraphNeighborsPerCentroid);
    graph.offsets[cid] = static_cast<uint64_t>(graph.neighbors.size());
    if (row.empty()) {
      continue;
    }

    uint64_t max_count = 0;
    for (const auto& item : row) {
      max_count = std::max(max_count, item.second);
    }
    if (max_count == 0) {
      continue;
    }

    std::vector<std::pair<uint32_t, uint64_t>> entries;
    entries.reserve(row.size());
    for (const auto& item : row) {
      entries.emplace_back(item.first, item.second);
    }
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });

    for (const auto& entry : entries) {
      graph.neighbors.push_back(entry.first);
      graph.weights.push_back(static_cast<float>(
          static_cast<double>(entry.second) / static_cast<double>(max_count)));
    }
  }
  graph.offsets[k] = static_cast<uint64_t>(graph.neighbors.size());
  return graph;
}

std::vector<float> gather_skeleton(const vector_io::VectorStorage<float>& dataset,
                                   const std::vector<uint64_t>& node_ids) {
  std::size_t dim = dataset.dim;
  std::vector<float> out(node_ids.size() * dim);
  for (std::size_t i = 0; i < node_ids.size(); ++i) {
    uint64_t idx = node_ids[i];
    if (idx >= dataset.count) {
      throw std::runtime_error("CSR node id out of range");
    }
    const float* src = dataset.values.data() + idx * dim;
    std::copy(src, src + dim, out.data() + i * dim);
  }
  return out;
}

} // namespace

namespace training_internal {

uint64_t VortexTrainingBase::memory_bytes() const {
  uint64_t total = byte_count(centroids.size(), sizeof(float));
  total = saturating_add(total, byte_count(mass.size(), sizeof(uint64_t)));
  total = saturating_add(total, byte_count(distances.size(), sizeof(std::vector<float>)));
  for (const auto& bucket : distances) {
    total = saturating_add(total, byte_count(bucket.size(), sizeof(float)));
  }
  total = saturating_add(total, byte_count(centroid_graph_offsets.size(), sizeof(uint64_t)));
  total = saturating_add(total, byte_count(centroid_graph_neighbors.size(), sizeof(uint32_t)));
  total = saturating_add(total, byte_count(centroid_graph_weights.size(), sizeof(float)));
  return total;
}

std::shared_ptr<const VortexTrainingBase> build_vortex_training_base(
    const TrainOptions& options,
    const vector_io::VectorStorage<float>& dataset,
    const NswCsr& csr,
    VortexTrainingProfile* profile) {
  set_threads(options.threads);
  rm_model::set_thread_count(options.threads);

  validate_train_options(options);

  if (dataset.empty()) {
    throw std::runtime_error("Dataset is empty");
  }

  if (csr.metric != Metric::L2) {
    throw std::runtime_error("Only L2 metric is supported in v1");
  }
  if (csr.dim != dataset.dim) {
    throw std::runtime_error("CSR dim does not match dataset");
  }

  std::size_t n_skel = csr.node_ids.size();
  if (n_skel == 0) {
    throw std::runtime_error("NSW skeleton is empty");
  }
  if (n_skel < options.K) {
    throw std::runtime_error("Skeleton size must be >= K");
  }

  auto build_start = TrainingClock::now();
  auto gather_start = TrainingClock::now();
  std::vector<float> skeleton = gather_skeleton(dataset, csr.node_ids);
  if (profile != nullptr) {
    profile->gather_skeleton_ms += elapsed_ms(gather_start);
  }

  auto cluster_start = TrainingClock::now();
  auto clustering = cluster_and_assign(skeleton, dataset.dim, options.K, options.seed);
  if (profile != nullptr) {
    profile->cluster_assign_ms += elapsed_ms(cluster_start);
  }

  CentroidGraphAffinity centroid_graph;
  if (options.enable_graph_centroid_order) {
    centroid_graph = build_centroid_graph_affinity(csr, clustering.labels, options.K);
  }

  auto assign_start = TrainingClock::now();
  bool use_full_dataset_assignments =
      options.use_full_dataset_for_assignments && dataset.count > n_skel;
  auto assignments = use_full_dataset_assignments
      ? assign_points_to_centroids(dataset.values.data(),
                                   dataset.count,
                                   dataset.dim,
                                   clustering.centroids,
                                   options.K,
                                   options.assignment_sample_limit)
      : assign_centroids(clustering.labels, clustering.distances, options.K);
  if (profile != nullptr) {
    profile->assign_centroids_ms += elapsed_ms(assign_start);
  }

  auto base = std::make_shared<VortexTrainingBase>();
  base->dim = dataset.dim;
  base->K = options.K;
  base->seed = options.seed;
  base->skeleton_node_count = static_cast<uint64_t>(n_skel);
  base->centroids = std::move(clustering.centroids);
  base->mass = std::move(assignments.mass);
  base->distances = std::move(assignments.distances);
  base->centroid_graph_offsets = std::move(centroid_graph.offsets);
  base->centroid_graph_neighbors = std::move(centroid_graph.neighbors);
  base->centroid_graph_weights = std::move(centroid_graph.weights);
  if (profile != nullptr) {
    profile->base_build_ms += elapsed_ms(build_start);
  }
  return base;
}

} // namespace training_internal
} // namespace vortex
