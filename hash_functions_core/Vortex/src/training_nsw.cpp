#include "vortex_v1/training.h"

#include "training_common_internal.h"

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <faiss/impl/HNSW.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace vortex {
namespace {

std::unique_ptr<faiss::IndexHNSW> read_hnsw_index(const std::filesystem::path& path) {
  std::unique_ptr<faiss::Index> idx(faiss::read_index(path.string().c_str()));
  if (!idx) {
    throw std::runtime_error("Failed to read FAISS index: " + path.string());
  }
  auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(idx.get());
  if (!hnsw) {
    throw std::runtime_error("Index is not an HNSW index: " + path.string());
  }
  idx.release();
  return std::unique_ptr<faiss::IndexHNSW>(hnsw);
}

std::unique_ptr<faiss::IndexHNSW> build_hnsw(const BuildHnswOptions& options) {
  auto vectors = training_internal::load_vectors_or_throw(options.dataset_path);
  if (vectors.empty()) {
    throw std::runtime_error("Dataset is empty");
  }

  faiss::MetricType metric = faiss::METRIC_L2;
  if (options.metric != Metric::L2) {
    throw std::runtime_error("Only L2 metric is supported in v1");
  }

  auto index = std::make_unique<faiss::IndexHNSWFlat>(static_cast<int>(vectors.dim),
                                                      static_cast<int>(options.M),
                                                      metric);
  index->hnsw.efConstruction = static_cast<int>(options.ef_construction);
  index->add(static_cast<faiss::idx_t>(vectors.count), vectors.data());
  return index;
}

int choose_layer(const faiss::HNSW& hnsw, uint64_t target) {
  int max_layer = hnsw.max_level;
  if (target == 0) {
    throw std::runtime_error("target_skeleton must be > 0");
  }
  for (int layer = max_layer; layer >= 0; --layer) {
    uint64_t count = 0;
    for (int level : hnsw.levels) {
      // FAISS stores node level as (pt_level + 1), so membership at layer L is levels[i] > L.
      if (level > layer) {
        ++count;
      }
    }
    if (count >= target) {
      return layer;
    }
  }
  return 0;
}

NswCsr extract_layer(const faiss::IndexHNSW& index, int layer) {
  const faiss::HNSW& hnsw = index.hnsw;
  std::size_t ntotal = static_cast<std::size_t>(index.ntotal);

  std::vector<uint64_t> node_ids;
  node_ids.reserve(ntotal);
  std::vector<int64_t> local(ntotal, -1);

  for (std::size_t i = 0; i < ntotal; ++i) {
    if (hnsw.levels[i] > layer) {
      local[i] = static_cast<int64_t>(node_ids.size());
      node_ids.push_back(static_cast<uint64_t>(i));
    }
  }
  if (node_ids.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Skeleton too large for uint32 neighbors");
  }

  std::vector<uint64_t> offsets(node_ids.size() + 1, 0);
  std::vector<uint32_t> neighbors;
  if (!node_ids.empty()) {
    std::size_t per_node = static_cast<std::size_t>(hnsw.nb_neighbors(layer));
    if (per_node > 0 &&
        per_node <= std::numeric_limits<std::size_t>::max() / node_ids.size()) {
      neighbors.reserve(node_ids.size() * per_node);
    }
  }

  for (std::size_t idx = 0; idx < node_ids.size(); ++idx) {
    std::size_t global = static_cast<std::size_t>(node_ids[idx]);
    size_t begin = 0;
    size_t end = 0;
    hnsw.neighbor_range(static_cast<faiss::idx_t>(global), layer, &begin, &end);
    offsets[idx] = neighbors.size();
    for (size_t pos = begin; pos < end; ++pos) {
      auto neighbor = hnsw.neighbors[pos];
      if (neighbor < 0) {
        continue;
      }
      std::size_t nb = static_cast<std::size_t>(neighbor);
      if (nb >= ntotal) {
        continue;
      }
      int64_t local_idx = local[nb];
      if (local_idx < 0) {
        continue;
      }
      neighbors.push_back(static_cast<uint32_t>(local_idx));
    }
  }
  offsets.back() = neighbors.size();

  NswCsr csr;
  csr.dim = static_cast<uint32_t>(index.d);
  csr.metric = Metric::L2;
  csr.layer = layer;
  csr.node_ids = std::move(node_ids);
  csr.offsets = std::move(offsets);
  csr.neighbors = std::move(neighbors);
  return csr;
}

} // namespace

std::filesystem::path build_hnsw_index(const BuildHnswOptions& options,
                                      const std::filesystem::path& output_path) {
  auto index = build_hnsw(options);
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  faiss::write_index(index.get(), output_path.string().c_str());
  return output_path;
}

NswCsr extract_nsw(const ExtractNswOptions& options,
                   const std::filesystem::path& output_path) {
  std::unique_ptr<faiss::IndexHNSW> index;
  if (options.index_path.has_value()) {
    index = read_hnsw_index(*options.index_path);
  } else {
    if (!options.dataset_path.has_value()) {
      throw std::runtime_error("extract_nsw requires --index or --dataset");
    }
    BuildHnswOptions build_opts;
    build_opts.dataset_path = *options.dataset_path;
    build_opts.M = options.M;
    build_opts.ef_construction = options.ef_construction;
    build_opts.metric = options.metric;
    index = build_hnsw(build_opts);
  }

  if (index->metric_type != faiss::METRIC_L2) {
    throw std::runtime_error("Only L2 metric is supported in v1");
  }

  int layer = 0;
  if (options.layer.has_value()) {
    layer = *options.layer;
    if (layer < 0 || layer > index->hnsw.max_level) {
      throw std::runtime_error("Requested layer is out of range [0, max_level]");
    }
  } else {
    layer = choose_layer(index->hnsw, options.target_skeleton);
  }

  NswCsr csr = extract_layer(*index, layer);
  csr.write(output_path);
  return csr;
}

} // namespace vortex
