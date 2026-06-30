#include "training_common_internal.h"
#include "training_internal.h"

#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <queue>
#include <stdexcept>

namespace vortex {
namespace {

struct Edge {
  uint32_t to = 0;
  double weight = 0.0;
};

struct MstEdge {
  uint32_t a = 0;
  uint32_t b = 0;
  double weight = 0.0;
};

std::vector<std::vector<Edge>> build_knn_edges(const std::vector<float>& centroids,
                                               uint32_t dim,
                                               uint32_t knn) {
  std::size_t k = centroids.empty() ? 0 : centroids.size() / dim;
  if (k == 0) {
    throw std::runtime_error("No centroids to order");
  }

  faiss::IndexFlatL2 index(static_cast<int>(dim));
  index.add(static_cast<faiss::idx_t>(k), centroids.data());

  uint32_t search_k = std::min<uint32_t>(knn + 1, static_cast<uint32_t>(k));
  std::vector<faiss::idx_t> labels(k * search_k);
  std::vector<float> distances(k * search_k);

  index.search(static_cast<faiss::idx_t>(k), centroids.data(),
               static_cast<int>(search_k), distances.data(), labels.data());

  std::vector<std::vector<Edge>> adj(k);
  for (std::size_t i = 0; i < k; ++i) {
    adj[i].reserve(knn);
    for (uint32_t j = 0; j < search_k; ++j) {
      faiss::idx_t nb = labels[i * search_k + j];
      if (nb < 0 || static_cast<std::size_t>(nb) == i) {
        continue;
      }
      Edge e;
      e.to = static_cast<uint32_t>(nb);
      e.weight = static_cast<double>(distances[i * search_k + j]);
      adj[i].push_back(e);
    }
  }

  return adj;
}

std::vector<MstEdge> build_mst(const std::vector<std::vector<Edge>>& adj,
                               const std::vector<float>& centroids,
                               uint32_t dim) {
  std::size_t n = adj.size();
  std::vector<bool> in_mst(n, false);
  std::vector<MstEdge> mst;
  mst.reserve(n ? n - 1 : 0);

  auto dist = [&](uint32_t a, uint32_t b) {
    const float* pa = centroids.data() + static_cast<std::size_t>(a) * dim;
    const float* pb = centroids.data() + static_cast<std::size_t>(b) * dim;
    double acc = 0.0;
    for (uint32_t i = 0; i < dim; ++i) {
      double diff = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
      acc += diff * diff;
    }
    return acc;
  };

  struct HeapItem {
    double weight;
    uint32_t from;
    uint32_t to;
  };

  auto cmp = [](const HeapItem& a, const HeapItem& b) {
    if (a.weight != b.weight) return a.weight > b.weight;
    if (a.from != b.from) return a.from > b.from;
    return a.to > b.to;
  };

  std::priority_queue<HeapItem, std::vector<HeapItem>, decltype(cmp)> pq(cmp);

  if (n == 0) {
    return mst;
  }

  auto add_edges = [&](uint32_t node) {
    for (const auto& edge : adj[node]) {
      if (!in_mst[edge.to]) {
        pq.push({edge.weight, node, edge.to});
      }
    }
  };

  uint32_t visited = 0;
  in_mst[0] = true;
  visited = 1;
  add_edges(0);

  while (visited < n && !pq.empty()) {
    auto item = pq.top();
    pq.pop();
    if (in_mst[item.to]) {
      continue;
    }
    in_mst[item.to] = true;
    visited++;
    mst.push_back({item.from, item.to, item.weight});
    add_edges(item.to);
  }

  if (visited < n) {
    for (uint32_t node = 0; node < n; ++node) {
      if (in_mst[node]) {
        continue;
      }
      double best = std::numeric_limits<double>::infinity();
      uint32_t best_from = 0;
      for (uint32_t v = 0; v < n; ++v) {
        if (!in_mst[v]) {
          continue;
        }
        double d = dist(v, node);
        if (d < best) {
          best = d;
          best_from = v;
        }
      }
      in_mst[node] = true;
      visited++;
      mst.push_back({best_from, node, best});
    }
  }

  return mst;
}

std::vector<uint32_t> dfs_order_from_mst(uint32_t n, const std::vector<MstEdge>& mst) {
  std::vector<std::vector<Edge>> adj(n);
  for (const auto& edge : mst) {
    adj[edge.a].push_back({edge.b, edge.weight});
    adj[edge.b].push_back({edge.a, edge.weight});
  }
  for (auto& list : adj) {
    std::sort(list.begin(), list.end(), [](const Edge& a, const Edge& b) {
      if (a.weight != b.weight) return a.weight < b.weight;
      return a.to < b.to;
    });
  }

  std::vector<uint32_t> order;
  order.reserve(n);
  std::vector<bool> visited(n, false);
  std::vector<uint32_t> stack;
  stack.push_back(0);

  while (!stack.empty()) {
    uint32_t node = stack.back();
    stack.pop_back();
    if (visited[node]) {
      continue;
    }
    visited[node] = true;
    order.push_back(node);
    auto& neighbors = adj[node];
    for (auto it = neighbors.rbegin(); it != neighbors.rend(); ++it) {
      if (!visited[it->to]) {
        stack.push_back(it->to);
      }
    }
  }

  return order;
}

double centroid_dist(const std::vector<float>& centroids, uint32_t dim, uint32_t a, uint32_t b) {
  const float* pa = centroids.data() + static_cast<std::size_t>(a) * dim;
  const float* pb = centroids.data() + static_cast<std::size_t>(b) * dim;
  double acc = 0.0;
  for (uint32_t i = 0; i < dim; ++i) {
    double diff = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
    acc += diff * diff;
  }
  return acc;
}

uint64_t square_byte_count(std::size_t count, std::size_t item_size) {
  if (count != 0 && count > std::numeric_limits<std::size_t>::max() / count) {
    return std::numeric_limits<uint64_t>::max();
  }
  return training_internal::byte_count(count * count, item_size);
}

class CentroidDistanceLookup {
 public:
  CentroidDistanceLookup(const std::vector<float>& centroids,
                         uint32_t dim,
                         bool enable_table,
                         bool enable_graph_order,
                         const std::vector<uint64_t>* graph_offsets = nullptr,
                         const std::vector<uint32_t>* graph_neighbors = nullptr,
                         const std::vector<float>* graph_weights = nullptr)
      : centroids_(centroids),
        dim_(dim),
        graph_offsets_(graph_offsets),
        graph_neighbors_(graph_neighbors),
        graph_weights_(graph_weights) {
    n_ = dim == 0 ? 0 : centroids.size() / dim;
    graph_enabled_ =
        enable_graph_order &&
        graph_offsets_ != nullptr &&
        graph_neighbors_ != nullptr &&
        graph_weights_ != nullptr &&
        graph_offsets_->size() == n_ + 1 &&
        graph_neighbors_->size() == graph_weights_->size();
    if (!enable_table || n_ == 0) {
      return;
    }

    constexpr uint64_t kMaxTableBytes = 128ULL * 1024ULL * 1024ULL;
    if (square_byte_count(n_, sizeof(double)) > kMaxTableBytes) {
      return;
    }

    try {
      table_.assign(n_ * n_, 0.0);
    } catch (const std::bad_alloc&) {
      table_.clear();
      return;
    }

    for (std::size_t i = 0; i < n_; ++i) {
      for (std::size_t j = i + 1; j < n_; ++j) {
        double d = centroid_dist(centroids_, dim_, static_cast<uint32_t>(i),
                                 static_cast<uint32_t>(j));
        table_[i * n_ + j] = d;
        table_[j * n_ + i] = d;
      }
    }
  }

  double operator()(uint32_t a, uint32_t b) const {
    double base = raw_distance(a, b);
    if (!graph_enabled_) {
      return base;
    }
    double affinity = graph_affinity(a, b);
    if (!(affinity > 0.0)) {
      return base;
    }
    constexpr double kGraphOrderCostDiscount = 0.25;
    double multiplier = 1.0 - kGraphOrderCostDiscount * std::min(1.0, affinity);
    return base * std::max(0.05, multiplier);
  }

  double raw_distance(uint32_t a, uint32_t b) const {
    if (!table_.empty()) {
      return table_[static_cast<std::size_t>(a) * n_ + b];
    }
    return centroid_dist(centroids_, dim_, a, b);
  }

  double graph_affinity(uint32_t a, uint32_t b) const {
    if (!graph_enabled_ || static_cast<std::size_t>(a) >= n_ ||
        static_cast<std::size_t>(b) >= n_) {
      return 0.0;
    }
    return std::min(1.0, std::max(row_affinity(a, b), row_affinity(b, a)));
  }

 private:
  double row_affinity(uint32_t from, uint32_t to) const {
    uint64_t begin = (*graph_offsets_)[from];
    uint64_t end = (*graph_offsets_)[static_cast<std::size_t>(from) + 1];
    if (end <= begin || end > graph_neighbors_->size()) {
      return 0.0;
    }
    auto begin_it = graph_neighbors_->begin() + static_cast<std::ptrdiff_t>(begin);
    auto end_it = graph_neighbors_->begin() + static_cast<std::ptrdiff_t>(end);
    auto found = std::lower_bound(begin_it, end_it, to);
    if (found == end_it || *found != to) {
      return 0.0;
    }
    std::size_t index = static_cast<std::size_t>(found - graph_neighbors_->begin());
    return std::max(0.0, static_cast<double>((*graph_weights_)[index]));
  }

  const std::vector<float>& centroids_;
  uint32_t dim_ = 0;
  std::size_t n_ = 0;
  bool graph_enabled_ = false;
  const std::vector<uint64_t>* graph_offsets_ = nullptr;
  const std::vector<uint32_t>* graph_neighbors_ = nullptr;
  const std::vector<float>* graph_weights_ = nullptr;
  std::vector<double> table_;
};

void reweight_edges(std::vector<std::vector<Edge>>* adj,
                    const CentroidDistanceLookup& distances) {
  if (adj == nullptr) {
    return;
  }
  for (uint32_t from = 0; from < adj->size(); ++from) {
    for (auto& edge : (*adj)[from]) {
      edge.weight = distances(from, edge.to);
    }
  }
}

std::vector<double> raw_edge_caps(const std::vector<std::vector<Edge>>& adj) {
  std::vector<double> caps(adj.size(), 0.0);
  for (std::size_t i = 0; i < adj.size(); ++i) {
    for (const auto& edge : adj[i]) {
      caps[i] = std::max(caps[i], edge.weight);
    }
  }
  return caps;
}

void add_graph_order_edges(std::vector<std::vector<Edge>>* adj,
                           const CentroidDistanceLookup& distances,
                           const std::vector<uint64_t>& graph_offsets,
                           const std::vector<uint32_t>& graph_neighbors,
                           const std::vector<double>& raw_caps) {
  if (adj == nullptr || graph_offsets.size() != adj->size() + 1) {
    return;
  }
  constexpr double kGraphEdgeRawCapMultiplier = 1.25;
  for (uint32_t from = 0; from < adj->size(); ++from) {
    uint64_t begin = graph_offsets[from];
    uint64_t end = graph_offsets[static_cast<std::size_t>(from) + 1];
    if (end <= begin || end > graph_neighbors.size()) {
      continue;
    }
    auto& row = (*adj)[from];
    row.reserve(row.size() + static_cast<std::size_t>(end - begin));
    for (uint64_t pos = begin; pos < end; ++pos) {
      uint32_t to = graph_neighbors[static_cast<std::size_t>(pos)];
      if (to == from || static_cast<std::size_t>(to) >= adj->size()) {
        continue;
      }
      if (from < raw_caps.size() && raw_caps[from] > 0.0) {
        double raw = distances.raw_distance(from, to);
        if (raw > raw_caps[from] * kGraphEdgeRawCapMultiplier) {
          continue;
        }
      }
      row.push_back({to, distances(from, to)});
    }
  }
}

void deduplicate_edges(std::vector<std::vector<Edge>>* adj) {
  if (adj == nullptr) {
    return;
  }
  for (auto& row : *adj) {
    std::sort(row.begin(), row.end(), [](const Edge& a, const Edge& b) {
      if (a.to != b.to) {
        return a.to < b.to;
      }
      return a.weight < b.weight;
    });
    std::size_t write = 0;
    for (std::size_t read = 0; read < row.size(); ++read) {
      if (write == 0 || row[read].to != row[write - 1].to) {
        row[write++] = row[read];
      }
    }
    row.resize(write);
    std::sort(row.begin(), row.end(), [](const Edge& a, const Edge& b) {
      if (a.weight != b.weight) {
        return a.weight < b.weight;
      }
      return a.to < b.to;
    });
  }
}

std::vector<uint32_t> cut_cycle_order(const std::vector<uint32_t>& cycle,
                                      const CentroidDistanceLookup& distances) {
  std::size_t n = cycle.size();
  if (n == 0) {
    return {};
  }
  double max_edge = -1.0;
  std::size_t cut_idx = 0;
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t j = (i + 1) % n;
    double d = distances(cycle[i], cycle[j]);
    if (d > max_edge) {
      max_edge = d;
      cut_idx = j;
    }
  }
  std::vector<uint32_t> order;
  order.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    order.push_back(cycle[(cut_idx + i) % n]);
  }
  return order;
}

bool two_opt_refine_cycle(std::vector<uint32_t>& cycle,
                          const CentroidDistanceLookup& distances,
                          uint32_t max_iterations) {
  std::size_t n = cycle.size();
  if (n < 4 || max_iterations == 0) {
    return false;
  }

  auto edge_dist = [&](std::size_t i, std::size_t j) {
    return distances(cycle[i], cycle[j]);
  };

  constexpr double kImproveEpsilon = 1e-12;
  bool any_improved = false;

  for (uint32_t iter = 0; iter < max_iterations; ++iter) {
    double best_gain = 0.0;
    std::size_t best_i = 0;
    std::size_t best_j = 0;

    for (std::size_t i = 0; i < n; ++i) {
      std::size_t i2 = (i + 1) % n;
      std::size_t j_start = i + 2;
      if (j_start >= n) {
        continue;
      }
      for (std::size_t j = j_start; j < n; ++j) {
        std::size_t j2 = (j + 1) % n;
        // Skip adjacent edges and the wrap-around pair that is also adjacent in the cycle.
        if (i == 0 && j2 == 0) {
          continue;
        }
        double old_cost = edge_dist(i, i2) + edge_dist(j, j2);
        double new_cost = edge_dist(i, j) + edge_dist(i2, j2);
        double gain = old_cost - new_cost;
        if (gain > best_gain + kImproveEpsilon) {
          best_gain = gain;
          best_i = i;
          best_j = j;
        }
      }
    }

    if (best_gain <= kImproveEpsilon) {
      break;
    }

    std::size_t left = best_i + 1;
    std::size_t right = best_j;
    if (left < n && right < n && left <= right) {
      std::reverse(cycle.begin() + static_cast<std::ptrdiff_t>(left),
                   cycle.begin() + static_cast<std::ptrdiff_t>(right + 1));
      any_improved = true;
    } else {
      break;
    }
  }

  return any_improved;
}

std::vector<uint32_t> compute_centroid_order(const std::vector<float>& centroids,
                                             uint32_t dim,
                                             uint32_t knn,
                                             bool enable_2opt,
                                             uint32_t two_opt_iterations,
                                             bool enable_graph_order,
                                             const std::vector<uint64_t>& graph_offsets,
                                             const std::vector<uint32_t>& graph_neighbors,
                                             const std::vector<float>& graph_weights) {
  std::size_t k = centroids.empty() ? 0 : centroids.size() / dim;
  if (k == 0) {
    return {};
  }

  CentroidDistanceLookup distances(centroids,
                                   dim,
                                   enable_2opt && two_opt_iterations > 0,
                                   enable_graph_order,
                                   &graph_offsets,
                                   &graph_neighbors,
                                   &graph_weights);
  auto adj = build_knn_edges(centroids, dim, knn);
  auto graph_raw_caps = raw_edge_caps(adj);
  reweight_edges(&adj, distances);
  if (enable_graph_order && graph_offsets.size() == k + 1) {
    add_graph_order_edges(&adj, distances, graph_offsets, graph_neighbors, graph_raw_caps);
  }
  deduplicate_edges(&adj);
  auto mst = build_mst(adj, centroids, dim);
  auto cycle = dfs_order_from_mst(static_cast<uint32_t>(k), mst);
  if (enable_2opt) {
    two_opt_refine_cycle(cycle, distances, two_opt_iterations);
  }
  return cut_cycle_order(cycle, distances);
}

} // namespace

namespace training_internal {

std::vector<uint32_t> build_vortex_centroid_order(const TrainOptions& options,
                                                  const VortexTrainingBase& base,
                                                  VortexTrainingProfile* profile) {
  validate_training_base(options, base);

  auto order_start = TrainingClock::now();
  std::vector<uint32_t> order = compute_centroid_order(base.centroids,
                                                       base.dim,
                                                       options.centroid_knn,
                                                       options.enable_2opt,
                                                       options.two_opt_iterations,
                                                       options.enable_graph_centroid_order,
                                                       base.centroid_graph_offsets,
                                                       base.centroid_graph_neighbors,
                                                       base.centroid_graph_weights);
  if (profile != nullptr) {
    double order_ms = elapsed_ms(order_start);
    profile->centroid_order_ms += order_ms;
    profile->order_build_ms += order_ms;
  }
  if (order.size() != options.K) {
    throw std::runtime_error("Centroid ordering failed");
  }
  return order;
}

} // namespace training_internal
} // namespace vortex
