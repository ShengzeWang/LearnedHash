#ifndef RM_MODEL_BENCHMARK_H
#define RM_MODEL_BENCHMARK_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace rm_model {

inline std::string format_duration_ns(uint64_t ns) {
  if (ns < 1000ULL) {
    return std::to_string(ns) + " ns";
  }
  if (ns < 1000ULL * 1000ULL) {
    double us = static_cast<double>(ns) / 1000.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << us << " us";
    return oss.str();
  }
  double ms = static_cast<double>(ns) / 1000000.0;
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << ms << " ms";
  return oss.str();
}

class P2Quantile {
 public:
  explicit P2Quantile(double quantile) : q_(quantile) {}

  void add(double x) {
    if (count_ < kMarkerCount) {
      init_.push_back(x);
      count_ += 1;
      if (count_ == kMarkerCount) {
        initialize();
      }
      return;
    }

    count_ += 1;
    int k = 0;
    if (x < h_[0]) {
      h_[0] = x;
      k = 0;
    } else if (x < h_[1]) {
      k = 0;
    } else if (x < h_[2]) {
      k = 1;
    } else if (x < h_[3]) {
      k = 2;
    } else if (x <= h_[4]) {
      k = 3;
    } else {
      h_[4] = x;
      k = 3;
    }

    for (int i = k + 1; i < kMarkerCount; ++i) {
      n_[i] += 1;
    }

    np_[0] += 0.0;
    np_[1] += q_ / 2.0;
    np_[2] += q_;
    np_[3] += (1.0 + q_) / 2.0;
    np_[4] += 1.0;

    for (int i = 1; i <= 3; ++i) {
      double d = np_[i] - static_cast<double>(n_[i]);
      if ((d >= 1.0 && n_[i + 1] - n_[i] > 1) ||
          (d <= -1.0 && n_[i - 1] - n_[i] < -1)) {
        int sign = d > 0.0 ? 1 : -1;
        double q_new = parabolic(i, sign);
        if (q_new > h_[i - 1] && q_new < h_[i + 1]) {
          h_[i] = q_new;
        } else {
          h_[i] = linear(i, sign);
        }
        n_[i] += sign;
      }
    }
  }

  bool empty() const { return count_ == 0; }

  double value() const {
    if (count_ == 0) {
      return 0.0;
    }
    if (count_ < kMarkerCount) {
      std::vector<double> sorted = init_;
      std::sort(sorted.begin(), sorted.end());
      std::size_t idx = static_cast<std::size_t>(q_ * static_cast<double>(sorted.size() - 1));
      return sorted[idx];
    }
    return h_[2];
  }

 private:
  static constexpr int kMarkerCount = 5;

  void initialize() {
    std::sort(init_.begin(), init_.end());
    for (int i = 0; i < kMarkerCount; ++i) {
      h_[i] = init_[static_cast<std::size_t>(i)];
      n_[i] = i + 1;
    }

    np_[0] = 1.0;
    np_[1] = 1.0 + 2.0 * q_;
    np_[2] = 1.0 + 4.0 * q_;
    np_[3] = 3.0 + 2.0 * q_;
    np_[4] = 5.0;
    init_.clear();
  }

  double parabolic(int i, int d) const {
    double n_i = static_cast<double>(n_[i]);
    double n_i1 = static_cast<double>(n_[i + 1]);
    double n_i_1 = static_cast<double>(n_[i - 1]);
    double d_f = static_cast<double>(d);
    double a = (n_i - n_i_1 + d_f) * (h_[i + 1] - h_[i]) / (n_i1 - n_i);
    double b = (n_i1 - n_i - d_f) * (h_[i] - h_[i - 1]) / (n_i - n_i_1);
    return h_[i] + d_f * (a + b) / (n_i1 - n_i_1);
  }

  double linear(int i, int d) const {
    int idx = i + d;
    double n_i = static_cast<double>(n_[i]);
    double n_idx = static_cast<double>(n_[idx]);
    return h_[i] + static_cast<double>(d) * (h_[idx] - h_[i]) / (n_idx - n_i);
  }

  double q_;
  std::size_t count_ = 0;
  std::vector<double> init_;
  std::array<double, kMarkerCount> h_{};
  std::array<double, kMarkerCount> np_{};
  std::array<int, kMarkerCount> n_{};
};

class TimingStats {
 public:
  void add(uint64_t ns) {
    if (count_ == 0) {
      min_ = ns;
      max_ = ns;
    } else {
      min_ = std::min(min_, ns);
      max_ = std::max(max_, ns);
    }
    total_ += ns;
    count_ += 1;
    q50_.add(static_cast<double>(ns));
    q95_.add(static_cast<double>(ns));
    q99_.add(static_cast<double>(ns));
  }

  bool empty() const { return count_ == 0; }

  uint64_t total_ns() const { return total_; }
  uint64_t avg_ns() const { return count_ == 0 ? 0 : total_ / count_; }
  uint64_t min_ns() const { return min_; }
  uint64_t max_ns() const { return max_; }
  uint64_t p50_ns() const { return static_cast<uint64_t>(q50_.value()); }
  uint64_t p95_ns() const { return static_cast<uint64_t>(q95_.value()); }
  uint64_t p99_ns() const { return static_cast<uint64_t>(q99_.value()); }

 private:
  uint64_t total_ = 0;
  uint64_t min_ = 0;
  uint64_t max_ = 0;
  std::size_t count_ = 0;
  P2Quantile q50_{0.50};
  P2Quantile q95_{0.95};
  P2Quantile q99_{0.99};
};

} // namespace rm_model

#endif // RM_MODEL_BENCHMARK_H
