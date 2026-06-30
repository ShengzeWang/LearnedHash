#ifndef VORTEX_V1_TYPES_H
#define VORTEX_V1_TYPES_H

#include <cstdint>
#include <stdexcept>
#include <string>

namespace vortex {

enum class Metric : uint32_t {
  L2 = 1,
};

inline const char* metric_name(Metric metric) {
  switch (metric) {
    case Metric::L2:
      return "L2";
  }
  return "L2";
}

inline Metric parse_metric(const std::string& value) {
  if (value == "L2" || value == "l2") {
    return Metric::L2;
  }
  throw std::runtime_error("Unsupported metric: " + value);
}

} // namespace vortex

#endif // VORTEX_V1_TYPES_H
