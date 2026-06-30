#include "vortex_v1/nsw_csr.h"

#include "vortex_v1/detail/binary_io.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace vortex {

namespace {

constexpr char kMagic[8] = {'V','T','X','N','S','W','1','\0'};

} // namespace

void NswCsr::write(const std::filesystem::path& path) const {
  if (offsets.size() != node_ids.size() + 1) {
    throw std::runtime_error("CSR offsets must be node_count + 1");
  }
  if (!offsets.empty() && offsets.back() != neighbors.size()) {
    throw std::runtime_error("CSR offsets do not match neighbor count");
  }

  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Unable to open CSR file for writing: " + path.string());
  }

  NswCsrHeader header;
  std::memcpy(header.magic, kMagic, sizeof(kMagic));
  header.dim = dim;
  header.metric = static_cast<uint32_t>(metric);
  header.layer = layer;
  header.node_count = static_cast<uint64_t>(node_ids.size());
  header.edge_count = static_cast<uint64_t>(neighbors.size());

  detail::write_bytes(out, &header, sizeof(header), "Failed to write NSW CSR file");
  detail::write_vector(out, node_ids, "Failed to write NSW CSR file");
  detail::write_vector(out, offsets, "Failed to write NSW CSR file");
  detail::write_vector(out, neighbors, "Failed to write NSW CSR file");
}

NswCsr NswCsr::read(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Unable to open CSR file: " + path.string());
  }

  NswCsrHeader header{};
  detail::read_bytes(in, &header, sizeof(header), "Failed to read NSW CSR file");
  if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("Invalid NSW CSR magic");
  }
  if (header.version != 1) {
    throw std::runtime_error("Unsupported NSW CSR version");
  }

  NswCsr csr;
  csr.dim = header.dim;
  csr.metric = static_cast<Metric>(header.metric);
  csr.layer = header.layer;

  detail::read_vector(in,
                      csr.node_ids,
                      static_cast<std::size_t>(header.node_count),
                      "Failed to read NSW CSR file");
  detail::read_vector(in,
                      csr.offsets,
                      static_cast<std::size_t>(header.node_count + 1),
                      "Failed to read NSW CSR file");
  detail::read_vector(in,
                      csr.neighbors,
                      static_cast<std::size_t>(header.edge_count),
                      "Failed to read NSW CSR file");

  if (csr.offsets.size() != csr.node_ids.size() + 1) {
    throw std::runtime_error("CSR offsets must be node_count + 1");
  }
  if (!csr.offsets.empty() && csr.offsets.back() != csr.neighbors.size()) {
    throw std::runtime_error("CSR offsets do not match neighbor count");
  }

  return csr;
}

} // namespace vortex
