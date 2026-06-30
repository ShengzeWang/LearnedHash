#include "lead/hasher.h"

#include "rm_model/detail/cli_args.h"
#include "rm_model/logging.h"

#include "rm_model/benchmark.h"

#include "rm_model/data_loader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using rm_model::DataType;
using rm_model::MappedDataset;
using rm_model::load_data;
using rm_model::detail::has_help_flag;
using rm_model::detail::is_flag;
using rm_model::detail::parse_double_arg;
using rm_model::detail::parse_size_arg;
using rm_model::detail::parse_u32_arg;
using rm_model::detail::parse_u64_arg;
using rm_model::detail::require_value;

namespace {

enum class OutputFormat {
  Hex,
  Raw,
};

using SteadyClock = std::chrono::steady_clock;
using rm_model::TimingStats;
using rm_model::format_duration_ns;

std::string format_hex_u64(uint64_t value) {
  std::ostringstream oss;
  oss << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << value;
  return oss.str();
}

void write_csv_header(std::ostream& out, uint32_t space_bits, bool include_pred) {
  if (space_bits <= 64) {
    if (include_pred) {
      out << "key_dec,rm_pred,hash_dec,hash_hex\n";
    } else {
      out << "key_dec,hash_dec,hash_hex\n";
    }
  } else {
    if (include_pred) {
      out << "key_dec,rm_pred,hash_msb64_hex,hash_hex\n";
    } else {
      out << "key_dec,hash_msb64_hex,hash_hex\n";
    }
  }
}

template <typename KeyT>
void write_csv_row(std::ostream& out, const KeyT& key, double pred,
                   const lead::HashOutput& hash, uint32_t space_bits, bool include_pred) {
  if (space_bits <= 64) {
    out << key;
    if (include_pred) {
      out << "," << pred;
    }
    out << "," << hash.to_u64_msb() << ",0x" << hash.to_hex() << '\n';
  } else {
    out << key;
    if (include_pred) {
      out << "," << pred;
    }
    out << ",0x" << format_hex_u64(hash.to_u64_msb())
        << ",0x" << hash.to_hex() << '\n';
  }
}

struct Options {
  std::filesystem::path model_dir;
  std::optional<std::filesystem::path> dataset;
  std::optional<std::string> key_value;
  std::optional<std::string> key_type;
  std::optional<std::filesystem::path> output;
  std::size_t limit = 0;
  OutputFormat format = OutputFormat::Hex;
  bool csv = false;
  bool csv_pred = false;
  bool bench = false;
};

void print_usage(std::ostream& out) {
  out << "Usage: lead_hasher --model-dir <dir> (--dataset <path> | --key <value>) [options]\n";
  out << "Options:\n";
  out << "  --key-type <u64|u32|f64>       Key type override for single-key mode\n";
  out << "  --out <file>                   Output file (default: stdout)\n";
  out << "  --limit <n>                    Max number of records for dataset mode\n";
  out << "  --format <hex|raw>             Output format\n";
  out << "  --csv                          Emit CSV (dataset mode defaults to CSV for hex format)\n";
  out << "  --csv-pred                     Emit CSV with rm_model prediction column\n";
  out << "  --bench                        Print latency stats\n";
  out << "  --help, -h                     Show this help\n";
}

Options parse_args(int argc, char** argv) {
  Options opts;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (!is_flag(arg)) {
      throw std::runtime_error("Unexpected positional argument: " + arg);
    }

    if (arg == "--model-dir") {
      opts.model_dir = require_value(i, argc, argv, arg);
    } else if (arg == "--dataset") {
      opts.dataset = require_value(i, argc, argv, arg);
    } else if (arg == "--key") {
      opts.key_value = require_value(i, argc, argv, arg);
    } else if (arg == "--key-type") {
      opts.key_type = require_value(i, argc, argv, arg);
    } else if (arg == "--out") {
      opts.output = require_value(i, argc, argv, arg);
    } else if (arg == "--limit") {
      opts.limit = parse_size_arg(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--format") {
      std::string fmt = require_value(i, argc, argv, arg);
      if (fmt == "hex") {
        opts.format = OutputFormat::Hex;
      } else if (fmt == "raw") {
        opts.format = OutputFormat::Raw;
      } else {
        throw std::runtime_error("Unknown format: " + fmt);
      }
    } else if (arg == "--csv") {
      opts.csv = true;
    } else if (arg == "--csv-pred") {
      opts.csv = true;
      opts.csv_pred = true;
    } else if (arg == "--bench") {
      opts.bench = true;
    } else if (arg == "--width") {
      throw std::runtime_error("--width is configured at hashgen via --space-bits");
    } else if (arg == "--help" || arg == "-h") {
      throw std::runtime_error("Usage: lead_hasher --model-dir <dir> (--dataset <path> | --key <value>) [--csv] [--csv-pred] [options]");
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  if (opts.model_dir.empty()) {
    throw std::runtime_error("--model-dir is required");
  }
  if (!opts.dataset && !opts.key_value) {
    throw std::runtime_error("Must provide --dataset or --key");
  }
  if (opts.dataset && opts.key_value) {
    throw std::runtime_error("Specify only one of --dataset or --key");
  }
  if (opts.dataset.has_value() && opts.format != OutputFormat::Raw) {
    opts.csv = true;
  }
  return opts;
}

lead::RmKeyType parse_key_type_or_throw(const std::string& value) {
  if (value == "u64") return lead::RmKeyType::U64;
  if (value == "u32") return lead::RmKeyType::U32;
  if (value == "f64") return lead::RmKeyType::F64;
  throw std::runtime_error("Unknown key type: " + value);
}

DataType data_type_from_key(lead::RmKeyType kt) {
  if (kt == lead::RmKeyType::U32) return DataType::UINT32;
  if (kt == lead::RmKeyType::F64) return DataType::FLOAT64;
  return DataType::UINT64;
}

void write_raw(std::ostream& out, const lead::HashOutput& hash) {
  std::array<uint8_t, 32> bytes = hash.bytes_be();
  size_t byte_len = (hash.bits + 7) / 8;
  if (byte_len == 0) {
    return;
  }
  size_t start = bytes.size() - byte_len;
  if (hash.bits % 8 != 0) {
    uint8_t mask = static_cast<uint8_t>((1u << (hash.bits % 8)) - 1u);
    bytes[start] &= mask;
  }
  out.write(reinterpret_cast<const char*>(bytes.data() + start), static_cast<std::streamsize>(byte_len));
}

} // namespace

int main(int argc, char** argv) {
  if (has_help_flag(argc, argv)) {
    print_usage(std::cout);
    return 0;
  }

  try {
    rm_model::init_logging();

    Options opts = parse_args(argc, argv);

    lead::Hasher hasher;
    hasher.load(opts.model_dir);

    if ((opts.csv || opts.csv_pred) && opts.format == OutputFormat::Raw) {
      throw std::runtime_error("--csv/--csv-pred requires --format hex");
    }

    const lead::LeadManifest& manifest = hasher.manifest();
    RM_MODEL_LOG_INFO("LEAD model: name=" << manifest.name
                                          << ", key_type=" << lead::key_type_name(manifest.key_type)
                                          << ", model_spec=" << manifest.model_spec
                                          << ", bf=" << manifest.branching_factor
                                          << ", size=" << manifest.model_size_bytes << " bytes");
    RM_MODEL_LOG_INFO("Mapping: max_index=" << manifest.mapping.max_index
                                            << ", space_bits=" << manifest.mapping.space_bits
                                            << ", scale_factor=" << manifest.mapping.scale_factor);
    if (hasher.calibration_enabled()) {
      RM_MODEL_LOG_INFO("Calibration: enabled=true, raw_predict="
                        << (hasher.raw_predict_available() ? "true" : "false")
                        << ", pred_min=" << hasher.calibration().min_pred
                        << ", pred_max=" << hasher.calibration().max_pred
                        << ", scale=" << hasher.calibration_scale());
    } else {
      RM_MODEL_LOG_INFO("Calibration: enabled=false, raw_predict="
                        << (hasher.raw_predict_available() ? "true" : "false"));
    }

    const char* output_label = opts.format == OutputFormat::Raw
        ? "raw"
        : (opts.csv ? "csv" : "hex");
    RM_MODEL_LOG_INFO("Output: format=" << output_label
                                        << ", space_bits=" << hasher.space_bits()
                                        << ", csv=" << (opts.csv ? "true" : "false")
                                        << ", csv_pred=" << (opts.csv_pred ? "true" : "false"));

    uint32_t space_bits = hasher.space_bits();
    lead::RmKeyType key_type = hasher.key_type();
    if (opts.key_type.has_value()) {
      key_type = parse_key_type_or_throw(*opts.key_type);
      if (key_type != hasher.key_type()) {
        throw std::runtime_error("Override key type does not match model key type");
      }
    }

    bool timing_enabled = opts.bench;
    TimingStats timing;

    if (opts.key_value.has_value()) {
      lead::HashOutput hash;
      double pred = 0.0;
      if (opts.csv) {
        std::cout << std::setprecision(17);
        write_csv_header(std::cout, space_bits, opts.csv_pred);
      }
      if (key_type == lead::RmKeyType::U64) {
        uint64_t key = parse_u64_arg(*opts.key_value, "--key");
        if (timing_enabled) {
          auto start = SteadyClock::now();
          if (opts.csv_pred) {
            hash = hasher.hash_with_pred(key, &pred);
          } else {
            hash = hasher.hash(key);
          }
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
        } else {
          if (opts.csv_pred) {
            hash = hasher.hash_with_pred(key, &pred);
          } else {
            hash = hasher.hash(key);
          }
        }
        if (opts.csv) {
          write_csv_row(std::cout, key, pred, hash, space_bits, opts.csv_pred);
        }
      } else if (key_type == lead::RmKeyType::U32) {
        uint32_t key = parse_u32_arg(*opts.key_value, "--key");
        if (timing_enabled) {
          auto start = SteadyClock::now();
          if (opts.csv_pred) {
            hash = hasher.hash_with_pred(key, &pred);
          } else {
            hash = hasher.hash(key);
          }
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
        } else {
          if (opts.csv_pred) {
            hash = hasher.hash_with_pred(key, &pred);
          } else {
            hash = hasher.hash(key);
          }
        }
        if (opts.csv) {
          write_csv_row(std::cout, key, pred, hash, space_bits, opts.csv_pred);
        }
      } else {
        double key = parse_double_arg(*opts.key_value, "--key");
        if (timing_enabled) {
          auto start = SteadyClock::now();
          if (opts.csv_pred) {
            hash = hasher.hash_with_pred(key, &pred);
          } else {
            hash = hasher.hash(key);
          }
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
        } else {
          if (opts.csv_pred) {
            hash = hasher.hash_with_pred(key, &pred);
          } else {
            hash = hasher.hash(key);
          }
        }
        if (opts.csv) {
          write_csv_row(std::cout, key, pred, hash, space_bits, opts.csv_pred);
        }
      }

      if (opts.csv) {
        // CSV already emitted.
      } else if (opts.format == OutputFormat::Hex) {
        std::cout << hash.to_hex() << '\n';
      } else {
        write_raw(std::cout, hash);
      }
      if (timing_enabled && !timing.empty()) {
        RM_MODEL_LOG_INFO("Hash time (model+mapping): total "
                          << format_duration_ns(timing.total_ns())
                          << ", avg " << format_duration_ns(timing.avg_ns())
                          << " per key, min " << format_duration_ns(timing.min_ns())
                          << ", p50 " << format_duration_ns(timing.p50_ns())
                          << ", p95 " << format_duration_ns(timing.p95_ns())
                          << ", p99 " << format_duration_ns(timing.p99_ns())
                          << ", max " << format_duration_ns(timing.max_ns()));
      }
      return 0;
    }

    DataType dt = data_type_from_key(key_type);
    auto load_result = load_data(opts.dataset->string(), dt);
    MappedDataset dataset = std::move(load_result.second);
    std::size_t num_rows = load_result.first;
    RM_MODEL_LOG_INFO("Loaded dataset (" << num_rows << " rows)");

    std::optional<std::ofstream> out_file;
    std::ostream* out = &std::cout;
    if (opts.output.has_value()) {
      if (opts.format == OutputFormat::Raw) {
        out_file.emplace(*opts.output, std::ios::binary);
      } else {
        out_file.emplace(*opts.output);
      }
      if (!out_file->good()) {
        throw std::runtime_error("Unable to open output file");
      }
      out = &*out_file;
    }

    std::size_t limit = opts.limit == 0 ? num_rows : std::min(opts.limit, num_rows);

    auto start = SteadyClock::now();

    if (opts.csv) {
      (*out) << std::setprecision(17);
      write_csv_header(*out, space_bits, opts.csv_pred);
    }

    if (timing_enabled) {
      dataset.visit([&](auto& typed_data) {
        using KeyT = typename std::decay_t<decltype(typed_data)>::value_type::first_type;
        for (std::size_t i = 0; i < limit; ++i) {
          auto item = typed_data.get(i);
          lead::HashOutput hash;
          double pred = 0.0;
          auto start = SteadyClock::now();
          if constexpr (std::is_same_v<KeyT, uint64_t>) {
            if (opts.csv_pred) {
              hash = hasher.hash_with_pred(static_cast<uint64_t>(item.first), &pred);
            } else {
              hash = hasher.hash(static_cast<uint64_t>(item.first));
            }
          } else if constexpr (std::is_same_v<KeyT, uint32_t>) {
            if (opts.csv_pred) {
              hash = hasher.hash_with_pred(static_cast<uint32_t>(item.first), &pred);
            } else {
              hash = hasher.hash(static_cast<uint32_t>(item.first));
            }
          } else {
            if (opts.csv_pred) {
              hash = hasher.hash_with_pred(static_cast<double>(item.first), &pred);
            } else {
              hash = hasher.hash(static_cast<double>(item.first));
            }
          }
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));

          if (out) {
            if (opts.csv) {
              write_csv_row(*out, item.first, pred, hash, space_bits, opts.csv_pred);
            } else if (opts.format == OutputFormat::Hex) {
              (*out) << hash.to_hex() << '\n';
            } else {
              write_raw(*out, hash);
            }
          }
        }
      });
    } else {
      dataset.visit([&](auto& typed_data) {
        using KeyT = typename std::decay_t<decltype(typed_data)>::value_type::first_type;
        for (std::size_t i = 0; i < limit; ++i) {
          auto item = typed_data.get(i);
          lead::HashOutput hash;
          double pred = 0.0;
          if constexpr (std::is_same_v<KeyT, uint64_t>) {
            if (opts.csv_pred) {
              hash = hasher.hash_with_pred(static_cast<uint64_t>(item.first), &pred);
            } else {
              hash = hasher.hash(static_cast<uint64_t>(item.first));
            }
          } else if constexpr (std::is_same_v<KeyT, uint32_t>) {
            if (opts.csv_pred) {
              hash = hasher.hash_with_pred(static_cast<uint32_t>(item.first), &pred);
            } else {
              hash = hasher.hash(static_cast<uint32_t>(item.first));
            }
          } else {
            if (opts.csv_pred) {
              hash = hasher.hash_with_pred(static_cast<double>(item.first), &pred);
            } else {
              hash = hasher.hash(static_cast<double>(item.first));
            }
          }

          if (out) {
            if (opts.csv) {
              write_csv_row(*out, item.first, pred, hash, space_bits, opts.csv_pred);
            } else if (opts.format == OutputFormat::Hex) {
              (*out) << hash.to_hex() << '\n';
            } else {
              write_raw(*out, hash);
            }
          }
        }
      });
    }

    auto end = SteadyClock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    if (opts.bench || !opts.output.has_value()) {
      double throughput = elapsed > 0.0 ? static_cast<double>(limit) / elapsed : 0.0;
      RM_MODEL_LOG_INFO("Hashed " << limit << " keys in " << elapsed << " s (" << throughput << " keys/s)");
    }
    if (timing_enabled && !timing.empty()) {
      RM_MODEL_LOG_INFO("Hash time (model+mapping): total "
                        << format_duration_ns(timing.total_ns())
                        << ", avg " << format_duration_ns(timing.avg_ns())
                        << " per key, min " << format_duration_ns(timing.min_ns())
                        << ", p50 " << format_duration_ns(timing.p50_ns())
                        << ", p95 " << format_duration_ns(timing.p95_ns())
                        << ", p99 " << format_duration_ns(timing.p99_ns())
                        << ", max " << format_duration_ns(timing.max_ns()));
    }

  } catch (const std::exception& ex) {
    RM_MODEL_LOG_ERROR(ex.what());
    return 1;
  }

  return 0;
}
