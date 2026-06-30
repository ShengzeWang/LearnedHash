#include "lead/trainer.h"

#include "lead/mapping.h"
#include "lead/manifest.h"

#include "rm_model/codegen.h"
#include "rm_model/learned_model_selector.h"
#include "rm_model/logging.h"
#include "rm_model/parallel.h"
#include "rm_model/train.h"

#include "rm_model/data_loader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

using rm_model::DataType;
using rm_model::MappedDataset;
using rm_model::load_data;

namespace lead {

namespace {

std::string sanitize_component(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  bool last_sep = false;
  for (unsigned char ch : value) {
    if (std::isalnum(ch) || ch == '-' || ch == '_') {
      out.push_back(static_cast<char>(ch));
      last_sep = false;
      continue;
    }
    if (!last_sep) {
      out.push_back('_');
      last_sep = true;
    }
  }
  if (out.empty()) {
    out = "model";
  }
  return out;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    throw std::runtime_error("Unable to read file: " + path.string());
  }
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
}

void write_file(const std::filesystem::path& path, const std::string& data) {
  std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Unable to write file: " + path.string());
  }
  out << data;
}

std::size_t find_matching_brace(const std::string& text, std::size_t open_pos) {
  std::size_t depth = 0;
  for (std::size_t i = open_pos; i < text.size(); ++i) {
    if (text[i] == '{') {
      depth += 1;
    } else if (text[i] == '}') {
      if (depth == 0) {
        throw std::runtime_error("Unbalanced braces in generated source");
      }
      depth -= 1;
      if (depth == 0) {
        return i;
      }
    }
  }
  throw std::runtime_error("Unbalanced braces in generated source");
}

bool is_ident_char(char ch) {
  return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
}

std::string trim_copy(const std::string& value) {
  std::size_t start = 0;
  while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
    start += 1;
  }
  std::size_t end = value.size();
  while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    end -= 1;
  }
  return value.substr(start, end - start);
}

struct FunctionBlock {
  std::size_t start = 0;
  std::size_t end = 0;
  std::string text;
};

bool try_extract_function_block(const std::string& text,
                                const std::string& name,
                                FunctionBlock* out) {
  std::size_t pos = 0;
  while (true) {
    pos = text.find(name, pos);
    if (pos == std::string::npos) {
      return false;
    }
    if (pos > 0 && is_ident_char(text[pos - 1])) {
      pos += name.size();
      continue;
    }
    std::size_t after = pos + name.size();
    if (after < text.size() && is_ident_char(text[after])) {
      pos = after;
      continue;
    }
    while (after < text.size() && std::isspace(static_cast<unsigned char>(text[after]))) {
      after += 1;
    }
    if (after >= text.size() || text[after] != '(') {
      pos = after;
      continue;
    }

    std::size_t brace_pos = text.find('{', after);
    if (brace_pos == std::string::npos) {
      throw std::runtime_error("Missing function body for: " + name);
    }
    std::size_t semi_pos = text.find(';', after);
    if (semi_pos != std::string::npos && semi_pos < brace_pos) {
      pos = after;
      continue;
    }

    std::size_t line_start = text.rfind('\n', pos);
    line_start = (line_start == std::string::npos) ? 0 : (line_start + 1);
    std::size_t end_pos = find_matching_brace(text, brace_pos);
    if (out) {
      out->start = line_start;
      out->end = end_pos;
      out->text = text.substr(line_start, end_pos - line_start + 1);
    }
    return true;
  }
}

FunctionBlock extract_function_block(const std::string& text, const std::string& name) {
  FunctionBlock block;
  if (!try_extract_function_block(text, name, &block)) {
    throw std::runtime_error("Missing function: " + name);
  }
  return block;
}

std::size_t find_matching_paren(const std::string& text, std::size_t open_pos) {
  std::size_t depth = 0;
  for (std::size_t i = open_pos; i < text.size(); ++i) {
    if (text[i] == '(') {
      depth += 1;
    } else if (text[i] == ')') {
      if (depth == 0) {
        throw std::runtime_error("Unbalanced parentheses in generated source");
      }
      depth -= 1;
      if (depth == 0) {
        return i;
      }
    }
  }
  throw std::runtime_error("Unbalanced parentheses in generated source");
}

std::string extract_dclamp_arg(const std::string& text) {
  std::size_t dclamp_pos = text.find("DCLAMP");
  if (dclamp_pos == std::string::npos) {
    throw std::runtime_error("predict() does not use DCLAMP");
  }
  std::size_t open = text.find('(', dclamp_pos);
  if (open == std::string::npos) {
    throw std::runtime_error("Malformed DCLAMP call");
  }
  std::size_t close = find_matching_paren(text, open);
  std::string inner = text.substr(open + 1, close - open - 1);

  std::size_t depth = 0;
  std::size_t comma = std::string::npos;
  for (std::size_t i = 0; i < inner.size(); ++i) {
    char ch = inner[i];
    if (ch == '(') {
      depth += 1;
    } else if (ch == ')') {
      if (depth == 0) {
        throw std::runtime_error("Unbalanced parentheses in DCLAMP args");
      }
      depth -= 1;
    } else if (ch == ',' && depth == 0) {
      comma = i;
      break;
    }
  }
  if (comma == std::string::npos) {
    throw std::runtime_error("Unexpected DCLAMP expression in predict()");
  }
  return trim_copy(inner.substr(0, comma));
}

void replace_return_with_raw(std::string& fn, const std::string& expr) {
  std::size_t dclamp_pos = fn.find("DCLAMP");
  if (dclamp_pos == std::string::npos) {
    throw std::runtime_error("Missing DCLAMP call in predict()");
  }
  std::size_t open = fn.find('(', dclamp_pos);
  if (open == std::string::npos) {
    throw std::runtime_error("Malformed DCLAMP call");
  }
  std::size_t close = find_matching_paren(fn, open);
  std::size_t return_pos = fn.rfind("return", dclamp_pos);
  if (return_pos == std::string::npos) {
    throw std::runtime_error("Missing return in predict()");
  }
  std::size_t semi_pos = fn.find(';', close);
  if (semi_pos == std::string::npos) {
    throw std::runtime_error("Missing return terminator in predict()");
  }
  fn.replace(return_pos, semi_pos - return_pos + 1, "return " + expr + ";");
}

void replace_function_name(std::string& text,
                           const std::string& from,
                           const std::string& to) {
  std::size_t pos = text.find(from);
  if (pos == std::string::npos) {
    return;
  }
  if (pos > 0 && is_ident_char(text[pos - 1])) {
    return;
  }
  std::size_t after = pos + from.size();
  if (after < text.size() && is_ident_char(text[after])) {
    return;
  }
  text.replace(pos, from.size(), to);
}

bool replace_predict_call(std::string& text) {
  std::size_t pos = text.find("::predict(");
  if (pos != std::string::npos) {
    text.replace(pos, std::string("::predict(").size(), "::predict_raw(");
    return true;
  }
  pos = text.find("predict(");
  while (pos != std::string::npos) {
    if ((pos == 0 || !is_ident_char(text[pos - 1])) &&
        (pos + 7 >= text.size() || !is_ident_char(text[pos + 7]))) {
      text.replace(pos, std::string("predict(").size(), "predict_raw(");
      return true;
    }
    pos = text.find("predict(", pos + 7);
  }
  return false;
}

void inject_raw_predict_support(const std::filesystem::path& rm_model_dir,
                                const std::string& namespace_name) {
  std::filesystem::path cpp_path = rm_model_dir / "src" / (namespace_name + ".cpp");
  std::filesystem::path header_path = rm_model_dir / "include" / (namespace_name + ".h");

  std::string cpp = read_file(cpp_path);
  FunctionBlock predict_block = extract_function_block(cpp, "predict");
  bool has_predict_raw = try_extract_function_block(cpp, "predict_raw", nullptr);

  std::string predict_fn = predict_block.text;
  std::string raw_fn = predict_fn;
  replace_function_name(raw_fn, "predict", "predict_raw");
  std::string expr = extract_dclamp_arg(raw_fn);
  replace_return_with_raw(raw_fn, expr);

  if (!has_predict_raw) {
    cpp.insert(predict_block.end + 1, "\n\n" + raw_fn + "\n");
  }

  FunctionBlock pred_u64_block = extract_function_block(cpp, "rm_model_infer_predict_u64");
  FunctionBlock pred_u32_block = extract_function_block(cpp, "rm_model_infer_predict_u32");
  FunctionBlock pred_f64_block = extract_function_block(cpp, "rm_model_infer_predict_f64");

  auto make_raw_wrapper = [](const std::string& block, const std::string& from,
                             const std::string& to) {
    std::string out = block;
    replace_function_name(out, from, to);
    if (!replace_predict_call(out)) {
      throw std::runtime_error("Missing predict() call in wrapper: " + from);
    }
    return out;
  };

  auto raw_wrapper_valid = [](const FunctionBlock& block, const std::string& name) {
    std::size_t line_end = block.text.find('\n');
    std::string sig = block.text.substr(0, line_end == std::string::npos
                                                ? block.text.size()
                                                : line_end);
    if (sig.find("RM_MODEL_INFER_EXPORT") == std::string::npos) {
      return false;
    }
    if (sig.find("double") == std::string::npos) {
      return false;
    }
    if (sig.find(name) == std::string::npos) {
      return false;
    }
    return block.text.find("predict_raw") != std::string::npos;
  };

  std::vector<FunctionBlock> existing_raw;
  bool raw_ok = true;
  for (const auto& name : {"rm_model_infer_predict_raw_u64",
                           "rm_model_infer_predict_raw_u32",
                           "rm_model_infer_predict_raw_f64"}) {
    FunctionBlock block;
    if (try_extract_function_block(cpp, name, &block)) {
      existing_raw.push_back(block);
      if (!raw_wrapper_valid(block, name)) {
        raw_ok = false;
      }
    } else {
      raw_ok = false;
    }
  }

  if (!raw_ok) {
    std::string raw_u64 = make_raw_wrapper(pred_u64_block.text, "rm_model_infer_predict_u64",
                                           "rm_model_infer_predict_raw_u64");
    std::string raw_u32 = make_raw_wrapper(pred_u32_block.text, "rm_model_infer_predict_u32",
                                           "rm_model_infer_predict_raw_u32");
    std::string raw_f64 = make_raw_wrapper(pred_f64_block.text, "rm_model_infer_predict_f64",
                                           "rm_model_infer_predict_raw_f64");

    std::sort(existing_raw.begin(), existing_raw.end(),
              [](const FunctionBlock& a, const FunctionBlock& b) {
                return a.start > b.start;
              });
    for (const auto& block : existing_raw) {
      cpp.erase(block.start, block.end - block.start + 1);
    }

    FunctionBlock insert_block = extract_function_block(cpp, "rm_model_infer_predict_f64");
    cpp.insert(insert_block.end + 1,
               "\n\n" + raw_u64 + "\n\n" + raw_u32 + "\n\n" + raw_f64 + "\n");
  }

  write_file(cpp_path, cpp);

  std::string header = read_file(header_path);
  if (header.find("predict_raw(") != std::string::npos) {
    return;
  }
  std::size_t predict_decl = header.find("double predict(");
  if (predict_decl == std::string::npos) {
    throw std::runtime_error("Missing predict() declaration in header");
  }
  std::size_t decl_end = header.find(";", predict_decl);
  if (decl_end == std::string::npos) {
    throw std::runtime_error("Malformed predict() declaration in header");
  }
  std::string predict_line = header.substr(predict_decl, decl_end - predict_decl + 1);
  std::string raw_line = predict_line;
  std::size_t decl_replace = raw_line.find("double predict(");
  if (decl_replace != std::string::npos) {
    raw_line.replace(decl_replace, std::string("double predict(").size(),
                     "double predict_raw(");
  }
  header.insert(decl_end + 1, "\n" + raw_line);
  write_file(header_path, header);

}

std::filesystem::path output_root() {
  return std::filesystem::path("lead_output");
}

std::filesystem::path auto_output_dir_for_model(const std::string& namespace_name,
                                                const std::string& models,
                                                uint64_t branch_factor) {
  std::ostringstream tag;
  tag << sanitize_component(namespace_name) << "_"
      << (models.empty() ? "auto" : sanitize_component(models))
      << "_bf" << branch_factor;
  return output_root() / tag.str();
}

struct OutputPaths {
  std::filesystem::path output_dir;
  std::filesystem::path data_dir;
};

OutputPaths resolve_output_paths(const TrainOptions& opts,
                                 const std::filesystem::path& default_output_dir) {
  std::filesystem::path output_dir = opts.output_dir.has_value()
      ? *opts.output_dir
      : default_output_dir;
  std::filesystem::path data_dir = opts.data_path_explicit
      ? opts.data_path
      : (output_dir / "rm_model" / "data");
  return {output_dir, data_dir};
}

rm_model::KeyType detect_key_type(const std::string& path, DataType& dt) {
  rm_model::KeyType key_type = rm_model::KeyType::U64;
  dt = DataType::UINT64;
  if (path.find("uint32") != std::string::npos) {
    dt = DataType::UINT32;
    key_type = rm_model::KeyType::U32;
  } else if (path.find("f64") != std::string::npos) {
    dt = DataType::FLOAT64;
    key_type = rm_model::KeyType::F64;
  } else if (path.find("uint64") == std::string::npos) {
    throw std::runtime_error("Data file must contain uint64, uint32, or f64.");
  }
  return key_type;
}

std::string namespace_from_path(const std::string& path) {
  auto name = std::filesystem::path(path).filename().string();
  return name.empty() ? std::string("lead") : name;
}

rm_model::ModelSelectionStats select_balanced(
    const std::vector<rm_model::ModelSelectionStats>& stats,
    double size_weight,
    double error_weight) {
  if (stats.empty()) {
    throw std::runtime_error("Selector returned no configs");
  }

  double min_size = static_cast<double>(stats.front().size);
  double max_size = static_cast<double>(stats.front().size);
  double min_err = stats.front().average_log2_error;
  double max_err = stats.front().average_log2_error;

  for (const auto& item : stats) {
    min_size = std::min(min_size, static_cast<double>(item.size));
    max_size = std::max(max_size, static_cast<double>(item.size));
    min_err = std::min(min_err, item.average_log2_error);
    max_err = std::max(max_err, item.average_log2_error);
  }

  double size_range = std::max(1e-12, max_size - min_size);
  double err_range = std::max(1e-12, max_err - min_err);

  auto score = [&](const rm_model::ModelSelectionStats& item) {
    double size_norm = (static_cast<double>(item.size) - min_size) / size_range;
    double err_norm = (item.average_log2_error - min_err) / err_range;
    return size_weight * size_norm + error_weight * err_norm;
  };

  const rm_model::ModelSelectionStats* best = &stats.front();
  double best_score = score(*best);

  for (const auto& item : stats) {
    double s = score(item);
    if (s < best_score) {
      best_score = s;
      best = &item;
    }
  }

  return *best;
}

struct CalibrationStats {
  double min_pred = std::numeric_limits<double>::infinity();
  double max_pred = -std::numeric_limits<double>::infinity();
  std::size_t samples = 0;
};

std::size_t model_index_from_output(rm_model::ModelDataType from,
                                    std::size_t bound,
                                    bool needs_check,
                                    double fpred,
                                    uint64_t ipred) {
  std::size_t max_index = bound > 0 ? bound - 1 : 0;
  if (from == rm_model::ModelDataType::Float) {
    if (!needs_check) {
      return static_cast<std::size_t>(fpred);
    }
    if (fpred < 0.0) {
      return 0;
    }
    double max_d = static_cast<double>(max_index);
    if (fpred > max_d) {
      return max_index;
    }
    return static_cast<std::size_t>(fpred);
  }

  if (from == rm_model::ModelDataType::Int) {
    if (!needs_check) {
      return static_cast<std::size_t>(ipred);
    }
    return ipred > max_index ? max_index : static_cast<std::size_t>(ipred);
  }

  if (!needs_check) {
    return static_cast<std::size_t>(ipred);
  }
  return ipred > max_index ? max_index : static_cast<std::size_t>(ipred);
}

template <typename KeyT>
double predict_raw(const rm_model::TrainedModel& model,
                   rm_model::KeyType key_type,
                   KeyT key) {
  rm_model::ModelInput input = rm_model::TrainingKeyOps<KeyT>::to_model_input(key);
  rm_model::ModelDataType last_output = rm_model::to_model_data_type(key_type);
  bool needs_check = true;
  double fpred = input.as_float();
  uint64_t ipred = input.as_int();
  std::size_t model_index = 0;

  for (const auto& layer : model.model_layers) {
    if (layer.empty()) {
      break;
    }
    if (layer.size() > 1) {
      model_index = model_index_from_output(last_output, layer.size(), needs_check, fpred, ipred);
      if (model_index >= layer.size()) {
        model_index = layer.size() - 1;
      }
    } else {
      model_index = 0;
    }

    const rm_model::Model& layer_model = *layer[model_index];
    if (layer_model.output_type() == rm_model::ModelDataType::Float) {
      fpred = layer_model.predict_to_float(input);
    } else {
      ipred = layer_model.predict_to_int(input);
    }
    last_output = layer_model.output_type();
    needs_check = layer_model.needs_bounds_check();
  }

  double pred = (last_output == rm_model::ModelDataType::Float)
      ? fpred
      : static_cast<double>(ipred);
  return pred;
}

template <typename KeyT>
CalibrationStats compute_calibration(const rm_model::TrainedModel& model,
                                     rm_model::KeyType key_type,
                                     const rm_model::TrainingData<KeyT>& data) {
  CalibrationStats stats;
  std::size_t len = data.len();
  for (std::size_t i = 0; i < len; ++i) {
    KeyT key = data.get_key(i);
    double pred = predict_raw(model, key_type, key);
    if (!std::isfinite(pred)) {
      continue;
    }
    stats.min_pred = std::min(stats.min_pred, pred);
    stats.max_pred = std::max(stats.max_pred, pred);
    stats.samples += 1;
  }
  return stats;
}

} // namespace

TrainResult train_and_generate(const TrainOptions& options) {
  rm_model::set_thread_count(options.threads);
  if (options.size_weight < 0.0 || options.error_weight < 0.0) {
    throw std::runtime_error("size_weight and error_weight must be >= 0");
  }
  if (options.selector_limit < 2) {
    throw std::runtime_error("selector_limit must be >= 2");
  }
  if (options.model_spec.has_value() != options.branching_factor.has_value()) {
    throw std::runtime_error("model_spec and branching_factor must be set together");
  }

  DataType dt = DataType::UINT64;
  rm_model::KeyType key_type = detect_key_type(options.input, dt);

  auto load_result = load_data(options.input, dt);
  std::size_t num_rows = load_result.first;
  if (num_rows == 0) {
    throw std::runtime_error("Cannot train on empty dataset");
  }

  MappedDataset data = std::move(load_result.second);

  std::string namespace_name = options.namespace_name.value_or(namespace_from_path(options.input));
  RM_MODEL_LOG_INFO("Loaded dataset " << options.input << " (" << num_rows
                                      << " rows, key_type=" << key_type_name(key_type) << ")");

  rm_model::TrainedModel trained_model;
  if (options.max_size_bytes.has_value()) {
    RM_MODEL_LOG_INFO("Training mode: max-size (" << *options.max_size_bytes << " bytes)");
    trained_model = data.visit([&](auto& typed_data) {
      return rm_model::train_for_size(typed_data, *options.max_size_bytes);
    });
  } else if (options.model_spec.has_value() && options.branching_factor.has_value()) {
    RM_MODEL_LOG_INFO("Training mode: explicit models (" << *options.model_spec
                                                         << ", bf=" << *options.branching_factor << ")");
    trained_model = data.visit([&](auto& typed_data) {
      return rm_model::train(typed_data, *options.model_spec, *options.branching_factor);
    });
  } else {
    RM_MODEL_LOG_INFO("Training mode: auto selector (limit " << options.selector_limit
                                                             << ", size_weight=" << options.size_weight
                                                             << ", error_weight=" << options.error_weight << ")");
    auto stats = data.visit([&](auto& typed_data) {
      return rm_model::select_pareto_configs(typed_data, options.selector_limit);
    });
    auto selected = select_balanced(stats, options.size_weight, options.error_weight);
    trained_model = data.visit([&](auto& typed_data) {
      return rm_model::train(typed_data, selected.model_spec, selected.branching_factor);
    });
  }

  if (options.zero_build_time) {
    trained_model.build_time_ns = 0;
  }

  RM_MODEL_LOG_INFO("Model spec: " << trained_model.model_spec
                                   << " (bf=" << trained_model.branching_factor << "), "
                                   << "size=" << rm_model::model_size_bytes(trained_model) << " bytes, "
                                   << "build_time_ns=" << trained_model.build_time_ns);

  uint64_t max_index = trained_model.num_model_rows == 0
      ? 0
      : static_cast<uint64_t>(trained_model.num_model_rows - 1);

  std::string warning;
  MappingConfig mapping = MappingConfig::compute(max_index,
                                                 options.space_bits,
                                                 options.scale_factor,
                                                 &warning);
  if (!warning.empty()) {
    RM_MODEL_LOG_INFO("LEAD mapping: " << warning);
  }
  RM_MODEL_LOG_INFO("Mapping: max_index=" << mapping.max_index
                                          << ", space_bits=" << mapping.space_bits
                                          << ", scale_factor=" << mapping.scale_factor);

  PredictionCalibration calibration;
  if (options.calibrate) {
    CalibrationStats stats = data.visit([&](auto& typed_data) {
      return compute_calibration(trained_model, key_type, typed_data);
    });
    if (stats.samples > 0 && std::isfinite(stats.min_pred) && std::isfinite(stats.max_pred) &&
        stats.max_pred > stats.min_pred) {
      calibration.enabled = true;
      calibration.min_pred = stats.min_pred;
      calibration.max_pred = stats.max_pred;
      RM_MODEL_LOG_INFO("Calibration: pred_min=" << calibration.min_pred
                                                 << ", pred_max=" << calibration.max_pred
                                                 << ", samples=" << stats.samples);
    } else {
      RM_MODEL_LOG_WARN("Calibration disabled (insufficient range or samples).");
    }
  } else {
    RM_MODEL_LOG_INFO("Calibration disabled (--no-calibration).");
  }

  std::filesystem::path default_output_dir = auto_output_dir_for_model(
      namespace_name,
      trained_model.model_spec,
      trained_model.branching_factor);

  auto output_paths = resolve_output_paths(options, default_output_dir);
  std::filesystem::path lead_dir = output_paths.output_dir;
  std::filesystem::path rm_model_dir = lead_dir / "rm_model";

  LeadManifest manifest;
  manifest.name = namespace_name;
  manifest.key_type = key_type;
  manifest.mapping = mapping;
  manifest.calibration = calibration;
  manifest.model_spec = trained_model.model_spec;
  manifest.branching_factor = trained_model.branching_factor;
  manifest.model_size_bytes = rm_model::model_size_bytes(trained_model);
  manifest.build_time_ns = trained_model.build_time_ns;
  manifest.loadable = options.emit_code;
  manifest.rm_model_dir = rm_model_dir;
  manifest.rm_model_manifest = rm_model_dir / "model.json";

  if (options.emit_code) {
    std::filesystem::create_directories(output_paths.data_dir);
    rm_model::emit_model(namespace_name,
                         std::move(trained_model),
                         rm_model_dir.string(),
                         output_paths.data_dir.string(),
                         key_type,
                         options.include_errors);
    inject_raw_predict_support(rm_model_dir, namespace_name);
  } else {
    RM_MODEL_LOG_WARN("LEAD artifacts not emitted (--no-code); output is not loadable.");
  }

  std::filesystem::path manifest_path = lead_dir / "lead.json";
  manifest.write(manifest_path);

  TrainResult result;
  result.manifest = manifest;
  result.lead_dir = lead_dir;
  result.rm_model_dir = rm_model_dir;
  return result;
}

} // namespace lead
