#include "vortex_v1/codegen.h"
#include "vortex_v1/codegen_loader.h"
#include "vortex_v1/model.h"
#include "vortex_v1/nsw_csr.h"
#include "vortex_v1/training.h"
#include "vortex_v1/types.h"

#include "cli_common.h"

#include "rm_model/logging.h"

#include "vector_io/vector_io.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using vortex::tools::ArgList;
using vortex::tools::parse_i32;
using vortex::tools::parse_u32;
using vortex::tools::parse_u64;
using vortex::tools::validate_known_flags;
using vortex::tools::validate_required_flag_values;

std::string default_name_from_path(const std::filesystem::path& path) {
  if (path.empty()) return "vortex";
  return path.stem().string();
}

void print_main_usage() {
  std::cout << "Usage: vortex_v1_cli <command> [options]\n";
  std::cout << "Commands:\n";
  std::cout << "  build_hnsw  Build FAISS HNSW index\n";
  std::cout << "  extract_nsw Extract NSW skeleton CSR\n";
  std::cout << "  train       Train Vortex v1 model\n";
  std::cout << "  hash        Hash vectors using generated code\n";
  std::cout << "  codegen     Generate reusable code artifact from model\n";
  std::cout << "Run 'vortex_v1_cli <command> --help' for command options.\n";
}

void print_build_hnsw_usage() {
  std::cout << "Usage: vortex_v1_cli build_hnsw --dataset <fvecs> [options]\n";
  std::cout << "Options: --name --M --efConstruction --metric --output\n";
}

void print_extract_nsw_usage() {
  std::cout << "Usage: vortex_v1_cli extract_nsw (--index <file> | --dataset <fvecs>) "
               "--target_skeleton <n> [options]\n";
  std::cout << "Options: --M --efConstruction --metric --layer --output\n";
}

void print_train_usage() {
  std::cout << "Usage: vortex_v1_cli train --dataset <fvecs> --nsw <csr> --K <n> "
               "--hash_bits <bits> --centroid_knn <n> [options]\n";
  std::cout << "Options: --cdf_models --cdf_model_spec --cdf_branch --cdf_branching_factor "
               "--enable_2opt --disable_2opt --two_opt_iters "
               "--disable_graph_centroid_order "
               "--disable-full-dataset-assignments --assignment-sample-limit "
               "--seed --threads --output "
               "--codegen --codegen-dir --codegen-name --no-cmake\n";
}

void print_hash_usage() {
  std::cout << "Usage: vortex_v1_cli hash --codegen-dir <dir> (--dataset <fvecs> | --query <fvecs>) "
               "[options]\n";
  std::cout << "Options: --out | --output --limit "
               "--disable-centroid-index --centroid-index-pivots --disable-query-cache\n";
}

void print_codegen_usage() {
  std::cout << "Usage: vortex_v1_cli codegen --model <model.bin> [options]\n";
  std::cout << "Options: --name --output-dir --no-cmake\n";
}

void cmd_build_hnsw(const ArgList& args) {
  if (args.has("--help")) {
    print_build_hnsw_usage();
    return;
  }
  validate_known_flags(args, {"--dataset", "--name", "--M", "--efConstruction",
                              "--metric", "--output"});

  std::filesystem::path dataset = args.require("--dataset");
  std::string name = args.value("--name").value_or(default_name_from_path(dataset));
  uint32_t M = args.value("--M").has_value() ? parse_u32(*args.value("--M"), "--M") : 32;
  uint32_t efc = args.value("--efConstruction").has_value()
      ? parse_u32(*args.value("--efConstruction"), "--efConstruction")
      : 200;
  vortex::Metric metric = vortex::Metric::L2;
  if (args.value("--metric").has_value()) {
    metric = vortex::parse_metric(*args.value("--metric"));
  }

  std::filesystem::path output = args.value("--output").has_value()
      ? std::filesystem::path(*args.value("--output"))
      : (std::filesystem::path("vortex_v1_output") / (name + "_hnsw.index"));

  vortex::BuildHnswOptions opts;
  opts.dataset_path = dataset;
  opts.M = M;
  opts.ef_construction = efc;
  opts.metric = metric;

  rm_model::init_logging();
  RM_MODEL_LOG_INFO("Building HNSW index: " << output.string());
  vortex::build_hnsw_index(opts, output);
}

void cmd_extract_nsw(const ArgList& args) {
  if (args.has("--help")) {
    print_extract_nsw_usage();
    return;
  }
  validate_known_flags(args, {"--index", "--dataset", "--M", "--efConstruction",
                              "--metric", "--target_skeleton", "--layer", "--output"});

  vortex::ExtractNswOptions opts;
  if (args.value("--index").has_value()) {
    opts.index_path = std::filesystem::path(*args.value("--index"));
  }
  if (args.value("--dataset").has_value()) {
    opts.dataset_path = std::filesystem::path(*args.value("--dataset"));
  }
  if (args.value("--M").has_value()) {
    opts.M = parse_u32(*args.value("--M"), "--M");
  }
  if (args.value("--efConstruction").has_value()) {
    opts.ef_construction = parse_u32(*args.value("--efConstruction"), "--efConstruction");
  }
  if (args.value("--metric").has_value()) {
    opts.metric = vortex::parse_metric(*args.value("--metric"));
  }
  opts.target_skeleton = parse_u64(args.require("--target_skeleton"), "--target_skeleton");
  if (args.value("--layer").has_value()) {
    std::string layer_val = *args.value("--layer");
    if (layer_val != "auto") {
      opts.layer = parse_i32(layer_val, "--layer");
    }
  }

  std::string name = "vortex";
  if (opts.index_path.has_value()) {
    name = default_name_from_path(*opts.index_path);
  } else if (opts.dataset_path.has_value()) {
    name = default_name_from_path(*opts.dataset_path);
  }

  std::filesystem::path output = args.value("--output").has_value()
      ? std::filesystem::path(*args.value("--output"))
      : (std::filesystem::path("vortex_v1_output") / (name + "_nsw.csr"));

  rm_model::init_logging();
  RM_MODEL_LOG_INFO("Extracting NSW skeleton: " << output.string());
  vortex::extract_nsw(opts, output);
}

void cmd_train(const ArgList& args) {
  if (args.has("--help")) {
    print_train_usage();
    return;
  }
  validate_known_flags(args, {"--dataset", "--nsw", "--K", "--hash_bits", "--centroid_knn",
                              "--cdf_models", "--cdf_model_spec", "--cdf_branch",
                              "--cdf_branching_factor", "--enable_2opt", "--disable_2opt",
                              "--two_opt_iters", "--disable_graph_centroid_order",
                              "--disable-full-dataset-assignments",
                              "--assignment-sample-limit", "--seed", "--threads", "--output",
                              "--codegen", "--codegen-dir", "--codegen-name", "--no-cmake"});
  validate_required_flag_values(
      args,
      {"--dataset", "--nsw", "--K", "--hash_bits", "--centroid_knn",
       "--cdf_models", "--cdf_model_spec", "--cdf_branch", "--cdf_branching_factor",
       "--two_opt_iters", "--assignment-sample-limit", "--seed", "--threads", "--output",
       "--codegen-dir", "--codegen-name"});

  vortex::TrainOptions opts;
  opts.dataset_path = args.require("--dataset");
  opts.nsw_path = args.require("--nsw");
  opts.K = parse_u32(args.require("--K"), "--K");
  opts.hash_bits = parse_u32(args.require("--hash_bits"), "--hash_bits");
  opts.centroid_knn = parse_u32(args.require("--centroid_knn"), "--centroid_knn");
  if (args.has("--enable_2opt") && args.has("--disable_2opt")) {
    throw std::runtime_error("Use only one of --enable_2opt or --disable_2opt");
  }
  opts.enable_2opt = !args.has("--disable_2opt");
  if (args.value("--two_opt_iters").has_value()) {
    opts.two_opt_iterations = parse_u32(*args.value("--two_opt_iters"), "--two_opt_iters");
  }
  if (args.has("--disable_graph_centroid_order")) {
    opts.enable_graph_centroid_order = false;
  }
  if (args.value("--cdf_models").has_value()) {
    opts.cdf_model_spec = *args.value("--cdf_models");
  } else if (args.value("--cdf_model_spec").has_value()) {
    opts.cdf_model_spec = *args.value("--cdf_model_spec");
  }
  if (args.value("--cdf_branch").has_value()) {
    opts.cdf_branching_factor = parse_u64(*args.value("--cdf_branch"), "--cdf_branch");
  } else if (args.value("--cdf_branching_factor").has_value()) {
    opts.cdf_branching_factor =
        parse_u64(*args.value("--cdf_branching_factor"), "--cdf_branching_factor");
  }
  if (args.value("--seed").has_value()) {
    opts.seed = parse_u64(*args.value("--seed"), "--seed");
  }
  if (args.value("--threads").has_value()) {
    opts.threads = parse_u32(*args.value("--threads"), "--threads");
  }
  if (args.has("--disable-full-dataset-assignments")) {
    opts.use_full_dataset_for_assignments = false;
  }
  if (args.value("--assignment-sample-limit").has_value()) {
    opts.assignment_sample_limit =
        parse_u64(*args.value("--assignment-sample-limit"), "--assignment-sample-limit");
  }

  std::string name = default_name_from_path(opts.dataset_path);
  std::filesystem::path output = args.value("--output").has_value()
      ? std::filesystem::path(*args.value("--output"))
      : (std::filesystem::path("vortex_v1_output") / (name + "_v1.model"));

  rm_model::init_logging();
  RM_MODEL_LOG_INFO("Training Vortex v1 model: " << output.string());
  auto model = vortex::train_vortex(opts);
  model.write(output);

  bool emit_codegen = args.has("--codegen") || args.value("--codegen-dir").has_value();
  if (emit_codegen) {
    std::filesystem::path codegen_dir = args.value("--codegen-dir").has_value()
        ? std::filesystem::path(*args.value("--codegen-dir"))
        : (std::filesystem::path("vortex_v1_output") / (name + "_codegen"));
    std::string codegen_name = args.value("--codegen-name").value_or(name);
    vortex::CodegenOptions cg_opts;
    cg_opts.output_dir = codegen_dir;
    cg_opts.name = codegen_name;
    cg_opts.emit_cmake = !args.has("--no-cmake");
    auto result = vortex::emit_codegen(model, cg_opts);
    RM_MODEL_LOG_INFO("Vortex codegen output dir: " << result.output_dir.string());
  }
}

void cmd_hash(const ArgList& args) {
  if (args.has("--help")) {
    print_hash_usage();
    return;
  }
  validate_known_flags(args, {"--codegen-dir", "--dataset", "--query", "--out",
                              "--output", "--limit", "--disable-centroid-index",
                              "--centroid-index-pivots", "--disable-query-cache"});

  std::filesystem::path codegen_dir = args.require("--codegen-dir");
  bool has_dataset = args.value("--dataset").has_value();
  bool has_query = args.value("--query").has_value();
  if (has_dataset && has_query) {
    throw std::runtime_error("Specify only one of --dataset or --query");
  }
  if (!has_dataset && !has_query) {
    throw std::runtime_error("Missing --dataset or --query");
  }
  std::filesystem::path dataset = has_dataset
      ? std::filesystem::path(*args.value("--dataset"))
      : std::filesystem::path(*args.value("--query"));

  std::optional<std::filesystem::path> output;
  if (args.value("--out").has_value()) {
    output = std::filesystem::path(*args.value("--out"));
  } else if (args.value("--output").has_value()) {
    output = std::filesystem::path(*args.value("--output"));
  }

  std::size_t limit = 0;
  if (args.value("--limit").has_value()) {
    limit = static_cast<std::size_t>(parse_u64(*args.value("--limit"), "--limit"));
  }

  vortex::CodegenLoadOptions load_opts;
  load_opts.enable_centroid_index = !args.has("--disable-centroid-index");
  if (args.value("--centroid-index-pivots").has_value()) {
    load_opts.centroid_index_pivots =
        parse_u32(*args.value("--centroid-index-pivots"), "--centroid-index-pivots");
  }
  load_opts.enable_query_cache = !args.has("--disable-query-cache");

  vortex::CodegenHasher hasher;
  rm_model::init_logging();
  RM_MODEL_LOG_INFO("Loading Vortex codegen from: " << codegen_dir.string());
  hasher.load(codegen_dir, load_opts);
  auto vectors = vector_io::read_fvecs(dataset.string());
  if (vectors.dim != hasher.dim()) {
    throw std::runtime_error("Dataset dim does not match model");
  }

  std::ostream* out = &std::cout;
  std::ofstream file;
  if (output.has_value()) {
    if (!output->parent_path().empty()) {
      std::filesystem::create_directories(output->parent_path());
    }
    file.open(*output);
    if (!file) {
      throw std::runtime_error("Unable to open output file: " + output->string());
    }
    out = &file;
  }

  std::size_t count = vectors.count;
  if (limit > 0 && limit < count) {
    count = limit;
  }

  for (std::size_t i = 0; i < count; ++i) {
    const float* vec = vectors.values.data() + i * vectors.dim;
    uint32_t bits = hasher.hash_bits();
    if (bits <= 64) {
      bool ok = false;
      uint64_t value = hasher.hash64(vec, &ok);
      if (!ok) {
        throw std::runtime_error("Codegen hash64 failed");
      }
      (*out) << value << "\n";
    } else {
      vortex::UInt256 value{};
      if (!hasher.hash(vec, &value)) {
        throw std::runtime_error("Codegen hash failed");
      }
      vortex::HashOutput out_hash;
      out_hash.value = value;
      out_hash.bits = bits;
      (*out) << out_hash.to_hex() << "\n";
    }
  }
}

void cmd_codegen(const ArgList& args) {
  if (args.has("--help")) {
    print_codegen_usage();
    return;
  }
  validate_known_flags(args, {"--model", "--name", "--output-dir", "--no-cmake"});

  std::filesystem::path model_path = args.require("--model");
  std::string name = args.value("--name").value_or(default_name_from_path(model_path));
  std::filesystem::path output = args.value("--output-dir").has_value()
      ? std::filesystem::path(*args.value("--output-dir"))
      : (std::filesystem::path("vortex_v1_output") / (name + "_codegen"));
  vortex::CodegenOptions opts;
  opts.output_dir = output;
  opts.name = name;
  opts.emit_cmake = !args.has("--no-cmake");
  rm_model::init_logging();
  RM_MODEL_LOG_INFO("Generating Vortex codegen artifacts: " << output.string());
  vortex::emit_codegen_from_file(model_path, opts);
}

} // namespace

int main(int argc, char** argv) {
  try {
    ArgList args(argc, argv);
    if (argc < 2 || args.args[1] == "--help" || args.args[1] == "help") {
      print_main_usage();
      return 0;
    }
    const std::string& cmd = args.args[1];
    if (cmd == "build_hnsw") {
      cmd_build_hnsw(args);
    } else if (cmd == "extract_nsw") {
      cmd_extract_nsw(args);
    } else if (cmd == "train") {
      cmd_train(args);
    } else if (cmd == "hash") {
      cmd_hash(args);
    } else if (cmd == "codegen") {
      cmd_codegen(args);
    } else {
      throw std::runtime_error("Unknown command: " + cmd);
    }
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
