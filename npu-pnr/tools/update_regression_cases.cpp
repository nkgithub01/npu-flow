#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <format>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "argparse/argparse.hpp"

#include "base/common.hpp"
#include "base/engine.hpp"
#include "base/routing/routing_state.hpp"
#include "engine/engine.hpp"

namespace {

struct RegressionConfig {
  std::string placer;
  std::string router;
  int max_placement_iterations;
};

struct RegressionResult {
  std::size_t placement_size;
  double routing_cost;
};

struct ManifestCase {
  std::string name;
  std::filesystem::path benchmark_path;
  RegressionConfig config;
  RegressionResult result;
};

base::Config createEngineConfig(const RegressionConfig &regression_config) {
  using base::Config;

  Config cfg =
      Config{}
          .set<int>("timeout_secs", 3600)
          .set<bool>("dump_rr_graph", false)
          .set("logger", Config{}.set<std::string>("verbose", "silent"))
          .set("telemetry", Config{}.set<bool>("enable", false));

  cfg.set("arch", Config{}.set("type", std::string("npu")));

  Config default_placer_cfg =
      Config{}
          .set<int>("random_seed", 24)
          .set<int>("max_iters", regression_config.max_placement_iterations)
          .set<bool>("enable_initial_placement_randomization", true)
          .set<bool>("enable_milp_router_lock_constraints", true)
          .set<int>("greedy_stage_max_iters", 100)
          .set<int>("max_move_attempts", 10000)
          .set<int>("num_moves_per_iter", 100)
          .set<bool>("enable_dynamic_temperature_scheduling", true)
          .set<std::string>("initial_temperature_estimate_method",
                            "equilibrium")
          .set<int>("initial_temperature_num_sampling_moves", -1)
          .set<double>("initial_temperature_multiplier", 10.0)
          .set<std::string>("cost_estimator", "prob")
          .set<bool>("enable_aggressive_local_search", true)
          .set<double>(
              "max_consecutive_iters_no_best_cost_improvement_scaling_factor",
              1.5)
          .set<std::string>("neighbor_region_shape", "cross")
          .set<std::string>("type", regression_config.placer);

  Config default_router_cfg =
      Config{}
          .set<bool>("enable_lock_constraints", true)
          .set<double>("congestion_penalty_scaling_factor", 1e5)
          .set<std::string>("type", regression_config.router);

  cfg.set("placer", default_placer_cfg);
  cfg.set("router", default_router_cfg);

  return cfg;
}

RegressionResult runRegressionCase(const std::filesystem::path &benchmark_path,
                                   const RegressionConfig &regression_config) {
  engine::PnREngine engine(createEngineConfig(regression_config));
  const auto result =
      engine.run(base::PnRNetlistReader::fromTOML(std::ifstream(benchmark_path)));

  if (result.routing.getLegality() != base::RoutingState::Legality::Legal) {
    throw std::runtime_error(std::format(
        "Routing is not legal for '{}' with placer='{}' and router='{}'",
        benchmark_path.string(), regression_config.placer,
        regression_config.router));
  }

  return RegressionResult{.placement_size = result.placement.size(),
                          .routing_cost = result.routing.getRoutingCost()};
}

std::size_t getLogicalCoreCount(const std::filesystem::path &benchmark_path) {
  return base::PnRNetlistReader::fromTOML(std::ifstream(benchmark_path))
      .getNodes()
      .size();
}

bool isQuickProfileBenchmark(const std::filesystem::path &benchmarks_root,
                             const std::filesystem::path &benchmark_path) {
  const std::filesystem::path relative_path =
      std::filesystem::relative(benchmark_path, benchmarks_root)
          .lexically_normal();
  const std::string rel = relative_path.generic_string();

  static const std::regex edge_detection_regex(
      R"(^edge_detection/edge_detection__col_[12]\.toml$)");
  static const std::regex pipelined_line_or_tree_regex(
      R"(^microbenchmark/pipelined/microbenchmark__(line|tree)_.*\.toml$)");

  if (std::regex_match(rel, edge_detection_regex)) {
    return true;
  }

  if (std::regex_match(rel, pipelined_line_or_tree_regex)) {
    return getLogicalCoreCount(benchmark_path) <= 8;
  }

  return false;
}

std::vector<std::filesystem::path>
collectBenchmarkPaths(const std::filesystem::path &benchmarks_root,
                      std::string_view profile) {
  if (!std::filesystem::exists(benchmarks_root)) {
    throw std::runtime_error(std::format(
        "Benchmarks root does not exist: {}", benchmarks_root.string()));
  }
  if (!std::filesystem::is_directory(benchmarks_root)) {
    throw std::runtime_error(std::format(
        "Benchmarks root is not a directory: {}", benchmarks_root.string()));
  }

  std::vector<std::filesystem::path> benchmark_paths;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(benchmarks_root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() != ".toml") {
      continue;
    }

    const std::filesystem::path benchmark_path = entry.path().lexically_normal();
    if (profile == "all" ||
        (profile == "quick" &&
         isQuickProfileBenchmark(benchmarks_root, benchmark_path))) {
      benchmark_paths.push_back(benchmark_path);
    }
  }

  std::ranges::sort(benchmark_paths);
  return benchmark_paths;
}

std::string toUpperSnakeCase(std::string value) {
  for (char &ch : value) {
    const unsigned char uchar = static_cast<unsigned char>(ch);
    if (std::isalnum(uchar) != 0) {
      ch = static_cast<char>(std::toupper(uchar));
    } else {
      ch = '_';
    }
  }
  return value;
}

std::string makeCaseName(const std::filesystem::path &benchmarks_root,
                         const std::filesystem::path &benchmark_path,
                         const RegressionConfig &regression_config) {
  std::string relative_name =
      std::filesystem::relative(benchmark_path, benchmarks_root)
          .replace_extension()
          .generic_string();
  relative_name = toUpperSnakeCase(relative_name);
  return std::format("{}_{}_{}", relative_name,
                     toUpperSnakeCase(regression_config.placer),
                     toUpperSnakeCase(regression_config.router));
}

std::string formatManifest(const std::filesystem::path &manifest_path,
                           const std::filesystem::path &benchmarks_root,
                           const std::vector<ManifestCase> &manifest_cases,
                           std::string_view profile,
                           double routing_cost_epsilon) {
  std::ostringstream oss;
  oss << "# Auto-generated by update_regression_cases\n";
  oss << std::format("# profile = {}\n", profile);
  oss << std::format("# cases = {}\n\n", manifest_cases.size());
  oss << std::fixed << std::setprecision(6);

  for (std::size_t index = 0; index < manifest_cases.size(); ++index) {
    const auto &manifest_case = manifest_cases[index];
    const std::filesystem::path relative_benchmark_path =
        std::filesystem::relative(manifest_case.benchmark_path,
                                  manifest_path.parent_path())
            .lexically_normal();

    oss << "[[case]]\n";
    oss << std::format("name = \"{}\"\n", manifest_case.name);
    oss << std::format("path = \"{}\"\n",
                       relative_benchmark_path.generic_string());
    oss << std::format("expected_placement_size = {}\n",
                       manifest_case.result.placement_size);
    oss << std::format("expected_routing_cost = {:.6f}\n",
                       manifest_case.result.routing_cost);
    oss << std::format("routing_cost_epsilon = {:.6f}\n\n",
                       routing_cost_epsilon);

    oss << "[case.config]\n";
    oss << std::format("placer = \"{}\"\n", manifest_case.config.placer);
    oss << std::format("router = \"{}\"\n", manifest_case.config.router);
    oss << std::format("max_placement_iterations = {}\n",
                       manifest_case.config.max_placement_iterations);

    if (index + 1 != manifest_cases.size()) {
      oss << "\n";
    }
  }

  return oss.str();
}

} // namespace

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("update_regression_cases");

  program.add_description(
      "Generate tests/regression_cases.toml from benchmark netlists.");

  program.add_argument("--benchmarks-root")
      .default_value(std::string{"benchmarks"})
      .help("path to the benchmarks root directory");
  program.add_argument("--manifest")
      .default_value(std::string{"tests/regression_cases.toml"})
      .help("output regression manifest path");
  program.add_argument("--profile")
      .default_value(std::string{"quick"})
      .help("benchmark selection profile [quick|all]");
  program.add_argument("--sa-max-iters")
      .default_value(3)
      .scan<'i', int>()
      .help("max placement iterations for the SA+MILP configuration");
  program.add_argument("--ls-max-iters")
      .default_value(1)
      .scan<'i', int>()
      .help("max placement iterations for the LS+MILP configuration");
  program.add_argument("--routing-cost-epsilon")
      .default_value(1e-3)
      .scan<'g', double>()
      .help("routing cost tolerance written to each manifest case");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }

  try {
    const std::filesystem::path benchmarks_root =
        std::filesystem::absolute(program.get<std::string>("--benchmarks-root"))
            .lexically_normal();
    const std::filesystem::path manifest_path =
        std::filesystem::absolute(program.get<std::string>("--manifest"))
            .lexically_normal();
    const std::string profile = program.get<std::string>("--profile");
    const int sa_max_iters = program.get<int>("--sa-max-iters");
    const int ls_max_iters = program.get<int>("--ls-max-iters");
    const double routing_cost_epsilon =
        program.get<double>("--routing-cost-epsilon");

    if (profile != "quick" && profile != "all") {
      throw std::runtime_error(
          std::format("Unsupported profile '{}'. Use quick or all.", profile));
    }
    if (sa_max_iters < 0 || ls_max_iters < 0) {
      throw std::runtime_error("Placement iteration counts must be non-negative");
    }
    if (routing_cost_epsilon < 0.0) {
      throw std::runtime_error("routing-cost-epsilon must be non-negative");
    }

    const std::vector<std::filesystem::path> benchmark_paths =
        collectBenchmarkPaths(benchmarks_root, profile);
    if (benchmark_paths.empty()) {
      throw std::runtime_error(std::format(
          "No benchmark TOML files matched profile '{}' under {}", profile,
          benchmarks_root.string()));
    }

    const std::vector<RegressionConfig> configs = {
        RegressionConfig{.placer = "sa",
                         .router = "milp",
                         .max_placement_iterations = sa_max_iters},
        RegressionConfig{.placer = "ls",
                         .router = "milp",
                         .max_placement_iterations = ls_max_iters},
    };

    std::vector<ManifestCase> manifest_cases;
    manifest_cases.reserve(benchmark_paths.size() * configs.size());

    std::cout << std::format(
        "Generating '{}' from {} benchmark(s) using profile '{}'...\n",
        manifest_path.string(), benchmark_paths.size(), profile);

    for (const auto &benchmark_path : benchmark_paths) {
      const std::filesystem::path relative_path =
          std::filesystem::relative(benchmark_path, benchmarks_root)
              .lexically_normal();
      for (const auto &config : configs) {
        std::cout << std::format("  - {} [{}+{}, max_iters={}]\n",
                                 relative_path.generic_string(), config.placer,
                                 config.router,
                                 config.max_placement_iterations);
        manifest_cases.push_back(ManifestCase{
            .name = makeCaseName(benchmarks_root, benchmark_path, config),
            .benchmark_path = benchmark_path,
            .config = config,
            .result = runRegressionCase(benchmark_path, config),
        });
      }
    }

    std::filesystem::create_directories(manifest_path.parent_path());
    std::ofstream(manifest_path)
        << formatManifest(manifest_path, benchmarks_root, manifest_cases,
                          profile, routing_cost_epsilon);

    std::cout << std::format(
        "Wrote {} regression case(s) to {}\n", manifest_cases.size(),
        manifest_path.string());
    return EXIT_SUCCESS;
  } catch (const std::exception &err) {
    std::cerr << "update_regression_cases failed: " << err.what() << std::endl;
    return EXIT_FAILURE;
  }
}
