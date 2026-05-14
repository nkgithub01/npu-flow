#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "base/common.hpp"
#include "base/placement.hpp"
#include "engine/engine.hpp"
#include "utils/misc.hpp"

namespace test_utils {

struct EndToEndConfig {
  std::string placer;
  std::string router;
  int max_placement_iterations = 100;
};

inline base::Config createEngineConfig(const EndToEndConfig &test_config) {
  using base::Config;

  if (test_config.placer.empty()) {
    throw std::runtime_error("End-to-end config requires a non-empty placer");
  }
  if (test_config.router.empty()) {
    throw std::runtime_error("End-to-end config requires a non-empty router");
  }

  Config cfg =
      Config{}
          .set<int>("timeout_secs", 3600)
          .set<bool>("dump_rr_graph", false)
          .set("logger", Config{}.set<std::string>("verbose", "minimal"))
          .set("telemetry", Config{}.set<bool>("enable", false));

  cfg.set("arch", Config{}.set("type", std::string("npu")));

  Config default_placer_cfg =
      Config{}
          .set<int>("random_seed", 24)
          .set<int>("max_iters", test_config.max_placement_iterations)
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
          .set<std::string>("type", test_config.placer);

  Config default_router_cfg =
      Config{}
          .set<bool>("enable_lock_constraints", true)
          .set<double>("congestion_penalty_scaling_factor", 1e5)
          .set<std::string>("type", test_config.router);

  cfg.set("placer", default_placer_cfg);
  cfg.set("router", default_router_cfg);

  return cfg;
}

class EndToEndRunner {
private:
  std::unique_ptr<engine::PnREngine> engine_ptr_;

public:
  explicit EndToEndRunner(const EndToEndConfig &test_config)
      : engine_ptr_(std::make_unique<engine::PnREngine>(
            createEngineConfig(test_config))) {}

  engine::PnRState run(std::string_view input_netlist) const {
    return engine_ptr_->run(base::PnRNetlistReader::fromTOML(
        std::string(input_netlist)));
  }

  engine::PnRState run(const std::filesystem::path &input_netlist_path) const {
    return engine_ptr_->run(
        base::PnRNetlistReader::fromTOML(std::ifstream(input_netlist_path)));
  }
};

[[nodiscard]] inline base::Placement
deserializeStdoutLoggedPlacement(const std::string &raw_str) {
  base::Placement placement;
  std::regex pattern(R"(^\s*(\S+)\s+(.+?)\s*$)");

  for (const auto &line : utils::split(utils::trim(raw_str), '\n')) {
    std::smatch matches;
    if (std::regex_search(line, matches, pattern)) {
      placement.add(base::LogicalCore::deserialize(matches[1]),
                    base::PhysicalCore::deserialize(matches[2]));
    } else {
      throw std::runtime_error(
          std::format("Invalid placement format '{}'", line));
    }
  }
  return placement;
}

[[nodiscard]] inline bool
isExpectedPlacement(const base::Placement &placement,
                    const std::string &serialized_expected_placement) {
  return placement ==
         deserializeStdoutLoggedPlacement(serialized_expected_placement);
}

[[nodiscard]] inline bool
isLegalRoutingStateWithExpectedCost(const base::RoutingState &routing,
                                    double expected_cost,
                                    double epsilon = 1e-3) {
  return routing.getLegality() == base::RoutingState::Legality::Legal &&
         std::abs(routing.getRoutingCost() - expected_cost) <= epsilon;
}

} // namespace test_utils
