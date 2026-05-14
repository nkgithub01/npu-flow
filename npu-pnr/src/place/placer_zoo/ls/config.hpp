#pragma once

#include <stdexcept>

#include "base/common.hpp"
#include "base/placement.hpp"
#include "base/routing/routing_mode.hpp"
#include "base/routing/routing_net.hpp"
#include "base/tf_graph.hpp"

namespace place::placer_zoo::ls {

using base::Placement;
using base::RoutingMode;
using base::RoutingNetList;
using base::TrafficFlowGraph;

const std::vector<base::GridPositionOffset> kRegion3x3ShapeKernelOffsets = {
    {{-1, -1},
     {-1, 0},
     {-1, 1},
     {0, -1},
     {0, 0},
     {0, 1},
     {1, -1},
     {1, 0},
     {1, 1}}};
const std::vector<base::GridPositionOffset> kRegionCrossShapeKernelOffsets = {
    {{-1, 0}, {0, -1}, {0, 0}, {0, 1}, {1, 0}}};
const std::vector<base::GridPositionOffset> kRegionXShapeKernelOffsets = {
    {{-1, -1}, {-1, 1}, {0, 0}, {1, -1}, {1, 1}}};

class Config {
public:
  enum class Mode {
    Aggressive,   // explore neighborhood regions for *all* logical cores per
                  // iteration
    Conservative, // explore neighborhood regions for *one* logical core
                  // (greedily chosen from route trials) per iteration
  };

  TrafficFlowGraph tf_graph;
  RoutingNetList netlist;
  Placement init_state;

  Mode mode;
  utils::RandomSeed random_seed;
  size_t max_iterations;
  size_t max_consecutive_iterations_no_best_cost_improvement;

  bool enable_initial_placement_randomization;

  std::vector<base::GridPositionOffset> region_offset_list;

  RoutingMode routing_mode;

  Config() = delete;

  explicit Config(const base::TrafficFlowGraph &tf_graph,
                  const base::Placement &init_placement,
                  const base::LogicalCoreCompatibilitySet &l_core_compat_set) {
    // Initialize basic fields
    this->netlist = base::RoutingNetList{tf_graph};
    this->init_state = init_placement;

    this->routing_mode = RoutingMode{};
    this->routing_mode.set(RoutingMode::LoggingLevel::Silent);
    this->routing_mode.set(RoutingMode::LogicalCorePacking::Enforced,
                           l_core_compat_set);
    RoutingMode::HyperParameters hyper_params;
    // TODO: scale the congestion penalty as the iterations progress
    // TODO: tune the following scaling factor via CLI (also num of iterations?)
    // hyper_params.objective_congestion_penalty_scaling_factor = 0.5;
    this->routing_mode.set(hyper_params);
  }

  Config &parse(const base::Config &cfg) {
    // Parse configuration options
    setMode(cfg.getOrDefault<bool>("enable_aggressive_local_search", false));
    setRandomSeed(cfg.getOrDefault<int>("random_seed", 0));
    setMaxNumIterations(cfg.getOrDefault<int>("max_iters", 100));
    setMaxConsecutiveIterationsNoBestCostImprovement(cfg.getOrDefault<double>(
        "max_consecutive_iters_no_best_cost_improvement_scaling_factor", 2.0));
    setRegionOffsets(
        cfg.getOrDefault<std::string>("neighbor_region_shape", "3x3"));

    this->enable_initial_placement_randomization =
        cfg.getOrDefault<bool>("enable_initial_placement_randomization", false);

    this->routing_mode.set(
        cfg.getOrDefault<bool>("enable_milp_router_lock_constraints", false)
            ? RoutingMode::LockCapacityConstraint::Enforced
            : RoutingMode::LockCapacityConstraint::Ignored);

    // Not available for public use yet
    setPackingDisallowed(cfg.getOrDefault<bool>("disable_packing", false));

    return *this;
  }

  void setMode(bool is_aggressive_mode) {
    this->mode = is_aggressive_mode ? Mode::Aggressive : Mode::Conservative;
  }

  void setRandomSeed(const utils::RandomSeed &seed) {
    this->random_seed = seed;
  }

  void setMaxNumIterations(size_t max_iters) {
    this->max_iterations = max_iters;
  }

  void setMaxConsecutiveIterationsNoBestCostImprovement(
      double max_no_improve_iters_scaling_factor) {
    this->max_consecutive_iterations_no_best_cost_improvement =
        static_cast<size_t>(max_no_improve_iters_scaling_factor *
                            static_cast<double>(this->init_state.size()));
    if (this->max_consecutive_iterations_no_best_cost_improvement < 1) {
        this->max_consecutive_iterations_no_best_cost_improvement = 1;
    }
  }

  void setRegionOffsets(std::string region_shape_name) {
    if (region_shape_name == "3x3") {
      this->region_offset_list = kRegion3x3ShapeKernelOffsets;
    } else if (region_shape_name == "cross") {
      this->region_offset_list = kRegionCrossShapeKernelOffsets;
    } else if (region_shape_name == "x") {
      this->region_offset_list = kRegionXShapeKernelOffsets;
    } else {
      throw std::runtime_error(std::format(
          "Unsupported neighbor region shape name: {}", region_shape_name));
    }
  }

  void setPackingDisallowed(bool is_disallowed) {
    if (is_disallowed) {
      std::vector<base::LogicalCoreCompatibilitySet::Subset> subsets;
      for (const auto &l_core : this->init_state.keys()) {
        subsets.push_back({l_core});
      }
      // TODO: refactor the packing option in RoutingMode
      this->routing_mode.set(RoutingMode::LogicalCorePacking::Enforced,
                             base::LogicalCoreCompatibilitySet{subsets});
    }
  }
};

} // namespace place::placer_zoo::ls
