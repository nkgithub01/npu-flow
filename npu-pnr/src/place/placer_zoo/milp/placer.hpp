#pragma once

#include "arch/arch.hpp"
#include "base/abstraction.hpp"
#include "base/placement.hpp"
#include "base/routing.hpp"
#include "base/rr_graph.hpp"
#include "base/tf_graph.hpp"
#include "route/router.hpp"
#include "utils/misc.hpp"

namespace place::placer_zoo::milp {

using base::LogicalCore;
using base::LogicalCoreSet;
using base::PhysicalCore;
using base::PhysicalCoreSet;
using base::Placement;
using base::RoutingMode;
using base::RoutingNetList;
using base::RoutingState;
using base::RRNode;
using base::TrafficFlow;
using base::TrafficFlowEndpoint;
using base::TrafficFlowGraph;
using base::Verbosity;
using route::Router;

class MILPPlacer {
private:
  const arch::Arch &npu_;
  const route::Router &router_;

public:
  MILPPlacer() = delete;

  explicit MILPPlacer(const arch::Arch &npu, const route::Router &router)
      : npu_(npu), router_(router) {}

  Placement place(engine::Components &engine, const base::Config &cfg,
                  const base::TrafficFlowGraph &tf_graph,
                  const base::Placement &initial_placement) const {
    // Process tf graph and initial placement
    RoutingNetList netlist{tf_graph};

    // Initialize routing mode
    RoutingMode routing_mode{};

    // Time limit
    if (!engine.timer.isUnlimited()) {
      int time_limit_sec = engine.timer.getTimeoutSecs();
      engine.logger.minimal(
          std::format("[MILP Placer] Time limit set to {} seconds for solving "
                      "the MILP placement problem",
                      time_limit_sec));
      routing_mode.set(RoutingMode::TimeLimit::Limited,
                       static_cast<double>(time_limit_sec));
    }

    // Set high penalty for illegal congestion to avoid it as much as possible
    // in hope of getting a legal routing solution at early stage
    RoutingMode::HyperParameters hyper_params{};
    hyper_params.objective_congestion_penalty_scaling_factor = 1e6;
    routing_mode.set(hyper_params);

    // Enable lock capacity constraints or not
    bool enable_lock_constraints =
        cfg.getOrDefault<bool>("enable_milp_router_lock_constraints", false);
    routing_mode.set(enable_lock_constraints
                         ? RoutingMode::LockCapacityConstraint::Enforced
                         : RoutingMode::LockCapacityConstraint::Ignored);

    // Logical core packing
    // Not available for public use yet
    bool packing_disabled = cfg.getOrDefault<bool>("disable_packing", false);
    auto get_packing_disabled_compat_set = [&]() {
      std::vector<base::LogicalCoreCompatibilitySet::Subset> subsets;
      for (const auto &l_core : initial_placement.keys()) {
        subsets.push_back({l_core});
      }
      return base::LogicalCoreCompatibilitySet{subsets};
    };
    auto get_packing_enabled_compat_set = [&]() {
      return npu_.getLogicalCoreCompatibilitySet(initial_placement);
    };

    base::LogicalCoreCompatibilitySet l_core_compat_set =
        packing_disabled ? get_packing_disabled_compat_set()
                         : get_packing_enabled_compat_set();
    engine.logger.verbose(std::format(
        "[MILP Placer] Logical core compatibility set (size={}):\n{}\n",
        l_core_compat_set.getSubsets().size(),
        utils::toString(l_core_compat_set)));
    // TODO: refactor the packing option in RoutingMode
    routing_mode.set(RoutingMode::LogicalCorePacking::Enforced,
                     l_core_compat_set);

    // Logical core placement
    base::LogicalToPhysicalCoreAvailablityTable l_to_p_core_avail_table =
        npu_.getLogicalToPhysicalCoreAvailablityTable(initial_placement);
    engine.logger.verbose(
        std::format("[MILP Placer] Logical to physical core "
                    "availability table (size={}):\n{}\n",
                    l_to_p_core_avail_table.data().size(),
                    utils::toString(l_to_p_core_avail_table)));
    routing_mode.set(RoutingMode::LogicalCorePlacement::Explorable,
                     l_to_p_core_avail_table);

    // Router logging
    // TODO: consider merging the logging level of router with the engine logger
    routing_mode.set(RoutingMode::LoggingLevel::Verbose);

    // MILP placement main routine
    const auto routing_state = router_.route(engine, routing_mode, netlist);
    if (const auto legality = routing_state.getLegality();
        legality != RoutingState::Legality::Legal) {
      throw std::runtime_error(std::format(
          "Routing resulted in illegal routing state with legality = {}",
          utils::toString(legality)));
    }

    const auto &route_trees = routing_state.getRouteTrees();
    if (!route_trees.has_value()) {
      throw std::runtime_error(
          std::format("Routing cost = {}\nIf not infinite, expect route trees "
                      "to be present after routing",
                      routing_state.getRoutingCost()));
    }

    engine.logger.verbose([&](auto &log) {
      log << "Route trees:\n";
      for (const auto &tree : route_trees.value()) {
        log << tree.getAssociatedInputNet().getName() << ":\n";
        for (const auto &edge : tree.getEdges()) {
          log << "  " << edge.getName() << "\n";
        }
      }
    });

    return routing_state.getLogicalToPhysicalCoreMapping().value();
  }
};

} // namespace place::placer_zoo::milp
