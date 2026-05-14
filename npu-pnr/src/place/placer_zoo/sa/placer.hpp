#pragma once

#include "arch/arch.hpp"
#include "base/abstraction.hpp"
#include "base/placement.hpp"
#include "base/routing.hpp"
#include "base/routing/routing_mode.hpp"
#include "base/tf_graph.hpp"
#include "engine/component.hpp"
#include "place/placer_zoo/sa/config.hpp"
#include "place/placer_zoo/sa/schedule.hpp"
#include "route/router.hpp"
#include <utility>

namespace place::placer_zoo::sa {

using base::LogicalCore;
using base::LogicalCoreCompatibilitySet;
using base::PhysicalCore;
using base::Placement;
using base::RoutingNetList;
using base::RoutingState;
using base::TrafficFlowEndpoint;
using base::TrafficFlowGraph;
using base::Verbosity;
using route::Router;

class SAPlacer {
private:
  const arch::Arch &npu_;
  const route::Router &router_;

private:
  void generateNewStateWithRandomSampling(
    Config &cfg,
    const Placement &prev_state,
    Placement &new_state
  ) const {
    npu_.generateInitialPlacement(
      prev_state,
      new_state,
      cfg.netlist,
      cfg.random_seed
    );
  }

  void generateNewState(
    Config &cfg,
    Placement &prev_state,
    Placement &new_state,
    std::vector<base::LogicalCore> &moved_l_cores
  ) const {
    npu_.generateLegalPlacementWithBoundedRandomSwap(
      prev_state,
      new_state,
      moved_l_cores,
      cfg.random_seed + (++cfg.current_move_attempts),
      cfg.sa_swap_bound
    );
  }

  TemperatureSchedule getDynamicTemperatureSchedule(engine::Components &engine,
                                                    Config &cfg) const;

  Placement runSimulatedAnnealing(engine::Components &engine,
                                  Config &cfg) const;

public:
  SAPlacer() = delete;

  explicit SAPlacer(const arch::Arch &npu, const route::Router &router)
      : npu_(npu), router_(router) {}

  Placement place(engine::Components &engine, const base::Config &cfg,
                  const base::TrafficFlowGraph &tf_graph,
                  const base::Placement &init_placement) const {
    const base::LogicalToPhysicalCoreAvailablityTable &full_avail_table =
        npu_.getLogicalToPhysicalCoreAvailablityTable(init_placement);

    auto parsed = Config(tf_graph, init_placement, full_avail_table).parse(cfg);

    generateNewStateWithRandomSampling(parsed, init_placement, parsed.init_state);

    // Setup cost estimator if needed
    if (parsed.cost_estimator == "bb" || parsed.cost_estimator == "prob") {
      std::unordered_map<PhysicalCore, std::pair<int, int>, utils::NamedClassHash> p_core_to_x_y;
      for (int col = 0; col < npu_.getDimensions().getNumCols(); ++col) {
        for (int row = 0; row < npu_.getDimensions().getNumRows(); ++row) {
          p_core_to_x_y.emplace(npu_.getPhysicalCore({row, col}),
                                         std::make_pair(row, col));
        }
      }
      // TODO: use a generic cost estimator concept
      parsed.bb_cost_estimator = BoundingBoxBasedCostEstimator(
          npu_.getDimensions().getNumCols(), npu_.getDimensions().getNumRows(),
          parsed.cost_estimator == "prob",
          parsed.cost_estimator_probability_distribution,
          parsed.enable_cost_legality_estimator,
          p_core_to_x_y,
          parsed.netlist,
          parsed.init_state
      );
      // bb_cost_estimator.printBinomicalCoefficientLookupTable();
    } else if (parsed.cost_estimator == "milp") {
      // Setup MILP cost estimator's lazy evaluation mode
      parsed.milp_cost_estimator = base::RoutingMode();
      parsed.milp_cost_estimator.setLazyEvaluation();
      parsed.milp_cost_estimator.set(base::RoutingMode::LoggingLevel::Silent);
      parsed.milp_cost_estimator.set(
          parsed.packing_disabled
              ? base::RoutingMode::LogicalCorePacking::Enforced
              : base::RoutingMode::LogicalCorePacking::Ignored,
          parsed.l_core_compatibility_set);
      parsed.milp_cost_estimator.set(base::RoutingMode::TimeLimit::Limited,
                                     parsed.milp_cost_estimator_timeout_secs);
      parsed.milp_cost_estimator.set(
          parsed.milp_cost_estimator_enable_lock_constraints
              ? base::RoutingMode::LockCapacityConstraint::Enforced
              : base::RoutingMode::LockCapacityConstraint::Ignored);
    } else {
      throw std::runtime_error(std::format(
          "[SAPlacer] Unknown cost estimator type: {}", parsed.cost_estimator));
    }

    // Set the SA move range limit to the max manhattan distance on the board initially
    // NOTE: This bound must be set before the first call to generateNewState, 
    // which is used in the initial temperature estimation when dynamic scheduling is enabled. 
    // After the initial temperature is estimated, the bound will be dynamically adjusted based on the acceptance ratio.
    parsed.sa_swap_bound = npu_.getDimensions().getNumCols() + npu_.getDimensions().getNumRows();

    // Dynamic adjustment of initial temperature based on initial placement
    if (parsed.enable_dynamic_temperature_scheduling) {
      parsed.temp_schedule = getDynamicTemperatureSchedule(engine, parsed);
    }

    // Simulated annealing main loop
    return runSimulatedAnnealing(engine, parsed);
  }

  std::pair<float, base::RoutingState::Legality>
  evaluatePlacement(engine::Components &engine, Config &cfg,
                    const Placement &state) const {
    if (cfg.cost_estimator == "bb") {
      return evaluateBoundingBoxBasedCost(engine, cfg, state);
    } else if (cfg.cost_estimator == "prob") {
      return evaluateProbabilisticModelBasedCost(engine, cfg, state);
    } else if (cfg.cost_estimator == "milp") {
      return evaluateRoutingBasedCost(engine, cfg, state);
    } else {
      throw std::runtime_error(std::format(
          "[SAPlacer] Unknown cost estimator type: {}", cfg.cost_estimator));
    }
  }

  std::pair<float, base::RoutingState::Legality>
  evaluatePlacementWithDeltaChange(engine::Components &engine, Config &cfg,
                    const Placement &state, const std::vector<base::LogicalCore> &moved_l_cores) const {
    if (cfg.cost_estimator == "bb") {
      return evaluateBoundingBoxBasedCostWithPlacementDeltaChange(engine, cfg, state, moved_l_cores);
    } else if (cfg.cost_estimator == "prob") {
      return evaluateProbabilisticModelBasedCostWithPlacementDeltaChange(engine, cfg, state, moved_l_cores);
    } else if (cfg.cost_estimator == "milp") {
      return evaluateRoutingBasedCost(engine, cfg, state);
    } else {
      throw std::runtime_error(std::format(
          "[SAPlacer] Unknown cost estimator type: {}", cfg.cost_estimator));
    }
  }

  void updateEstimatorStateWithDeltaChange(engine::Components &engine, Config &cfg, const Placement &state, const std::vector<base::LogicalCore> &moved_l_cores) const {
    if (cfg.cost_estimator == "bb" || cfg.cost_estimator == "prob") {
      cfg.bb_cost_estimator.updateEstimatorStateWithPlacementDeltaChange(cfg.netlist, state, moved_l_cores);
    }
    // For MILP cost estimator, since we are using lazy evaluation mode,
    // we don't need to update any internal state for the cost estimator.
    // The MILP model will be reconstructed from scratch when evaluateRoutingBasedCost is called.
  }

private:
  std::pair<float, base::RoutingState::Legality>
  evaluateRoutingBasedCost(engine::Components &engine, Config &cfg,
                           const Placement &state) const {
    cfg.milp_cost_estimator.set(base::RoutingMode::LogicalCorePlacement::Fixed,
                                base::cast(state));
    auto routing_state =
        router_.route(engine, cfg.milp_cost_estimator, cfg.netlist);
    return {routing_state.getRoutingCost(), routing_state.getLegality()};
  }

  std::pair<float, base::RoutingState::Legality>
  evaluateBoundingBoxBasedCost(engine::Components &engine, Config &cfg, const Placement &state) const {
    return cfg.bb_cost_estimator.estimatePlacementCost(
        cfg.netlist, state);
  }

  std::pair<float, base::RoutingState::Legality>
  evaluateProbabilisticModelBasedCost(engine::Components &engine, Config &cfg, const Placement &state) const {
    return cfg.bb_cost_estimator.estimatePlacementCost(
        cfg.netlist, state);
  }

  std::pair<float, base::RoutingState::Legality>
  evaluateBoundingBoxBasedCostWithPlacementDeltaChange(engine::Components &engine, Config &cfg, const Placement &state, const std::vector<base::LogicalCore> &moved_l_cores) const {
    return cfg.bb_cost_estimator.estimatePlacementCostWithPlacementDeltaChange(
        cfg.netlist, state, moved_l_cores);
  }

  std::pair<float, base::RoutingState::Legality>
  evaluateProbabilisticModelBasedCostWithPlacementDeltaChange(engine::Components &engine, Config &cfg, const Placement &state, const std::vector<base::LogicalCore> &moved_l_cores) const {
    return cfg.bb_cost_estimator.estimatePlacementCostWithPlacementDeltaChange(
        cfg.netlist, state, moved_l_cores);
  }
};

} // namespace place::placer_zoo::sa
