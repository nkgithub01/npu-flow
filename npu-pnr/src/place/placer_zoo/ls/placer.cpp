#include "place/placer_zoo/ls/placer.hpp"

#include "base/placement.hpp"
#include "base/routing/routing_state.hpp"
#include "engine/component.hpp"
#include "utils/misc.hpp"

namespace place::placer_zoo::ls {

// Conservative mode: try routing for each logical core's region placement and
// return the best routing state found
base::RoutingState LSPlacer::runConservativeRouting(
    engine::Components &engine,
    const base::LogicalToPhysicalCoreMapping &initial_core_mapping,
    Config &cfg) const {
  float best_cost = std::numeric_limits<float>::max();
  base::RoutingState best_state =
      RoutingState(cfg.routing_mode, RoutingState::Legality::Fatal,
                   RoutingState::kInvalidRoutingCost);

  for (const auto &l_core : initial_core_mapping.keys()) {
    // Extend placement for each logical core to region placement
    base::RegionPlacement extended = npu_.extendPlacementToRegionPlacement(
        initial_core_mapping, cfg.region_offset_list, {l_core});

    engine.logger.verbose(std::format("[LS] Routing trial for logical core {} "
                                      "with region placement:\n{}\n",
                                      l_core.getName(),
                                      utils::toString(extended)));

    cfg.routing_mode.set(RoutingMode::LogicalCorePlacement::Explorable,
                         base::cast(extended));
    cfg.routing_mode.set(RoutingMode::TimeLimit::Limited,
                         engine.timer.secondsRemaining());

    if (const auto state = router_.route(engine, cfg.routing_mode, cfg.netlist);
        // Choose the best legal or congested state
        state.getLegality() != RoutingState::Legality::Fatal &&
        state.getRoutingCost() < best_cost) {
      best_cost = state.getRoutingCost();
      best_state = state;
    }

    if (engine.timer.hasExpired()) {
      break;
    }
  }

  return best_state;
}

// Aggressive mode: try routing once with all logical cores exploring their
// neighborhood regions and return the routing state
base::RoutingState LSPlacer::runAggressiveRouting(
    engine::Components &engine,
    const base::LogicalToPhysicalCoreMapping &initial_core_mapping,
    Config &cfg) const {
  base::RegionPlacement extended = npu_.extendPlacementToRegionPlacement(
      initial_core_mapping, cfg.region_offset_list,
      initial_core_mapping.keys());
  engine.logger.verbose(std::format(
      "[LS] Routing for ALL logical cores with region placement:\n{}\n",
      utils::toString(extended)));
  cfg.routing_mode.set(RoutingMode::LogicalCorePlacement::Explorable,
                       base::cast(extended));
  cfg.routing_mode.set(RoutingMode::TimeLimit::Limited,
                       engine.timer.secondsRemaining());
  // For debugging, use fixed placement
  // cfg.routing_mode.set(RoutingMode::LogicalCorePlacement::Fixed,
  //                      base::cast(initial_core_mapping));
  return router_.route(engine, cfg.routing_mode, cfg.netlist);
}

Placement LSPlacer::runLocalSearch(engine::Components &engine,
                                   Config &cfg) const {
  float best_cost = std::numeric_limits<float>::max();
  base::LogicalToPhysicalCoreMapping best_mapping = cfg.init_state;

  // Used for hill climbing
  std::vector<base::LogicalToPhysicalCoreMapping> blacklist_solutions;
  // +1 for solvable iteration, +0.5 for unsolvable iteration (resume by
  // shuffling the placement)
  float num_consecutive_iters_no_best_cost_improvement = 0;
  base::LogicalToPhysicalCoreMapping oracle_mapping = best_mapping;

  engine.logger.minimal(
      std::format("[LS] Starting Local Search placer: max_iters: {}, early "
                  "exit after {} iters without best cost improvement.\n",
                  cfg.max_iterations,
                  cfg.max_consecutive_iterations_no_best_cost_improvement));
  engine.logger.normal(std::format("[LS] Initial placement:\n{}\n",
                                   utils::toString(cfg.init_state)));

  for (int i = 0; i < cfg.max_iterations &&
                  num_consecutive_iters_no_best_cost_improvement <
                      cfg.max_consecutive_iterations_no_best_cost_improvement &&
                  !engine.timer.hasExpired();
       i++) {

    engine.logger.normal(std::format("[LS] Starting iteration {}\n", i));

    base::RoutingState routing_state =
        cfg.mode == Config::Mode::Conservative
            ? runConservativeRouting(engine, oracle_mapping, cfg)
            : runAggressiveRouting(engine, oracle_mapping, cfg);

    auto legality = routing_state.getLegality();

    if (legality == RoutingState::Legality::Fatal) {
      engine.logger.normal(std::format(
          "[LS] Encountering fatal routing legality at iteration {}. Shuffling "
          "placement to resume. Best cost found so far: {:.2f}\n",
          i, best_cost));
      // TODO: double check if the random state is the same everytime the
      // std::mt19937 is initialized with the same seed
      oracle_mapping = npu_.getLegalPlacementWithRandomSampling(
          oracle_mapping,
          // +1 to differ from the random seed used for initial placement if any
          cfg.random_seed + i + 1);
      num_consecutive_iters_no_best_cost_improvement += 0.25;
      continue;
    }

    auto cost = routing_state.getRoutingCost();
    auto core_mapping = routing_state.getLogicalToPhysicalCoreMapping().value();

    engine.logger.normal(
        std::format("[LS] Routing cost {:.2f} ({}), prev best: {:.2f}\n", cost,
                    utils::toString(legality), best_cost));
    engine.logger.verbose(
        std::format("[LS] Oracle placement at iteration {}:\n{}\n", i,
                    utils::toString(core_mapping)));

    // Only update best mapping if the new one is legal and has better cost
    if (best_cost > cost && legality == RoutingState::Legality::Legal) {
      engine.logger.normal(std::format(
          "[LS] Updating BEST legal cost {:.2f} -> {:.2f}\n", best_cost, cost));
      best_cost = cost;
      best_mapping = core_mapping;
      num_consecutive_iters_no_best_cost_improvement = 0;
    } else {
      num_consecutive_iters_no_best_cost_improvement += 1;
    }

    blacklist_solutions.push_back(core_mapping);
    cfg.routing_mode.set(RoutingMode::LogicalCorePlacementBlacklist::Enforced,
                         blacklist_solutions);
    oracle_mapping = core_mapping;
  }

  if (engine.timer.hasExpired()) {
    engine.logger.minimal("[LS] Timer expired during Local Search placer.\n");
  }

  engine.logger.minimal(std::format(
      "[LS] Finished Local Search placer. Best cost found: {:.4f}\n",
      best_cost));

  return best_mapping;
}

} // namespace place::placer_zoo::ls
