#pragma once

#include "arch/arch.hpp"
#include "base/abstraction.hpp"
#include "base/common.hpp"
#include "base/placement.hpp"
#include "base/routing.hpp"
#include "base/routing/routing_mode.hpp"
#include "base/routing/routing_state.hpp"
#include "base/rr_graph.hpp"
#include "base/tf_graph.hpp"
#include "place/placer_zoo/ls/config.hpp"
#include "route/router.hpp"

namespace place::placer_zoo::ls {

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

class LSPlacer {

private:
  const arch::Arch &npu_;
  const route::Router &router_;

  RoutingState runAggressiveRouting(engine::Components &,
                                    const base::LogicalToPhysicalCoreMapping &,
                                    Config &) const;

  RoutingState
  runConservativeRouting(engine::Components &,
                         const base::LogicalToPhysicalCoreMapping &,
                         Config &) const;

  Placement runLocalSearch(engine::Components &, Config &) const;

public:
  LSPlacer() = delete;

  explicit LSPlacer(const arch::Arch &npu, const route::Router &router)
      : npu_(npu), router_(router) {}

  Placement place(engine::Components &engine, const base::Config &cfg,
                  const base::TrafficFlowGraph &tf_graph,
                  const base::Placement &initial_placement) const {
    base::LogicalCoreCompatibilitySet l_core_compat_set =
        npu_.getLogicalCoreCompatibilitySet(initial_placement);
    auto parsed =
        Config(tf_graph, initial_placement, l_core_compat_set).parse(cfg);

    if (parsed.enable_initial_placement_randomization) {
      parsed.init_state = npu_.getLegalPlacementWithRandomSampling(
          parsed.init_state, parsed.random_seed);
    }

    // Local search main loop
    return runLocalSearch(engine, parsed);
  }
};

} // namespace place::placer_zoo::ls
