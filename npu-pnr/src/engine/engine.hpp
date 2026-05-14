#pragma once

#include "arch/arch.hpp"
#include "arch/registry.hpp"
#include "base/engine.hpp"
#include "base/routing/routing_state.hpp"
#include "base/rr_graph.hpp"
#include "engine/logger.hpp"
#include "place/placer.hpp"
#include "place/registry.hpp"
#include "route/registry.hpp"
#include "route/router.hpp"

namespace engine {

struct PnRState {
  base::TrafficFlowGraph tf_graph;
  base::Placement placement;
  base::RoutingState routing;
};

class PnREngine {
private:
  arch::Arch arch_;
  route::Router router_;
  place::Placer placer_;

  engine::Logger logger_; // TODO: use async logger

  base::Config cfg_;

public:
  PnREngine() = delete;
  PnREngine(base::Config cfg)
      : arch_(std::move(arch::createArch(cfg.get<base::Config>("arch")))),
        router_(std::move(route::createRouter(cfg.get<base::Config>("router"),
                                              arch_.getRRGraph()))),
        placer_(std::move(place::createPlacer(cfg.get<base::Config>("placer"),
                                              arch_, router_))),
        logger_(cfg.get<base::Config>("logger")), cfg_(std::move(cfg)) {}

  PnRState run(base::PnRPlacedNetlist placed_netlist) const;

  PnRState run(base::PnRNetlistReader reader) const {
    return run(std::move(arch_.parseNetlist(std::move(reader))));
  }

  base::PnRNetlistWriter
  write(const base::PnRPlacedNetlist &placed_netlist) const {
    return arch_.dumpNetlist(placed_netlist);
  }

  base::PnRNetlistWriter write(const PnRState &state) const {
    return arch_.dumpNetlist({state.tf_graph, state.placement}, state.routing);
  }

  // Expose RRGraph for external use (e.g., debugging, visualization)
  base::RRGraph getRRGraph() const { return arch_.getRRGraph(); }
};

} // namespace engine
