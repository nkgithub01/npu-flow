#pragma once

#include "base/routing.hpp"
#include "base/rr_graph.hpp"
#include "engine/component.hpp"

namespace route::router_zoo::milp {

class MILPRouter {

private:
  base::RRGraph graph_;

public:
  MILPRouter() = delete;
  explicit MILPRouter(base::RRGraph graph) : graph_(graph) {}

  base::RoutingState route(engine::Components &, const base::RoutingMode &,
                           const base::RoutingNetList &) const;
};

} // namespace route::router_zoo::milp
