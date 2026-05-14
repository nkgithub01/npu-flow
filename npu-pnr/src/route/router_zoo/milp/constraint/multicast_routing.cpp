#include "base/rr_graph.hpp"
#include "route/router_zoo/milp/constraint.hpp"

#include <format>

#include "base/abstraction.hpp"

namespace route::router_zoo::milp {

void addMulticastRoutingConstraints(
    math_opt::Model &model, const NetEdgeToVarLookup &x,
    const base::RRGraph &graph, const base::RoutingNetList &netlist,
    const base::LogicalToPhysicalCoreAvailablityTable &l_to_p_core_avail_table,
    const base::RoutingMode::AllowedRREdgeTypeOrderedSet &allowed_edge_types) {

  using base::LogicalCore;
  using base::PhysicalCore;
  using base::RREdgeType;
  using base::RRNode;

  // TODO: when the backend routing pass supports multicast routing over a mix
  // of switching types, update this function based on the implementation in the
  // previous commits.
  if (allowed_edge_types.size() != 1 || allowed_edge_types.at(0).size() != 1 ||
      !allowed_edge_types.at(0).contains(RREdgeType::CircuitSwitching)) {
    throw std::runtime_error("Multicast routing only support circuit-switching "
                             "edges in current MILP router.");
  }

  for (const auto &net : netlist.getNets()) {
    if (!net.isMulticast()) {
      continue;
    }

    for (const auto &edge : graph.getEdgePtrs()) {
      if (edge->getType() != RREdgeType::CircuitSwitching) {
        model.AddLinearConstraint(
            x.at({net, *edge}) == 0,
            std::format("Multicast routing constraint for net {} on edge {}",
                        net.getName(), edge->getName()));
      }
    }
  }
}

} // namespace route::router_zoo::milp
