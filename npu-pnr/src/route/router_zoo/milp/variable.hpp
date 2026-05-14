#pragma once

#include <format>
#include <iostream>

#include "ortools/math_opt/cpp/math_opt.h"

#include "base/abstraction.hpp"
#include "base/routing.hpp"
#include "base/rr_graph.hpp"

namespace math_opt = ::operations_research::math_opt;

namespace route::router_zoo::milp {

// If using pointers for the key, the results should be the same but the
// routing might not be deterministic.
// TODO: use unordered_map for variable lookups
using NetNodeToVarLookup =
    std::map<std::pair<base::RoutingNet, base::RRNode>, math_opt::Variable>;
using NetEdgeToVarLookup =
    std::map<std::pair<base::RoutingNet, base::RREdge>, math_opt::Variable>;

using EdgeToVarLookup = utils::Lookup<base::RREdge, math_opt::Variable>;

using LogicalPhysicalCoreToVarLookup =
    std::map<std::pair<base::LogicalCore, base::PhysicalCore>,
             math_opt::Variable>;

inline NetEdgeToVarLookup createRREdgeSelectionBinaryDecisionVariables(
    math_opt::Model &model, const base::RRGraph &graph,
    const base::RoutingNetList &netlist, bool is_debug_mode) {
  NetEdgeToVarLookup x;
  for (const auto &net : netlist.getNets()) {
    for (const auto &edge : graph.getEdgePtrs()) {
      // TODO: avoid using pointers
      x.emplace(std::make_pair(net, *edge),
                model.AddBinaryVariable(
                    std::format("x_{}_{}", net.getName(), edge->getName())));
      if (is_debug_mode) {
        std::cout << "Added variable: " << x.at({net, *edge}).name()
                  << std::endl;
      }
    }
  }
  return x;
}

inline NetEdgeToVarLookup createRouteFlowContinuousDecisionVariables(
    math_opt::Model &model, const base::RRGraph &graph,
    const base::RoutingNetList &netlist, bool is_debug_mode) {
  NetEdgeToVarLookup f;
  for (const auto &net : netlist.getNets()) {
    for (const auto &edge : graph.getEdgePtrs()) {
      f.emplace(std::make_pair(net, *edge),
                model.AddContinuousVariable(
                    0.0, net.getNumTargetCores(), // flow range: [0, |T^k|]
                    std::format("f_{}_{}", net.getName(), edge->getName())));
      if (is_debug_mode) {
        std::cout << "Added variable: " << f.at({net, *edge}).name()
                  << std::endl;
      }
    }
  }
  return f;
}

inline EdgeToVarLookup createEdgeCongestionContinuousAuxiliaryVariables(
    math_opt::Model &model, const base::RRGraph &graph, bool is_debug_mode) {
  EdgeToVarLookup c;
  for (const auto &edge : graph.getEdgePtrs()) {
    if (edge->getCapacity().isInfinite()) {
      continue; // No congestion variable needed for infinite capacity edges
    }
    c.emplace(*edge, model.AddContinuousVariable(
                         0.0, base::RREdgeCapacity::kInfiniteCapacity,
                         std::format("c_{}", edge->getName())));
    if (is_debug_mode) {
      std::cout << "Added variable: " << c.at(*edge).name() << std::endl;
    }
  }
  return c;
}

inline LogicalPhysicalCoreToVarLookup createCoreMappingBinaryDecisionVariables(
    math_opt::Model &model, const base::LogicalCoreSet &l_cores,
    const base::PhysicalCoreSet &p_cores, bool is_debug_mode) {
  LogicalPhysicalCoreToVarLookup d;
  for (const auto &l : l_cores) {
    for (const auto &p : p_cores) {
      d.emplace(std::make_pair(l, p),
                model.AddBinaryVariable(
                    std::format("d_{}_{}", l.getName(), p.getName())));
      if (is_debug_mode) {
        std::cout << "Added variable: " << d.at({l, p}).name() << std::endl;
      }
    }
  }
  return d;
}

inline LogicalPhysicalCoreToVarLookup
createMemoryUsageContinuousAuxiliaryVariables(
    math_opt::Model &model, const base::LogicalCoreSet &l_cores,
    const base::PhysicalCoreSet &p_cores, bool is_debug_mode) {
  LogicalPhysicalCoreToVarLookup m;
  for (const auto &l : l_cores) {
    for (const auto &p : p_cores) {
      m.emplace(std::make_pair(l, p),
                model.AddContinuousVariable(
                    0.0, p.getMemoryCapacity(),
                    std::format("m_{}_{}", l.getName(), p.getName())));
      if (is_debug_mode) {
        std::cout << "Added variable: " << m.at({l, p}).name() << std::endl;
      }
    }
  }
  return m;
}

inline LogicalPhysicalCoreToVarLookup
createLockUsageContinuousAuxiliaryVariables(
    math_opt::Model &model, const base::LogicalCoreSet &l_cores,
    const base::PhysicalCoreSet &p_cores, bool is_debug_mode) {
  LogicalPhysicalCoreToVarLookup lock;
  for (const auto &l : l_cores) {
    for (const auto &p : p_cores) {
      lock.emplace(std::make_pair(l, p),
                   model.AddContinuousVariable(
                       0.0, p.getLockCapacity(),
                       std::format("lock_{}_{}", l.getName(), p.getName())));
      if (is_debug_mode) {
        std::cout << "Added variable: " << lock.at({l, p}).name() << std::endl;
      }
    }
  }
  return lock;
}

} // namespace route::router_zoo::milp
