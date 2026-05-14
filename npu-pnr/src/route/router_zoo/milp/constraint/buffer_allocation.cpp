#include "route/router_zoo/milp/constraint.hpp"

#include <format>

#include "base/buffer_alloc.hpp"
#include "base/routing.hpp"
#include "base/rr_graph.hpp"

namespace route::router_zoo::milp {

inline math_opt::Variable
reductionOr(math_opt::Model &model,
            const std::vector<math_opt::Variable> &vars) {
  // Create an auxiliary variable that is 1 if any of the vars is 1 and 0 if all
  // of the vars are 0 (i.e., logical OR)
  math_opt::Variable reduction = model.AddContinuousVariable(0.0, 1.0);
  model.AddLinearConstraint(reduction <= math_opt::Sum(vars));
  for (const auto &var : vars) {
    model.AddLinearConstraint(reduction >= var);
  }
  return reduction;
}

inline std::pair<std::vector<math_opt::Variable>,
                 std::vector<base::RoutingNet::BufferDemand>>
getLogicalCoreBufferDemandAtPhysicalCore(
    math_opt::Model &model, const NetEdgeToVarLookup &x,
    const base::LogicalCore &l_core, const base::PhysicalCore &p_core,
    const base::RRGraph &graph, const base::RoutingNetList &netlist,
    const base::RoutingMode::NetLink net_link_mode) {

  std::vector<math_opt::Variable> vars;
  std::vector<base::RoutingNet::BufferDemand> coeffs;

  const std::vector<base::RoutingNet> &nets_at_l_core =
      net_link_mode == base::RoutingMode::NetLink::Ignored
          ? netlist.getNetsAtLogicalCore(l_core)
          : netlist.getNetWithNoLinkCenterAtLogicalCore(l_core);

  for (const auto &net : nets_at_l_core) {
    { // Sink side
      std::vector<math_opt::Variable> or_reduce_vars;
      for (const auto &entering_edge :
           graph.getEdgesEnteringNode(p_core.getSink())) {
        if (base::isBufferNeeded(base::BufferType::Sink, *entering_edge)) {
          or_reduce_vars.push_back(x.at({net, *entering_edge}));
        }
      }
      if (!or_reduce_vars.empty()) {
        vars.push_back(reductionOr(model, or_reduce_vars));
        coeffs.push_back(net.getBufferDemand(l_core));
      }
    }
    { // Source side
      std::vector<math_opt::Variable> or_reduce_vars;
      for (const auto &leaving_edge :
           graph.getEdgesLeavingNode(p_core.getSource())) {
        if (base::isBufferNeeded(base::BufferType::Source, *leaving_edge)) {
          or_reduce_vars.push_back(x.at({net, *leaving_edge}));
        }
      }
      if (!or_reduce_vars.empty()) {
        vars.push_back(reductionOr(model, or_reduce_vars));
        coeffs.push_back(net.getBufferDemand(l_core));
      }
    }
  }

  if (net_link_mode != base::RoutingMode::NetLink::Ignored) {
    for (const auto &link : netlist.getNetLinksAtLogicalCore(l_core)) {
      const auto &from_nets = link.getFromNets();
      const auto &to_nets = link.getToNets();

      std::vector<math_opt::Variable> or_reduce_vars;

      for (const auto &entering_edge :
           graph.getEdgesEnteringNode(p_core.getSink())) {
        if (base::isBufferNeeded(base::BufferType::Sink, *entering_edge)) {
          for (const auto &net : from_nets) {
            or_reduce_vars.push_back(x.at({net, *entering_edge}));
          }
        }
      }
      for (const auto &leaving_edge :
           graph.getEdgesLeavingNode(p_core.getSource())) {
        if (base::isBufferNeeded(base::BufferType::Source, *leaving_edge)) {
          for (const auto &net : to_nets) {
            or_reduce_vars.push_back(x.at({net, *leaving_edge}));
          }
        }
      }

      if (!or_reduce_vars.empty()) {
        vars.push_back(reductionOr(model, or_reduce_vars));
        coeffs.push_back(
            netlist.getNetLinkBufferDemandAtLogicalCore(link, l_core));
      }
    }
  }

  return {vars, coeffs};
}

void addBufferAllocationConstraints(
    math_opt::Model &model, const LogicalPhysicalCoreToVarLookup &m,
    const LogicalPhysicalCoreToVarLookup &lock, const NetEdgeToVarLookup &x,
    const base::RRGraph &graph, const base::RoutingNetList &netlist,
    const base::PhysicalToLogicalCoreAvailablityTable &p_to_l_core_avail_table,
    const base::RoutingMode::MemoryCapacityConstraint mem_cap_constraint_mode,
    const base::RoutingMode::LockCapacityConstraint lock_cap_constraint_mode,
    const base::RoutingMode::NetLink net_link_mode) {

  // Cannot have a multicast net that multiple targets are mapped to the same
  // logical core (therefore the same physical core)

  using base::LogicalCore;
  using base::PhysicalCore;
  using base::RREdge;
  using base::RRNode;

  if (mem_cap_constraint_mode ==
      base::RoutingMode::MemoryCapacityConstraint::Enforced) {
    // 1. Memory capacity constraints
    for (const auto &[p_core, l_core_set] : p_to_l_core_avail_table.data()) {
      std::vector<math_opt::Variable> m_vars;
      for (const auto &l_core : l_core_set) {
        m_vars.push_back(m.at({l_core, p_core}));
      }
      model.AddLinearConstraint(
          // Sum_{l in L_p} m_{l,p} <= MemoryCapacity(p)
          math_opt::Sum(m_vars) <= int(p_core.getMemoryCapacity()),
          std::format("Memory capacity constraint for {}", p_core.getName()));
    }

    // 2. Auxiliary equality constraints
    for (const auto &[p_core, l_core_set] : p_to_l_core_avail_table.data()) {
      for (const auto &l_core : l_core_set) {
        const auto [vars, bufs] = getLogicalCoreBufferDemandAtPhysicalCore(
            model, x, l_core, p_core, graph, netlist, net_link_mode);
        std::vector<int> coeffs;
        for (const auto &buf : bufs) {
          coeffs.push_back(static_cast<int>(buf.getTotalBytes()));
        }
        if (!vars.empty()) {
          model.AddLinearConstraint(
              m.at({l_core, p_core}) == math_opt::InnerProduct(vars, coeffs),
              std::format("Memory demand for {} at {}", l_core.getName(),
                          p_core.getName()));
        }
      }
    }
  }

  if (lock_cap_constraint_mode ==
      base::RoutingMode::LockCapacityConstraint::Enforced) {
    // 3. Lock capacity constraints
    for (const auto &[p_core, l_core_set] : p_to_l_core_avail_table.data()) {
      std::vector<math_opt::Variable> lock_vars;
      for (const auto &l_core : l_core_set) {
        lock_vars.push_back(lock.at({l_core, p_core}));
      }
      model.AddLinearConstraint(
          // Sum_{l in L_p} lock_{l,p} <= LockCapacity(p)
          math_opt::Sum(lock_vars) <= int(p_core.getLockCapacity()),
          std::format("Lock capacity constraint for {}", p_core.getName()));
    }

    // 4. Auxiliary equality constraints
    for (const auto &[p_core, l_core_set] : p_to_l_core_avail_table.data()) {
      for (const auto &l_core : l_core_set) {
        const auto [vars, bufs] = getLogicalCoreBufferDemandAtPhysicalCore(
            model, x, l_core, p_core, graph, netlist, net_link_mode);
        std::vector<int> coeffs;
        for (const auto &buf : bufs) {
          // Note: each depth of buffer requires two locks
          coeffs.push_back(static_cast<int>(buf.depth) * 2);
        }
        if (!vars.empty()) {
          model.AddLinearConstraint(
              lock.at({l_core, p_core}) == math_opt::InnerProduct(vars, coeffs),
              std::format("Lock demand for {} at {}", l_core.getName(),
                          p_core.getName()));
        }
      }
    }
  }

  // 3. Enforce circuit-switched routing for linked nets
  if (net_link_mode ==
      base::RoutingMode::NetLink::ForceLinkedNetsToCircuitSwitching) {
    // For each net link, enforce that all from/to nets in the net link use only
    // circuit-switching edges when routed through the physical cores.
    for (const auto &link : netlist.getNetLinks()) {
      for (const auto &p_core :
           std::views::keys(p_to_l_core_avail_table.data())) {
        for (const auto &net : link.getFromNets()) {
          for (const auto &entering_edge :
               graph.getEdgesEnteringNode(p_core.getSink())) {
            if (entering_edge->getType() !=
                base::RREdgeType::CircuitSwitching) {
              model.AddLinearConstraint(x.at({net, *entering_edge}) == 0);
            }
          }
        }
        for (const auto &net : link.getToNets()) {
          for (const auto &leaving_edge :
               graph.getEdgesLeavingNode(p_core.getSource())) {
            if (leaving_edge->getType() != base::RREdgeType::CircuitSwitching) {
              model.AddLinearConstraint(x.at({net, *leaving_edge}) == 0);
            }
          }
        }
      }
    }
  }
}

} // namespace route::router_zoo::milp
