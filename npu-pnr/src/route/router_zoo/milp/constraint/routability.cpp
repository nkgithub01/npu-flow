#include "route/router_zoo/milp/constraint.hpp"

#include <format>

#include "base/routing.hpp"
#include "base/rr_graph.hpp"
#include "utils/misc.hpp"

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

void addRoutabilityConstraints(
    math_opt::Model &model, const NetEdgeToVarLookup &x,
    const NetEdgeToVarLookup &f, const EdgeToVarLookup &c,
    const LogicalPhysicalCoreToVarLookup &d, const base::RRGraph &graph,
    const base::RoutingNetList &netlist, const base::LogicalCoreSet &l_cores,
    const base::PhysicalCoreSet &p_cores,
    const utils::Lookup<base::RRNode, base::PhysicalCore> &rr_node_to_p_core) {

  using base::PhysicalCore;
  using base::RREdge;
  using base::RRNode;

  // 1. Flow conservation constraints
  auto routability_equality_constraint =
      [&](const base::RoutingNet &net, const RRNode &node,
          math_opt::LinearExpression rhs_expr) -> void {
    std::vector<math_opt::Variable> vars;
    std::vector<int> coeffs;
    for (const auto &entering_edge : graph.getEdgesEnteringNode(node)) {
      // TODO: consider refactoring using LinearExpression +=
      vars.push_back(f.at({net, *entering_edge}));
      coeffs.push_back(1);
    }
    for (const auto &leaving_edge : graph.getEdgesLeavingNode(node)) {
      vars.push_back(f.at({net, *leaving_edge}));
      coeffs.push_back(-1);
    }
    model.AddLinearConstraint(
        // Flow entering (node) - Flow leaving (node) == rhs_expr
        math_opt::InnerProduct(vars, coeffs) == rhs_expr,
        std::format("Routability constraint for {} on {}", net.getName(),
                    node.getName()));
    // TODO: find a clean way to print debug info
  };

  for (const auto &net : netlist.getNets()) {
    for (const auto &node_ptr : graph.getNodePtrs()) {
      const RRNode &node = *node_ptr;
      if (!rr_node_to_p_core.contains(node)) {
        // Intermediate node that does not correspond to any physical core
        routability_equality_constraint(net, node, 0);
        continue;
      }
      const PhysicalCore &p_core = rr_node_to_p_core.at(node);
      if (p_core.getSource() == node) {
        // Source node
        routability_equality_constraint(net, node,
                                        -int(net.getNumTargetCores()) *
                                            d.at({net.getStartCore(), p_core}));
      } else if (p_core.getSink() == node) {
        // Sink node
        std::vector<math_opt::Variable> d_vars;
        for (const auto &target_core : net.getTargetCores()) {
          d_vars.push_back(d.at({target_core, p_core}));
        }
        // The reason we use sum here is that multiple target logical cores
        // can be mapped to the same physical core
        routability_equality_constraint(net, node, math_opt::Sum(d_vars));
      } else {
        // Other node
        routability_equality_constraint(net, node, 0);
      }
    }
  }

  // 2. Tree-pattern constraints
  for (const auto &net : netlist.getNets()) {

    auto tree_pattern_inequality_constraint =
        [&](const RRNode &lhs_node) -> void {
      std::vector<math_opt::Variable> vars;
      for (const auto &entering_edge : graph.getEdgesEnteringNode(lhs_node)) {
        vars.push_back(x.at({net, *entering_edge}));
      }
      model.AddLinearConstraint(
          // Edges entering (lhs_node) <= 1
          math_opt::Sum(vars) <= 1,
          std::format("Tree-pattern constraint for {} on {}", net.getName(),
                      lhs_node.getName()));
    };

    for (const auto &node : graph.getNodePtrs()) {
      tree_pattern_inequality_constraint(*node);
    }
  }

  // 3. Flow on chosen edge constraints
  for (const auto &net : netlist.getNets()) {

    auto flow_on_chosen_edge_inequality_constraint =
        [&](const RREdge &lhs_edge) -> void {
      model.AddLinearConstraint(
          // f_e^k <= |T^k| * x_e^k
          f.at({net, lhs_edge}) -
                  int(net.getNumTargetCores()) * x.at({net, lhs_edge}) <=
              0,
          std::format("Flow on chosen edge constraints for {} on {}",
                      net.getName(), lhs_edge.getName()));
    };

    for (const auto &edge : graph.getEdgePtrs()) {
      flow_on_chosen_edge_inequality_constraint(*edge);
    }
  }

  // 4. Edge congestion (overuse) constraints
  //// 4.1 Exclusive edges (non-shared)
  for (const auto &edge : graph.getEdgePtrs()) {
    if (edge->getCapacity().isInfinite()) {
      continue;
    }
    if (edge->getCapacity().is(base::RREdgeCapacity::SharingBehavior::Shared)) {
      continue;
    }
    std::vector<math_opt::Variable> vars;
    for (const auto &net : netlist.getNets()) {
      vars.push_back(x.at({net, *edge}));
    }
    model.AddLinearConstraint(
        c.at(*edge) >= math_opt::Sum(vars) - edge->getCapacity().getValue(),
        std::format("Edge congestion constraint for {} (exclusive)",
                    edge->getName()));
  }
  //// 4.2 Shared edges (packet switching consolidation handled)
  for (const auto &edge_group : graph.getSharedEdgeGroups()) {
    if (edge_group.front()->getCapacity().isInfinite()) {
      throw std::runtime_error(
          "Shared edge group contains infinite capacity edge.");
    }
    size_t rep_capacity = edge_group.front()->getCapacity().getValue();
    if (std::ranges::any_of(edge_group, [&](const auto &edge) {
          return edge->getCapacity().getValue() != rep_capacity;
        })) {
      throw std::runtime_error(
          "Inconsistent capacities among edges in shared edge group.");
    }

    std::vector<math_opt::Variable> vars;
    for (const auto &edge : edge_group) {
      if (edge->getCapacity().is(
              base::RREdgeCapacity::ConsolidationBehavior::Consolidated)) {
        // Packet switching edge: use OR reduction
        std::vector<math_opt::Variable> or_reduce_vars;
        for (const auto &net : netlist.getNets()) {
          or_reduce_vars.push_back(x.at({net, *edge}));
        }
        vars.push_back(reductionOr(model, or_reduce_vars));
      } else {
        // Circuit switching edge: use sum directly
        for (const auto &net : netlist.getNets()) {
          vars.push_back(x.at({net, *edge}));
        }
      }
    }

    for (size_t i = 0; i < edge_group.size(); ++i) {
      if (i == 0) {
        model.AddLinearConstraint(
            c.at(*edge_group.at(i)) >= math_opt::Sum(vars) - rep_capacity,
            std::format("Shared edge congestion constraint for {}",
                        edge_group.at(i)->getName()));
      } else {
        // TODO: ensure no edge is in more than one shared edge group
        model.AddLinearConstraint(
            c.at(*edge_group.at(i)) == c.at(*edge_group.at(0)),
            std::format("Shared edge congestion constraint for {}",
                        edge_group.at(i)->getName()));
      }
    }
  }
  //// 4.3 The route tree can only use one type of edge
  for (const auto &net : netlist.getNets()) {
    utils::Lookup<base::RREdgeType, std::vector<math_opt::Variable>>
        edge_type_to_x_vars;
    for (const auto &edge : graph.getEdgePtrs()) {
      edge_type_to_x_vars[edge->getType()].push_back(x.at({net, *edge}));
    }

    std::vector<math_opt::Variable> or_reduction_vars;
    for (const auto &[type, x_vars] : edge_type_to_x_vars) {
      or_reduction_vars.push_back(reductionOr(model, x_vars));
    }

    model.AddLinearConstraint(
        // Sum over edge types of (OR reduction over x_e^k for edges of that
        // type) <= 1
        math_opt::Sum(or_reduction_vars) <= 1,
        std::format("Edge type exclusivity constraint for {}", net.getName()));
  }
}

} // namespace route::router_zoo::milp
