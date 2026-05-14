#include "route/router_zoo/milp/router.hpp"

#include <format>

#include "base/abstraction.hpp"
#include "base/routing.hpp"
#include "base/routing/routing_mode.hpp"
#include "base/rr_graph.hpp"
#include "route/router_zoo/milp/constraint.hpp"
#include "route/router_zoo/milp/variable.hpp"
#include "utils/misc.hpp"

namespace route::router_zoo::milp {

// TODO: split route() into multiple functions to improve maintainability
base::RoutingState
MILPRouter::route(engine::Components &engine, const base::RoutingMode &mode,
                  const base::RoutingNetList &netlist) const {
  using base::LogicalCore;
  using base::PhysicalCore;
  using base::RoutingMode;
  using base::RREdgeCapacity;
  using base::RRNode;

  // TODO: add precheck to error out for unsupported modes

  if (mode.get<RoutingMode::LoggingLevel>() >=
      RoutingMode::LoggingLevel::Minimal) {
    engine.logger.minimal(std::format("[MILP Router] Routing mode {{{}}}\n",
                                      utils::toString(mode)));
  }

  math_opt::Model model("MILP Router");

  // Constants
  const base::LogicalToPhysicalCoreAvailablityTable &l_to_p_core_avail_table =
      mode.getLogicalToPhysicalCoreAvailablityTable();
  const base::PhysicalToLogicalCoreAvailablityTable p_to_l_core_avail_table =
      base::cast(l_to_p_core_avail_table);

  const base::LogicalCoreSet l_cores = l_to_p_core_avail_table.keys();
  const base::PhysicalCoreSet p_cores = p_to_l_core_avail_table.keys();

  utils::Lookup<RRNode, PhysicalCore> rr_node_to_p_core;
  for (const auto &p_core : p_cores) {
    rr_node_to_p_core.insert_or_assign(p_core.getSource(), p_core);
    rr_node_to_p_core.insert_or_assign(p_core.getSink(), p_core);
  }

  bool is_debug_mode = mode.is(RoutingMode::LoggingLevel::Debug);

  // Variables
  NetEdgeToVarLookup x = createRREdgeSelectionBinaryDecisionVariables(
      model, graph_, netlist, is_debug_mode);
  NetEdgeToVarLookup f = createRouteFlowContinuousDecisionVariables(
      model, graph_, netlist, is_debug_mode);
  EdgeToVarLookup c = createEdgeCongestionContinuousAuxiliaryVariables(
      model, graph_, is_debug_mode);
  LogicalPhysicalCoreToVarLookup d = createCoreMappingBinaryDecisionVariables(
      model, l_cores, p_cores, is_debug_mode);
  LogicalPhysicalCoreToVarLookup m;
  LogicalPhysicalCoreToVarLookup lock;
  if (!mode.is(RoutingMode::MemoryCapacityConstraint::Ignored)) {
    m = createMemoryUsageContinuousAuxiliaryVariables(model, l_cores, p_cores,
                                                      is_debug_mode);
  }
  if (!mode.is(RoutingMode::LockCapacityConstraint::Ignored)) {
    lock = createLockUsageContinuousAuxiliaryVariables(model, l_cores, p_cores,
                                                       is_debug_mode);
  }

  // Constraints
  addRoutabilityConstraints(model, x, f, c, d, graph_, netlist, l_cores,
                            p_cores, rr_node_to_p_core);
  addCoreMappingConstraints(model, d, l_cores, p_cores, l_to_p_core_avail_table,
                            mode.getLogicalCoreCompatibilitySet(),
                            mode.getLogicalToPhysicalCoreMappingBlacklist());
  addBufferAllocationConstraints(
      model, m, lock, x, graph_, netlist, p_to_l_core_avail_table,
      mode.get<RoutingMode::MemoryCapacityConstraint>(),
      mode.get<RoutingMode::LockCapacityConstraint>(),
      mode.get<RoutingMode::NetLink>());
  addMulticastRoutingConstraints(
      model, x, graph_, netlist, l_to_p_core_avail_table,
      mode.getMulticastAllowedRREdgeTypeOrderedSet());

  // Objective
  std::vector<math_opt::Variable> obj_vars;
  // TODO: should be alpha * routing cost + beta * memory cost + gamma *
  // congestion penalty
  std::vector<float> cost_coeffs;

  RoutingMode::HyperParameters hyper_params =
      mode.get<RoutingMode::HyperParameters>();

  for (const auto &net : netlist.getNets()) {
    for (const auto &edge : graph_.getEdgePtrs()) {
      obj_vars.push_back(x.at({net, *edge}));
      cost_coeffs.push_back(edge->getCost());
    }
  }
  if (!mode.is(RoutingMode::MemoryCapacityConstraint::Ignored)) {
    for (const auto &m_var : std::views::values(m)) {
      obj_vars.push_back(m_var);
      cost_coeffs.push_back(
          hyper_params.objective_memory_capacity_scaling_factor);
    }
  }
  for (const auto &[edge, c_var] : c) {
    // TODO: consider add some properties at rr edge, e.g., congestion
    // estimation behaviors
    obj_vars.push_back(c_var);
    cost_coeffs.push_back(
        hyper_params.objective_congestion_penalty_scaling_factor);
  }

  model.Minimize(math_opt::InnerProduct(obj_vars, cost_coeffs));

  math_opt::SolveArguments args;
  args.parameters.enable_output = mode.get<RoutingMode::LoggingLevel>() >=
                                  RoutingMode::LoggingLevel::Verbose;
  args.parameters.time_limit =
      mode.is(RoutingMode::TimeLimit::Limited)
          ? (mode.getTimeLimitInSeconds() * absl::Seconds(1))
          : absl::InfiniteDuration();
  args.parameters.random_seed = 0; // TODO: use a configurable random seed
  // args.parameters.presolve = math_opt::Emphasis::kOff;
  // args.parameters.scaling = math_opt::Emphasis::kOff;

  const absl::StatusOr<math_opt::SolveResult> result =
      Solve(model, math_opt::SolverType::kGscip, args);

  if (!result.status().ok()) {
    throw std::runtime_error(std::format("Failed to solve MILP problem: {}",
                                         result.status().message()));
  }

  // Results
  // TODO: extract buffer allocation into for each net
  if (result->termination.EnsureIsOptimalOrFeasible().ok()) {
    base::RoutingState::Legality legality = base::RoutingState::Legality::Legal;

    // Helper: get sum of x variables on an edge
    utils::Lookup<base::RREdge, size_t> sum_x_on_edge;
    auto get_sum_x_on_edge = [&](const base::RREdge &edge) -> size_t {
      if (!sum_x_on_edge.contains(edge)) {
        size_t sum = 0;
        for (const auto &net : netlist.getNets()) {
          // TODO: consider consolidation here for any available type that is
          // not packet switching
          if (result->variable_values().at(x.at({net, edge})) > 0.5) {
            sum += 1;
          }
        }
        sum_x_on_edge.insert_or_assign(edge, sum);
      }
      return sum_x_on_edge.at(edge);
    };

    // Helper: get utilized capacity (consolidating packet switching into one
    // unit) on a shared edge group
    auto get_utilized_capacity_on_shared_edge_group =
        [&](const base::RREdge &edge) -> size_t {
      size_t utilized_capacity = 0;
      for (const auto shared_edge : graph_.getSharedEdgeGroupAtEdge(edge)) {
        const size_t sum_x = get_sum_x_on_edge(*shared_edge);
        if (sum_x > 0 &&
            shared_edge->getCapacity().is(
                base::RREdgeCapacity::ConsolidationBehavior::Consolidated)) {
          utilized_capacity += 1;
        } else {
          utilized_capacity += sum_x;
        }
      }
      return utilized_capacity;
    };

    // Helper: double check if an edge is overused and get the overuse after
    // solving (the congestion variables are genuinely enforced during the
    // solving, meaning even for shared edges the combined capacity utilization
    // and the packet switching consolidation are all considered)
    auto get_edge_overuse = [&](const base::RREdge &edge) -> size_t {
      const RREdgeCapacity cap = edge.getCapacity();
      if (cap.isInfinite()) {
        return 0;
      }
      const int cap_val = cap.getValue();
      if (cap.is(RREdgeCapacity::SharingBehavior::Exclusive)) {
        return std::max(int(get_sum_x_on_edge(edge)) - cap_val, 0);
      } else if (cap.is(RREdgeCapacity::SharingBehavior::Shared)) {
        return std::max(
            int(get_utilized_capacity_on_shared_edge_group(edge)) - cap_val, 0);
      } else {
        throw std::runtime_error("Unknown RREdgeCapacity sharing behavior.");
      }
    };

    for (const auto &[edge, c_var] : c) {
      const size_t congestion = std::round(result->variable_values().at(c_var));
      // Double check if the edge is actually overused
      if (const size_t actual = get_edge_overuse(edge); congestion != actual) {
        throw std::runtime_error(
            std::format("Inconsistent congestion variable for edge {}: "
                        "expected {}, got {}.",
                        edge.getName(), congestion, actual));
      }
      if (congestion > 0) {
        legality = base::RoutingState::Legality::Congested;
        break;
      }
    }

    if (mode.isLazyEvaluation()) {
      return base::RoutingState(mode, legality, result->objective_value());
    } else {
      // Even if the routing is congested, we still reconstruct everything
      std::vector<base::RouteTree> paths;
      if (mode.is(RoutingMode::RouteTreeReconstruction::Full)) {
        for (const auto &net : netlist.getNets()) {
          base::RouteTree path(net);
          for (const auto &edge : graph_.getEdgePtrs()) {
            if (result->variable_values().at(x.at({net, *edge})) > 0.5) {
              path.addEdge(*edge);
            }
          }
          paths.push_back(path);
        }
      }

      // Helper: get the nets that use an edge
      auto get_nets_using_edge =
          [&](const base::RREdge &edge) -> std::vector<base::RoutingNet> {
        std::vector<base::RoutingNet> nets_using_edge;
        for (const auto &net : netlist.getNets()) {
          if (result->variable_values().at(x.at({net, edge})) > 0.5) {
            nets_using_edge.push_back(net);
          }
        }
        return nets_using_edge;
      };
      base::CongestionMap congestion_map;
      if (mode.is(RoutingMode::CongestionMapReconstruction::Full)) {
        for (const auto &[edge, c_var] : c) {
          // Use rounding to avoid precision issues
          const size_t congestion =
              std::round(result->variable_values().at(c_var));
          if (congestion > 0) {
            congestion_map[edge] = congestion;
            if (mode.get<RoutingMode::LoggingLevel>() >=
                RoutingMode::LoggingLevel::Normal) {
              engine.logger.normal(
                  std::format("Edge {} has congestion of {}.\n", edge.getName(),
                              congestion));
              for (const auto &net : get_nets_using_edge(edge)) {
                engine.logger.normal(std::format(
                    " - Net {} uses congested edge.\n", net.getName()));
              }
            }
          }
        }
      }

      // TODO: refactor buffer allocation extraction
      base::BufferAllocationMap buffer_allocation;
      if (mode.is(RoutingMode::BufferAllocReconstruction::Full) &&
          !mode.is(RoutingMode::MemoryCapacityConstraint::Ignored)) {
        for (const auto &l_core : l_cores) {
          for (const auto &p_core : p_cores) {
            const size_t used_mem = std::round(
                result->variable_values().at(m.at({l_core, p_core})));
            if (used_mem > 0) {
              buffer_allocation[l_core].setTotalBufferSize(used_mem);
              if (mode.get<RoutingMode::LoggingLevel>() >=
                  RoutingMode::LoggingLevel::Normal) {
                engine.logger.normal(
                    std::format("Logical core {} uses {} bytes of "
                                "memory on physical core {} "
                                "(capacity: {}).\n",
                                l_core.getName(), used_mem, p_core.getName(),
                                p_core.getMemoryCapacity()));
              }
            }
          }
        }
      }

      base::LogicalToPhysicalCoreMapping l_to_p_core_mapping;
      if (mode.is(RoutingMode::LogicalCorePlacementReconstruction::Full)) {
        for (const auto &l_core : l_cores) {
          for (const auto &p_core : p_cores) {
            if (result->variable_values().at(d.at({l_core, p_core})) > 0.5) {
              l_to_p_core_mapping.add(l_core, p_core);
            }
          }
        }
      }

      return base::RoutingState(
          mode, netlist, legality, result->objective_value(), paths,
          congestion_map, buffer_allocation, l_to_p_core_mapping);
    }
  } else {
    return base::RoutingState(mode, base::RoutingState::Legality::Fatal,
                              base::RoutingState::kInvalidRoutingCost);
  }
}

} // namespace route::router_zoo::milp
