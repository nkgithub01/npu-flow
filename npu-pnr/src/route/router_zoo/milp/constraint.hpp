#pragma once

#include "base/abstraction.hpp"
#include "base/rr_graph.hpp"
#include "route/router_zoo/milp/variable.hpp"

namespace route::router_zoo::milp {

void addRoutabilityConstraints(
    math_opt::Model &model, const NetEdgeToVarLookup &x,
    const NetEdgeToVarLookup &f, const EdgeToVarLookup &c,
    const LogicalPhysicalCoreToVarLookup &d, const base::RRGraph &graph,
    const base::RoutingNetList &netlist, const base::LogicalCoreSet &l_cores,
    const base::PhysicalCoreSet &p_cores,
    const utils::Lookup<base::RRNode, base::PhysicalCore> &rr_node_to_p_core);

void addCoreMappingConstraints(
    math_opt::Model &model, const LogicalPhysicalCoreToVarLookup &d,
    const base::LogicalCoreSet &l_cores, const base::PhysicalCoreSet &p_cores,
    const base::LogicalToPhysicalCoreAvailablityTable &l_to_p_core_avail_table,
    const base::LogicalCoreCompatibilitySet &l_core_compat_set,
    const std::vector<base::LogicalToPhysicalCoreMapping>
        &l_to_p_core_mapping_blacklist);

void addBufferAllocationConstraints(
    math_opt::Model &model, const LogicalPhysicalCoreToVarLookup &m,
    const LogicalPhysicalCoreToVarLookup &lock, const NetEdgeToVarLookup &x,
    const base::RRGraph &graph, const base::RoutingNetList &netlist,
    const base::PhysicalToLogicalCoreAvailablityTable &p_to_l_core_avail_table,
    const base::RoutingMode::MemoryCapacityConstraint mem_cap_constraint_mode,
    const base::RoutingMode::LockCapacityConstraint lock_cap_constraint_mode,
    const base::RoutingMode::NetLink net_link_mode);

void addMulticastRoutingConstraints(
    math_opt::Model &model, const NetEdgeToVarLookup &x,
    const base::RRGraph &graph, const base::RoutingNetList &netlist,
    const base::LogicalToPhysicalCoreAvailablityTable &l_to_p_core_avail_table,
    const base::RoutingMode::AllowedRREdgeTypeOrderedSet &allowed_edge_types);

} // namespace route::router_zoo::milp
