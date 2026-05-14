#include "arch/npu/route_tree.hpp"

#include <algorithm>

#include "arch/npu/npu.hpp"
#include "base/abstraction.hpp"
#include "base/common.hpp"
#include "base/engine.hpp"
#include "base/routing/routing_state.hpp"
#include "base/rr_graph.hpp"
#include "utils/misc.hpp"

namespace arch {
namespace npu {

inline utils::Lookup<base::LogicalCore, std::vector<RRNode>>
findPathsFromCircuitOrPacketSwitchingRouteTree(
    const base::RouteTree &rt_tree,
    const base::LogicalToPhysicalCoreMapping &core_mapping) {
  using base::LogicalCore;
  using base::PhysicalCore;

  utils::Lookup<RRNode, RRNode> prev_rr_node_lookup;
  for (const base::RREdge &edge : rt_tree.getEdges()) {
    prev_rr_node_lookup.insert_or_assign(edge.getToNode(), edge.getFromNode());
  }

  utils::Lookup<LogicalCore, std::vector<RRNode>> path_from_source_to_sinks;
  const LogicalCore start = rt_tree.getAssociatedInputNet().getStartCore();
  const RRNode source = core_mapping.at(start).getSource();
  for (const LogicalCore &target :
       rt_tree.getAssociatedInputNet().getTargetCores()) {
    const RRNode sink = core_mapping.at(target).getSink();
    std::vector<RRNode> trace_back;
    RRNode current = sink;
    while (source != current) {
      trace_back.push_back(current);
      current = prev_rr_node_lookup.at(current);
    }
    trace_back.push_back(source);
    path_from_source_to_sinks.insert_or_assign(
        target, std::vector<RRNode>(trace_back.rbegin(), trace_back.rend()));
  }

  return path_from_source_to_sinks;
}

SupportedRouteTree NPU::parseRouteTree(
    const base::RouteTree &rt_tree,
    const npu::RouteTreeResourceAllocator &resource_allocator,
    const base::LogicalToPhysicalCoreMapping &core_mapping) const {
  using ResolvedRouteTreeType =
      RouteTreeResourceAllocator::ResolvedRouteTreeType;

  const ResolvedRouteTreeType type =
      resource_allocator.getResolvedRouteTreeType(rt_tree);

  if (type == ResolvedRouteTreeType::CircuitSwitching ||
      type == ResolvedRouteTreeType::PacketSwitching) {
    // Find intermediate nodes for each destination
    utils::Lookup<base::PnRNetlistFormat::ID, std::vector<base::GridPosition>>
        intermediates;
    for (const auto &[target_core, path] :
         findPathsFromCircuitOrPacketSwitchingRouteTree(rt_tree,
                                                        core_mapping)) {
      std::vector<base::GridPosition> intermediates_for_path;
      // Skip the first and last nodes (source and sink endpoints)
      for (size_t i = 1; i < path.size() - 1; ++i) {
        const auto tile = rr_node_to_tile_map_.at(path.at(i));
        intermediates_for_path.emplace_back(tile->getRowY(), tile->getColX());
      }
      intermediates.insert_or_assign(
          target_core.getTrafficFlowEndpoint().getID(), intermediates_for_path);
    }

    if (type == ResolvedRouteTreeType::CircuitSwitching) {
      return CircuitSwitchingRouteTree(intermediates);
    } else {
      return PacketSwitchingRouteTree(intermediates);
    }
  }

  if (type == ResolvedRouteTreeType::NeighborSharing) {
    utils::Lookup<base::PnRNetlistFormat::ID, base::GridPosition>
        allocation_tiles;
    for (const auto &target :
         rt_tree.getAssociatedInputNet().getTargetCores()) {
      const base::PnRNetlistFormat::ID target_id =
          target.getTrafficFlowEndpoint().getID();
      const RRNode sink = core_mapping.at(target).getSink();

      if (const auto it = std::ranges::find_if(
              rt_tree.getEdges(),
              [&](const auto &edge) { return edge.getToNode() == sink; });
          it != rt_tree.getEdges().end()) {
        if (it->getType() == RREdgeType::NeighborSharingOnSource) {
          const auto &tile = rr_node_to_tile_map_.at(it->getFromNode());
          allocation_tiles.insert_or_assign(
              target_id, base::GridPosition{tile->getRowY(), tile->getColX()});
        } else if (it->getType() == RREdgeType::NeighborSharingOnSink) {
          const auto &tile = rr_node_to_tile_map_.at(it->getToNode());
          allocation_tiles.insert_or_assign(
              target_id, base::GridPosition{tile->getRowY(), tile->getColX()});
        }
      } else {
        throw std::runtime_error(
            "Route tree with neighbor sharing edges should have all sinks "
            "connected by edges, but found a sink without incoming edge.");
      }
    }
    return NeighborSharingRouteTree(allocation_tiles);
  }

  if (type == ResolvedRouteTreeType::IntraTileKernelLinking) {
    if (rt_tree.getAssociatedInputNet().isMulticast()) {
      throw std::runtime_error(
          "Intra-tile kernel linking route tree should not be multicast.");
    }
    const base::LogicalCore target_core =
        rt_tree.getAssociatedInputNet().getTargetCores().front();
    const auto &tile =
        rr_node_to_tile_map_.at(core_mapping.at(target_core).getSink());
    return IntraTileKernelLinkingRouteTree(
        target_core.getTrafficFlowEndpoint().getID(),
        base::GridPosition{tile->getRowY(), tile->getColX()});
  }

  throw std::runtime_error(
      std::format("Unsupported RR edge type in route tree of net {}:\n{}",
                  rt_tree.getAssociatedInputNet().getName(),
                  utils::toString(rt_tree.getEdges(), "\n", "\t")));
}

} // namespace npu
} // namespace arch
