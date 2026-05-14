#include "arch/npu/npu.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>

#include "arch/npu/route_tree.hpp"
#include "base/abstraction.hpp"
#include "base/common.hpp"
#include "base/engine.hpp"
#include "base/placement.hpp"
#include "base/routing/routing_state.hpp"
#include "base/tf_graph.hpp"

namespace arch {
namespace npu {

base::PnRPlacedNetlist NPU::parseNetlist(base::PnRNetlistReader rd) const {
  using base::LogicalCore;
  using base::PhysicalCore;
  using base::PnRNetlistFormat;
  using base::TrafficFlow;
  using base::TrafficFlowEndpoint;

  auto get_tile_type = [](const PnRNetlistFormat::NodeInfo &node) -> TileType {
    std::string id = node.id;
    std::transform(id.begin(), id.end(), id.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (id.starts_with("comp")) {
      return TileType::Compute;
    } else if (id.starts_with("mem")) {
      return TileType::Memory;
    } else if (id.starts_with("shim")) {
      return TileType::Shim;
    } else {
      throw std::runtime_error("Unknown node type for node id: " + node.id);
    }
  };

  auto get_tile_position =
      [](const PnRNetlistFormat::NodeInfo &node) -> TilePos {
    if (!node.position.has_value()) {
      return {-1, -1}; // unplaced
    }
    return node.position.value();
  };

  auto get_resolved_nodes =
      [&]() -> utils::Lookup<PnRNetlistFormat::ID, TilePtr> {
    auto is_node_packable = [](TileType type) -> bool {
      return type == TileType::Memory || type == TileType::Shim;
    };

    utils::Lookup<TileType, std::vector<PnRNetlistFormat::NodeInfo>>
        tile_type_to_nodes;
    for (const auto &node : rd.getNodes()) {
      tile_type_to_nodes[get_tile_type(node)].push_back(node);
    }

    // Resolve the unplaced nodes (sequentially place them)
    const std::unordered_map<TileType, std::pair<int, int>> kRowBounds = {
        {TileType::Shim, {0, kRowsOfShimTilesInNPU - 1}},
        {TileType::Memory,
         {kRowsOfShimTilesInNPU,
          kRowsOfShimTilesInNPU + kRowsOfMemoryTilesInNPU - 1}},
        {TileType::Compute,
         {kRowsOfShimTilesInNPU + kRowsOfMemoryTilesInNPU, kRowsInNPU - 1}}};

    utils::Lookup<PnRNetlistFormat::ID, TilePtr> node_id_to_tile;

    for (const auto &type :
         {TileType::Shim, TileType::Memory, TileType::Compute}) {
      std::set<TilePos> placed_positions;

      for (const auto &node : tile_type_to_nodes[type]) {
        const TilePos pos = get_tile_position(node);
        const auto [row_y, col_x] = pos.getRowYColXPair();
        if (!((row_y > -1 && col_x > -1) || (row_y == -1 && col_x == -1))) {
          throw std::runtime_error("Node position should be either placed at "
                                   "(row=non-negative,col=non-negative) or "
                                   "unplaced using (row=-1,col=-1)");
        }
        if (row_y != -1 && col_x != -1) {
          // Already placed, check validity
          if ((row_y < kRowBounds.at(type).first ||
               row_y > kRowBounds.at(type).second) ||
              (col_x < 0 || col_x >= kColumnsInNPU)) {
            throw std::runtime_error(std::format(
                "Node of type {} at position (row={},col={}) is out of bounds",
                utils::toString(type), row_y, col_x));
          }
          if (placed_positions.contains(pos) && !is_node_packable(type)) {
            throw std::runtime_error(std::format(
                "Multiple un-packable nodes of type {} are placed at position "
                "(row={},col={})",
                utils::toString(type), row_y, col_x));
          }
          placed_positions.insert(pos);
          node_id_to_tile.insert_or_assign(node.id, getTile(pos).value());
        }
      }

      auto find_valid_position = [&]() -> TilePos {
        while (true) {
          // Find empty position first
          for (int row_idx = kRowBounds.at(type).first;
               row_idx <= kRowBounds.at(type).second; ++row_idx) {
            for (int col_idx = 0; col_idx < kColumnsInNPU; ++col_idx) {
              TilePos pos{row_idx, col_idx};
              if (!placed_positions.contains(pos)) {
                return pos;
              }
            }
          }
          // If no empty position is available and the node is not packable,
          // throw an runtime error
          if (!is_node_packable(type)) {
            throw std::runtime_error(std::format(
                "No valid position available for placing node of type {}",
                utils::toString(type)));
          }
          // If the node is packable, clear the placed positions and start over
          placed_positions.clear();
        }
      };

      for (const auto &node : tile_type_to_nodes[type]) {
        if (const TilePos pos = get_tile_position(node);
            pos.getColX() == -1 && pos.getRowY() == -1) {
          const TilePos resolved_pos = find_valid_position();
          placed_positions.insert(resolved_pos);
          node_id_to_tile.insert_or_assign(node.id,
                                           getTile(resolved_pos).value());
        }
      }
    }

    return node_id_to_tile;
  };

  utils::Lookup<PnRNetlistFormat::ID, TilePtr> node_id_to_tile =
      get_resolved_nodes();

  base::TrafficFlowGraph tf_graph;
  base::Placement placement;

  auto add_to_placement = [&](const PnRNetlistFormat::ID &node_id) {
    LogicalCore l_core{node_id};
    PhysicalCore p_core{tile_to_p_core_map_.at(node_id_to_tile.at(node_id))};
    placement.add(l_core, p_core);
  };

  for (const auto &net : rd.getEdges()) {
    const PnRNetlistFormat::ID &start_node_id = net.start_node;
    const std::vector<PnRNetlistFormat::ID> &target_node_ids = net.target_nodes;

    // TODO: consider merging tf ep with logical core concept
    add_to_placement(start_node_id);
    TrafficFlowEndpoint start_l_core{start_node_id};

    std::vector<TrafficFlowEndpoint> target_l_cores;
    for (const auto &target_node_id : target_node_ids) {
      add_to_placement(target_node_id);
      target_l_cores.push_back(TrafficFlowEndpoint{target_node_id});
    }

    std::vector<TrafficFlow::BufferDemand> buffer_demands; // depth x width
    if (net.depth.has_value()) {
      const size_t width = net.width.value_or(0);
      for (const auto depth : net.depth.value()) {
        buffer_demands.emplace_back(depth, width);
      }
    } else {
      buffer_demands = std::vector<TrafficFlow::BufferDemand>(
          target_l_cores.size() + 1, TrafficFlow::BufferDemand{0, 0});
    }

    // TODO: remove id field
    auto tf_ptr = tf_graph.addTrafficFlow(net.id, start_l_core, target_l_cores,
                                          buffer_demands);
    // TODO: remove width attribute since buffer demand already contains width
    tf_ptr->setAttribute("width", std::to_string(net.width.value_or(0)));
  }

  for (const auto &linking : rd.getLinkings()) {
    std::vector<TrafficFlow::ID> link_from_tf_ids;
    for (const auto &from_net_id : linking.from) {
      link_from_tf_ids.push_back(from_net_id);
    }
    std::vector<TrafficFlow::ID> link_to_tf_ids;
    for (const auto &to_net_id : linking.to) {
      link_to_tf_ids.push_back(to_net_id);
    }
    tf_graph.addTrafficFlowLink(linking.id, link_from_tf_ids, link_to_tf_ids);
  }

  // Note: avoid parsing route section in the netlist reader here, as rebuilding
  // route tree structure is complicated given the existance of route tree ILP
  // resolution in NPU architecture.

  return {.tf_graph = tf_graph, .placement = placement};
}

base::PnRNetlistWriter NPU::dumpNetlist(base::PnRPlacedNetlist p) const {
  using base::LogicalCore;
  using base::PhysicalCore;
  using base::PnRNetlistFormat;

  const auto &tf_graph = p.tf_graph;
  const auto &placement = p.placement;

  base::PnRNetlistWriter wr;

  for (const auto &node : tf_graph.getNodeViews()) {
    const LogicalCore l_core{*node};
    const PhysicalCore p_core = placement.at(l_core);
    const TilePtr tile = rr_node_to_tile_map_.at(p_core.getSink());

    PnRNetlistFormat::NodeInfo node_info;
    node_info.id = l_core.getTrafficFlowEndpoint().getID();
    node_info.position = base::GridPosition{tile->getRowY(), tile->getColX()};
    wr.addNode(node_info);
  }

  for (const auto &edge : tf_graph.getEdgeViews()) {
    PnRNetlistFormat::EdgeInfo edge_info;
    edge_info.id = edge->getId();
    edge_info.start_node = edge->getSource().getID();
    for (const auto &sink : edge->getSinks()) {
      edge_info.target_nodes.push_back(sink.getID());
    }
    // TODO: remove width attribute since buffer demand already contains width
    size_t width = std::stoi(edge->getAttribute("width"));
    if (width == 0) {
      edge_info.width = std::nullopt;
      edge_info.depth = std::nullopt;
    } else {
      edge_info.width = width;
      edge_info.depth = {edge->getBufferDemand(edge->getSource()).depth};
      for (const auto &sink : edge->getSinks()) {
        edge_info.depth.value().push_back(edge->getBufferDemand(sink).depth);
      }
    }

    wr.addEdge(edge_info);
  }

  for (const auto &tf_link : tf_graph.getTrafficFlowLinkings()) {
    PnRNetlistFormat::LinkingInfo linking_info{.id = tf_link.getId()};
    for (const auto &from_tf : tf_link.getFromTrafficFlows()) {
      linking_info.from.push_back(from_tf->getId());
    }
    for (const auto &to_tf : tf_link.getToTrafficFlows()) {
      linking_info.to.push_back(to_tf->getId());
    }
    wr.addLinking(linking_info);
  }

  return wr;
}

base::PnRNetlistWriter NPU::dumpNetlist(base::PnRPlacedNetlist p,
                                        base::RoutingState r) const {
  if (r.getLegality() != base::RoutingState::Legality::Legal) {
    throw std::runtime_error(
        "Cannot dump netlist with routing information when the routing "
        "state is not legal");
  }

  base::PnRNetlistWriter wr = dumpNetlist(std::move(p));

  RouteTreeResourceAllocator resource_allocator(
      r.getRouteTrees().value(), r.getRoutingNetList().value().getNetLinks());

  for (const auto &tf : p.tf_graph.getEdgeViews()) {
    base::PnRNetlistFormat::RouteInfo route;
    route.edge_id = tf->getId();

    if (auto rt_trees = r.getRouteTrees(); rt_trees.has_value()) {
      for (const auto &tree : rt_trees.value()) {
        if (tree.getAssociatedInputNet().getId() != tf->getId()) {
          continue;
        }
        auto parsed =
            parseRouteTree(tree, resource_allocator,
                           r.getLogicalToPhysicalCoreMapping().value());
        if (std::holds_alternative<CircuitSwitchingRouteTree>(parsed)) {
          const auto &cct =
              std::get<CircuitSwitchingRouteTree>(std::move(parsed));
          route.type = cct.getConnectionType();
          route.paths = cct.getIntermediates();
        } else if (std::holds_alternative<PacketSwitchingRouteTree>(parsed)) {
          const auto &pkt =
              std::get<PacketSwitchingRouteTree>(std::move(parsed));
          route.type = pkt.getConnectionType();
          route.paths = pkt.getIntermediates();
        } else if (std::holds_alternative<NeighborSharingRouteTree>(parsed)) {
          const auto &nbr =
              std::get<NeighborSharingRouteTree>(std::move(parsed));
          route.type = nbr.getConnectionType();
          // TODO: move this logic into NeighborSharingRouteTree class
          utils::Lookup<base::PnRNetlistFormat::ID,
                        std::vector<base::GridPosition>>
              paths;
          for (const auto &[t_node, alloc_node] : nbr.getAllocationTiles()) {
            paths.insert_or_assign(t_node,
                                   std::vector<base::GridPosition>{alloc_node});
          }
          route.paths = paths;
        } else if (std::holds_alternative<IntraTileKernelLinkingRouteTree>(
                       parsed)) {
          const auto &intra =
              std::get<IntraTileKernelLinkingRouteTree>(std::move(parsed));
          route.type = intra.getConnectionType();
          route.paths = intra.getAllocationTile();
        } else {
          throw std::runtime_error("Unsupported supported route tree variant");
        }

        wr.addRoute(route);
      }
    }
  }
  return wr;
}

} // namespace npu
} // namespace arch
