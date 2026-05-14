#ifndef _NPU_ROUTE_TREE_HPP_
#define _NPU_ROUTE_TREE_HPP_

#include <format>
#include <variant>

#include "base/common.hpp"
#include "base/engine.hpp"
#include "base/routing.hpp"
#include "base/routing/routing_net.hpp"
#include "base/routing/routing_state.hpp"
#include "utils/misc.hpp"

namespace arch {
namespace npu {

// TODO: refactor the route tree classes to reduce code duplication
class CircuitSwitchingRouteTree;
class PacketSwitchingRouteTree;
class NeighborSharingRouteTree;
class IntraTileKernelLinkingRouteTree;
using SupportedRouteTree =
    std::variant<CircuitSwitchingRouteTree, PacketSwitchingRouteTree,
                 NeighborSharingRouteTree, IntraTileKernelLinkingRouteTree>;

class CircuitOrPacketSwitchingRouteTree {
private:
  utils::Lookup<base::PnRNetlistFormat::ID, std::vector<base::GridPosition>>
      intermediates_;

public:
  CircuitOrPacketSwitchingRouteTree() = delete;
  CircuitOrPacketSwitchingRouteTree(
      utils::Lookup<base::PnRNetlistFormat::ID, std::vector<base::GridPosition>>
          intermediates)
      : intermediates_(intermediates) {}

  const auto &getIntermediates() const { return intermediates_; }
  virtual std::string getConnectionType() const = 0;
};

class CircuitSwitchingRouteTree : public CircuitOrPacketSwitchingRouteTree {
public:
  static inline const std::string kConnectionType = "circuit_switch";
  using CircuitOrPacketSwitchingRouteTree::CircuitOrPacketSwitchingRouteTree;
  std::string getConnectionType() const override { return kConnectionType; }
};

class PacketSwitchingRouteTree : public CircuitOrPacketSwitchingRouteTree {
public:
  static inline const std::string kConnectionType = "packet_switch";
  using CircuitOrPacketSwitchingRouteTree::CircuitOrPacketSwitchingRouteTree;
  std::string getConnectionType() const override { return kConnectionType; }
};

class NeighborSharingRouteTree {
private:
  utils::Lookup<base::PnRNetlistFormat::ID, base::GridPosition>
      allocation_tiles_;

public:
  static inline const std::string kConnectionType = "neighbor_sharing";
  NeighborSharingRouteTree() = delete;
  NeighborSharingRouteTree(
      utils::Lookup<base::PnRNetlistFormat::ID, base::GridPosition>
          allocation_tiles)
      : allocation_tiles_(allocation_tiles) {}

  const auto &getAllocationTiles() const { return allocation_tiles_; }
  std::string getConnectionType() const { return kConnectionType; }
};

class IntraTileKernelLinkingRouteTree {
private:
  utils::Lookup<base::PnRNetlistFormat::ID, std::vector<base::GridPosition>>
      allocation_tile_;

public:
  static inline const std::string kConnectionType = "intra_tile";
  IntraTileKernelLinkingRouteTree() = delete;
  IntraTileKernelLinkingRouteTree(base::PnRNetlistFormat::ID alloc_tile_id,
                                  base::GridPosition alloc_tile_pos) {
    allocation_tile_.insert_or_assign(
        alloc_tile_id, std::vector<base::GridPosition>{alloc_tile_pos});
  }

  const auto &getAllocationTile() const { return allocation_tile_; }
  std::string getConnectionType() const { return kConnectionType; }
};

class RouteTreeResourceAllocator {
public:
  enum class ResolvedRouteTreeType {
    Unsupported = -1,
    CircuitSwitching,
    PacketSwitching,
    NeighborSharing,
    IntraTileKernelLinking,
  };

private:
  utils::Lookup<base::RoutingNet, ResolvedRouteTreeType>
      preprocessed_route_tree_type_map_;

public:
  RouteTreeResourceAllocator() = delete;
  RouteTreeResourceAllocator(
      const std::vector<base::RouteTree> &route_trees,
      const std::vector<base::RoutingNetLink> &net_links) {
    using base::RoutingNet;
    using base::RREdgeType;

    auto get_error_details = [](const base::RouteTree &tree) -> std::string {
      std::string details(
          std::format("\tNet {}\n", tree.getAssociatedInputNet().getName()));
      for (const base::RREdge &edge : tree.getEdges()) {
        details += std::format("\tRREdge {}\n", edge.getName());
      }
      return details;
    };

    // Preprocess route tree types for faster and optimized resource allocation
    // and route tree reconstruction
    for (const auto &tree : route_trees) {
      const RREdgeType first_edge_type = tree.getEdges().front().getType();
      const RoutingNet &associated_net = tree.getAssociatedInputNet();

      if (first_edge_type == RREdgeType::NeighborSharingOnSource ||
          first_edge_type == RREdgeType::NeighborSharingOnSink) {
        // 1. Neighbor-sharing route trees
        if (std::ranges::any_of(tree.getEdges(), [](const auto &edge) {
              return edge.getType() != RREdgeType::NeighborSharingOnSource &&
                     edge.getType() != RREdgeType::NeighborSharingOnSink;
            })) {
          throw std::runtime_error(
              "Route tree for a neighbor-sharing net can only contain either "
              "RREdgeType::NeighborSharingOnSource or NeighborSharingOnSink, "
              "but found other rr edge types:\n" +
              get_error_details(tree));
        }

        if (preprocessed_route_tree_type_map_.contains(associated_net)) {
          throw std::runtime_error(
              "Multiple route trees found for the same net when "
              "preprocessing neighbor-sharing route trees");
        }
        preprocessed_route_tree_type_map_.insert_or_assign(
            associated_net, ResolvedRouteTreeType::NeighborSharing);

      } else if (first_edge_type == RREdgeType::IntraTileKernelLinking) {
        // 2. Intra-tile kernel linking route trees
        if (tree.getEdges().size() != 1) {
          throw std::runtime_error(
              "Intra-tile kernel linking route tree should have exactly one "
              "rr edge, but get:\n" +
              get_error_details(tree));
        }

        if (preprocessed_route_tree_type_map_.contains(associated_net)) {
          throw std::runtime_error(
              "Multiple route trees found for the same net when "
              "preprocessing intra-tile kernel linking route trees");
        }
        preprocessed_route_tree_type_map_.insert_or_assign(
            associated_net, ResolvedRouteTreeType::IntraTileKernelLinking);

      } else if (first_edge_type == RREdgeType::CircuitSwitching) {
        // 3. Circuit-switching route trees
        if (std::ranges::any_of(tree.getEdges(), [](const auto &edge) {
              return edge.getType() != RREdgeType::CircuitSwitching;
            })) {
          throw std::runtime_error(
              "Route tree for a circuit-switching net can only contain "
              "RREdgeType::CircuitSwitching, but found other rr edge types:\n" +
              get_error_details(tree));
        }

        if (preprocessed_route_tree_type_map_.contains(associated_net)) {
          throw std::runtime_error(
              "Multiple route trees found for the same net when "
              "preprocessing circuit-switching route trees");
        }
        preprocessed_route_tree_type_map_.insert_or_assign(
            associated_net, ResolvedRouteTreeType::CircuitSwitching);

      } else if (first_edge_type == RREdgeType::PacketSwitching) {
        // 4. Packet-switching route trees
        if (std::ranges::any_of(tree.getEdges(), [](const auto &edge) {
              return edge.getType() != RREdgeType::PacketSwitching;
            })) {
          throw std::runtime_error(
              "Route tree for a packet-switching net can only contain "
              "RREdgeType::PacketSwitching, but found other rr edge types:\n" +
              get_error_details(tree));
        }

        if (preprocessed_route_tree_type_map_.contains(associated_net)) {
          throw std::runtime_error(
              "Multiple route trees found for the same net when "
              "preprocessing packet-switching route trees");
        }
        preprocessed_route_tree_type_map_.insert_or_assign(
            associated_net, ResolvedRouteTreeType::PacketSwitching);

      } else {
        // 5. Unsupported route tree types
        if (preprocessed_route_tree_type_map_.contains(associated_net)) {
          throw std::runtime_error(
              "Multiple route trees found for the same net when "
              "preprocessing unsupported route tree types");
        }
        preprocessed_route_tree_type_map_.insert_or_assign(
            associated_net, ResolvedRouteTreeType::Unsupported);
      }
    }
  }

  ResolvedRouteTreeType
  getResolvedRouteTreeType(const base::RouteTree &tree) const {
    const auto &associated_net = tree.getAssociatedInputNet();
    if (!preprocessed_route_tree_type_map_.contains(associated_net)) {
      throw std::runtime_error(
          std::format("Associated net {} of route tree is not found in "
                      "preprocessed route tree type map:\n{}",
                      associated_net.getName(),
                      utils::toString(tree.getEdges(), "\n", "\t")));
    }
    return preprocessed_route_tree_type_map_.at(associated_net);
  }
};

} // namespace npu
} // namespace arch

#endif
