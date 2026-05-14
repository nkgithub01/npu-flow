#include "arch/npu/npu.hpp"

#include <memory>
#include <unordered_map>

#include "arch/npu/route_tree.hpp"
#include "base/abstraction.hpp"
#include "base/engine.hpp"
#include "base/routing/routing_net.hpp"
#include "utils/visualization.hpp"

namespace arch {
namespace npu {

inline const utils::ColorScheme &
getColorSchemeByTileType(const TileType &tile_type) {
  if (tile_type == TileType::Compute) {
    return utils::highlighted_color_scheme.at("green");
  } else if (tile_type == TileType::Memory) {
    return utils::highlighted_color_scheme.at("steelblue");
  } else if (tile_type == TileType::Shim) {
    return utils::highlighted_color_scheme.at("purple");
  } else {
    return utils::highlighted_color_scheme.at("grey");
  }
}

std::string NPU::visualizePlacement(const base::PnRNetlistReader &rd) const {

  constexpr double kNodeLabelFontSize = 25;
  constexpr double kNodeShapeWidth = 1.25;
  constexpr double kNodeSpacingFactor = 1.5;
  constexpr double kNodeStrokeWidth = 2;
  constexpr double kEdgeStrokeWidth = 5.5;

  const auto &parsed = parseNetlist(rd);
  const auto &placement = parsed.placement;
  const auto &tf_graph = parsed.tf_graph;

  std::stringstream ss;

  ss << std::format("digraph NPUPlacement{{\n"
                    "  layout=neato;\n"
                    "  graph[outputMode=nodesfirst;splines=polyline];\n"
                    "  node[shape=square;width={}];\n"
                    "  edge[headclip=false;tailclip=false];\n",
                    kNodeShapeWidth);

  ss << "\n  // Nodes\n";

  // added 2 invisible nodes as dynamic adjustment for graphviz layout to prevent edge text extend the drawing boundry
  ss << std::format("  \"bottom left anchor\" [style=filled;pin=true;fontcolor=\"#FFFFFF\";color=\"#FFFFFF\";fillcolor=\"#FFFFFF\"; pos=\"{},{}!\";];\n", -1 * kNodeShapeWidth * kNodeSpacingFactor, -1 * kNodeShapeWidth * kNodeSpacingFactor);
  ss << std::format("  \"top right anchor\" [style=filled;pin=true;fontcolor=\"#FFFFFF\";color=\"#FFFFFF\";fillcolor=\"#FFFFFF\"; pos=\"{},{}!\";];\n", (kColumnsInNPU) * kNodeShapeWidth * kNodeSpacingFactor, (kRowsInNPU) * kNodeShapeWidth * kNodeSpacingFactor);
  ss << "\n";

  for (int i = kRowsInNPU - 1; i >= 0; --i) {
    for (int j = 0; j < kColumnsInNPU; ++j) {
      std::shared_ptr<Tile> tile = tiles_[i][j];
      const utils::ColorScheme &color_scheme =
          getColorSchemeByTileType(tile->getTileType());

      ss << std::format("  \"{}\" [", tile->getName());
      // ss << "label=\"\\N\"";
      ss << "style=filled;pin=true;";
      ss << std::format("label=\"{}, {}\";", i, j);
      ss << std::format("fontsize=\"{}\";", kNodeLabelFontSize);
      ss << std::format("color=\"{}\";", color_scheme.stroke);
      ss << std::format("fillcolor=\"{}\";", color_scheme.fill);
      ss << std::format("penwidth=\"{}\";", kNodeStrokeWidth);
      ss << std::format("pos=\"{},{}!\";",
                        j * kNodeShapeWidth * kNodeSpacingFactor,
                        i * kNodeShapeWidth * kNodeSpacingFactor);
      ss << "];\n";
    }
    ss << "\n";
  }

  ss << "\n  // Edges\n";
  base::RoutingNetList netlist{tf_graph};
  for (const auto &net : netlist.getNets()) {
    RRNode src_rr_node =
        placement.at(base::LogicalCore{net.getStartCore()}).getSource();
    std::vector<RRNode> sink_rr_nodes;
    for (const auto &target_core : net.getTargetCores()) {
      sink_rr_nodes.push_back(
          placement.at(base::LogicalCore{target_core}).getSink());
    }

    const std::string src = rr_node_to_tile_map_.at(src_rr_node)->getName();
    const std::string stroke_color =
        net.getNumTargetCores() > 1
            ? utils::highlighted_color_scheme.at("steelblue").stroke
            : utils::highlighted_color_scheme.at("red").stroke;
    for (const auto &sink_rr_node : sink_rr_nodes) {
      const std::string sink = rr_node_to_tile_map_.at(sink_rr_node)->getName();
      ss << std::format("  \"{}\" -> \"{}\" [", src, sink);
      ss << std::format("color=\"{}\";", stroke_color);
      ss << std::format("label=\"#{}\";", net.getId());
      ss << std::format("penwidth=\"{}\";", kEdgeStrokeWidth);
      ss << "];\n";
    }
  }

  ss << "\n}\n";

  return ss.str();
}

std::string NPU::visualizeRouting(const base::PnRNetlistReader &rd) const {

  constexpr double kNodeLabelFontSize = 35;
  constexpr double kNodeShapeWidth = 2.0;
  constexpr double kNodeSpacingFactor = 2.75;
  constexpr double kNodeStrokeWidth = 2.5;
  constexpr double kEdgeStrokeWidth = 6.0;
  constexpr double kSwitchBoxShapeWidth = 1.5;

  if (!rd.getRoutes().has_value()) {
    throw std::runtime_error(
        "Cannot visualize routing: no routing information in the netlist");
  }
  const std::vector<base::PnRNetlistFormat::RouteInfo> route_infos =
      rd.getRoutes().value();

  const auto &parsed = parseNetlist(rd);
  const auto &placement = parsed.placement;
  const auto &tf_graph = parsed.tf_graph;
  const auto &routing_netlist = base::RoutingNetList{tf_graph};

  auto get_tile_from_l_core = [&](const base::LogicalCore &l_core) -> TilePtr {
    const RRNode rr_node =
        placement.at(l_core).getSource(); // doesn't matter if source or sink
    return rr_node_to_tile_map_.at(rr_node);
  };

  // Build route trees for visualization (without using *RoutingTree classes)
  using RouteTreeType = std::string;
  using GraphVizNode = std::string;
  std::vector<std::tuple<base::RoutingNet, RouteTreeType,
                         std::vector<std::pair<GraphVizNode, GraphVizNode>>>>
      rt_trees;
  for (const auto &rt : route_infos) {
    const base::RoutingNet net = routing_netlist.getNetById(rt.edge_id);
    const RouteTreeType type = rt.type;
    for (const auto &[target_node, path] : rt.paths) {
      std::vector<std::pair<GraphVizNode, GraphVizNode>> route_path;

      if (type == CircuitSwitchingRouteTree::kConnectionType ||
          type == PacketSwitchingRouteTree::kConnectionType) {

        const GraphVizNode src_ep = get_tile_from_l_core(net.getStartCore())
                                        ->getSourceEndpoint()
                                        .getName();
        const GraphVizNode src_sw =
            getTile(path.at(0)).value()->getSwitchBox().getName();
        route_path.emplace_back(std::make_pair(src_ep, src_sw));

        for (size_t i = 0; i < path.size() - 1; ++i) {
          const auto from_tile = getTile(path.at(i)).value();
          const auto to_tile = getTile(path.at(i + 1)).value();
          route_path.emplace_back(
              std::make_pair(from_tile->getSwitchBox().getName(),
                             to_tile->getSwitchBox().getName()));
        }

        const GraphVizNode sink_sw =
            getTile(path.back()).value()->getSwitchBox().getName();
        const GraphVizNode sink_ep =
            get_tile_from_l_core(base::LogicalCore{target_node})
                ->getSinkEndpoint()
                .getName();
        route_path.emplace_back(std::make_pair(sink_sw, sink_ep));

      } else if (type == IntraTileKernelLinkingRouteTree::kConnectionType) {

        if (path.size() != 1 ||
            getPhysicalCore(path.at(0)) != placement.at(net.getStartCore()) ||
            base::LogicalCore{target_node} != net.getStartCore() ||
            net.getStartCore() != net.getTargetCores().front() ||
            net.isMulticast()) {
          throw std::runtime_error("Intra-tile kernel linking route tree "
                                   "should be a unicast self-loop");
        }
        const auto tile = get_tile_from_l_core(net.getStartCore());
        route_path.emplace_back(
            std::make_pair(tile->getSourceEndpoint().getName(),
                           tile->getSinkEndpoint().getName()));

      } else if (type == NeighborSharingRouteTree::kConnectionType) {

        if (path.size() != 1) {
          throw std::runtime_error(
              "Neighbor-sharing route tree path should contain exactly "
              "one tile position");
        }
        // Note: ignore the buffer tile for visualization
        route_path.emplace_back(
            std::make_pair(get_tile_from_l_core(net.getStartCore())
                               ->getSourceEndpoint()
                               .getName(),
                           get_tile_from_l_core(base::LogicalCore{target_node})
                               ->getSinkEndpoint()
                               .getName()));

      } else {
        throw std::runtime_error(
            "Unsupported route tree type for visualization: " + type);
      }

      rt_trees.emplace_back(net, type, route_path);
    }
  }

  const utils::ColorScheme &sb_color_scheme =
      utils::highlighted_color_scheme.at("grey");

  std::stringstream ss;

  ss << std::format("digraph NPURouting{{\n");
  ss << std::format("  layout=neato;\n");
  ss << std::format("  graph[outputMode=nodesfirst;splines=false];\n");
  ss << std::format("  node[shape=square;penwidth=\"{}\";fontsize=\"{}\"];\n",
                    kNodeStrokeWidth, kNodeLabelFontSize);

  ss << "\n  // Nodes\n";
  for (int i = kRowsInNPU - 1; i >= 0; --i) {
    for (int j = 0; j < kColumnsInNPU; ++j) {
      std::shared_ptr<Tile> tile = tiles_[i][j];
      const utils::ColorScheme &color_scheme =
          getColorSchemeByTileType(tile->getTileType());
      // Tile nodes (source endpoints)
      ss << std::format("  \"{}\" [", tile->getSourceEndpoint().getName());
      ss << "style=filled;pin=true;";
      ss << std::format("label=\"{}, {}\\nSRC\";", i, j);
      ss << std::format("color=\"{}\";", color_scheme.stroke);
      ss << std::format("fillcolor=\"{}\";", color_scheme.fill);
      ss << std::format("pos=\"{},{}!\";",
                        j * kNodeShapeWidth * kNodeSpacingFactor -
                            0.5 * kNodeShapeWidth,
                        i * kNodeShapeWidth * kNodeSpacingFactor);
      ss << std::format("shape=rectangle;");
      ss << std::format("height={};", kNodeShapeWidth);
      ss << std::format("width={};", 0.5 * kNodeShapeWidth);
      ss << "];\n";
      // Tile nodes (sink endpoints)
      ss << std::format("  \"{}\" [", tile->getSinkEndpoint().getName());
      ss << "style=filled;pin=true;";
      ss << std::format("label=\"{}, {}\\nSNK\";", i, j);
      ss << std::format("color=\"{}\";", color_scheme.stroke);
      ss << std::format("fillcolor=\"{}\";", color_scheme.fill);
      ss << std::format("pos=\"{},{}!\";",
                        j * kNodeShapeWidth * kNodeSpacingFactor +
                            0.5 * kNodeShapeWidth,
                        i * kNodeShapeWidth * kNodeSpacingFactor);
      ss << std::format("shape=rectangle;");
      ss << std::format("height={};", kNodeShapeWidth);
      ss << std::format("width={};", 0.5 * kNodeShapeWidth);
      ss << "];\n";
      // SwitchBox nodes
      ss << std::format("  \"{}\" [", tile->getSwitchBox().getName());
      ss << "style=filled;pin=true;";
      ss << std::format("label=\"{}, {}\";", i, j);
      ss << std::format("color=\"{}\";", sb_color_scheme.stroke);
      ss << std::format("fillcolor=\"{}\";", sb_color_scheme.fill);
      ss << std::format(
          "pos=\"{},{}!\";",
          j * kNodeShapeWidth * kNodeSpacingFactor - kNodeShapeWidth,
          i * kNodeShapeWidth * kNodeSpacingFactor + kNodeShapeWidth);
      ss << std::format("width={};", kSwitchBoxShapeWidth);
      ss << "];\n";
    }
    ss << "\n";
  }

  ss << "\n  // Edges\n";
  for (size_t tree_i = 0; const auto &tree : rt_trees) {
    for (const auto &[from_node, to_node] : std::get<2>(tree)) {
      ss << std::format("  \"{}\" -> \"{}\" [", from_node, to_node);
      ss << std::format("label=\"#{}\";", std::get<0>(tree).getId());
      ss << std::format("tooltip=\"{}\";", std::get<1>(tree));
      ss << std::format("color=\"{}\";", utils::getStrokeColorLoop(tree_i));
      ss << std::format("penwidth=\"{}\";", kEdgeStrokeWidth);
      ss << "];\n";
    }
    tree_i++;
  }

  ss << "}\n";

  return ss.str();
}

} // namespace npu
} // namespace arch
