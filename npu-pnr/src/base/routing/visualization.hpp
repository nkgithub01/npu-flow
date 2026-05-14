#ifndef _BASE_ROUTING_VISUALIZATION_HPP_
#define _BASE_ROUTING_VISUALIZATION_HPP_

#include <format>
#include <sstream>
#include <string>

#include "base/abstraction.hpp"
#include "base/routing/routing_net.hpp"
#include "base/routing/routing_state.hpp"
#include "utils/misc.hpp"

// TODO: split this file into multiple files under base/routing directory
namespace base {

// TODO: dummy class for now to make the header file happy
class RoutingVisualizer {};

// TODO: refactor the following function to avoid using RouteTree class
// TODO: visualize routing state as dot file
inline std::string visualizeAsDotFile(const std::vector<RouteTree> &rt_trees) {
  std::stringstream ss;
  double total_rt_tree_cost = 0.0;

  ss << "digraph RouteTrees {\n";
  ss << "  layout=dot;\n";

  for (const auto &tree : rt_trees) {
    const RoutingNet &net = tree.getAssociatedInputNet();
    const auto &net_id = net.getId();

    double rt_tree_cost = 0.0;
    utils::Set<RRNode> nodes;
    for (const auto &edge : tree.getEdges()) {
      nodes.insert(edge.getFromNode());
      nodes.insert(edge.getToNode());
      rt_tree_cost += edge.getCost();
    }
    total_rt_tree_cost += rt_tree_cost;

    ss << std::format("  subgraph cluster_{} {{\n", net_id);
    ss << std::format("    label=\"{}\\nRoute tree cost:{:.2f}\";\n",
                      net.getName(), rt_tree_cost);

    // Nodes
    for (const auto &node : nodes) {
      std::string fill_color =
          // TODO: fix buffer allocation visualization
          false ? "style=filled;fillcolor=gray88" : "";
      ss << std::format("    \"{}_{}\" [label=\"{}\";{}];\n", node.getName(),
                        net_id, node.getName(), fill_color);
    }

    // Edges
    for (const auto &edge : tree.getEdges()) {
      ss << std::format("    \"{}_{}\" -> \"{}_{}\" [label=\"{}\"];\n",
                        edge.getFromNode().getName(), net_id,
                        edge.getToNode().getName(), net_id,
                        utils::concateAttributes(edge.getAttributes()));
    }

    // Buffer Allocation
    const auto &start = net.getStartCore();
    ss << std::format("    \"buf_{}_{}\" [shape=\"box\";"
                      "label=\"Source buffer allocated "
                      "at {}\n{}/{}(alloc/demand)\"];\n",
                      start.getName(), net_id, start.getName(),
                      // TODO: fix buffer allocation visualization
                      0, // tree.getAllocatedBufferSize(src_ep),
                      net.getBufferDemand(start).getTotalBytes());
    for (const auto &target : net.getTargetCores()) {
      ss << std::format("    \"buf_{}_{}\" [shape=\"box\";"
                        "label=\"Sink buffer allocated "
                        "at {}\n{}/{}(alloc/demand)\"];\n",
                        target.getName(), net_id, target.getName(),
                        // TODO: fix buffer allocation visualization
                        0, // tree.getAllocatedBufferSize(sink_ep),
                        net.getBufferDemand(target).getTotalBytes());
    }

    ss << "  }\n";
  }
  ss << std::format("  label=\"Total route tree cost: {:.2f}\";\n",
                    total_rt_tree_cost);
  ss << "}\n";

  return ss.str();
}

} // namespace base

#endif
