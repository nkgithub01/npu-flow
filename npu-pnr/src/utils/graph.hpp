#ifndef _UTILS_GRAPH_HPP_
#define _UTILS_GRAPH_HPP_

#include <algorithm>
#include <concepts>
#include <format>
#include <map>
#include <memory>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#include "utils/misc.hpp"
#include "utils/visualization.hpp"

namespace utils {

enum class GraphTopology { Any = 0, Grid };

class GraphNode : public AttributableClass {
public:
  GraphNode() = default;
  virtual ~GraphNode() = default;
};

enum class GraphEdgeType {
  RegularEdge = 0, // connects two nodes
  HyperEdge = 1,   // connects multiple nodes
};

template <typename Node, GraphEdgeType EdgeType = GraphEdgeType::RegularEdge>
  requires std::derived_from<Node, GraphNode>
class GraphEdge : public AttributableClass {
private:
  using NodePtr = std::shared_ptr<Node>;
  using FromNodePtr = NodePtr;
  using ToNodePtrOrPtrs =
      std::conditional_t<EdgeType == GraphEdgeType::HyperEdge,
                         std::vector<NodePtr>, NodePtr>;

  FromNodePtr from_;
  ToNodePtrOrPtrs to_;

public:
  GraphEdge() = delete;
  GraphEdge(FromNodePtr from, ToNodePtrOrPtrs to)
      : from_(std::move(from)), to_(std::move(to)) {}
  virtual ~GraphEdge() = default;

  [[nodiscard]] static constexpr bool isHyperEdge() {
    return EdgeType == GraphEdgeType::HyperEdge;
  }

  FromNodePtr getFromNodePtr() const { return from_; }
  ToNodePtrOrPtrs getToNodePtr() const
    requires(EdgeType == GraphEdgeType::RegularEdge)
  {
    return to_;
  }
  ToNodePtrOrPtrs getToNodePtrs() const
    requires(EdgeType == GraphEdgeType::HyperEdge)
  {
    return to_;
  }

  [[nodiscard]] bool hasNode(const Node &node) const {
    if constexpr (EdgeType == GraphEdgeType::RegularEdge) {
      return *from_ == node || *to_ == node;
    } else {
      if (*from_ == node) {
        return true;
      }
      for (const auto &to_ptr : to_) {
        if (*to_ptr == node) {
          return true;
        }
      }
      return false;
    }
  }
};

template <typename Node, typename Edge,
          GraphTopology kTopo = GraphTopology::Any>
  requires requires(Node a, Node b, Edge) {
    { a < b } -> std::same_as<bool>; // used for std::map
    { Edge::isHyperEdge() } -> std::same_as<bool>;
  } && (std::constructible_from<Edge, std::shared_ptr<Node>,
                                std::shared_ptr<Node>> ||
        std::constructible_from<Edge, std::shared_ptr<Node>,
                                std::vector<std::shared_ptr<Node>>>)
class GraphClass : public AttributableClass {
private:
  using NodePtr = std::shared_ptr<Node>;
  using AnyTopoNodeContainer = std::vector<NodePtr>;
  // GridTopoNodeContainer[row][col][first node is backbone node, e.g.,
  // switchbox; others are tile local components]
  using GridTopoNodeContainer = std::vector<std::vector<std::vector<NodePtr>>>;
  using NodeContainer =
      std::conditional_t<kTopo == GraphTopology::Grid, GridTopoNodeContainer,
                         AnyTopoNodeContainer>;

  NodeContainer nodes_;
  // TODO: consider using std::unordered_map for better performance
  std::map<Node, std::conditional_t<kTopo == GraphTopology::Grid,
                                    std::tuple<size_t, size_t, NodePtr>,
                                    NodePtr>>
      node_ptr_lookup_; // for fast lookup by name

  using EdgePtr = std::shared_ptr<Edge>;
  using EdgeContainer = std::vector<EdgePtr>;

  EdgeContainer edges_;

protected: // only called by a derived class
  NodePtr addNode(const Node &node)
    requires(kTopo == GraphTopology::Any)
  {
    if (node_ptr_lookup_.contains(node)) {
      return node_ptr_lookup_[node];
    }
    NodePtr ptr = std::make_shared<Node>(node);
    nodes_.push_back(ptr);
    node_ptr_lookup_[node] = ptr; // store the pointer for fast lookup
    return ptr;
  }

  NodePtr addNode(const Node &node, size_t row, size_t col,
                  bool is_grid_backbone_node = false)
    requires(kTopo == GraphTopology::Grid)
  {
    if (!(row < nodes_.size() && col < nodes_[row].size())) {
      throw std::runtime_error("Grid row or column index out of bounds");
    }

    auto &grid_tile = nodes_[row][col];

    if (node_ptr_lookup_.contains(node)) {
      // Node already exists on a specific position in the grid
      // Check if the input position matches the existing one
      const auto [exist_row, exist_col, ptr] = node_ptr_lookup_[node];
      if (!(exist_row == row && exist_col == col)) {
        throw std::runtime_error(
            "Node exists on a different position in the grid");
      }
      if (is_grid_backbone_node) {
        if (grid_tile[0] != ptr) {
          throw std::runtime_error(
              "Node exists on a different position in the grid");
        }
      } else {
        if (std::find(grid_tile.begin() + 1, grid_tile.end(), ptr) ==
            grid_tile.end()) {
          throw std::runtime_error(
              "Node exists on a different position in the grid");
        }
      }
      return ptr;
    }

    NodePtr ptr = std::make_shared<Node>(node);
    if (is_grid_backbone_node) {
      if (grid_tile[0] != nullptr) {
        throw std::runtime_error(
            "Grid backbone node already exists at this position");
      }
      grid_tile[0] = ptr; // add a backbone node
    } else {
      grid_tile.push_back(ptr); // add a local component
    }
    node_ptr_lookup_[node] =
        std::make_tuple(row, col, ptr); // store the pointer for fast lookup
    return ptr;
  }

  NodePtr getNodePtr(const Node &node) const {
    if (!node_ptr_lookup_.contains(node)) {
      throw std::runtime_error(std::format(
          "Node {} not found in the graph. Ensure that the node is added "
          "before accessing it.",
          node.getName()));
    }
    if constexpr (kTopo == GraphTopology::Grid) {
      return std::get<2>(node_ptr_lookup_.at(node));
    } else {
      return node_ptr_lookup_.at(node);
    }
  }

  // TODO: consider using std::ranges::random_access_range and merge it with
  // getNodeViews() to allow random access to nodes
  NodePtr getNodePtrByInternalIndex(size_t node_index_in_graph_class) const {
    if (node_index_in_graph_class >= getNumNodes()) {
      throw std::runtime_error("Node index out of bounds");
    }
    auto it = node_ptr_lookup_.begin();
    std::advance(it, node_index_in_graph_class);
    if constexpr (kTopo == GraphTopology::Grid) {
      return std::get<2>(it->second);
    } else {
      return it->second;
    }
  }

  EdgePtr addEdge(const NodePtr from, const NodePtr to)
    requires(!Edge::isHyperEdge())
  {
    // TODO: add support for hyper edges
    const EdgePtr ptr = std::make_shared<Edge>(from, to);
    edges_.push_back(ptr);
    return ptr;
  }

  EdgePtr addEdge(const NodePtr from, const std::vector<NodePtr> tos)
    requires(Edge::isHyperEdge())
  {
    // TODO: add support for hyper edges
    const EdgePtr ptr = std::make_shared<Edge>(from, tos);
    edges_.push_back(ptr);
    return ptr;
  }

public:
  GraphClass() = default;
  GraphClass(const size_t num_grid_rows, const size_t num_grid_cols)
    requires(kTopo == GraphTopology::Grid)
  {
    std::vector<NodePtr> grid_tile{nullptr}; // first node is a backbone
    std::vector<std::vector<NodePtr>> grid_row(num_grid_cols, grid_tile);
    nodes_.assign(num_grid_rows, grid_row);
  }
  virtual ~GraphClass() = default;
  // GraphClass(const GraphClass &) = delete;
  // GraphClass &operator=(const GraphClass &) = delete;

  size_t getNumNodes() const { return node_ptr_lookup_.size(); }
  size_t getNumEdges() const { return edges_.size(); }

  auto getNodeViews() const -> std::ranges::input_range auto {
    if constexpr (kTopo == GraphTopology::Grid) {
      // Flatten the grid-like 3d vector into a single view
      // TODO: consider using std::views::transform to apply a function
      // to each node on the node_ptr_lookup_ map
      return nodes_ | std::views::join | std::views::join;
    } else {
      return nodes_ | std::views::all;
    }
  }

  auto getEdgeViews() const -> std::ranges::input_range auto { return edges_; }

  // TODO: refactor visualization code (probably to a separate file)
  std::string
  visualizeAsDotFile(const std::string selected_highlighted_color_scheme =
                         "red" /*TODO: make color scheme config cleaner*/) const
    requires requires(NodePtr n) {
      { n->getName() } -> std::convertible_to<std::string>;
      { n->getAttributes() } -> std::convertible_to<const AttributeMap &>;
    } && requires(EdgePtr e) {
      { e->getName() } -> std::convertible_to<std::string>;
      { e->getAttributes() } -> std::convertible_to<const AttributeMap &>;
      { e->getFromNodePtr() } -> std::convertible_to<NodePtr>;
      { e->getToNodePtr() } -> std::convertible_to<NodePtr>; // TODO: HyperEdges
    }
  {
    auto clean_escape_characters = [](const std::string &str) -> std::string {
      std::string result;
      // Escape double quotes and backslashes in the string
      // TODO: consider escaping other characters if needed
      for (const char &c : str) {
        if (c == '"' || c == '\\') {
          result += '\\'; // add backslash before the character
        }
        result += c;
      }
      return result;
    };

    // TODO: consider using AttributeMap and merging with concateAttributes
    auto append_vis_attr = [](std::string &dot_vis_attr, const std::string key,
                              const std::string value) {
      if (!dot_vis_attr.empty()) {
        dot_vis_attr += ",";
      }
      dot_vis_attr += std::format("{}=\"{}\"", key, value);
    };

    // TODO: clean up this lambda function, e.g., is_highlighted_node?
    auto get_grid_backbone_node_pos_if_any =
        [this](
            const NodePtr &node) -> std::optional<std::pair<size_t, size_t>> {
      if constexpr (kTopo == GraphTopology::Grid) {
        auto [row_x, col_y, ptr] = this->node_ptr_lookup_.at(*node);
        if (this->nodes_[row_x][col_y][0] == ptr) {
          return std::make_pair(row_x, col_y);
        }
        return std::nullopt;
      } else {
        return std::nullopt;
      }
    };

    // TODO: make this nicely configurable
    const auto [highlighted_fill_color, highlighted_stroke_color] =
        utils::highlighted_color_scheme.at(selected_highlighted_color_scheme);
    const std::string highlighted_stroke_width = "3";

    std::stringstream ss;

    ss << "digraph G {\n";

    if constexpr (kTopo == GraphTopology::Grid) {
      // TODO: Uncomment the following lines if using neato layout
      // ss << "  layout=neato;\n";
      // ss << "  normalize=true;\n";
    } else {
      ss << "  layout=dot;\n";
      ss << "  rankdir=LR;\n"; // left to right
    }

    // TODO: use record-based nodes
    // ss << "  node [shape=record];\n\n";

    ss << "\n  // Nodes\n";

    for (const NodePtr &node : getNodeViews()) {
      std::string node_vis_attrs;

      if (const auto attrs =
              clean_escape_characters(concateAttributes(node->getAttributes()));
          !attrs.empty()) {
        append_vis_attr(node_vis_attrs, "label",
                        std::format("\\N\\n{}", attrs));
      } else {
        append_vis_attr(node_vis_attrs, "label", "\\N");
      }

      if constexpr (kTopo == GraphTopology::Grid) {
        if (get_grid_backbone_node_pos_if_any(node)
                .has_value()) { // backbone node
          append_vis_attr(node_vis_attrs, "shape", "box");
          append_vis_attr(node_vis_attrs, "color", highlighted_stroke_color);
          append_vis_attr(node_vis_attrs, "style", "filled");
          append_vis_attr(node_vis_attrs, "fillcolor", highlighted_fill_color);
          append_vis_attr(node_vis_attrs, "penwidth", highlighted_stroke_width);
        } else { // local component
          append_vis_attr(node_vis_attrs, "shape", "ellipse");
        }
      }

      ss << std::format(
          "  \"{}\"{};\n", clean_escape_characters(node->getName()),
          node_vis_attrs.empty() ? "" : std::format(" [{}]", node_vis_attrs));
    }

    ss << "\n  // Edges\n";

    for (const EdgePtr &edge : getEdgeViews()) {
      // TODO: support hyper edges in the visualization. Reference:
      // https://gitlab.com/graphviz/graphviz/-/issues/1911
      static_assert(!Edge::isHyperEdge(),
                    "HyperEdges are not supported yet in the visualization.");

      NodePtr from_node = edge->getFromNodePtr();
      NodePtr to_node = edge->getToNodePtr();

      std::string edge_vis_attrs;

      if constexpr (kTopo == GraphTopology::Grid) {
        if (const auto from_pos = get_grid_backbone_node_pos_if_any(from_node),
            to_pos = get_grid_backbone_node_pos_if_any(to_node);
            from_pos.has_value() && to_pos.has_value()) {
          // TODO: uncomment the following code if using neato layout
          // auto [from_row, from_col] = from_pos.value();
          // auto [to_row, to_col] = to_pos.value();
          // size_t avg_num_nodes = (nodes_[from_row][from_col].size() - 1 +
          //                         nodes_[to_row][to_col].size() - 1) /
          //                        2;
          // append_vis_attr(edge_vis_attrs,
          //                 "len" /*default is 1.0 according to DOT format*/,
          //                 std::to_string(5));
          append_vis_attr(edge_vis_attrs, "color", highlighted_stroke_color);
          append_vis_attr(edge_vis_attrs, "penwidth", highlighted_stroke_width);
        }
      }

      if (const auto attrs =
              clean_escape_characters(concateAttributes(edge->getAttributes()));
          !attrs.empty()) {
        append_vis_attr(edge_vis_attrs, "label", attrs);
      }

      ss << std::format(
          "  \"{}\" -> \"{}\"{};\n",
          clean_escape_characters(from_node->getName()),
          clean_escape_characters(to_node->getName()),
          edge_vis_attrs.empty() ? "" : std::format(" [{}]", edge_vis_attrs));
    }

    ss << "}\n";

    return ss.str();
  }
};

// Adjacency list representation of a directed graph
template <typename Node, typename EdgePayload> class AdjacencyList {
public:
  using Edge = std::pair<Node, EdgePayload>;

private:
  utils::Lookup<Node, std::vector<Edge>> adj_list_;
  static constexpr std::vector<Edge> kEmptyEdgeList{};

public:
  AdjacencyList() = default;

  void addEdge(const Node &from, const Node &to, const EdgePayload &payload) {
    adj_list_[from].emplace_back(to, payload);
  }

  const utils::Lookup<Node, std::vector<Edge>> &data() const {
    return adj_list_;
  }

  const std::vector<Edge> &getOutgoingEdges(const Node &node) const {
    if (!adj_list_.contains(node)) {
      return kEmptyEdgeList;
    }
    return adj_list_.at(node);
  }
};

} // namespace utils

#endif
