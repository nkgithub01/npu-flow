#ifndef _BASE_ENGINE_HPP_
#define _BASE_ENGINE_HPP_

#include <algorithm>
#include <fstream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>

#include "toml++/toml.hpp"

#include "base/common.hpp"
#include "base/placement.hpp"
#include "base/tf_graph.hpp"
#include "utils/misc.hpp"

namespace base {

class PnRNetlistFormat {
public:
  using ID = std::string;
  struct NodeInfo {
    ID id;
    std::optional<GridPosition> position;
  };
  struct EdgeInfo {
    ID id;
    ID start_node;
    std::vector<ID> target_nodes;
    std::optional<std::vector<size_t>> depth;
    std::optional<size_t> width;
  };
  struct LinkingInfo {
    ID id;
    std::vector<ID> from;
    std::vector<ID> to;
  };
  struct RouteInfo {
    // TODO: use the type to replace all related occurences
    using Path = std::vector<base::GridPosition>;
    using PathLookup = utils::Lookup<ID, Path>;
    ID edge_id;
    ID type;
    // Mapping target_node_id -> path of rr node positions (i.e., switch
    // positions for circuit/packet switching, and allocated tile position for
    // neighbor-sharing interconnects)
    PathLookup paths;
    // TODO: add more route info fields (e.g., memory usage) if needed
  };

protected:
  utils::Lookup<ID, NodeInfo> nodes_;
  utils::Lookup<ID, EdgeInfo> edges_;
  utils::Lookup<ID, LinkingInfo> linkings_;
  std::optional<utils::Lookup<ID, RouteInfo>>
      routes_; // Mapping edge_id -> route info

  static std::pair<std::vector<std::string>, std::vector<std::string>>
  parseHyperEdgeKey(const std::string &key) {
    size_t arrow_pos = key.find("->");
    if (arrow_pos == std::string::npos) {
      throw std::runtime_error(
          "Invalid hyper edge key format: missing '->' in " + key);
    }
    std::string froms_str = utils::trim(key.substr(0, arrow_pos));
    std::string tos_str = utils::trim(key.substr(arrow_pos + 2));

    if (froms_str.empty() || tos_str.empty()) {
      throw std::runtime_error(
          "Invalid hyper edge key format: empty froms or tos in " + key);
    }

    auto parse = [](const std::string &str) -> std::vector<ID> {
      std::vector<ID> ids;
      std::stringstream ss(str);
      for (ID id; std::getline(ss, id, ',');) {
        ids.push_back(utils::trim(id));
      }
      return ids;
    };

    return {parse(froms_str), parse(tos_str)};
  }

  static std::string generateHyperEdgeKey(const std::vector<ID> &froms,
                                          const std::vector<ID> &tos) {
    std::stringstream ss;
    ss << utils::toString(froms, ",") << "->" << utils::toString(tos, ",");
    return ss.str();
  }

  // Validate the structural integrity of the parsed data. Note: some of the
  // validation is duplicated in PnRNetlistReader since we assume this method
  // can also be called from PnRNetlistWriter.
  void validateStructuralData() const {
    // Validate no isolated nodes. Note: All nodes that appear in edges have
    // already been added to nodes_ in reader parsing or writer adding edges.
    for (const auto &node : std::views::keys(nodes_)) {
      bool is_start_node = std::ranges::any_of(
          std::views::values(edges_),
          [&](const auto &edge) { return edge.start_node == node; });
      bool is_target_node = std::ranges::any_of(
          std::views::values(edges_), [&](const auto &edge) {
            return std::ranges::find(edge.target_nodes, node) !=
                   edge.target_nodes.end();
          });
      if (!is_start_node && !is_target_node) {
        throw std::runtime_error(
            std::format("Node {} is not connected by any edge", node));
      }
    }

    // Validate edges has valid depth and width fields
    for (const auto &edge : std::views::values(edges_)) {
      // Check for zero values
      if (edge.width.has_value() && edge.width.value() == 0) {
        throw std::runtime_error(
            std::format("Edge {} has zero buffer width specified", edge.id));
      }
      if (edge.depth.has_value() &&
          std::ranges::any_of(*edge.depth, [](size_t d) { return d == 0; })) {
        throw std::runtime_error(
            std::format("Edge {} has zero buffer depth specified", edge.id));
      }
      // Check for nodes/depth count mismatch
      if (edge.depth.has_value() && edge.depth->size() != 1 &&
          (edge.depth->size() != edge.target_nodes.size() + 1)) {
        throw std::runtime_error(
            std::format("Edge {} depth field if provided must have exactly 1 "
                        "(uniform depth) or {} (per-node depth; first depth is "
                        "for start_node) entries",
                        edge.id, edge.target_nodes.size() + 1));
      }
      // Check for missing depth when width is specified and vice versa
      if (edge.depth.has_value() && !edge.width.has_value()) {
        throw std::runtime_error(std::format(
            "Edge {} has depth specified but missing width", edge.id));
      }
      if (edge.width.has_value() && !edge.depth.has_value()) {
        throw std::runtime_error(std::format(
            "Edge {} has width specified but missing depth", edge.id));
      }
    }

    // Validate linking types
    for (const auto &linking : std::views::values(linkings_)) {
      if (!((linking.from.size() == 1 && linking.to.size() == 1) ||
            (linking.from.size() >= 1 && linking.to.size() == 1) ||
            (linking.from.size() == 1 && linking.to.size() >= 1))) {
        throw std::runtime_error(std::format(
            "Linking {} must be one-to-one, one-to-many, or many-to-one",
            linking.id));
      }
    }

    // Validate linkings refer to existing edges
    for (const auto &linking : std::views::values(linkings_)) {
      for (const auto &from_edge : linking.from) {
        if (!edges_.contains(from_edge)) {
          throw std::runtime_error(
              std::format("Linking 'from' edge {} does not exist", from_edge));
        }
      }
      for (const auto &to_edge : linking.to) {
        if (!edges_.contains(to_edge)) {
          throw std::runtime_error(
              std::format("Linking 'to' edge {} does not exist", to_edge));
        }
      }
    }

    // Validate linking edges have a single joint node
    for (const auto &linking : std::views::values(linkings_)) {
      const ID &expected_joint_node = edges_.at(linking.to.front()).start_node;
      if (std::ranges::any_of(linking.to, [&](const auto &to_edge) {
            return edges_.at(to_edge).start_node != expected_joint_node;
          })) {
        throw std::runtime_error(std::format(
            "Linking {} 'to' edges do not share the same start node",
            linking.id));
      }
      if (std::ranges::any_of(linking.from, [&](const auto &from_edge) {
            return std::ranges::find(edges_.at(from_edge).target_nodes,
                                     expected_joint_node) ==
                   edges_.at(from_edge).target_nodes.end();
          })) {
        throw std::runtime_error(std::format(
            "Linking {} 'from' edges do not share the same target node",
            linking.id));
      }
    }

    // Validate linking edges have compatible widths or depths
    // TODO: consider relaxing this constraint in the future if needed, for
    // example, we can allow linking edges with different widths as long as the
    // the buffer demands (width * depth) are compatible.
    auto get_joint_node_of_linking = [&](const LinkingInfo &linking) -> ID {
      // Already validated to have a single joint node
      return edges_.at(linking.to.front()).start_node;
    };
    auto get_depth_of_edge_at_node = [&](const ID &edge_id,
                                         const ID &node_id) -> size_t {
      const auto &edge = edges_.at(edge_id);
      if (!edge.depth.has_value()) {
        return 0;
      }
      if (edge.start_node == node_id) {
        return edge.depth.value().front();
      }
      for (size_t i = 0; i < edge.target_nodes.size(); ++i) {
        if (edge.target_nodes.at(i) == node_id) {
          return edge.depth.value().at(i + 1);
        }
      }
      throw std::runtime_error(
          std::format("Edge {} does not connect to node {}", edge.id, node_id));
    };
    auto get_width_of_edge = [&](const ID &edge_id) -> size_t {
      const auto &edge = edges_.at(edge_id);
      return edge.width.value_or(0);
    };
    // TODO: refactor to avoid code redundancy
    for (const auto &linking : std::views::values(linkings_)) {
      const ID joint_node_id = get_joint_node_of_linking(linking);
      auto get_buffer_demand_at_joint_node = [&](const ID &edge_id) -> size_t {
        return get_width_of_edge(edge_id) *
               get_depth_of_edge_at_node(edge_id, joint_node_id);
      };

      // One-to-one linking (one pattern: data copying): buffer demand (width *
      // depth) must match between from and to edges
      if (linking.from.size() == 1 && linking.to.size() == 1) {
        const ID from_edge_id = linking.from.front();
        const ID to_edge_id = linking.to.front();

        if (get_buffer_demand_at_joint_node(from_edge_id) !=
            get_buffer_demand_at_joint_node(to_edge_id)) {
          throw std::runtime_error(std::format(
              "Linking {} passthrough 'from' edge {} and 'to' edge {} have "
              "incompatible buffer demands at joint node {}",
              linking.id, from_edge_id, to_edge_id, joint_node_id));
        }
      }
      // Many-to-one linking (one pattern: data merging): buffer demand of the
      // to edge at the joint node must equal the total buffer demand of from
      // edges at the joint node
      else if (linking.from.size() >= 1 && linking.to.size() == 1) {
        size_t total_buffer_demand_of_from_edges = 0;
        for (const auto &from_edge : linking.from) {
          total_buffer_demand_of_from_edges +=
              get_buffer_demand_at_joint_node(from_edge);
        }
        if (total_buffer_demand_of_from_edges !=
            get_buffer_demand_at_joint_node(linking.to.front())) {
          throw std::runtime_error(std::format(
              "Linking {} many-to-one 'from' edges and 'to' edge have "
              "incompatible buffer demands at joint node {}",
              linking.id, joint_node_id));
        }
      }
      // One-to-many linking (two patterns: data splitting and copying): either
      // (1) buffer demand of the from edge equals the total buffer demand of to
      // edges at the joint node, or (2) each to edge has the same buffer demand
      // as the from edge at the joint node
      else if (linking.from.size() == 1 && linking.to.size() >= 1) {
        auto is_data_splitting_case = [&]() -> bool {
          size_t total_buffer_demand_of_to_edges = 0;
          for (const auto &to_edge : linking.to) {
            total_buffer_demand_of_to_edges +=
                get_buffer_demand_at_joint_node(to_edge);
          }
          return get_buffer_demand_at_joint_node(linking.from.front()) ==
                 total_buffer_demand_of_to_edges;
        };
        auto is_data_copying_case = [&]() -> bool {
          for (const auto &to_edge : linking.to) {
            if (get_buffer_demand_at_joint_node(to_edge) !=
                get_buffer_demand_at_joint_node(linking.from.front())) {
              return false;
            }
          }
          return true;
        };
        if (!is_data_splitting_case() && !is_data_copying_case()) {
          throw std::runtime_error(std::format(
              "Linking {} one-to-many 'from' edge and 'to' edges have "
              "incompatible buffer demands at joint node {}",
              linking.id, joint_node_id));
        }
      }
    }

    // Validate routes if exist
    if (routes_.has_value()) {
      for (const auto &route : std::views::values(routes_.value())) {
        // Check route edge existence
        if (!edges_.contains(route.edge_id)) {
          throw std::runtime_error(std::format(
              "Route for edge {} refers to non-existent edge", route.edge_id));
        }
        // Check route paths cover all target nodes
        const auto &edge = edges_.at(route.edge_id);
        for (const auto &target_node : edge.target_nodes) {
          if (!route.paths.contains(target_node)) {
            throw std::runtime_error(
                std::format("Route for edge {} missing path for target node {}",
                            route.edge_id, target_node));
          }
        }
        // Check route paths do not contain non-target nodes
        for (const auto &path_node : std::views::keys(route.paths)) {
          if (std::ranges::none_of(edge.target_nodes, [&](const auto &target) {
                return target == path_node;
              })) {
            throw std::runtime_error(
                std::format("Route for edge {} has path for non-target node {}",
                            route.edge_id, path_node));
          }
        }
      }
    }
  }

protected:
  PnRNetlistFormat() = default; // only constructible by derived classes
};

class PnRNetlistReader : protected PnRNetlistFormat {
private:
  PnRNetlistReader() : PnRNetlistFormat() {}

public:
  static PnRNetlistReader fromTOML(const std::string &raw_str) {
    /*
      [node]
      shim   = "0,0"
      mem    = "1,0"
      comp_1 = "2,0"
      comp_2 = "3,0"
      comp_3 = "4,0"
      comp_4 = "5,0"

      [edge]
      0 = [ "shim->mem,comp_1", [2, 7, 2], 7680 ]
      1 = [ "comp_1->comp_2",   [4],       1920 ]
      2 = [ "comp_2->comp_3",   [2],       1920 ]
      3 = [ "comp_3->comp_4",   [7, 2],    7680 ]
      4 = [ "mem->comp_4",      [7, 2],    7680 ]
      5 = [ "comp_4->comp_4",   [1],       7680 ]
      6 = [ "comp_4->mem",      [2],       7680 ]
      7 = [ "mem->shim",        [2],       7680 ]

      [linking]
      link_0 = "0->4"
      link_1 = "6->7"

      [route] # optional
      0 = [ "circuit_switch", { mem=["0,0","1,0"],comp_1=["0,0","1,0","2,0"] } ]
      1 = [ "neighbor_sharing", { comp_2=["2,0"] } ]
      2 = [ "neighbor_sharing", { comp_3=["3,0"] } ]
      3 = [ "neighbor_sharing", { comp_4=["4,0"] } ]
      4 = [ "circuit_switch", { comp_4=["1,0","2,0","3,0","4,0","5,0"] } ]
      5 = [ "intra_tile", { comp_4=["5,0"] } ]
      6 = [ "circuit_switch", { mem=["5,0","4,0","3,0","2,0","1,0"] } ]
      7 = [ "circuit_switch", { shim=["1,0","0,0"] } ]
     */

    auto parsed = PnRNetlistReader();

    try {
      toml::table tbl = toml::parse(raw_str);

      // Parse nodes
      if (tbl.contains("node") && tbl["node"].is_table()) {
        const toml::table *node_tbl = tbl["node"].as_table();
        for (const auto &[k, v] : *node_tbl) {
          // If the key exists, there must be a value of string type
          parsed.nodes_.insert_or_assign(
              ID{k}, NodeInfo{.id = ID{k},
                              .position = v.value<std::string>().value()});
        }
      }

      // Parse edges
      if (tbl.contains("edge") && tbl["edge"].is_table()) {
        const toml::table *edge_tbl = tbl["edge"].as_table();
        for (const auto &[k, v] : *edge_tbl) {
          const toml::array *edge_info_arr = v.as_array();
          if (edge_info_arr->size() != 1 && edge_info_arr->size() != 3) {
            throw std::runtime_error(std::format(
                "Edge {} must have exactly 1 (only containing the path) or 3 "
                "(containing the path, depth, and the width in order) fields",
                std::string{k}));
          }

          EdgeInfo edge = {.id = ID{k}};
          // Parse path to get start_node and target_nodes
          auto parse_result = parseHyperEdgeKey(
              edge_info_arr->front().value<std::string>().value());
          edge.start_node = parse_result.first.front();
          edge.target_nodes = parse_result.second;
          // Insert start and target nodes of the edge into nodes_ if not exist
          auto check_node = [&](const ID &node_id) {
            if (!parsed.nodes_.contains(node_id)) {
              parsed.nodes_.insert_or_assign(
                  node_id, NodeInfo{.id = node_id, .position = std::nullopt});
            }
          };
          check_node(edge.start_node);
          for (const auto &t_node : edge.target_nodes) {
            check_node(t_node);
          }
          // Parse depth and width if exist
          if (edge_info_arr->size() == 3) {
            edge.depth = std::vector<size_t>{};
            const toml::array *depth_arr = edge_info_arr->at(1).as_array();
            if (depth_arr->size() != 1 &&
                (depth_arr->size() != edge.target_nodes.size() + 1)) {
              throw std::runtime_error(std::format(
                  "Edge {} depth field must have exactly 1 (uniform depth) or "
                  "{} (per-node depth; first depth is for start_node) entries",
                  std::string{k}, edge.target_nodes.size() + 1));
            }
            if (depth_arr->size() == 1) {
              // Uniform depth
              size_t uniform_depth = depth_arr->front().value<size_t>().value();
              for (size_t i = 0; i < edge.target_nodes.size() + 1; ++i) {
                edge.depth.value().push_back(uniform_depth);
              }
            } else {
              // Per-node depth
              for (const auto &d : *depth_arr) {
                edge.depth.value().push_back(d.value<size_t>().value());
              }
            }

            edge.width = edge_info_arr->at(2).value<size_t>().value();
          } else {
            edge.depth = std::nullopt;
            edge.width = std::nullopt;
          }

          parsed.edges_.insert_or_assign(ID{k}, edge);
        }
      }

      // Parse linkings
      if (tbl.contains("linking") && tbl["linking"].is_table()) {
        const toml::table *linking_tbl = tbl["linking"].as_table();
        for (const auto &[k, v] : *linking_tbl) {
          const std::string link_info = v.value<std::string>().value();
          auto parse_result = parseHyperEdgeKey(link_info);

          LinkingInfo linking{.id = ID{k},
                              .from = parse_result.first,
                              .to = parse_result.second};

          if (!((linking.from.size() == 1 && linking.to.size() == 1) ||
                (linking.from.size() >= 1 && linking.to.size() == 1) ||
                (linking.from.size() == 1 && linking.to.size() >= 1))) {
            throw std::runtime_error(std::format(
                "Linking {} must be one-to-one, one-to-many, or many-to-one",
                std::string{k}));
          }

          parsed.linkings_.insert_or_assign(ID{k}, linking);
        }
      }

      // Parse routes (optional)
      if (tbl.contains("route") && tbl["route"].is_table()) {
        const toml::table *route_tbl = tbl["route"].as_table();
        for (const auto &[k, v] : *route_tbl) {
          const toml::array *route_info_arr = v.as_array();
          if (route_info_arr->size() != 2) {
            throw std::runtime_error(std::format(
                "Route {} must have exactly 2 fields (type and paths)",
                std::string{k}));
          }

          RouteInfo route;
          route.edge_id = ID{k};
          route.type = route_info_arr->at(0).value<std::string>().value();

          const toml::table *paths_tbl = route_info_arr->at(1).as_table();
          for (const auto &[node_k, node_v] : *paths_tbl) {
            const toml::array *path_arr = node_v.as_array();
            std::vector<GridPosition> path;
            for (const auto &pos_val : *path_arr) {
              path.push_back(
                  GridPosition{pos_val.value<std::string>().value()});
            }
            route.paths.insert_or_assign(ID{node_k}, path);
          }

          if (!parsed.routes_.has_value()) {
            parsed.routes_ = utils::Lookup<ID, RouteInfo>{};
          }
          parsed.routes_->insert_or_assign(route.edge_id, route);
        }
      }
    } catch (const toml::parse_error &err) {
      // TOML parser will handle format errors like key duplications
      std::stringstream ss;
      ss << "PnRNetlistReader parsing failed:\n" << err;
      throw std::runtime_error(ss.str());
    }

    parsed.validateStructuralData();
    return parsed;
  }

  static PnRNetlistReader fromTOML(std::ifstream file) {
    return fromTOML(utils::readFileToString(std::move(file)));
  }

  std::vector<NodeInfo> getNodes() const {
    std::vector<NodeInfo> result;
    for (const auto &node : std::views::values(nodes_)) {
      result.push_back(node);
    }
    return result;
  }

  std::vector<EdgeInfo> getEdges() const {
    std::vector<EdgeInfo> result;
    for (const auto &edge : std::views::values(edges_)) {
      result.push_back(edge);
    }
    return result;
  }

  std::vector<LinkingInfo> getLinkings() const {
    std::vector<LinkingInfo> result;
    for (const auto &linking : std::views::values(linkings_)) {
      result.push_back(linking);
    }
    return result;
  }

  // TODO: add unit tests for this method
  std::optional<std::vector<RouteInfo>> getRoutes() const {
    if (!routes_.has_value()) {
      return std::nullopt;
    }
    std::vector<RouteInfo> result;
    for (const auto &route : std::views::values(routes_.value())) {
      result.push_back(route);
    }
    return result;
  }
};

class PnRNetlistWriter : protected PnRNetlistFormat {
public:
  PnRNetlistWriter() : PnRNetlistFormat() {}

  void addNode(const NodeInfo &node) {
    if (nodes_.contains(node.id)) {
      throw std::runtime_error(std::format("Node {} already exists", node.id));
    }
    nodes_.insert_or_assign(node.id, node);
  }

  void addEdge(const EdgeInfo &edge) {
    auto check_node = [&](const ID &node_id) {
      if (!nodes_.contains(node_id)) {
        addNode(NodeInfo{.id = node_id, .position = std::nullopt});
      }
    };
    edges_.insert_or_assign(edge.id, edge);
    check_node(edge.start_node);
    for (const auto &t_node : edge.target_nodes) {
      check_node(t_node);
    }
  }

  void addLinking(const LinkingInfo &linking) {
    linkings_.insert_or_assign(linking.id, linking);
  }

  void addRoute(const RouteInfo &route) {
    if (!routes_.has_value()) {
      routes_ = utils::Lookup<ID, RouteInfo>{};
    }
    if (routes_.value().contains(route.edge_id)) {
      throw std::runtime_error(
          std::format("Route for edge {} already exists", route.edge_id));
    }
    routes_.value().insert_or_assign(route.edge_id, route);
  }

  void validate() const { validateStructuralData(); }

  std::string toTOML(bool need_validate = true) const {
    // Set need_validate to false for debugging purpose only
    if (need_validate) {
      validate();
    }

    std::stringstream ss;

    if (!nodes_.empty()) {
      ss << "[node]\n";
      for (const auto &node : std::views::values(nodes_)) {
        if (node.position.has_value()) {
          ss << std::format("{} = \"{}\"\n", node.id,
                            node.position.value().toString());
        }
      }
    }
    if (!edges_.empty()) {
      ss << "\n[edge]\n";
      for (const auto &edge : std::views::values(edges_)) {
        if (edge.depth.has_value()) {
          ss << std::format(
              "{} = [ \"{}\", [ {} ], {} ]\n", edge.id,
              generateHyperEdgeKey({edge.start_node}, edge.target_nodes),
              utils::toString(edge.depth.value(), ", "), edge.width.value());
        } else {
          ss << std::format(
              "{} = [ \"{}\" ]\n", edge.id,
              generateHyperEdgeKey({edge.start_node}, edge.target_nodes));
        }
      }
    }
    if (!linkings_.empty()) {
      ss << "\n[linking]\n";
      for (const auto &linking : std::views::values(linkings_)) {
        ss << std::format("{} = \"{}\"\n", linking.id,
                          generateHyperEdgeKey(linking.from, linking.to));
      }
    }
    if (routes_.has_value() && !routes_->empty()) {
      ss << "\n[route]\n";
      for (const auto &route : std::views::values(routes_.value())) {
        ss << std::format("{} = [ \"{}\", {{ ", route.edge_id, route.type);
        size_t count = 0;
        for (const auto &[t_node, path] : route.paths) {
          ss << std::format("{} = [ {} ]", t_node,
                            utils::toString(path, ", ", "\"", "\""));
          if (count != route.paths.size() - 1) {
            ss << ", ";
          }
          ++count;
        }
        ss << " } ]\n";
      }
    }
    return ss.str();
  }
};

struct PnRPlacedNetlist {
  base::TrafficFlowGraph tf_graph;
  base::Placement placement;
};

} // namespace base

#endif // _BASE_ENGINE_HPP_
