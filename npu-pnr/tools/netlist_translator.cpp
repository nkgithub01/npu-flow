#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "argparse/argparse.hpp"
#include "nlohmann/json.hpp"

#include "base/common.hpp"
#include "base/engine.hpp"

using json = nlohmann::ordered_json;

class NetlistTranslator {
public:
  static void translateJSONtoTOML(const std::string &json_path,
                                  const std::string &toml_path) {
    // Read JSON file
    json j = json::parse(std::ifstream(json_path));

    // Create TOML writer
    base::PnRNetlistWriter writer;

    // Map node IDs to node names
    std::unordered_map<int, std::string> node_id_to_name;
    std::unordered_map<int, base::GridPosition> node_id_to_position;

    // Process nodes
    if (j.contains("nodes") && j["nodes"].is_array()) {
      for (const auto &node : j["nodes"]) {
        int node_id = node["id"];
        std::string node_type = node["type"];

        // Convert type to lowercase for naming
        std::string type_lower = node_type;
        std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(),
                       ::tolower);

        // Generate unique node name
        std::string node_name = std::format("{}_{}", type_lower, node_id);
        node_id_to_name[node_id] = node_name;

        // Create node with position if available
        base::PnRNetlistFormat::NodeInfo node_info;
        node_info.id = node_name;

        if (node.contains("col_x") && node.contains("row_y")) {
          int col_x = node["col_x"];
          int row_y = node["row_y"];
          node_info.position = base::GridPosition{row_y, col_x};
        } else {
          node_info.position = std::nullopt;
        }
        // TODO: switch to a better way to represent undefined positions
        node_id_to_position.insert_or_assign(
            node_id, node_info.position.value_or(base::GridPosition{-1, -1}));

        writer.addNode(node_info);
      }
    }

    // Map net IDs to edge IDs for linking processing
    std::unordered_map<int, std::string> net_id_to_edge_id;

    // Process nets (edges)
    if (j.contains("nets") && j["nets"].is_array()) {
      for (const auto &net : j["nets"]) {
        int net_id = net["net_id"];
        int src_id = net["src_id"];
        std::vector<int> dst_ids = net["dst_ids"].get<std::vector<int>>();

        // Create edge ID
        std::string edge_id = std::to_string(net_id);
        net_id_to_edge_id[net_id] = edge_id;

        base::PnRNetlistFormat::EdgeInfo edge_info;
        edge_info.id = edge_id;
        edge_info.start_node = node_id_to_name.at(src_id);

        for (int dst_id : dst_ids) {
          edge_info.target_nodes.push_back(node_id_to_name.at(dst_id));
        }

        // Process depths and width
        if (net.contains("depths") && net.contains("byte_size_per_depth")) {
          std::vector<size_t> depths = net["depths"].get<std::vector<size_t>>();
          size_t width = net["byte_size_per_depth"].get<size_t>();
          if (depths.size() != 1 && depths.size() != (dst_ids.size() + 1)) {
            throw std::runtime_error(
                "Mismatch in depths size for net ID " + std::to_string(net_id) +
                ". Expected size 1 or " + std::to_string(dst_ids.size() + 1) +
                ", got " + std::to_string(depths.size()) + ".");
          }
          std::vector<size_t> adjusted_depths;
          if (depths.size() == 1) {
            adjusted_depths.resize(dst_ids.size() + 1, depths[0]);
          } else {
            adjusted_depths = depths;
          }
          edge_info.depth = adjusted_depths;
          edge_info.width = width;
        } else {
          edge_info.depth = std::nullopt;
          edge_info.width = std::nullopt;
        }

        // Process routing_info if available
        if (net.contains("routing_info")) {
          const auto &routing_info_obj = net["routing_info"];
          base::PnRNetlistFormat::RouteInfo route_info;
          route_info.edge_id = edge_id;
          route_info.type = routing_info_obj["connection_type"];

          if (route_info.type == "neighbor_sharing") {
            // "allocation_tiles": [{"col_x": <>, "row_y": <>}, ...]
            for (int path_id = 0;
                 const auto &path_obj : routing_info_obj["allocation_tiles"]) {
              if (path_obj.size() != 2) {
                throw std::runtime_error(
                    "Invalid path size for neighbor_sharing route.");
              }
              base::PnRNetlistFormat::RouteInfo::Path path;
              path.push_back(base::GridPosition{path_obj["row_y"].get<int>(),
                                                path_obj["col_x"].get<int>()});
              route_info.paths.insert_or_assign(
                  node_id_to_name.at(net["dst_ids"][path_id++]), path);
            }
          } else if (route_info.type == "circuit_switch" ||
                     route_info.type == "packet_switch") {
            // "intermediates": [[[<col_x>, <row_y>], ...], ...]
            for (int path_id = 0;
                 const auto &path_obj : routing_info_obj["intermediates"]) {
              base::PnRNetlistFormat::RouteInfo::Path path;
              for (const auto &pos_array : path_obj) {
                if (pos_array.size() != 2) {
                  throw std::runtime_error(
                      "Invalid position array size in intermediates.");
                }
                int col_x = pos_array[0].get<int>();
                int row_y = pos_array[1].get<int>();
                path.push_back(base::GridPosition{row_y, col_x});
              }
              route_info.paths.insert_or_assign(
                  node_id_to_name.at(net["dst_ids"][path_id++]), path);
            }
          } else if (route_info.type == "intra_tile") {
            base::PnRNetlistFormat::RouteInfo::Path path;
            if (net["dst_ids"].size() != 1) {
              throw std::runtime_error(
                  "Intra-tile route must have exactly one destination.");
            }
            int node_id = net["dst_ids"][0].get<int>();
            path.push_back(node_id_to_position.at(node_id));
            route_info.paths.insert_or_assign(node_id_to_name.at(node_id),
                                              path);
          }

          writer.addRoute(route_info);
        }

        writer.addEdge(edge_info);
      }
    }

    // Process links (linkings)
    if (j.contains("links") && j["links"].is_array()) {
      int link_count = 0;
      for (const auto &link : j["links"]) {
        std::vector<int> src_net_ids =
            link["src_net_ids"].get<std::vector<int>>();
        std::vector<int> dst_net_ids =
            link["dst_net_ids"].get<std::vector<int>>();

        base::PnRNetlistFormat::LinkingInfo linking_info;
        linking_info.id = "link_" + std::to_string(link_count++);

        for (int src_net_id : src_net_ids) {
          linking_info.from.push_back(net_id_to_edge_id.at(src_net_id));
        }

        for (int dst_net_id : dst_net_ids) {
          linking_info.to.push_back(net_id_to_edge_id.at(dst_net_id));
        }

        writer.addLinking(linking_info);
      }
    }

    // Write to TOML file
    std::string toml_content = writer.toTOML();

    std::ofstream toml_file(toml_path);
    if (!toml_file.is_open()) {
      throw std::runtime_error("Failed to open output file: " + toml_path);
    }
    toml_file << toml_content;
    toml_file.close();

    std::cout << "Successfully translated " << json_path << " to " << toml_path
              << std::endl;
  }

  static void translateTOMLtoJSON(const std::string &toml_path,
                                  const std::string &json_path) {
    // Read TOML file
    std::ifstream toml_file(toml_path);
    if (!toml_file.is_open()) {
      throw std::runtime_error("Failed to open TOML file: " + toml_path);
    }

    // Parse TOML using PnRNetlistReader
    base::PnRNetlistReader reader =
        base::PnRNetlistReader::fromTOML(std::move(toml_file));

    // Create JSON object
    json j;

    // Extract nodes, edges, and linkings
    auto nodes = reader.getNodes();
    auto edges = reader.getEdges();
    auto linkings = reader.getLinkings();

    // Map node names to node IDs (we make the same assumption as in
    // JSON->TOML translation that node names are in the format "[type]_[id]")
    auto parse_node_name =
        [](const std::string &node_name) -> std::pair<std::string, int> {
      size_t underscore_pos = node_name.find('_');
      if (underscore_pos == std::string::npos) {
        throw std::runtime_error("Invalid node name format: " + node_name);
      }
      // Extract type from name (e.g., "shim_0" -> "SHIM", "comp_1" -> "COMP")
      std::string node_type = node_name.substr(0, underscore_pos);
      std::transform(node_type.begin(), node_type.end(), node_type.begin(),
                     ::toupper);
      // Extract ID from name
      std::string id_str = node_name.substr(underscore_pos + 1);
      return {node_type, std::stoi(id_str)};
    };

    // Process nodes
    std::unordered_map<std::string, int> node_name_to_id_lookup;
    json nodes_array = json::array();
    for (const auto &node : nodes) {
      json node_obj;

      auto [node_type, node_id] = parse_node_name(node.id);
      node_name_to_id_lookup[node.id] = node_id;

      node_obj["id"] = node_id;
      node_obj["type"] = node_type;

      // Add position if available
      if (node.position.has_value()) {
        base::GridPosition pos = node.position.value();
        node_obj["col_x"] = pos.getColX();
        node_obj["row_y"] = pos.getRowY();
      } else {
        node_obj["col_x"] = -1;
        node_obj["row_y"] = -1;
      }

      nodes_array.push_back(node_obj);
    }
    j["nodes"] = nodes_array;

    // Map edge IDs to net IDs (we make the same assumption as in JSON->TOML
    // translation that edge IDs are integers in string format)
    auto get_edge_id_from_name = [](const std::string &edge_name) -> int {
      try {
        return std::stoi(edge_name);
      } catch (...) {
        throw std::runtime_error("Invalid edge ID format: " + edge_name);
      }
    };
    auto check_and_get_node_id_from_name =
        [&node_name_to_id_lookup](const std::string &node_name) -> int {
      if (node_name_to_id_lookup.contains(node_name)) {
        return node_name_to_id_lookup.at(node_name);
      } else {
        throw std::runtime_error("Node name not found: " + node_name);
      }
    };

    // Process edges (nets)
    std::unordered_map<std::string, int> edge_name_to_id_lookup;
    json nets_array = json::array();
    for (const auto &edge : edges) {
      json net_obj;

      int edge_id = get_edge_id_from_name(edge.id);
      net_obj["net_id"] = edge_id;
      edge_name_to_id_lookup[edge.id] = edge_id;

      // Map node names to IDs
      net_obj["src_id"] = check_and_get_node_id_from_name(edge.start_node);

      json dst_ids_array = json::array();
      for (const auto &target : edge.target_nodes) {
        dst_ids_array.push_back(check_and_get_node_id_from_name(target));
      }
      net_obj["dst_ids"] = dst_ids_array;

      // Add depth and width if available
      if (edge.depth.has_value() && edge.width.has_value()) {
        net_obj["depths"] = edge.depth.value();
        net_obj["byte_size_per_depth"] = edge.width.value();
      }

      nets_array.push_back(net_obj);
    }
    j["nets"] = nets_array;

    // Process linkings (links)
    auto check_and_get_edge_id_from_name =
        [&edge_name_to_id_lookup](const std::string &edge_name) -> int {
      if (edge_name_to_id_lookup.contains(edge_name)) {
        return edge_name_to_id_lookup.at(edge_name);
      } else {
        throw std::runtime_error("Edge name not found: " + edge_name);
      }
    };
    json links_array = json::array();
    for (const auto &_ : linkings) {
      // Used to enforce the linking order based on linking.id
      links_array.push_back(json::object());
    }
    for (const auto &linking : linkings) {
      json link_obj;

      json src_net_ids_array = json::array();
      for (const auto &from_edge : linking.from) {
        src_net_ids_array.push_back(check_and_get_edge_id_from_name(from_edge));
      }
      link_obj["src_net_ids"] = src_net_ids_array;

      json dst_net_ids_array = json::array();
      for (const auto &to_edge : linking.to) {
        dst_net_ids_array.push_back(check_and_get_edge_id_from_name(to_edge));
      }
      link_obj["dst_net_ids"] = dst_net_ids_array;

      int link_id = std::stoi(linking.id.substr(
          std::string("link_").length())); // assuming format "link_X"
      links_array.at(link_id) = link_obj;
    }
    j["links"] = links_array;

    // Optional: Process routes (routing_infos)
    if (reader.getRoutes().has_value()) {
      auto routes = reader.getRoutes().value();

      auto check_and_get_net_json = [&j](int net_id) -> json & {
        for (auto &net : j["nets"]) {
          if (net["net_id"] == net_id) {
            return net;
          }
        }
        throw std::runtime_error("Net ID not found in JSON: " +
                                 std::to_string(net_id));
      };

      auto get_order_enforced_path_vec =
          [&](int net_id,
              const base::PnRNetlistFormat::RouteInfo::PathLookup &path) {
            auto get_node_name_from_id =
                [&node_name_to_id_lookup](int node_id) -> std::string {
              for (const auto &[name, id] : node_name_to_id_lookup) {
                if (id == node_id) {
                  return name;
                }
              }
              throw std::runtime_error("Node name not found for ID: " +
                                       std::to_string(node_id));
            };
            json &net_obj = check_and_get_net_json(net_id);
            std::vector<std::vector<base::GridPosition>> ordered_paths;
            for (const auto &dst : net_obj["dst_ids"]) {
              std::string dst_node_name = get_node_name_from_id(dst.get<int>());
              if (!path.contains(dst_node_name)) {
                throw std::runtime_error(
                    "Path not found for destination node: " + dst_node_name);
              }
              ordered_paths.push_back(path.at(dst_node_name));
            }
            return ordered_paths;
          };

      for (const auto &route : routes) {
        const int edge_id = std::stoi(route.edge_id);
        const std::string route_type = route.type;
        const auto paths = route.paths;
        json &net_obj = check_and_get_net_json(edge_id);

        json routing_info = json::object();
        // "connection_type": <type>
        routing_info["connection_type"] = route_type;

        auto ordered_paths = get_order_enforced_path_vec(edge_id, paths);
        if (route_type == "neighbor_sharing") {
          // "allocation_tiles": [{"col_x": <>, "row_y": <>}, ...]
          routing_info["allocation_tiles"] = json::array();
          for (const auto &path : ordered_paths) {
            if (path.size() != 1) {
              throw std::runtime_error(
                  "Invalid path size for neighbor_sharing route.");
            }
            routing_info["allocation_tiles"].push_back(
                {{"col_x", path[0].getColX()}, {"row_y", path[0].getRowY()}});
          }
        } else if (route_type == "circuit_switch" ||
                   route_type == "packet_switch") {
          // "intermediates": [[[<col_x>, <row_y>], ...], ...]
          routing_info["intermediates"] = json::array();
          for (const auto &path : ordered_paths) {
            json intermediate_array = json::array();
            // First and last positions (source and sink ep) are already
            // skipped
            for (const base::GridPosition &pos : path) {
              intermediate_array.push_back(
                  json::array({pos.getColX(), pos.getRowY()}));
            }
            routing_info["intermediates"].push_back(intermediate_array);
          }
        } else if (route_type == "intra_tile") {
          // Skip for intra_tile routes
        }

        net_obj["routing_info"] = routing_info;
      }
    }

    std::sort(
        j["nodes"].begin(), j["nodes"].end(), [](const json &a, const json &b) {
          return a["id"].template get<int>() < b["id"].template get<int>();
        });

    std::sort(j["nets"].begin(), j["nets"].end(),
              [](const json &a, const json &b) {
                return a["net_id"].template get<int>() <
                       b["net_id"].template get<int>();
              });

    // Write to JSON file
    std::ofstream json_file(json_path);
    if (!json_file.is_open()) {
      throw std::runtime_error("Failed to open output file: " + json_path);
    }
    json_file << j.dump(2); // Pretty print with 2-space indentation
    json_file.close();

    std::cout << "Successfully translated " << toml_path << " to " << json_path
              << std::endl;
  }
};

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("netlist_translator");

  program.add_argument("input").help(
      "path to the input netlist file (JSON or TOML)");

  program.add_argument("-o", "--output")
      .help("path to the output netlist file (format auto-detected from "
            "extension)");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }

  std::string input_path = program.get<std::string>("input");

  // Determine input file extension
  std::string input_ext;
  size_t dot_pos = input_path.find_last_of('.');
  if (dot_pos != std::string::npos) {
    input_ext = input_path.substr(dot_pos);
    std::transform(input_ext.begin(), input_ext.end(), input_ext.begin(),
                   ::tolower);
  }

  // Determine output path and extension
  std::string output_path;
  std::string output_ext;

  if (program.is_used("--output")) {
    output_path = program.get<std::string>("--output");
    size_t out_dot_pos = output_path.find_last_of('.');
    if (out_dot_pos != std::string::npos) {
      output_ext = output_path.substr(out_dot_pos);
      std::transform(output_ext.begin(), output_ext.end(), output_ext.begin(),
                     ::tolower);
    }
  } else {
    // Auto-generate output filename based on input format
    std::string base_name = input_path.substr(0, dot_pos);
    if (input_ext == ".json") {
      output_path = base_name + ".toml";
      output_ext = ".toml";
    } else if (input_ext == ".toml") {
      output_path = base_name + ".json";
      output_ext = ".json";
    } else {
      std::cerr
          << "Error: Could not determine input file format from extension. "
          << "Please use .json or .toml extension." << std::endl;
      return EXIT_FAILURE;
    }
  }

  try {
    // Determine translation direction based on file extensions
    if (input_ext == ".json" && output_ext == ".toml") {
      NetlistTranslator::translateJSONtoTOML(input_path, output_path);
    } else if (input_ext == ".toml" && output_ext == ".json") {
      NetlistTranslator::translateTOMLtoJSON(input_path, output_path);
    } else {
      std::cerr << "Error: Unsupported translation direction. "
                << "Supported: .json -> .toml or .toml -> .json" << std::endl;
      return EXIT_FAILURE;
    }
  } catch (const std::exception &err) {
    std::cerr << "Translation failed: " << err.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
