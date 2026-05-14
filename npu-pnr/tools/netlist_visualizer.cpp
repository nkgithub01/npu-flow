#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "argparse/argparse.hpp"

#include "arch/npu/npu.hpp"
#include "base/common.hpp"
#include "base/engine.hpp"

class NetlistVisualizer {
public:
  enum class VisualizationMode { Placement, Routing };

  static void visualizeNetlist(VisualizationMode mode,
                               const std::string &toml_path,
                               const std::string &dot_path,
                               const base::Config &arch_cfg) {
    // Read TOML netlist file
    std::ifstream toml_file(toml_path);
    if (!toml_file.is_open()) {
      throw std::runtime_error("Failed to open TOML file: " + toml_path);
    }

    auto reader = base::PnRNetlistReader::fromTOML(std::move(toml_file));

    // Create NPU instance with default configuration
    arch::npu::NPU npu{arch_cfg};

    // Generate visualization based on mode
    std::string dot_content;
    try {
      if (mode == VisualizationMode::Placement) {
        std::cout << "Generating placement visualization..." << std::endl;
        dot_content = npu.visualizePlacement(reader);
      } else if (mode == VisualizationMode::Routing) {
        std::cout << "Generating routing visualization..." << std::endl;
        dot_content = npu.visualizeRouting(reader);
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("Visualization failed: " +
                               std::string(e.what()));
    }

    // Write to output dot file
    std::ofstream dot_file(dot_path);
    if (!dot_file.is_open()) {
      throw std::runtime_error("Failed to open output file: " + dot_path);
    }
    dot_file << dot_content;
    dot_file.close();

    std::cout << "Successfully generated " << dot_path << std::endl;
  }
};

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("netlist_visualizer");

  program.add_description(
      "Visualize TOML netlist as GraphViz dot file for placement or routing");

  // Input and output files
  program.add_argument("input").help("path to the input TOML netlist file");
  program.add_argument("-o", "--output")
      .help("path to the output GraphViz dot file (default: <input_mode>.dot)");

  // Visualization mode
  program.add_argument("-m", "--mode")
      .help("visualization mode: 'placement' or 'routing' (default: placement)")
      .default_value(std::string("placement"));

  // Architecture parameters
  program.add_argument("--num_cols")
      .help("number of columns in the NPU architecture")
      .default_value(8)
      .scan<'i', int>();
  program.add_argument("--num_rows")
      .help("number of rows in the NPU architecture")
      .default_value(6)
      .scan<'i', int>();
  program.add_argument("--num_compute_rows")
      .help("number of compute tile rows in the NPU")
      .default_value(4)
      .scan<'i', int>();
  program.add_argument("--num_memory_rows")
      .help("number of memory tile rows in the NPU")
      .default_value(1)
      .scan<'i', int>();
  program.add_argument("--num_shim_rows")
      .help("number of shim tile rows in the NPU")
      .default_value(1)
      .scan<'i', int>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }

  std::string input_path = program.get<std::string>("input");
  std::string mode_str = program.get<std::string>("--mode");

  base::Config arch_cfg;
  arch_cfg.set<int>("num_cols", program.get<int>("--num_cols"));
  arch_cfg.set<int>("num_rows", program.get<int>("--num_rows"));
  arch_cfg.set<int>("num_compute_rows", program.get<int>("--num_compute_rows"));
  arch_cfg.set<int>("num_memory_rows", program.get<int>("--num_memory_rows"));
  arch_cfg.set<int>("num_shim_rows", program.get<int>("--num_shim_rows"));

  // Parse visualization mode
  std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::tolower);
  NetlistVisualizer::VisualizationMode mode;

  if (mode_str == "placement" || mode_str == "place" || mode_str == "p") {
    mode = NetlistVisualizer::VisualizationMode::Placement;
    mode_str = "placement";
  } else if (mode_str == "routing" || mode_str == "route" || mode_str == "r") {
    mode = NetlistVisualizer::VisualizationMode::Routing;
    mode_str = "routing";
  } else {
    std::cerr << "Error: Invalid visualization mode '" << mode_str << "'. "
              << "Must be 'placement' or 'routing'." << std::endl;
    return EXIT_FAILURE;
  }

  // Validate input file extension
  size_t dot_pos = input_path.find_last_of('.');
  std::string input_ext;
  if (dot_pos != std::string::npos) {
    input_ext = input_path.substr(dot_pos);
    std::transform(input_ext.begin(), input_ext.end(), input_ext.begin(),
                   ::tolower);
  }

  if (input_ext != ".toml") {
    std::cerr << "Warning: Input file does not have .toml extension. "
              << "Attempting to parse as TOML anyway." << std::endl;
  }

  // Determine output path
  std::string output_path;
  if (program.is_used("--output")) {
    output_path = program.get<std::string>("--output");
  } else {
    // Auto-generate output filename based on input and mode
    std::string base_name = (dot_pos != std::string::npos)
                                ? input_path.substr(0, dot_pos)
                                : input_path;
    output_path = base_name + "_" + mode_str + ".dot";
  }

  try {
    NetlistVisualizer::visualizeNetlist(mode, input_path, output_path,
                                        arch_cfg);
  } catch (const std::exception &err) {
    std::cerr << "Visualization failed: " << err.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
