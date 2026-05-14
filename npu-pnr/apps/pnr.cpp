#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include "argparse/argparse.hpp"

#include "base/common.hpp"
#include "base/engine.hpp"
#include "engine/engine.hpp"
#include "utils/misc.hpp"

template <typename T>
concept CLIConfigValue = utils::either<T, int, double, bool, std::string,
                                       std::vector<int>, std::vector<double>>;
class CLIOptions {
private:
  base::Config cfg_;
  std::string input_path_;
  std::string output_path_;
  argparse::ArgumentParser program_;
  std::vector<std::pair<std::string, std::string>> added_args_;

private:
  // Helper function to wrap help text at specified width
  std::string wrapHelpText(const std::string &raw_text, size_t width) const {
    std::string result;

    for (const auto text : utils::split(raw_text, '\n')) {
      size_t pos = 0;

      while (pos < text.length()) {
        // If remaining text is shorter than width, just add it
        if (pos + width >= text.length()) {
          result += text.substr(pos);
          result += "\n"; // Add newline at the end of the \n splited text chunk
          break;
        }

        // Find the last space within the width limit
        size_t end = pos + width;
        size_t last_space = text.rfind(' ', end);

        if (last_space != std::string::npos && last_space > pos) {
          result += text.substr(pos, last_space - pos) + "\n";
          pos = last_space + 1; // Skip the space
        } else {
          // No space found, force break at width
          result += text.substr(pos, width) + "\n";
          pos += width;
        }
      }
    }

    if (result.ends_with('\n')) {
      result.pop_back(); // Remove the last newline for cleaner output
    }

    return result;
  }

  template <typename T>
    requires CLIConfigValue<T>
  void setConfig(std::string hierarchical_name, const T &value) {
    // Split hierarchical name by '.', maximum 2 levels for now
    size_t dot_pos = hierarchical_name.find('.');
    if (dot_pos == std::string::npos) {
      cfg_.set(hierarchical_name, value);
    } else {
      std::string section = hierarchical_name.substr(0, dot_pos);
      std::string param = hierarchical_name.substr(dot_pos + 1);
      if (!cfg_.has(section)) {
        cfg_.set(section, base::Config());
      }
      base::Config section_cfg = cfg_.get<base::Config>(section);
      section_cfg.set(param, value);
      cfg_.set(section, section_cfg);
    }
  }

  template <typename T>
    requires CLIConfigValue<T>
  void populateConfig(std::string hierarchical_name) {
    T value = program_.get<T>("--" + hierarchical_name);
    setConfig<T>(hierarchical_name, value);
  }

public:
  CLIOptions() : program_("pnr") {
    program_.add_argument("netlist").help(
        "the path to the input netlist file with the initial solution");
    program_.add_argument("--output", "-o")
        .default_value(std::string{"solution.toml"})
        .help("the output path where the PnR solution will be saved");
  }

  template <typename T>
    requires CLIConfigValue<T>
  void addArgument(const std::string &name, const T &default_value,
                   const std::string &help,
                   const std::string &short_name = "") {
    argparse::Argument &arg =
        short_name.empty()
            ? program_.add_argument("--" + name)
            : program_.add_argument("--" + name, "-" + short_name);
    arg.default_value(default_value).help(wrapHelpText(help, 70));
    if constexpr (std::is_same_v<T, bool>) {
      arg.implicit_value(!default_value);
      if (default_value != false) {
        // Enforcing this to avoid any potential bugs for boolean flags
        // Lesson learned in the commit of this comment :)
        throw std::runtime_error(
            "Boolean CLI arguments must have default value false");
      }
      added_args_.push_back({name, "bool"});
    } else if constexpr (std::is_same_v<T, int>) {
      arg.scan<'i', int>();
      added_args_.push_back({name, "int"});
    } else if constexpr (std::is_same_v<T, double>) {
      arg.scan<'g', double>();
      added_args_.push_back({name, "double"});
    } else if constexpr (std::is_same_v<T, std::string>) {
      added_args_.push_back({name, "string"});
    } else if constexpr (std::is_same_v<T, std::vector<int>>) {
      arg.scan<'i', int>().nargs(argparse::nargs_pattern::at_least_one);
      added_args_.push_back({name, "vector<int>"});
    } else if constexpr (std::is_same_v<T, std::vector<double>>) {
      arg.scan<'g', double>().nargs(argparse::nargs_pattern::at_least_one);
      added_args_.push_back({name, "vector<double>"});
    } else {
      static_assert(false, "Unsupported type for CLI argument");
    }
  }

  void parse(int argc, char **argv) {
    try {
      program_.parse_args(argc, argv);
    } catch (const std::exception &err) {
      std::cerr << err.what() << std::endl;
      std::cerr << program_;
      exit(1);
    }

    input_path_ = program_.get<std::string>("netlist");
    output_path_ = program_.get<std::string>("--output");

    // Populate cfg_ with all defined arguments
    for (const auto &[arg_name, arg_type] : added_args_) {
      if (arg_type == "bool") {
        populateConfig<bool>(arg_name);
      } else if (arg_type == "int") {
        populateConfig<int>(arg_name);
      } else if (arg_type == "double") {
        populateConfig<double>(arg_name);
      } else if (arg_type == "string") {
        populateConfig<std::string>(arg_name);
      } else if (arg_type == "vector<int>") {
        populateConfig<std::vector<int>>(arg_name);
      } else if (arg_type == "vector<double>") {
        populateConfig<std::vector<double>>(arg_name);
      } else {
        throw std::runtime_error(
            "Unsupported argument type found during config population");
      }
    }

    // Save a copy of the raw command-line string
    setConfig<std::string>("engine.cli_string",
                           utils::getCommandLineString(argc, argv));
    setConfig<std::string>("engine.input_netlist",
                           utils::readFileToString(input_path_));
  }

  [[nodiscard]] base::Config getConfig() const { return cfg_; }
  [[nodiscard]] std::string getInputPath() const { return input_path_; }
  [[nodiscard]] std::string getOutputPath() const { return output_path_; }
};

int main(int argc, char **argv) {
  utils::CumulativeTimer global_timer;
  global_timer.start();
  CLIOptions cli;

  //
  // General PnR options
  //
  cli.addArgument<int>(
      "timeout_secs", 36000,
      "the maximum time (in seconds) allowed for each PnR stage (either "
      "placement or routing).\n"
      "If set to -1, there is no time limit. If set to 0, the PnR stage will "
      "stop immediately after initialization. If set to a positive value, the "
      "PnR stage will be terminated after the specified time limit (may "
      "slightly exceed this value due to internal processing).");
  cli.addArgument<bool>("dump_rr_graph", false,
                        "whether to dump the RR graph to a rr_graph.dot in the "
                        "current working directory for debugging purposes");

  //
  // Architecture
  //
  cli.addArgument<std::string>("arch.type", "npu",
                               "the type of architecture to use [npu]");
  cli.addArgument<int>("arch.num_cols", 8, "the number of columns in the NPU");
  cli.addArgument<int>("arch.num_rows", 6, "the number of rows in the NPU");
  cli.addArgument<int>("arch.num_compute_rows", 4,
                       "the number of compute tile rows in the NPU");
  cli.addArgument<int>("arch.num_memory_rows", 1,
                       "the number of memory tile rows in the NPU");
  cli.addArgument<int>("arch.num_shim_rows", 1,
                       "the number of shim tile rows in the NPU");

  //
  // Placer
  //
  cli.addArgument<std::string>("placer.type", "sa",
                               "the type of placer to use [sa|ls|milp|noop]");
  cli.addArgument<int>(
      "placer.random_seed", 24,
      "the random seed for the placer (only for stochastic placers like "
      "simulated annealing), use -1 for std::random_device");
  cli.addArgument<int>("placer.max_iters", 2e9,
                       "the maximum number of iterations to perform during "
                       "placement (mainly for debugging purposes)",
                       "n");
  cli.addArgument<bool>(
      "placer.enable_initial_placement_randomization", false,
      "whether to randomize the initial placement state before starting the "
      "placer process to avoid bias from the initial placement. If set, the "
      "random placement will overwrite any initial placement made by the "
      "netlist parser or the placer itself. If not set, the placer will start "
      "from the initial placement provided in the input netlist and/or the "
      "sequentially resolved placement for any unplaced logical cores when "
      "parsing the input netlist and/or any initial placement made by the "
      "placer itself depending the corresponding placer-type-specific options "
      "used.",
      "r");
  cli.addArgument<bool>(
      "placer.enable_milp_router_lock_constraints", false,
      "whether to enable lock constraints when using MILP router as either cost"
      "estimator in the SA placer or oracle in the LS placer, or using MILP "
      "joint-placement-and-routing mode for the one-shot MILP placer."
      "Effective only when (1) --placer.cost_estimator is set to 'milp' or (2) "
      "--placer.type is set to 'ls' or 'milp'.");

  // SA Placer
  // TODO: refactor command-line arguments to explicitly specify static or
  // dynamic temperature schedule
  cli.addArgument<std::vector<double>>(
      "placer.start_temps", {1000.0, 0.5, 0.001},
      "the start temperatures for different stages of the SA process. "
      "The length of this list determines how many stages will be "
      "performed (the last stage is always a greedy search with "
      "temperature 0). Example: --start-temperatures 1000.0 0.5 0.001, "
      "together with the example cooling factors in --cooling-factors "
      "below, which means the first stage (fast cooling) will start at "
      "1000.0, the second stage (slow cooling) will start at 0.5, and "
      "the last stage (greedy) will start at 0.001. When setting both "
      "--start-temperatures 0 and --cooling-factors 0, the placer will "
      "perform a pure greedy search without any bad move acceptance.");
  cli.addArgument<std::vector<double>>(
      "placer.cooling_factors", {0.95, 0.99, 0},
      "the cooling factors for different stages of the SA process. The "
      "length of this list must be equal to the length of the "
      "--start-temperatures list. The last stage is always a greedy search "
      "with cooling factor 0. Example: --cooling-factors 0.95 0.99 0, "
      "which means the first stage (fast cooling) will use cooling factor "
      "0.95, the second stage (slow cooling) will use cooling factor 0.99, "
      "and the last stage (greedy) will use cooling factor 0. When setting "
      "both --start-temperatures 0 and --cooling-factors 0, the placer "
      "will perform a pure greedy search without any bad move acceptance.");
  cli.addArgument<double>(
      "placer.greedy_stage_entering_temperature", 0.1,
      "the temperature threshold to enter the greedy stage in the SA process. "
      "When the current temperature drops below this threshold, the placer "
      "will enter the greedy stage where only cost-improving moves are "
      "accepted.\n"
      "Effective only when --placer.enable_dynamic_temperature_scheduling is "
      "set. Together with --placer.greedy_stage_entering_acceptance_ratio, "
      "whichever condition is met first will trigger the greedy stage.");
  cli.addArgument<double>(
      "placer.greedy_stage_entering_acceptance_ratio", 0.001,
      "the acceptance ratio threshold to enter the greedy stage in the SA "
      "process. When the acceptance ratio (number of accepted moves / number "
      "of attempted moves) within the acceptance window drops below this "
      "threshold, the placer will enter the greedy stage where only cost-"
      "improving moves are accepted.\n"
      "Effective only when --placer.enable_dynamic_temperature_scheduling is "
      "set. Together with --placer.greedy_stage_entering_temperature, "
      "whichever condition is met first will trigger the greedy stage.");
  cli.addArgument<double>(
      "placer.greedy_stage_num_iters_scaling_factor", 5.0,
      "the scaling factor to determine the number of greedy iterations at the "
      "end of the SA process. The number of greedy iterations will be equal to "
      "this scaling factor multiplied by the circuit size (i.e., number of "
      "logical cores in the netlist). This allows larger circuits to have more "
      "greedy iterations to refine the SA process.\n"
      "Effective for both static and dynamic temperature scheduling.\n"
      "Together with --placer.greedy_stage_max_iters, whichever condition is "
      "met first will end the greedy stage.");
  cli.addArgument<int>(
      "placer.greedy_stage_max_iters", 2e9,
      "the maximum number (hard-limit) of iterations to perform in the greedy "
      "stage at the end of the SA process.\n"
      "Effective for both static and dynamic temperature scheduling.\n"
      "Together with --placer.greedy_stage_num_iters_scaling_factor, whichever "
      "condition is met first will end the greedy stage.");
  cli.addArgument<int>("placer.max_move_attempts", 2e9,
                       "the maximum number of move attempts (or cost "
                       "evaluations) to perform during the SA process");
  cli.addArgument<int>(
      // TODO: consider using equilibrium method
      "placer.num_moves_per_iter", 1000,
      "the number of moves to attempt in each iteration of the SA process",
      "m");
  cli.addArgument<bool>(
      // TODO: should default be true after fixing CLI parser for boolean flags
      "placer.enable_dynamic_temperature_scheduling", false,
      "whether to use dynamic temperature scheduling to determine the start "
      "temperatures and cooling factors for SA placer. If set, the placer will "
      "sample the costs of the first (default) 100 random moves to find mean "
      "and standard deviation of move costs, and use that to set the start "
      "temperatures and cooling factors automatically.\n"
      "This option overrides the --placer.start_temps and "
      "--placer.cooling_factors.\n"
      "This option can be used with "
      "--placer.initial_temperature_estimate_method, "
      "--placer.initial_temperature_num_sampling_moves, and "
      "--placer.initial_temperature_multiplier to control how many random move "
      "costs to sample.");
  cli.addArgument<std::string>(
      "placer.initial_temperature_estimate_method", "equilibrium",
      "the initial temperature estimation method when using dynamic "
      "temperature [manual|std-dev|equilibrium]. 'manual' uses the provided "
      "start temperatures directly; 'std-dev' uses the standard deviation of "
      "sampled move costs to estimate starting temperature; 'equilibrium' uses "
      "binary search to find the equilibrium temperature where the expected "
      "delta cost is close to zero.\n"
      "Only used when --placer.enable_dynamic_temperature_scheduling is set.");
  cli.addArgument<int>(
      "placer.initial_temperature_num_sampling_moves", -1,
      "the number <N> of random move costs to sample when using dynamic "
      "temperature scheduling. Setting this to -1 means sampling relative to "
      "the number of logical cores in the circuit (i.e., sample <N> = number "
      "of logical cores).\n"
      "Only used when --placer.enable_dynamic_temperature_scheduling is set.");
  cli.addArgument<double>(
      "placer.initial_temperature_multiplier", 1.0,
      "the multiplier to scale the estimated initial temperature when using "
      "dynamic temperature scheduling.\n"
      "Only used when --placer.enable_dynamic_temperature_scheduling is set.");
  cli.addArgument<std::string>(
      "placer.cost_estimator", "bb",
      "the cost estimation method to use during placement [bb|prob|milp]. 'bb' "
      "uses bounding-box cost estimation during the SA process. If set, the "
      "placer will use a fast cost estimation method to evaluate placement "
      "costs, which greatly speeds up the placement process with sacrifice in "
      "placement quality; 'prob' uses probabilistic congestion model-based "
      "cost estimation built on top of bounding-box cost estimator during the "
      "SA process; 'milp' uses MILP full routing-based cost estimation during "
      "the SA process.");
  cli.addArgument<std::string>(
      "placer.cost_estimator_probability_distribution", "path_uniform",
      "the probability distribution to use for the probabilistic cost estimator "
      "during placement [path_uniform|channel_uniform]. 'channel_uniform' assigns equal "
      "probability to all routing channel; 'path_uniform' assigns probability "
      "proportional to the number of routing paths passing through each edge.");
  cli.addArgument<bool>(
      "placer.enable_cost_legality_estimator", false,
      "whether to enable legality estimation in the cost estimator. If set, the "
      "cost estimator will not only estimate the cost of a placement solution, "
      "but also estimate its legality (i.e., whether it is congested or not). "
      "This can help the SA placer to better navigate the solution space by "
      "providing more guidance on whether a move leads to an illegal (e.g., "
      "congested) solution or not, rather than just providing cost estimation."
      "Effective only when --placer.cost_estimator is set to 'bb' or 'prob'."
      "Default to false.");
  cli.addArgument<double>(
      "placer.milp_cost_estimator_timeout_secs", 1.0,
      "the time limit (in seconds; can be set to floating-point value) for "
      "each MILP cost estimation call in the SA placer with MILP-based cost "
      "estimator. If the MILP solver fails to find an optimal solution within "
      "this time limit, the best feasible solution found so far will be used "
      "as the estimated cost.\n"
      "Effective only when --placer.cost_estimator is set to 'milp'.");

  // LS Placer
  cli.addArgument<bool>(
      // TODO: should default be true after fixing CLI parser for boolean flags
      "placer.enable_aggressive_local_search", false,
      "whether to use aggressive local search mode, where in each iteration "
      "all logical cores explore their neighborhood regions for possible "
      "improvements. If not set, the placer will use conservative local search "
      "mode, where in each iteration only one logical core (greedily chosen "
      "from route trials) explores its neighborhood region for possible "
      "improvements.");
  cli.addArgument<double>(
      "placer.max_consecutive_iters_no_best_cost_improvement_scaling_factor",
      1.5,
      "the scaling factor to determine the maximum number of consecutive "
      "iterations without improvement to the best placement cost before "
      "terminating the LS placer.");
  cli.addArgument<std::string>(
      "placer.neighbor_region_shape", "cross",
      "the shape of the neighbor regions to explore during LS placement "
      "[3x3|cross|x]. '3x3' explores all 9 cells in a 3x3 grid around the "
      "current cell; 'cross' explores the 5 cells in a cross shape (north, "
      "south, east, west, and its current cell) around the current cell; 'x' "
      "explores the 5 cells in an X shape (NW, NE, SW, SE, and current) around "
      "the current cell.");

  //
  // Router
  //
  cli.addArgument<std::string>(
      "router.type", "milp",
      "the type of router to use [milp]. MILP router uses a MILP solver "
      "to find an optimal routing solution.");
  cli.addArgument<bool>(
      "router.enable_lock_constraints", false,
      "whether to enable lock constraints when using MILP router.");
  cli.addArgument<double>(
      "router.congestion_penalty_scaling_factor", 1e5,
      "the scaling factor for congestion penalty in the MILP router. Set to "
      "smaller values to smooth the routing cost function between legal and "
      "illegal solutions at the cost of settling for a congested (illegal) "
      "solution rather than detouring in the solution search. Set to larger "
      "values to strongly penalize congestion and favor legal (even detoured) "
      "solutions.");

  //
  // Logger
  //
  cli.addArgument<std::string>("logger.verbose", "normal",
                               "the verbosity level of the logger output "
                               "[silent|minimal|normal|verbose|debug]");

  //
  // Telemetry
  //
  cli.addArgument<bool>("telemetry.enable", false,
                        "whether to enable telemetry collection during the PnR "
                        "process for performance analysis and debugging");
  cli.addArgument<std::string>(
      "telemetry.save_path", "telemetry.db",
      "the file path to save the telemetry SQLite database");

  cli.parse(argc, argv);

  engine::PnREngine engine(cli.getConfig());
  auto result = engine.run(
      base::PnRNetlistReader::fromTOML(std::ifstream(cli.getInputPath())));

  base::PnRNetlistWriter wr;
  auto exit_code = EXIT_SUCCESS;
  if (result.routing.getLegality() == base::RoutingState::Legality::Legal) {
    wr = engine.write(result);
  } else {
    wr =
        engine.write(base::PnRPlacedNetlist{result.tf_graph, result.placement});
    exit_code = EXIT_FAILURE;
  }
  std::ofstream(cli.getOutputPath()) << wr.toTOML();

  global_timer.pause();
  std::cout << std::format("\nTotal PnR runtime: {:.3f} seconds\n",
                           global_timer.getSeconds());

  return exit_code;
}
