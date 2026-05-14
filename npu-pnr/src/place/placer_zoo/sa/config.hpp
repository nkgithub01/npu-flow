#pragma once

#include <iostream>

#include "base/common.hpp"
#include "base/placement.hpp"
#include "base/routing/routing_mode.hpp"
#include "base/routing/routing_net.hpp"
#include "base/tf_graph.hpp"
#include "place/placer_zoo/sa/bb_cost_estimator.hpp"
#include "place/placer_zoo/sa/schedule.hpp"
#include "utils/misc.hpp"

namespace place::placer_zoo::sa {

struct Config {
  base::RoutingNetList netlist;
  base::Placement init_state;

  utils::RandomSeed random_seed; // TODO: put random seed to engine components?
  int current_move_attempts = 0;
  int sa_swap_bound = 1; // The maximum Manhattan distance between the original tile and the new tile for a swap move.

  base::LogicalToPhysicalCoreAvailablityTable full_avail_table;
  base::LogicalCoreCompatibilitySet l_core_compatibility_set;

  TemperatureSchedule temp_schedule;
  bool enable_dynamic_temperature_scheduling;
  std::string initial_temperature_estimate_method;
  double initial_temperature_value;
  double initial_temperature_multiplier;
  int initial_temperature_num_sampling_moves;

  int max_iters;               // upper limit of N
  int num_moves_per_iteration; // M
  int max_move_attempts;       // upper limit of M*N

  int greedy_stage_num_iters;                    // static & dynamic scheduling
  double greedy_stage_entering_temperature;      // dynamic scheduling only
  double greedy_stage_entering_acceptance_ratio; // dynamic scheduling only

  bool enable_initial_placement_randomization;

  bool packing_disabled;

  std::string cost_estimator;
  std::string cost_estimator_probability_distribution;
  bool enable_cost_legality_estimator;
  base::RoutingMode milp_cost_estimator;
  double milp_cost_estimator_timeout_secs;
  bool milp_cost_estimator_enable_lock_constraints;
  BoundingBoxBasedCostEstimator bb_cost_estimator; // TODO: move out of Config?

  Config() = delete;

  explicit Config(
    const base::TrafficFlowGraph &tf_graph,
    const base::Placement &init_placement,
    const base::LogicalToPhysicalCoreAvailablityTable &full_avail_table
  ) {
    // Initialize basic fields
    this->netlist = base::RoutingNetList{tf_graph};
    this->init_state = init_placement;
    this->full_avail_table = full_avail_table;
  }

  Config &parse(const base::Config &cfg) {
    // TODO: use custom utils::RandomSeed parser?
    setRandomSeed(cfg.getOrDefault<int>("random_seed", 0));
    setMaxNumIterations(cfg.getOrDefault<int>("max_iters", 1e6));
    // Set staic temperature schedule by default, dynamic schedule can be set
    // later to override this if specified.
    // TODO: add if-else to refactor the setting of static/dynamic schedule
    setStaticTemperatureSchedule(
      cfg.getOrDefault<std::vector<double>>("start_temps",
                                            {1000.0, 0.5, 0.001}),
      cfg.getOrDefault<std::vector<double>>("cooling_factors",
                                            {0.95, 0.99, 0}),
      cfg.getOrDefault<int>("max_iters", 1e6),
      cfg.getOrDefault<int>("num_moves_per_iter", 1000),
      cfg.getOrDefault<int>("greedy_stage_max_iters", 1000),
      static_cast<int>(this->init_state.size() * cfg.getOrDefault<double>("greedy_stage_num_iters_scaling_factor", 10.0)),
      cfg.getOrDefault<int>("max_move_attempts", 1e6)
    );
    setDynamicTemperatureSchedule(
      cfg.getOrDefault<bool>("enable_dynamic_temperature_scheduling", true),
      // Initial temperature estimation parameters
      cfg.getOrDefault<std::string>("initial_temperature_estimate_method", "equilibrium"),
      // TODO: refactor it since the following should be exclusively used for
      // static scheduling. It is bad idea to borrow part of its values to be
      // used in dynamic scheduling.
      cfg.getOrDefault<std::vector<double>>("start_temps", {1000.0, 0.5, 0.001}).front(),
      cfg.getOrDefault<double>("initial_temperature_multiplier", 1.0),
      cfg.getOrDefault<int>("initial_temperature_num_sampling_moves", -1),
      // Greedy stage parameters
      cfg.getOrDefault<double>("greedy_stage_entering_temperature", 0.1),
      cfg.getOrDefault("greedy_stage_entering_acceptance_ratio", 0.01),
      cfg.getOrDefault<int>("greedy_stage_max_iters", 1000),
      static_cast<int>(this->init_state.size() * cfg.getOrDefault<double>("greedy_stage_num_iters_scaling_factor", 10.0)),
      cfg.getOrDefault<int>("max_move_attempts", 1e6)
    );
    setMaxMoveAttempts(cfg.getOrDefault<int>("max_move_attempts", 1e6));
    setNumMovesPerSAIteration(cfg.getOrDefault<int>("num_moves_per_iter", 1000));

    // TODO: consider moving to a class method to allow non-config construction
    this->cost_estimator = cfg.getOrDefault<std::string>("cost_estimator", "bb");
    this->cost_estimator_probability_distribution = cfg.getOrDefault<std::string>("cost_estimator_probability_distribution", "path_uniform");
    this->enable_cost_legality_estimator = cfg.getOrDefault<bool>("enable_cost_legality_estimator", false);
    this->milp_cost_estimator_timeout_secs = cfg.getOrDefault<double>("milp_cost_estimator_timeout_secs", 1.0);
    this->milp_cost_estimator_enable_lock_constraints = cfg.getOrDefault<bool>("enable_milp_router_lock_constraints", false);
    this->enable_initial_placement_randomization = cfg.getOrDefault<bool>("enable_initial_placement_randomization", false);

    // Not available for public use yet
    setPackingDisallowed(cfg.getOrDefault<bool>("disable_packing", false));

    return *this;
  }

  // Setter functions for configuration options; can only be called inside
  // place() or testers, not intended for public use.
  void setRandomSeed(const utils::RandomSeed &seed) {this->random_seed = seed;}

  void setStaticTemperatureSchedule(std::span<const double> start_temperatures,
                                    std::span<const double> cooling_factors,
                                    int num_iterations_limit,
                                    int num_moves_per_iteration_limit,
                                    int greedy_stage_max_iters,
                                    int greedy_stage_relative_num_iters,
                                    int total_move_limit
  ) {
    // TODO: no need to save this; need to refactor
    this->greedy_stage_num_iters = std::min(greedy_stage_max_iters, greedy_stage_relative_num_iters);
    this->temp_schedule = TemperatureSchedule(
      start_temperatures, cooling_factors, num_iterations_limit,
      num_moves_per_iteration_limit, greedy_stage_num_iters, total_move_limit
    );
  }

  void setDynamicTemperatureSchedule(
    bool enable_dynamic_temperature_scheduling,
    // Initial temperature estimation parameters
    std::string initial_temperature_estimate_method,
    double initial_temperature_value, 
    double initial_temperature_multiplier,
    int initial_temperature_num_sampling_moves,
    // Greedy stage parameters
    double greedy_stage_entering_temperature,
    double greedy_stage_entering_acceptance_ratio, 
    int greedy_stage_max_iters,
    int greedy_stage_relative_num_iters,
    int total_move_limit
  ) {
    this->enable_dynamic_temperature_scheduling = enable_dynamic_temperature_scheduling;
    // Initial temperature estimation parameters
    this->initial_temperature_estimate_method = initial_temperature_estimate_method;
    this->initial_temperature_value = initial_temperature_value;
    this->initial_temperature_multiplier = initial_temperature_multiplier;
    this->initial_temperature_num_sampling_moves = initial_temperature_num_sampling_moves;
    // Greedy stage parameters
    this->greedy_stage_entering_temperature = greedy_stage_entering_temperature;
    this->greedy_stage_entering_acceptance_ratio = greedy_stage_entering_acceptance_ratio;
    this->greedy_stage_num_iters = std::min(greedy_stage_max_iters, greedy_stage_relative_num_iters);
    this->max_move_attempts = total_move_limit;
  }

  void setMaxNumIterations(int max_iters) { this->max_iters = max_iters; }

  void setNumMovesPerSAIteration(int num_moves_per_iteration) {
    this->num_moves_per_iteration = num_moves_per_iteration;
  }

  void setPackingDisallowed(bool is_disallowed) {
    // TODO: need to change swap function to respect packing disallowed.
    // Currently, it works by forcing the routing of packed placement to be
    // fatal.
    this->packing_disabled = is_disallowed;
    if (this->packing_disabled) {
      std::cerr << "Warning: Packing disallowed is not fully supported yet."
                << std::endl;
      std::vector<base::LogicalCoreCompatibilitySet::Subset> subsets;
      for (const auto &l_core : this->init_state.keys()) {
        subsets.push_back({l_core});
      }
      this->l_core_compatibility_set = base::LogicalCoreCompatibilitySet{subsets};
    } else {
      this->l_core_compatibility_set = base::LogicalCoreCompatibilitySet{};
    }
  }

  void setMaxMoveAttempts(int max_move_attempts) {
    this->max_move_attempts = max_move_attempts;
  }
};

} // namespace place::placer_zoo::sa
