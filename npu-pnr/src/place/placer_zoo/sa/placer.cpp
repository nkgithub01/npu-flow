#include "place/placer_zoo/sa/placer.hpp"

#include <cmath>
#include <algorithm>

#include "base/placement.hpp"
#include "engine/component.hpp"
#include "engine/logger.hpp"
#include "place/placer_zoo/sa/schedule.hpp"
#include "utils/misc.hpp"

namespace place::placer_zoo::sa {

constexpr double kInitialTemperatureNumSamplingMovesScalingFactor = 10.0;
constexpr double kAcceptanceRatioTarget = 0.44;

Placement SAPlacer::runSimulatedAnnealing(engine::Components &engine,
                                          Config &cfg) const {
  utils::CumulativeTimer total_sa_timer;
  total_sa_timer.start();

  Placement accepted_state = cfg.init_state;
  Placement current_state = cfg.init_state;
  std::vector<base::LogicalCore> moved_l_cores; // for tracking moved logical cores in each move

  auto accept = [&](int delta_cost, double current_T) -> bool {
    double acceptance_probability = 0.0;
    bool is_accepted = false;
    if (delta_cost <= 0) {
      acceptance_probability = 1.0;
      is_accepted = true; // always accept if cost is lower or equal
    } else {
      if (current_T <= 0) {
        is_accepted = false; // always reject if T is zero or negative
      } else {
        // TODO: use better random number generator
        acceptance_probability = std::exp(-delta_cost / current_T);
        is_accepted = acceptance_probability > (static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
      }
    }
    
    if (is_accepted) {
      cfg.temp_schedule.updateDeltaCostExpectedValue(delta_cost, acceptance_probability);
    }
    return is_accepted;
  };

  auto updateAcceptedState = [&](Placement &accepted_state, const Placement &current_state, const std::vector<base::LogicalCore> &moved_l_cores) {
    for (const auto &l_core : moved_l_cores) {
      accepted_state.update(l_core) = current_state.at(l_core);
    }
  };

  auto restoreCurrentStateToAcceptedState = [&](Placement &current_state, const Placement &accepted_state, const std::vector<base::LogicalCore> &moved_l_cores) {
    for (const auto &l_core : moved_l_cores) {
      current_state.update(l_core) = accepted_state.at(l_core);
    }
  };

  if (cfg.cost_estimator == "bb" || cfg.cost_estimator == "prob") {
    engine.logger.debug([&](auto &) {
      cfg.bb_cost_estimator.printBinomialCoefficientLookupTable();
    });
  }

  // This evaluation is necessary to initialize the best cost and legality,
  // and it is also important to initialize the internal state of cost estimator,
  // such that it can operate on delta changes in the subsequent iterations efficiently.
  auto [min_cost, best_legality] = evaluatePlacement(engine, cfg, accepted_state);
  Placement best_state = accepted_state;
  float prev_cost = min_cost;

  const bool is_max_iters_set = (cfg.max_iters != 1e6);

  engine.logger.minimal([&](auto &log) {
    std::string max_iters_str =
        (is_max_iters_set ? std::format("max_iters set to {}", cfg.max_iters)
                          : "max_iters not set");
    log << std::format("[SA] Initial cost: {}, Initial T: {:.6f}, ({})\n",
                       prev_cost, cfg.temp_schedule.temperature, max_iters_str);
    log << "[SA] Temperature Schedule:\n";
    if (cfg.temp_schedule.isUsingStaticSchedule()) {
      log << "\tUsing static temperature schedule.\n";
    } else {
      log << "\tUsing dynamic temperature schedule.\n";
    }
  });

  // One move: one cost evaluation (or move attempt); each move may be legal or
  // illegal (either congested or fatal, where congested means some *routing*
  // resources are over-utilized but all hard constraints are satisfied, and
  // fatal means some hard constraints such as reachability or memory capacity
  // are violated).
  //
  // One iteration: one temperature update (after a certain number of moves,
  // which can be set by the user).
  //
  // #moves = #accepted + #rejected = #legal + #congested + #fatal, unless the
  // placer is set to only anneal on legal moves.
  int num_moves_accepted = 0, num_moves_rejected = 0;
  int num_moves_legal = 0, num_moves_congested = 0, num_moves_fatal = 0;

  auto get_total_moves = [&]() -> int {
    if (num_moves_accepted + num_moves_rejected !=
         num_moves_legal + num_moves_congested + num_moves_fatal) {
      throw std::runtime_error("Inconsistent move counts: #accepted + "
                               "#rejected != #legal + #congested + #fatal");
    }
    return num_moves_legal + num_moves_congested + num_moves_fatal;
  };
  auto get_accepted_move_ratio = [&]() -> double {
    return (double)num_moves_accepted / get_total_moves() * 100;
  };
  auto get_rejected_move_ratio = [&]() -> double {
    return (double)num_moves_rejected / get_total_moves() * 100;
  };
  auto get_legal_move_ratio = [&]() -> double {
    return (double)num_moves_legal / get_total_moves() * 100;
  };
  auto get_congested_move_ratio = [&]() -> double {
    return (double)num_moves_congested / get_total_moves() * 100;
  };
  auto get_fatal_move_ratio = [&]() -> double {
    return (double)num_moves_fatal / get_total_moves() * 100;
  };

  auto print_per_iter_stats = [&]() {
    engine.logger.normal([&](auto &log) {
      const auto &sched = cfg.temp_schedule;
      const std::string stats = std::format(
          "acc/rej={}/{},acc/rej\%={:.1f}\%/{:.1f}\%,"
          "legal/cong/fatal={}/{}/{},"
          "legal/cong/fatal\%={:.1f}\%/{:.1f}\%/{:.1f}\%",
          num_moves_accepted, num_moves_rejected, get_accepted_move_ratio(),
          get_rejected_move_ratio(), num_moves_legal, num_moves_congested,
          num_moves_fatal, get_legal_move_ratio(), get_congested_move_ratio(),
          get_fatal_move_ratio());
      const std::string T_str = sched.isInGreedyStage()
                                    ? "0 (Greedy Stage)"
                                    : std::format("{:.6f}", sched.temperature);
      log << std::format("\n[SA] Iteration completed {} ({})\n"
                         "[SA] Current best cost: {}, T: {}\n"
                         "[SA] R_acceptance: {:.2f}, Cooling Factor: {:.6f}\n",
                         sched.curr_iterations, stats, min_cost, T_str,
                         sched.getAcceptanceRatio(), sched.getCoolingFactor());
    });
  };

  // Main SA loop: for each iteration
  for (cfg.temp_schedule.curr_iterations = 0;
       !cfg.temp_schedule.outerLoopLimitReached() && !engine.timer.hasExpired();
       cfg.temp_schedule.curr_iterations++) {

    utils::CumulativeTimer iter_timer;
    iter_timer.start();

    engine.logger.verbose(std::format(
        "\n[SA] Starting Iteration {}: {}\n", cfg.temp_schedule.curr_iterations,
        cfg.temp_schedule.isInGreedyStage()
            ? "(Greedy Stage)"
            : std::format("T_start={:.6f}", cfg.temp_schedule.temperature)));

    engine.telemetry.startEvent(
        "sa_iter",
        base::Config()
            .set<int>("iter", cfg.temp_schedule.curr_iterations)
            .set<double>("T", cfg.temp_schedule.temperature)
            .set<double>("cooling_factor", cfg.temp_schedule.getCoolingFactor())
            .set<std::string>(
                "stage",
                (cfg.temp_schedule.isInGreedyStage() ? "greedy" : "sa")));

    // For each move in an iteration
    for (cfg.temp_schedule.curr_moves_in_iteration = 0;
         !cfg.temp_schedule.innerLoopLimitReached() && !engine.timer.hasExpired();
         cfg.temp_schedule.curr_moves_in_iteration++, cfg.temp_schedule.total_moves++) {

      float cur_cost;
      RoutingState::Legality cur_legality;

      engine.telemetry.startEvent(
          "sa_move", base::Config()
                         .set<int>("move_in_iter",
                                   cfg.temp_schedule.curr_moves_in_iteration)
                         .set<int>("iter", cfg.temp_schedule.curr_iterations));
      utils::CumulativeTimer cost_eval_timer;
      
      // Generate a new candidate state by making a random move from the current state
      generateNewState(cfg, accepted_state, current_state, moved_l_cores);
      cost_eval_timer.start();
      std::tie(cur_cost, cur_legality) = evaluatePlacementWithDeltaChange(engine, cfg, current_state, moved_l_cores);
      cost_eval_timer.pause();

      switch (cur_legality) {
      case RoutingState::Legality::Legal:
        num_moves_legal++;
        break;
      case RoutingState::Legality::Congested:
        num_moves_congested++;
        break;
      case RoutingState::Legality::Fatal:
        num_moves_fatal++;
        break;
      default:
        throw std::runtime_error("Unknown legality state encountered.");
      }

      engine.logger.debug(std::format(
        "\n[SA] Move Attempt: Iter {}, Move {}"
        "\n     Test Acceptance: Prev Cost: {}, Cur Cost: {}, "
        "Cur Legality: {}, T: {:.6f}\n",
        cfg.temp_schedule.curr_iterations, cfg.temp_schedule.curr_moves_in_iteration,
        prev_cost, cur_cost,
        utils::toString(cur_legality), cfg.temp_schedule.temperature
      ));
      
      bool is_move_accepted = false;
      double delta_cost = cur_cost - prev_cost;
      if (accept(delta_cost, cfg.temp_schedule.temperature)) {
        updateAcceptedState(accepted_state, current_state, moved_l_cores);
        prev_cost = cur_cost;
        num_moves_accepted++;
        cfg.temp_schedule.updateAcceptanceCount(true);
        is_move_accepted = true;

        engine.logger.debug(std::format(
          "\n[SA] ACCEPT Move: Iter {}, Cur Cost: {}, "
          "Cur Legality: {}, T: {:.6f}\n",
          cfg.temp_schedule.curr_iterations, cur_cost,
          utils::toString(cur_legality), cfg.temp_schedule.temperature
        ));
      } else {
        num_moves_rejected++;
        cfg.temp_schedule.updateAcceptanceCount(false);
        is_move_accepted = false;

        engine.logger.debug(std::format(
          "\n[SA] REJECT Move: Iter {}, Cur Cost: {}, "
          "Cur Legality: {}, T: {:.6f}\n",
          cfg.temp_schedule.curr_iterations, cur_cost,
          utils::toString(cur_legality), cfg.temp_schedule.temperature
        ));
      }

      // Update best legal placement found so far
      if (min_cost >= cur_cost &&
          cur_legality == RoutingState::Legality::Legal
      ) {
        engine.logger.verbose(std::format(
          "\n[SA] Found {} best cost {} ({}) "
          "(num_iters={}, num_move_attempts={})\n",
          (min_cost == cur_cost ? "EQL" : "NEW"), cur_cost,
          utils::toString(cur_legality), cfg.temp_schedule.curr_iterations,
          get_total_moves()
        ));
        best_state = accepted_state;
        min_cost = cur_cost;
        best_legality = cur_legality;
      }

      engine.telemetry.endEvent(
        "sa_move",
        base::Config()
          .set<double>("current_best_cost", min_cost)
          .set<double>("cost", cur_cost)
          .set<double>("delta_cost", delta_cost)
          .set<std::string>("legality", utils::toString(cur_legality))
          .set<double>("time_secs", cost_eval_timer.getSeconds())
          .set<double>("acceptance_rate", cfg.temp_schedule.getAcceptanceRatio())
          .set<std::string>("estimator", cfg.cost_estimator)
          .set<std::string>("placement", base::serializePlacement(current_state))
          .set<bool>("acceptance", is_move_accepted)
          .set<double>("temperature", cfg.temp_schedule.temperature)
          .set<int>("range_limit", cfg.sa_swap_bound)
      );

      // If the move is rejected, restore the current state to the last accepted state.
      // Also restore the cost estimator state by feeding it the accepted state again.
      // This is needed because the cost estimator state is updated based on the current state and the move attempt,
      // which may be different from the accepted state if the previous move is rejected.
      if (!is_move_accepted) {
        restoreCurrentStateToAcceptedState(current_state, accepted_state, moved_l_cores);
        updateEstimatorStateWithDeltaChange(engine, cfg, accepted_state, moved_l_cores);
      }
    } // for each move in an iteration
    print_per_iter_stats();

    // Update temperature according to schedule
    cfg.temp_schedule.updateTemperatureAccordingToSchedule();

    // Update the range limit base on the acceptance rate.
    double acceptance_ratio = cfg.temp_schedule.getAcceptanceRatio();
    int new_bound = static_cast<int>(std::round(cfg.sa_swap_bound * (1.0 - kAcceptanceRatioTarget + acceptance_ratio)));
    // Ensure the bound is at least 1
    new_bound = std::max(1, new_bound); 
    // Cap the bound to be the maximum Manhattan distance on the board
    new_bound = std::min(new_bound, npu_.getDimensions().getNumCols() + npu_.getDimensions().getNumRows()); 
    cfg.sa_swap_bound = new_bound;

    iter_timer.pause();
    engine.telemetry.endEvent(
      "sa_iter",
      base::Config()
        // TODO: consider use a more protected method rather than
        // direct access to curr_moves_in_iteration
        .set<int>("num_moves", cfg.temp_schedule.curr_moves_in_iteration)
        .set<double>("time_secs", iter_timer.getSeconds())
    );
  } // for each iteration

  // Don't pause total iteration or cost eval timer, otherwise will double count
  utils::pause(total_sa_timer);

  if (cfg.cost_estimator == "bb" || cfg.cost_estimator == "prob") {
    engine.logger.minimal([&](auto &log) {
      auto [estimate_cost, estimate_legality] = evaluateBoundingBoxBasedCost(engine, cfg, best_state);
      auto [true_cost, true_legality] = evaluateRoutingBasedCost(engine, cfg, best_state);
      log << std::format(
        "\n[SA] Final best placement legality check - Estimated Cost: {}, "
        "Estimated Legality: {}; True Cost: {}, True Legality: {}\n",
        estimate_cost, utils::toString(estimate_legality), true_cost,
        utils::toString(true_legality)
      );
    });
  }

  // TODO: refactor time reporting
  double total_sa_time_sec = total_sa_timer.getSeconds();

  if (engine.timer.hasExpired()) {
    engine.logger.minimal("\n[SA] Timer expired during SA placement, terminating SA.\n");
  }

  engine.logger.minimal(std::format(
    "\n[SA] Final best cost: {}\n"
    "[SA] Total # of iterations: {}\n"
    "[SA] Total # of move attempts: {}\n"
    "\t- Accepted moves:  {} ({:.1f} %)\n"
    "\t  Rejected moves:  {} ({:.1f} %)\n"
    "\t- Legal moves:     {} ({:.1f} %)\n"
    "\t  Congested moves: {} ({:.1f} %)\n"
    "\t  Fatal moves:     {} ({:.1f} %)\n"
    "[SA] Average # of move attempts per iteration: {:.1f}\n"
    "[SA] Total SA runtime: {:.3f} secs\n",
    min_cost, cfg.temp_schedule.curr_iterations, get_total_moves(),
    num_moves_accepted, get_accepted_move_ratio(), num_moves_rejected,
    get_rejected_move_ratio(), num_moves_legal, get_legal_move_ratio(),
    num_moves_congested, get_congested_move_ratio(), num_moves_fatal,
    get_fatal_move_ratio(),
    1.0f * get_total_moves() / cfg.temp_schedule.curr_iterations,
    total_sa_time_sec
  ));
    
  return best_state;
}

TemperatureSchedule
SAPlacer::getDynamicTemperatureSchedule(engine::Components &engine, Config &cfg) const {

  auto estimate_equilibrium_temperature =
      [](const std::vector<double> &move_costs) -> double {
    // Find delta cost
    double max_delta_cost = 0.0;
    std::vector<double> delta_costs;
    for (size_t i = 1; i < move_costs.size(); ++i) {
      double delta_cost = move_costs[i] - move_costs[0];
      delta_costs.push_back(delta_cost);
      max_delta_cost = std::max(max_delta_cost, std::abs(delta_cost));
    }

    auto getExpectedValue = [&](std::vector<double> &delta_costs, double current_T) -> double {
      double expected_value = 0.0;
      for (const double &delta_cost : delta_costs) {
        if (delta_cost <= 0) {
          expected_value += delta_cost;
        } else {
          expected_value += delta_cost * std::exp(-delta_cost / current_T);
        }
      }
      return expected_value;
    };

    // Use binary search to find the equilibrium temperature where the expected
    // delta cost is close to zero
    //      Initialize the upper bound temperature. It is possible for
    //      the equilibrium temperature to be infinite if the initial placement
    //      is so bad that no swaps are accepted. In that case this value will
    //      be returned instead of infinity.
    //      At this temperature, the probability of accepting this worst
    //      rejected swap would be 81.873% (e^(-1/5)).
    //      TODO: Investigate if this is a good initial temperature for these
    //      cases.
    double initial_temperature = 1;
    double low_T = 0.0;
    double high_T = 5.0 * max_delta_cost;
    double tolerance = 1e-6;
    double expected_value = getExpectedValue(delta_costs, initial_temperature);
    size_t max_binary_search_iterations = 100;
    size_t binary_search_iteration = 0;
    // Binary search for equilibrium temperature, stopping when the relative
    // difference between high_T and low_T is within 0.0001% of the final
    // temperature
    while (std::abs(high_T - low_T) / initial_temperature > tolerance) {
      if (expected_value > 0) {
        high_T = initial_temperature;
        initial_temperature = (low_T + high_T) / 2;
      } else {
        low_T = initial_temperature;
        initial_temperature = (low_T + high_T) / 2;
      }
      expected_value = getExpectedValue(delta_costs, initial_temperature);
      binary_search_iteration++;
      if (binary_search_iteration >= max_binary_search_iterations) {
        break;
      }
    }
    return initial_temperature;
  };

  auto estimate_starting_temp_using_cost_variance =
      [](const std::vector<double> &move_costs) -> double {
    double mean = 0.0;
    for (const double &cost : move_costs) {
      mean += cost;
    }
    mean /= static_cast<double>(move_costs.size());

    double variance = 0.0;
    for (const double &cost : move_costs) {
      variance += (cost - mean) * (cost - mean);
    }
    variance /= static_cast<double>(move_costs.size());

    double stddev = std::sqrt(variance);
    // get the temperature that accepts 90% of moves with cost difference
    // within 2*stddev
    return (-3.0 * stddev) / std::log(0.9);
  };

  double initial_temperature = 0.0;
  if (cfg.initial_temperature_estimate_method == "std-dev" ||
      cfg.initial_temperature_estimate_method == "equilibrium") {
    // Calculate initial temperature based random move costs
    if (cfg.initial_temperature_num_sampling_moves == -1) {
      cfg.initial_temperature_num_sampling_moves =
          static_cast<int>(cfg.init_state.size()) *
          kInitialTemperatureNumSamplingMovesScalingFactor;
    }
    if (cfg.initial_temperature_num_sampling_moves < 10) {
      throw std::runtime_error(
        "Error: Number of initial sampling moves must be at least 10 (this "
        "number can be set by the command-line option or calculated by the "
        "number of logical cores in the circuit if the command-line option is"
        "set to -1) to calculate initial temperature."
      );
    }
    std::vector<double> move_costs;
    auto [init_cost, init_legality] = evaluatePlacement(engine, cfg, cfg.init_state);
    move_costs.push_back(init_cost);
    Placement new_state = cfg.init_state;
    std::vector<base::LogicalCore> moved_l_cores; // for tracking moved logical cores in each move
    // TODO: consider adding a timeout check for this loop?
    for (size_t i = 0; i < cfg.initial_temperature_num_sampling_moves; ++i) {
      generateNewState(cfg, cfg.init_state, new_state, moved_l_cores);
      auto [new_cost, new_legality] = evaluatePlacement(engine, cfg, new_state);
      move_costs.push_back(new_cost);
    }

    if (cfg.initial_temperature_estimate_method == "std-dev") {
      initial_temperature = estimate_starting_temp_using_cost_variance(move_costs);
    } else if (cfg.initial_temperature_estimate_method == "equilibrium") {
      initial_temperature = estimate_equilibrium_temperature(move_costs);
    }
  } else {
    // Default case is manual
    initial_temperature = cfg.initial_temperature_value;
  }
  initial_temperature *= cfg.initial_temperature_multiplier;
  return TemperatureSchedule(
      initial_temperature, cfg.max_iters, cfg.num_moves_per_iteration,
      cfg.greedy_stage_num_iters, cfg.max_move_attempts, cfg.greedy_stage_entering_temperature,
      cfg.greedy_stage_entering_acceptance_ratio);
}

} // namespace place::placer_zoo::sa
