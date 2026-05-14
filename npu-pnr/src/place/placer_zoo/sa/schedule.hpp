#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <span>
#include <stdexcept>
#include <vector>

namespace place::placer_zoo::sa {

// TODO: refactor this class to separate static and dynamic temperature
// scheduling derived from a common interface
class TemperatureSchedule {
public:
  // TODO: refactor these public members
  double temperature = 0.0;
  int curr_iterations = 0;
  int curr_moves_in_iteration = 0;
  int curr_greedy_iterations = 0;
  int total_moves = 0;

private:
  int num_iteration_limit_;
  int num_moves_per_iteration_limit_;
  int num_greedy_iteration_limit_;
  int total_move_limit_;
  std::vector<double> static_schedule_temperatures_;
  std::vector<double> static_schedule_cooling_factors_;
  bool use_static_schedule_ = true;
  bool in_greedy_stage_ = false;

  int accept_count_ = 0;
  int reject_count_ = 0;
  double acceptance_ratio_ = 0.0;

  std::deque<int> delta_cost_expected_value_history_;
  int delta_cost_expected_value_window_size_ = 100;
  double delta_cost_expected_value_sum_ = 0.0;
  double equilibrium_tolerance_threshold_ = 1e-3;

  // Greedy stage parameters for dynamic scheduling
  double greedy_stage_entering_temperature_;
  double greedy_stage_entering_acceptance_ratio_;

public:
  TemperatureSchedule() = default;
  explicit TemperatureSchedule(std::span<const double> stage_start_temperatures,
                               std::span<const double> stage_cooling_factors,
                               int num_iterations_limit,
                               int num_moves_per_iteration_limit,
                               int num_greedy_iterations,
                               int total_move_limit)
      : static_schedule_temperatures_(stage_start_temperatures.begin(),
                                      stage_start_temperatures.end()),
        static_schedule_cooling_factors_(stage_cooling_factors.begin(),
                                         stage_cooling_factors.end()),
        num_iteration_limit_(num_iterations_limit),
        num_moves_per_iteration_limit_(num_moves_per_iteration_limit),
        num_greedy_iteration_limit_(num_greedy_iterations),
        total_move_limit_(total_move_limit),
        use_static_schedule_(true) {

    // For manually specified schedule, the number of cooling factors and the
    // number of start temperatures must match
    if (stage_start_temperatures.size() != stage_cooling_factors.size() ||
        stage_start_temperatures.size() < 1) {
      throw std::runtime_error("Error: Temperatures and cooling factors must "
                               "have the same size and be at least 1.");
    }

    // Purely greedy schedule check
    if (stage_start_temperatures.size() == 1.0 &&
        stage_start_temperatures.front() == 0.0 &&
        stage_cooling_factors.size() == 1.0 &&
        stage_cooling_factors.front() == 0.0) {
      temperature = 0.0;
      num_iteration_limit_ = num_greedy_iterations;
      in_greedy_stage_ = true;
      return;
    }

    // Non-greedy schedule checks
    if (std::ranges::any_of(stage_start_temperatures,
                            [](double T) { return T <= 0.0; })) {
      throw std::runtime_error("Error: Temperatures must be positive.");
    }
    for (int i = 0; i < stage_start_temperatures.size() - 1; ++i) {
      if (stage_start_temperatures[i] < stage_start_temperatures[i + 1]) {
        throw std::runtime_error(
            "Error: Start temperatures must be in strictly decreasing order.");
      }
    }

    if (std::any_of(stage_cooling_factors.begin(),
                    stage_cooling_factors.end() - 1,
                    [](double f) { return f < 0 || f >= 1; })) {
      throw std::runtime_error(
          "Error: Cooling factors must be in the range (0, 1) except the last "
          "one which might be 0; otherwise, the temperature will not *cool*. "
          "Use 0 for greedy stage.");
    }

    if (stage_cooling_factors.back() != 0) {
      throw std::runtime_error(
          "Error: The last cooling factor must be 0 to indicate the greedy "
          "stage.");
    }

    // Initialize temperature
    temperature = static_schedule_temperatures_.front();
  }

  explicit TemperatureSchedule(double start_temperature,
                               int num_iteration_limit,
                               int num_moves_per_iteration_limit,
                               int num_greedy_iterations,
                               int total_move_limit,
                               double greedy_stage_entering_temperature,
                               double greedy_stage_entering_acceptance_ratio)
      : num_iteration_limit_(num_iteration_limit),
        num_moves_per_iteration_limit_(num_moves_per_iteration_limit),
        num_greedy_iteration_limit_(num_greedy_iterations),
        total_move_limit_(total_move_limit),
        greedy_stage_entering_temperature_(greedy_stage_entering_temperature),
        greedy_stage_entering_acceptance_ratio_(
        greedy_stage_entering_acceptance_ratio),
        use_static_schedule_(false) {
    delta_cost_expected_value_window_size_ = std::max(num_moves_per_iteration_limit/10, 100);
    temperature = start_temperature;
  }

  int getNumIterationsLimit() const { return num_iteration_limit_; }
  int getNumMovesPerIterationLimit() const {
    return num_moves_per_iteration_limit_;
  }
  bool isUsingStaticSchedule() const { return use_static_schedule_; }
  bool isInGreedyStage() const { return in_greedy_stage_; }
  bool outerLoopLimitReached() const {
    return (curr_iterations >= num_iteration_limit_) ||
           (in_greedy_stage_ && curr_greedy_iterations >= num_greedy_iteration_limit_) ||
           (total_moves >= total_move_limit_);
  }

  bool innerLoopLimitReached() const {
    return (curr_moves_in_iteration >= num_moves_per_iteration_limit_) ||
           (total_moves >= total_move_limit_) ||
           (!in_greedy_stage_ && reachedEquilibrium());
  }

  bool maxMoveAttemptLimitReached() const {
    return total_moves >= total_move_limit_;
  }

  double getCoolingFactor() const {
    if (use_static_schedule_) {
      // Static schedule
      for (int i = 1; i < static_schedule_temperatures_.size(); ++i) {
        if (temperature > static_schedule_temperatures_[i]) {
          return static_schedule_cooling_factors_[i - 1];
        }
      }
      // If temperature is below the lowest stage temperature, return the last
      // cooling factor
      return static_schedule_cooling_factors_.back();
    } else {
      // Dynamic schedule based on acceptance ratio
      if (accept_count_ + reject_count_ == 0) {
        return 1.0;
      } else {
        return acceptanceRatioToCoolingFactorLookup(acceptance_ratio_);
      }
    }
  }

  void applyCooling() {
    double cooling_factor = getCoolingFactor();
    temperature *= cooling_factor;
  }

  void updateTemperatureAccordingToSchedule() {
    // Cool down temperature at the end of each iteration
    applyCooling();

    // Increment Greedy Iteration Count if in greedy stage
    if (in_greedy_stage_) {
      curr_greedy_iterations++;
    } else {
      // Check if we need to enter greedy stage
      if (use_static_schedule_) {
        // In static schedule, greedy stage has cooling factor of 0,
        // so we enter greedy stage when temperature reaches 0
        if (temperature == 0.0) {
          in_greedy_stage_ = true;
        }
      } else {
        // In dynamic schedule, we enter greedy stage when acceptance ratio is
        // very low
        // TODO: define a better criterion for entering greedy stage
        if (acceptance_ratio_ < greedy_stage_entering_acceptance_ratio_ ||
            temperature < greedy_stage_entering_temperature_) {
          in_greedy_stage_ = true;
          temperature = 0.0;
        }
      }
    }

    // Clear the expected value history when temperature changes
    clearDeltaCostExpectedValueHistory();
    clearAcceptanceHistory();
  }

  double getTemperature() const { return temperature; }

  void updateAcceptanceCount(bool accepted) {
    if (accepted) {
      accept_count_++;
    } else {
      reject_count_++;
    }

    // Recalculate acceptance ratio
    acceptance_ratio_ = static_cast<double>(accept_count_) /
                        static_cast<double>(accept_count_ + reject_count_);
  }

  double getAcceptanceRatio() const { return acceptance_ratio_; }

  void clearAcceptanceHistory() {
    accept_count_ = 0;
    reject_count_ = 0;
    acceptance_ratio_ = 1.0;
  }

  void updateDeltaCostExpectedValue(double delta_cost, double probability) {
    delta_cost_expected_value_history_.push_back(delta_cost * probability);
    delta_cost_expected_value_sum_ += delta_cost * probability;
    if (delta_cost_expected_value_history_.size() > delta_cost_expected_value_window_size_) {
      delta_cost_expected_value_sum_ -= delta_cost_expected_value_history_.front();
      delta_cost_expected_value_history_.pop_front();
    }
  }

  void clearDeltaCostExpectedValueHistory() {
    delta_cost_expected_value_history_.clear();
    delta_cost_expected_value_sum_ = 0.0;
  }

  bool reachedEquilibrium() const {
    // This function can be called to indicate that we have reached the
    // equilibrium at current temperature based on the expected value of delta cost.
    // We can use this as a signal to end current iteration if we are using
    // dynamic scheduling.
    return (delta_cost_expected_value_history_.size() == delta_cost_expected_value_window_size_) && 
           (std::abs(delta_cost_expected_value_sum_) < equilibrium_tolerance_threshold_);
  }

private:
  double acceptanceRatioToCoolingFactorLookup(double acceptance_ratio) const {
    if (acceptance_ratio > 0.96) {
      return 0.5;
    } else if (acceptance_ratio > 0.8) {
      return 0.9;
    } else if (acceptance_ratio > 0.15) {
      return 0.95;
    } else {
      return 0.8;
    }
  }
};

} // namespace place::placer_zoo::sa
