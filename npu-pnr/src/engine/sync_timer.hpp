#pragma once

#include <chrono>
#include <stdexcept>

namespace engine {

// Synchronous timer that can be polled to check if a timeout has elapsed.
// Designed to be used in iteration loops as part of the exit condition.
//
// A negative timeout_secs (e.g., -1) creates an unlimited timer that never
// expires.
class SyncTimer {
private:
  int timeout_secs_;
  bool started_{false};
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_check_time_;
  bool has_been_checked_{false};

public:
  explicit SyncTimer(int timeout_secs) : timeout_secs_(timeout_secs) {}

  ~SyncTimer() = default;

  // Starts the timer. Resets all internal state.
  void start() {
    start_time_ = std::chrono::steady_clock::now();
    last_check_time_ = start_time_;
    has_been_checked_ = false;
    started_ = true;
  }

  // Checks if the timeout has been reached.
  // Returns false if the timer has not been started.
  // Returns false always if timeout_secs is negative (unlimited timer).
  // Updates the internal timestamp used by secondsSinceLastCheck().
  bool hasExpired() {
    if (!started_) {
      return false;
    }

    auto now = std::chrono::steady_clock::now();
    last_check_time_ = now;
    has_been_checked_ = true;

    // Unlimited timer never expires
    if (timeout_secs_ < 0) {
      return false;
    }

    // Zero timeout expires immediately
    if (timeout_secs_ == 0) {
      return true;
    }

    auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(now - start_time_)
            .count();
    return elapsed >= timeout_secs_;
  }

  // Returns the time elapsed in seconds since the last hasExpired() call.
  // If hasExpired() has not been called yet, returns time since start().
  // If the timer has not been started, returns 0.0.
  double secondsSinceLastCheck() {
    if (!started_) {
      return 0.0;
    }

    auto now = std::chrono::steady_clock::now();
    auto reference_time = has_been_checked_ ? last_check_time_ : start_time_;

    return std::chrono::duration<double>(now - reference_time).count();
  }

  // Returns the remaining time in seconds before the timer expires.
  // Returns 0.0 if the timer has already expired.
  // Throws std::runtime_error if the timer has not been started.
  // Throws std::runtime_error if the timer is unlimited.
  double secondsRemaining() {
    if (!started_) {
      throw std::runtime_error(
          "SyncTimer::secondsRemaining() called before start()");
    }

    if (timeout_secs_ < 0) {
      throw std::runtime_error(
          "SyncTimer::secondsRemaining() called on unlimited timer");
    }

    if (timeout_secs_ == 0) {
      return 0.0;
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    double remaining = static_cast<double>(timeout_secs_) - elapsed;

    return remaining > 0.0 ? remaining : 0.0;
  }

  // Returns the configured timeout in seconds.
  int getTimeoutSecs() const { return timeout_secs_; }

  // Returns true if the timer is configured as unlimited (negative timeout).
  bool isUnlimited() const { return timeout_secs_ < 0; }
};

} // namespace engine
