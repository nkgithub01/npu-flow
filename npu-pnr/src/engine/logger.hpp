#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "base/common.hpp"

namespace engine {

class Logger {
private:
  base::Verbosity verbosity_;

  // TODO: consider adding scope-based logging with timestamps, etc.
  void log(base::Verbosity level, const std::string &message) const {
    if (verbosity_ >= level) {
      // TODO: Add timestamp, level prefix, etc. here
      std::cout << message << std::flush;
    }
  }

  void log(base::Verbosity level,
           std::function<void(std::stringstream &)> message_fn) const {
    if (verbosity_ >= level) {
      std::stringstream ss;
      message_fn(ss);
      log(level, ss.str());
    }
  }

  // TODO: consider forwarding std::format-style arguments

public:
  Logger(const base::Config &cfg)
      : verbosity_(base::getVerbosityFromString(
            cfg.getOrDefault<std::string>("verbose", "normal"))) {}

  template <typename T> void minimal(T &&message) const {
    log(base::Verbosity::Minimal, std::forward<T>(message));
  }

  template <typename T> void normal(T &&message) const {
    log(base::Verbosity::Normal, std::forward<T>(message));
  }

  template <typename T> void verbose(T &&message) const {
    log(base::Verbosity::Verbose, std::forward<T>(message));
  }

  template <typename T> void debug(T &&message) const {
    log(base::Verbosity::Debug, std::forward<T>(message));
  }
};

} // namespace engine
