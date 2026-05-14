#pragma once

#include "engine/logger.hpp"
#include "engine/sync_timer.hpp"
#include "engine/telemetry.hpp"

namespace engine {

struct Components {
  const engine::Logger &logger;
  engine::Telemetry &telemetry;
  engine::SyncTimer &timer;

  Components(const engine::Logger &logger_, engine::Telemetry &telemetry_,
             engine::SyncTimer &sync_timer_)
      : logger(logger_), telemetry(telemetry_), timer(sync_timer_) {}
};

} // namespace engine
