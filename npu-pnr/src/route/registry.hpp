#pragma once

#include "base/common.hpp"
#include "route/router.hpp"

#include "route/router_zoo/milp/router.hpp"

namespace route {

inline Router createRouter(const base::Config &cfg,
                           const base::RRGraph &graph) {
  std::string type = cfg.get<std::string>("type");

  if (type == "milp") {
    return Router::create<router_zoo::milp::MILPRouter>(graph);
  }

  throw std::runtime_error("Router type " + cfg.get<std::string>("type") +
                           " is not implemented.");
}

} // namespace route
