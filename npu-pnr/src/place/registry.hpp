#pragma once

#include "place/placer.hpp"

#include "place/placer_zoo/ls/placer.hpp"
#include "place/placer_zoo/milp/placer.hpp"
#include "place/placer_zoo/sa/placer.hpp"

namespace place {

class NoOpPlacer {
private:
  const arch::Arch &arch_;
  const route::Router &router_;

public:
  explicit NoOpPlacer(const arch::Arch &arch, const route::Router &router)
      : arch_(arch), router_(router) {}

  base::Placement place(engine::Components &engine, const base::Config &cfg,
                        const base::TrafficFlowGraph &tf_graph,
                        const base::Placement &initial_placement) const {
    // Return a copy of the initial placement to skip placement stage
    return initial_placement;
  }
};

inline Placer createPlacer(const base::Config &cfg, const arch::Arch &arch,
                           const route::Router &router) {
  std::string type = cfg.get<std::string>("type");

  if (type == "sa") {
    return Placer::create<placer_zoo::sa::SAPlacer>(arch, router);
  } else if (type == "ls") {
    return Placer::create<placer_zoo::ls::LSPlacer>(arch, router);
  } else if (type == "milp") {
    return Placer::create<placer_zoo::milp::MILPPlacer>(arch, router);
  } else if (type == "noop") {
    return Placer::create<NoOpPlacer>(arch, router);
  }

  throw std::runtime_error("Placer type " + cfg.get<std::string>("type") +
                           " is not implemented.");
}

} // namespace place
