#pragma once

#include <utility>

#include "arch/arch.hpp"
#include "base/placement.hpp"
#include "base/tf_graph.hpp"
#include "engine/component.hpp"
#include "route/router.hpp"

namespace place {

class Placer {
private:
  struct PlacerConcept {
    virtual ~PlacerConcept() = default;
    virtual base::Placement place(engine::Components &, const base::Config &,
                                  const base::TrafficFlowGraph &,
                                  const base::Placement &) const = 0;
  };

  template <typename T> struct PlacerModel : PlacerConcept {
    T impl_;

    explicit PlacerModel(const arch::Arch &arch, const route::Router &router)
        : impl_(arch, router) {}

    base::Placement
    place(engine::Components &engine, const base::Config &cfg,
          const base::TrafficFlowGraph &tf_graph,
          const base::Placement &initial_placement) const override {
      return impl_.place(engine, cfg, tf_graph, initial_placement);
    }
  };

  std::unique_ptr<PlacerConcept> pimpl_;

  // Private constructor
  template <typename T>
  explicit Placer(std::in_place_type_t<T>, const arch::Arch &arch,
                  const route::Router &router)
      : pimpl_(std::make_unique<PlacerModel<T>>(arch, router)) {}

public:
  // Static factory
  template <typename T>
  static Placer create(const arch::Arch &arch, const route::Router &router) {
    return Placer(std::in_place_type<T>, arch, router);
  }

  base::Placement place(engine::Components &engine, const base::Config &cfg,
                        const base::TrafficFlowGraph &tf_graph,
                        const base::Placement &initial_placement) const {
    return pimpl_->place(engine, cfg, tf_graph, initial_placement);
  }
};

}; // namespace place
