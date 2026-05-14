#pragma once

#include <utility>

#include "base/routing.hpp"
#include "base/rr_graph.hpp"
#include "engine/component.hpp"

namespace route {

class Router {
private:
  struct RouterConcept {
    virtual ~RouterConcept() = default;
    virtual base::RoutingState route(engine::Components &,
                                     const base::RoutingMode &,
                                     const base::RoutingNetList &) const = 0;
  };

  template <typename T> struct RouterModel : RouterConcept {
    T impl_;

    explicit RouterModel(const base::RRGraph &graph) : impl_(graph) {}

    base::RoutingState route(engine::Components &engine,
                             const base::RoutingMode &mode,
                             const base::RoutingNetList &nets) const override {
      return impl_.route(engine, mode, nets);
    }
  };

  std::unique_ptr<RouterConcept> pimpl_;

  // Private constructor
  template <typename T>
  explicit Router(std::in_place_type_t<T>, const base::RRGraph &graph)
      : pimpl_(std::make_unique<RouterModel<T>>(graph)) {}

public:
  // Static factory
  template <typename T> static Router create(const base::RRGraph &graph) {
    return Router(std::in_place_type<T>, graph);
  }

  base::RoutingState route(engine::Components &engine,
                           const base::RoutingMode &mode,
                           const base::RoutingNetList &nets) const {
    return pimpl_->route(engine, mode, nets);
  }
};

} // namespace route
