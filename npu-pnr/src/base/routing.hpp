#ifndef _BASE_ROUTING_HPP_
#define _BASE_ROUTING_HPP_

#include "base/routing/routing_mode.hpp"
#include "base/routing/routing_net.hpp"
#include "base/routing/routing_state.hpp"
#include "base/routing/visualization.hpp"

// TODO: refactor the following entrypoint header
namespace base {

// routing_mode.hpp
class RoutingMode;
using base::RoutingMode;

// routing_net.hpp
class RoutingNet;
using base::RoutingNet;

// routing_state.hpp
class RoutingState;
using base::RoutingState;

// visualization.hpp
using base::RoutingVisualizer;

} // namespace base

#endif
