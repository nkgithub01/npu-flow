#include "engine/engine.hpp"

#include <filesystem>

#include "base/engine.hpp"
#include "base/placement.hpp"
#include "base/routing/routing_mode.hpp"
#include "base/routing/routing_state.hpp"
#include "base/rr_graph.hpp"
#include "engine/component.hpp"
#include "engine/sync_timer.hpp"
#include "engine/telemetry.hpp"
#include "utils/misc.hpp"

namespace engine {

PnRState PnREngine::run(base::PnRPlacedNetlist placed_netlist) const {
  const base::PnRPlacedNetlist &input = placed_netlist;

  // TODO: right now the timeout is mainly for placement stage
  int timeout_secs = cfg_.getOrDefault<int>("timeout_secs", -1);
  engine::SyncTimer timer{timeout_secs};
  logger_.minimal(std::format(
      "[PnR] Timeout enabled: each PnR stage will be terminated after {} secs "
      "(may slightly exceed this value due to internal processing)\n",
      timeout_secs));
  timer.start();

  if (cfg_.getOrDefault<bool>("dump_rr_graph", false)) {
    std::filesystem::path p = "rr_graph.dot"; // TODO: make it configurable
    logger_.normal(std::format("Dumping RR graph to {}\n",
                               std::filesystem::absolute(p).string()));
    std::ofstream(p) << arch_.getRRGraph().visualizeAsDotFile();
  }

  engine::Telemetry telemetry =
      createTelemetry(cfg_.get<base::Config>("telemetry"));
  telemetry.startEvent("engine_run",
                       cfg_.getOrDefault<base::Config>("engine", {}));

  engine::Components engine_components{logger_, telemetry, timer};

  // Placement
  base::Placement final_placement =
      placer_.place(engine_components, cfg_.get<base::Config>("placer"),
                    input.tf_graph, input.placement);
  if (arch_.isLegalPlacement(final_placement)) {
    logger_.minimal(std::format("\n[PnR] Found legal placement.\n"));
  } else {
    logger_.minimal(std::format(
        "\n[PnR] Found illegal placement. Bailing out without routing.\n"));
    throw std::runtime_error(
        "Illegal placement found, cannot proceed to routing.");
  }

  // Routing
  base::Config router_cfg = cfg_.get<base::Config>("router");
  base::RoutingMode full_routing_mode = base::RoutingMode();
  full_routing_mode.set(base::RoutingMode::LogicalCorePlacement::Fixed,
                        {base::cast(final_placement)});
  // Assume packing constraints are already enforced in placing stage
  full_routing_mode.set(base::RoutingMode::LogicalCorePacking::Ignored, {});
  full_routing_mode.set(
      router_cfg.get<bool>("enable_lock_constraints")
          ? base::RoutingMode::LockCapacityConstraint::Enforced
          : base::RoutingMode::LockCapacityConstraint::Ignored);
  // Set time limit for routing stage based on the CLI timeout config
  full_routing_mode.set(timer.isUnlimited()
                            ? base::RoutingMode::TimeLimit::Unlimited
                            : base::RoutingMode::TimeLimit::Limited,
                        timer.isUnlimited() ? 0 : timer.getTimeoutSecs());
  base::RoutingMode::HyperParameters router_hyper_params;
  router_hyper_params.objective_congestion_penalty_scaling_factor =
      router_cfg.getOrDefault<double>("congestion_penalty_scaling_factor", 1e5);
  full_routing_mode.set(router_hyper_params);

  base::RoutingState final_routing =
      router_.route(engine_components, full_routing_mode,
                    base::RoutingNetList{input.tf_graph});

  if (timer.hasExpired()) {
    logger_.minimal(
        std::format("\n[PnR] Timeout reached during PnR process.\n"));
  }

  logger_.minimal(std::format("\n[PnR] Placement Summary:\n"
                              " - Total # of logical cores: {}\n"
                              " - Total # of physical cores: {}\n\n"
                              "[PnR] Serialized placement:\n{}\n",
                              final_placement.keys().size(),
                              final_placement.values().size(),
                              base::serializePlacement(final_placement)));

  logger_.minimal(std::format("[PnR] Routing Summary:\n"
                              " - Legality: {}\n"
                              " - Cost: {:.6f}\n",
                              utils::toString(final_routing.getLegality()),
                              final_routing.getRoutingCost()));

  // TODO: add telemetry event for placement and routing stats
  telemetry.endEvent("engine_run");

  return PnRState{.tf_graph = input.tf_graph,
                  .placement = final_placement,
                  .routing = final_routing};
}

} // namespace engine
