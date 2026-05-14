#include <utility>

#include <gtest/gtest.h>

#include "arch/npu/npu.hpp"
#include "base/common.hpp"
#include "base/engine.hpp"
#include "base/routing.hpp"
#include "base/rr_graph.hpp"
#include "engine/component.hpp"
#include "engine/sync_timer.hpp"
#include "engine/telemetry.hpp"
#include "gtest_helpers.hpp"
#include "route/registry.hpp"
#include "route/router.hpp"
#include "route/router_zoo/milp/router.hpp"

namespace {

base::RRGraph createDummyRRGraph() {
  return arch::npu::NPU{base::Config{}}.getRRGraph();
}

engine::Components createDummyComponents() {
  static engine::Logger logger{base::Config{}};
  static engine::Telemetry telemetry =
      engine::Telemetry::create<engine::NoOpTelemetry>(base::Config{});
  static engine::SyncTimer timer(-1);
  return engine::Components(logger, telemetry, timer);
}

base::PnRPlacedNetlist createDummyPnRPlacedNetlist() {
  constexpr const char *toml = R"(
    [edge] # Please use `edge` not `edges`!
    e1 = [ "comp1->comp2", [ 2 ], 1 ]
  )";
  arch::npu::NPU npu{base::Config{}};
  return npu.parseNetlist(base::PnRNetlistReader::fromTOML(toml));
}

base::RoutingMode createDummyRoutingMode() {
  base::RoutingMode mode;
  mode.setLazyEvaluation();
  mode.set(base::RoutingMode::LogicalCorePlacement::Fixed,
           {base::cast(createDummyPnRPlacedNetlist().placement)});
  mode.set(base::RoutingMode::LogicalCorePacking::Ignored, {});
  return mode;
}

base::RoutingNetList createDummyRoutingNetList() {
  return base::RoutingNetList{createDummyPnRPlacedNetlist().tf_graph};
}

class MockRouter {
private:
  base::RRGraph graph_;

public:
  explicit MockRouter(const base::RRGraph &graph) : graph_(graph) {}

  base::RoutingState route(engine::Components &, const base::RoutingMode &mode,
                           const base::RoutingNetList &) const {
    return base::RoutingState{mode, base::RoutingState::Legality::Legal, 0.0F};
  }
};

} // namespace

TEST(RouterTypeErasureTest, CreateRouterUsingStaticFactory) {
  const auto graph = createDummyRRGraph();
  auto components = createDummyComponents();
  const auto mode = createDummyRoutingMode();
  const auto nets = createDummyRoutingNetList();

  const auto router = route::Router::create<MockRouter>(graph);
  EXPECT_NO_THROW(router.route(components, mode, nets));
}

TEST(RouterTypeErasureTest, RouterCanBeMoved) {
  const auto graph = createDummyRRGraph();
  auto components = createDummyComponents();
  const auto mode = createDummyRoutingMode();
  const auto nets = createDummyRoutingNetList();

  auto router1 = route::Router::create<MockRouter>(graph);
  auto router2 = std::move(router1);

  EXPECT_NO_THROW(router2.route(components, mode, nets));
}

TEST(RouterTypeErasureTest, CanChainMoveOperations) {
  const auto graph = createDummyRRGraph();
  auto components = createDummyComponents();
  const auto mode = createDummyRoutingMode();
  const auto nets = createDummyRoutingNetList();

  auto router1 = route::Router::create<MockRouter>(graph);
  auto router2 = std::move(router1);
  auto router3 = std::move(router2);

  EXPECT_NO_THROW(router3.route(components, mode, nets));
}

TEST(RouterTypeErasureTest, RouterConstCorrectness) {
  const auto graph = createDummyRRGraph();
  auto components = createDummyComponents();
  const auto mode = createDummyRoutingMode();
  const auto nets = createDummyRoutingNetList();

  const auto router = route::Router::create<MockRouter>(graph);
  EXPECT_NO_THROW(router.route(components, mode, nets));
}

TEST(RouterTypeErasureTest, MultipleRoutersCanCoexistIndependently) {
  const auto graph = createDummyRRGraph();
  auto components = createDummyComponents();
  const auto mode = createDummyRoutingMode();
  const auto nets = createDummyRoutingNetList();

  const auto router1 = route::Router::create<MockRouter>(graph);
  const auto router2 =
      route::Router::create<route::router_zoo::milp::MILPRouter>(graph);

  EXPECT_NO_THROW(router1.route(components, mode, nets));
  EXPECT_NO_THROW(router2.route(components, mode, nets));
}

TEST(CreateRouterRegistryTest, CreateMilpRouterWithValidConfig) {
  const auto graph = createDummyRRGraph();
  base::Config cfg;
  cfg.set("type", std::string("milp"));

  EXPECT_NO_THROW(route::createRouter(cfg, graph));
}

TEST(CreateRouterRegistryTest, CreatedRouterIsFunctional) {
  const auto graph = createDummyRRGraph();
  auto components = createDummyComponents();
  const auto mode = createDummyRoutingMode();
  const auto nets = createDummyRoutingNetList();
  base::Config cfg;
  cfg.set("type", std::string("milp"));

  const auto router = route::createRouter(cfg, graph);
  EXPECT_NO_THROW(router.route(components, mode, nets));
}

TEST(CreateRouterRegistryTest, UnsupportedRouterTypeThrows) {
  const auto graph = createDummyRRGraph();
  base::Config cfg;
  cfg.set("type", std::string("astar"));

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)route::createRouter(cfg, graph); },
      {"astar", "not implemented"});
}

TEST(CreateRouterRegistryTest, MissingTypeFieldThrows) {
  const auto graph = createDummyRRGraph();
  base::Config cfg;

  EXPECT_THROW(route::createRouter(cfg, graph), std::runtime_error);
}

TEST(CreateRouterRegistryTest, EmptyTypeStringThrows) {
  const auto graph = createDummyRRGraph();
  base::Config cfg;
  cfg.set("type", std::string(""));

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)route::createRouter(cfg, graph); },
      {"not implemented"});
}

TEST(CreateRouterRegistryTest, MultipleRoutersFromRegistryAreIndependent) {
  const auto graph = createDummyRRGraph();
  auto components = createDummyComponents();
  const auto mode = createDummyRoutingMode();
  const auto nets = createDummyRoutingNetList();
  base::Config cfg;
  cfg.set("type", std::string("milp"));

  const auto router1 = route::createRouter(cfg, graph);
  const auto router2 = route::createRouter(cfg, graph);

  EXPECT_NO_THROW(router1.route(components, mode, nets));
  EXPECT_NO_THROW(router2.route(components, mode, nets));
}
