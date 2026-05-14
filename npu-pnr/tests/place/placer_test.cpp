#include <utility>

#include <gtest/gtest.h>

#include "arch/arch.hpp"
#include "arch/npu/npu.hpp"
#include "base/common.hpp"
#include "base/engine.hpp"
#include "base/placement.hpp"
#include "base/tf_graph.hpp"
#include "engine/component.hpp"
#include "engine/sync_timer.hpp"
#include "engine/telemetry.hpp"
#include "gtest_helpers.hpp"
#include "place/placer.hpp"
#include "place/placer_zoo/sa/placer.hpp"
#include "place/registry.hpp"
#include "route/registry.hpp"
#include "route/router.hpp"

namespace {

arch::Arch createDummyArch() {
  return arch::Arch::create<arch::npu::NPU>(base::Config{});
}

route::Router createDummyRouter() {
  base::RRGraph graph = createDummyArch().getRRGraph();
  base::Config cfg;
  cfg.set("type", std::string("milp"));
  return route::createRouter(cfg, graph);
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
    [edge]
    e1 = [ "comp1->comp2", [ 2 ], 1 ]
  )";
  return createDummyArch().parseNetlist(base::PnRNetlistReader::fromTOML(toml));
}

base::TrafficFlowGraph createDummyTrafficFlowGraph() {
  return createDummyPnRPlacedNetlist().tf_graph;
}

base::Placement createDummyPlacement() {
  return createDummyPnRPlacedNetlist().placement;
}

class MockPlacer {
private:
  const arch::Arch &arch_;
  const route::Router &router_;

public:
  explicit MockPlacer(const arch::Arch &arch, const route::Router &router)
      : arch_(arch), router_(router) {}

  base::Placement place(engine::Components &, const base::Config &,
                        const base::TrafficFlowGraph &,
                        const base::Placement &initial_placement) const {
    return initial_placement;
  }
};

} // namespace

TEST(PlacerTypeErasureTest, CreatePlacerUsingStaticFactory) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  base::Config cfg;
  const auto tf_graph = createDummyTrafficFlowGraph();
  const auto initial_placement = createDummyPlacement();

  const auto placer = place::Placer::create<MockPlacer>(arch, router);
  EXPECT_NO_THROW(placer.place(components, cfg, tf_graph, initial_placement));
}

TEST(PlacerTypeErasureTest, PlacerCanBeMoved) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  base::Config cfg;
  const auto tf_graph = createDummyTrafficFlowGraph();
  const auto initial_placement = createDummyPlacement();

  auto placer1 = place::Placer::create<MockPlacer>(arch, router);
  auto placer2 = std::move(placer1);

  EXPECT_NO_THROW(placer2.place(components, cfg, tf_graph, initial_placement));
}

TEST(PlacerTypeErasureTest, CanChainMoveOperations) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  base::Config cfg;
  const auto tf_graph = createDummyTrafficFlowGraph();
  const auto initial_placement = createDummyPlacement();

  auto placer1 = place::Placer::create<MockPlacer>(arch, router);
  auto placer2 = std::move(placer1);
  auto placer3 = std::move(placer2);

  EXPECT_NO_THROW(placer3.place(components, cfg, tf_graph, initial_placement));
}

TEST(PlacerTypeErasureTest, PlacerConstCorrectness) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  base::Config cfg;
  const auto tf_graph = createDummyTrafficFlowGraph();
  const auto initial_placement = createDummyPlacement();

  const auto placer = place::Placer::create<MockPlacer>(arch, router);
  EXPECT_NO_THROW(placer.place(components, cfg, tf_graph, initial_placement));
}

TEST(PlacerTypeErasureTest, MultiplePlacersCanCoexistIndependently) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  base::Config cfg;
  const auto tf_graph = createDummyTrafficFlowGraph();
  const auto initial_placement = createDummyPlacement();

  const auto placer1 = place::Placer::create<MockPlacer>(arch, router);
  const auto placer2 = place::Placer::create<MockPlacer>(arch, router);

  EXPECT_NO_THROW(placer1.place(components, cfg, tf_graph, initial_placement));
  EXPECT_NO_THROW(placer2.place(components, cfg, tf_graph, initial_placement));
}

TEST(PlacerTypeErasureTest, PlacerReturnsExpectedPlacement) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  base::Config cfg;
  const auto tf_graph = createDummyTrafficFlowGraph();
  const auto initial_placement = createDummyPlacement();

  const auto placer = place::Placer::create<MockPlacer>(arch, router);
  EXPECT_EQ(placer.place(components, cfg, tf_graph, initial_placement),
            initial_placement);
}

TEST(PlacerTypeErasureTest, PlacerCanBeCalledMultipleTimes) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  base::Config cfg;
  const auto tf_graph = createDummyTrafficFlowGraph();
  const auto initial_placement = createDummyPlacement();

  const auto placer = place::Placer::create<MockPlacer>(arch, router);
  EXPECT_NO_THROW(placer.place(components, cfg, tf_graph, initial_placement));
  EXPECT_NO_THROW(placer.place(components, cfg, tf_graph, initial_placement));
  EXPECT_NO_THROW(placer.place(components, cfg, tf_graph, initial_placement));
}

TEST(PlacerTypeErasureTest, DifferentPlacerTypesCanCoexist) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  base::Config cfg;
  const auto tf_graph = createDummyTrafficFlowGraph();
  const auto initial_placement = createDummyPlacement();

  const auto mock_placer = place::Placer::create<MockPlacer>(arch, router);
  const auto sa_placer =
      place::Placer::create<place::placer_zoo::sa::SAPlacer>(arch, router);

  EXPECT_NO_THROW(
      mock_placer.place(components, cfg, tf_graph, initial_placement));
  cfg.set("max_move_attempts", 5);
  EXPECT_NO_THROW(sa_placer.place(components, cfg, tf_graph, initial_placement));
}

TEST(CreatePlacerRegistryTest, CreateSaPlacerWithValidConfig) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  base::Config cfg;
  cfg.set("type", std::string("sa"));

  EXPECT_NO_THROW(place::createPlacer(cfg, arch, router));
}

TEST(CreatePlacerRegistryTest, CreatedPlacerIsFunctional) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  auto tf_graph = createDummyTrafficFlowGraph();
  auto initial_placement = createDummyPlacement();
  base::Config cfg;
  cfg.set("type", std::string("sa"));
  cfg.set("max_move_attempts", 5);

  const auto placer = place::createPlacer(cfg, arch, router);
  EXPECT_NO_THROW(placer.place(components, cfg, tf_graph, initial_placement));
}

TEST(CreatePlacerRegistryTest, UnsupportedPlacerTypeThrows) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  base::Config cfg;
  cfg.set("type", std::string("random"));

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)place::createPlacer(cfg, arch, router); },
      {"random", "not implemented"});
}

TEST(CreatePlacerRegistryTest, MissingTypeFieldThrows) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  base::Config cfg;

  EXPECT_THROW(place::createPlacer(cfg, arch, router), std::runtime_error);
}

TEST(CreatePlacerRegistryTest, EmptyTypeStringThrows) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  base::Config cfg;
  cfg.set("type", std::string(""));

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)place::createPlacer(cfg, arch, router); },
      {"not implemented"});
}

TEST(CreatePlacerRegistryTest, MultiplePlacersFromRegistryAreIndependent) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  auto tf_graph = createDummyTrafficFlowGraph();
  auto initial_placement = createDummyPlacement();
  base::Config cfg;
  cfg.set("type", std::string("sa"));
  cfg.set("max_move_attempts", 5);

  const auto placer1 = place::createPlacer(cfg, arch, router);
  const auto placer2 = place::createPlacer(cfg, arch, router);

  EXPECT_NO_THROW(placer1.place(components, cfg, tf_graph, initial_placement));
  EXPECT_NO_THROW(placer2.place(components, cfg, tf_graph, initial_placement));
}

TEST(CreatePlacerRegistryTest, CreatedPlacerCanBeMovedAndUsed) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  auto tf_graph = createDummyTrafficFlowGraph();
  auto initial_placement = createDummyPlacement();
  base::Config cfg;
  cfg.set("type", std::string("sa"));
  cfg.set("max_move_attempts", 5);

  auto placer1 = place::createPlacer(cfg, arch, router);
  auto placer2 = std::move(placer1);

  EXPECT_NO_THROW(placer2.place(components, cfg, tf_graph, initial_placement));
}

TEST(CreatePlacerRegistryTest, RegistryFunctionSupportsDifferentConfigurations) {
  auto arch = createDummyArch();
  auto router = createDummyRouter();
  auto components = createDummyComponents();
  auto tf_graph = createDummyTrafficFlowGraph();
  auto initial_placement = createDummyPlacement();

  base::Config cfg1;
  cfg1.set("type", std::string("sa"));
  cfg1.set("max_move_attempts", 5);

  base::Config cfg2;
  cfg2.set("type", std::string("sa"));
  cfg2.set("max_iters", 5);

  const auto placer1 = place::createPlacer(cfg1, arch, router);
  const auto placer2 = place::createPlacer(cfg2, arch, router);

  EXPECT_NO_THROW(placer1.place(components, cfg1, tf_graph, initial_placement));
  EXPECT_NO_THROW(placer2.place(components, cfg2, tf_graph, initial_placement));
}
