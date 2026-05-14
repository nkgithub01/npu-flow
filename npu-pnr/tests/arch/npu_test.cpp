#include <ranges>
#include <string>

#include <gtest/gtest.h>

#include "arch/npu/npu.hpp"
#include "base/common.hpp"
#include "base/engine.hpp"
#include "base/placement.hpp"
#include "base/rr_graph.hpp"
#include "base/tf_graph.hpp"
#include "gtest_helpers.hpp"

namespace {

arch::npu::NPU createSmallNPU() {
  base::Config cfg;
  cfg.set("num_cols", 4);
  cfg.set("num_rows", 5);
  cfg.set("num_compute_rows", 3);
  cfg.set("num_memory_rows", 1);
  cfg.set("num_shim_rows", 1);
  return arch::npu::NPU(cfg);
}

} // namespace

TEST(NpuConstructionTest, DefaultConfigurationCreatesStandardGrid) {
  const auto npu = arch::npu::NPU(base::Config{});

  EXPECT_EQ(npu.kColumnsInNPU, 8);
  EXPECT_EQ(npu.kRowsInNPU, 6);
  EXPECT_EQ(npu.kRowsOfComputeTilesInNPU, 4);
  EXPECT_EQ(npu.kRowsOfMemoryTilesInNPU, 1);
  EXPECT_EQ(npu.kRowsOfShimTilesInNPU, 1);
}

TEST(NpuConstructionTest, DefaultConfigurationInitializesAllTilePositions) {
  const auto npu = arch::npu::NPU(base::Config{});

  for (int row = 0; row < npu.kRowsInNPU; ++row) {
    for (int col = 0; col < npu.kColumnsInNPU; ++col) {
      EXPECT_NO_THROW(npu.getPhysicalCore({row, col}));
    }
  }
}

TEST(NpuConstructionTest, DefaultConfigurationCreatesExpectedRrGraphSize) {
  const auto npu = arch::npu::NPU(base::Config{});
  const auto rr_graph = npu.getRRGraph();

  const int expected_node_count = npu.kColumnsInNPU * npu.kRowsInNPU * 3;
  const auto nodes = rr_graph.getNodePtrs();
  EXPECT_EQ(std::ranges::distance(nodes), expected_node_count);
}

TEST(NpuConstructionTest, CustomConfigurationIsApplied) {
  const auto npu = createSmallNPU();

  EXPECT_EQ(npu.kColumnsInNPU, 4);
  EXPECT_EQ(npu.kRowsInNPU, 5);
  EXPECT_EQ(npu.kRowsOfComputeTilesInNPU, 3);
  EXPECT_EQ(npu.kRowsOfMemoryTilesInNPU, 1);
  EXPECT_EQ(npu.kRowsOfShimTilesInNPU, 1);
}

TEST(NpuConstructionTest, CustomConfigurationInitializesAllTilePositions) {
  const auto npu = createSmallNPU();

  for (int row = 0; row < npu.kRowsInNPU; ++row) {
    for (int col = 0; col < npu.kColumnsInNPU; ++col) {
      EXPECT_NO_THROW(npu.getPhysicalCore({row, col}));
    }
  }
}

TEST(NpuConstructionTest, CustomConfigurationCreatesExpectedRrGraphSize) {
  const auto npu = createSmallNPU();
  const auto rr_graph = npu.getRRGraph();

  const int expected_node_count = npu.kColumnsInNPU * npu.kRowsInNPU * 3;
  const auto nodes = rr_graph.getNodePtrs();
  EXPECT_EQ(std::ranges::distance(nodes), expected_node_count);
}

TEST(NpuConstructionValidationTest, RowCountMismatchThrows) {
  base::Config invalid_cfg;
  invalid_cfg.set("num_rows", 6);
  invalid_cfg.set("num_compute_rows", 3);
  invalid_cfg.set("num_memory_rows", 1);
  invalid_cfg.set("num_shim_rows", 1);

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)arch::npu::NPU(invalid_cfg); },
      {"row configuration mismatch", "must be equal to the sum"});
}

TEST(NpuNetlistParsingTest, AutoPositioningRespectsTileTypeBounds) {
  constexpr const char *toml = R"(
    [edge]
    e1 = [ "comp1->comp2" ]
    e2 = [ "mem1->comp1" ]
    e3 = [ "shim1->mem1" ]

    [linking]
    link1 = "e3->e2"
    link2 = "e2->e1"
  )";

  const auto npu = arch::npu::NPU(base::Config{});
  const auto reader = base::PnRNetlistReader::fromTOML(toml);
  const auto result = npu.parseNetlist(reader);

  EXPECT_GE(result.placement.size(), 3U);
}

TEST(NpuNetlistParsingTest, MultipleMemoryNodesCanSharePosition) {
  constexpr const char *toml = R"(
    [node]
    mem1 = "1,0"
    mem2 = "1,0"

    [edge]
    e1 = [ "mem1->mem2" ]
  )";

  const auto npu = createSmallNPU();
  const auto reader = base::PnRNetlistReader::fromTOML(toml);
  EXPECT_NO_THROW(npu.parseNetlist(reader));
}

TEST(NpuNetlistParsingTest, MultipleShimNodesCanSharePosition) {
  constexpr const char *toml = R"(
    [node]
    shim1 = "0,0"
    shim2 = "0,0"

    [edge]
    e1 = [ "shim1->shim2" ]
  )";

  const auto npu = createSmallNPU();
  const auto reader = base::PnRNetlistReader::fromTOML(toml);
  EXPECT_NO_THROW(npu.parseNetlist(reader));
}

TEST(NpuNetlistParsingTest, ComputeNodesCannotSharePosition) {
  constexpr const char *toml = R"(
    [node]
    comp1 = "2,0"
    comp2 = "2,0"

    [edge]
    e1 = [ "comp1->comp2" ]
  )";

  const auto npu = createSmallNPU();
  const auto reader = base::PnRNetlistReader::fromTOML(toml);
  EXPECT_THROW(npu.parseNetlist(reader), std::runtime_error);
}

TEST(NpuNetlistParsingValidationTest, NegativeRowThrows) {
  constexpr const char *toml = R"(
    [node]
    comp1 = "-1,1"

    [edge]
    e1 = [ "comp1->comp1" ]
  )";

  const auto npu = createSmallNPU();
  const auto reader = base::PnRNetlistReader::fromTOML(toml);
  EXPECT_THROW(npu.parseNetlist(reader), std::runtime_error);
}

TEST(NpuNetlistParsingValidationTest, OutOfBoundsRowThrows) {
  constexpr const char *toml = R"(
    [node]
    comp1 = "10,0"

    [edge]
    e1 = [ "comp1->comp1" ]
  )";

  const auto npu = createSmallNPU();
  const auto reader = base::PnRNetlistReader::fromTOML(toml);
  EXPECT_THROW(npu.parseNetlist(reader), std::runtime_error);
}

TEST(NpuNetlistParsingValidationTest, OutOfBoundsColumnThrows) {
  constexpr const char *toml = R"(
    [node]
    comp1 = "2,10"

    [edge]
    e1 = [ "comp1->comp1" ]
  )";

  const auto npu = createSmallNPU();
  const auto reader = base::PnRNetlistReader::fromTOML(toml);
  EXPECT_THROW(npu.parseNetlist(reader), std::runtime_error);
}

TEST(NpuPlacementLegalityTest, LegalPlacementIsAccepted) {
  const auto npu = createSmallNPU();
  base::Placement placement;

  const base::LogicalCore comp(base::TrafficFlowEndpoint("comp1"));
  const base::LogicalCore mem(base::TrafficFlowEndpoint("mem1"));
  const base::LogicalCore shim(base::TrafficFlowEndpoint("shim1"));

  placement.add(comp, npu.getPhysicalCore({2, 0}));
  placement.add(mem, npu.getPhysicalCore({1, 0}));
  placement.add(shim, npu.getPhysicalCore({0, 0}));

  EXPECT_TRUE(npu.isLegalPlacement(placement));
}

TEST(NpuPlacementLegalityTest, TwoComputeNodesOnSameCoreIsIllegal) {
  const auto npu = createSmallNPU();
  base::Placement placement;

  const base::LogicalCore l1(base::TrafficFlowEndpoint("comp1"));
  const base::LogicalCore l2(base::TrafficFlowEndpoint("comp2"));
  const auto physical_core = npu.getPhysicalCore({2, 0});

  placement.add(l1, physical_core);
  placement.add(l2, physical_core);

  EXPECT_NO_THROW(npu.isLegalPlacement(placement));
  EXPECT_FALSE(npu.isLegalPlacement(placement));
}

TEST(NpuPhysicalCoreTest, GetPhysicalCoreReturnsValidCoreForPosition) {
  const auto npu = createSmallNPU();
  const auto physical_core = npu.getPhysicalCore({2, 0});

  EXPECT_NO_THROW(physical_core.getMemoryCapacity());
}

TEST(NpuPhysicalCoreTest, GetPhysicalCoreThrowsForInvalidPosition) {
  const auto npu = createSmallNPU();

  EXPECT_THROW(npu.getPhysicalCore({10, 0}), std::runtime_error);
}

TEST(NpuPhysicalCoreTest, GetPhysicalCoresReturnsAllCores) {
  const auto npu = createSmallNPU();
  const auto cores = npu.getPhysicalCores();

  EXPECT_EQ(cores.size(), npu.kColumnsInNPU * npu.kRowsInNPU);
}
