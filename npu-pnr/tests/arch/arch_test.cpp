#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "arch/arch.hpp"
#include "arch/registry.hpp"
#include "base/abstraction.hpp"
#include "base/common.hpp"
#include "base/engine.hpp"
#include "base/rr_graph.hpp"
#include "gtest_helpers.hpp"

namespace {

class MockArch {
private:
  base::Config cfg_;

public:
  explicit MockArch(const base::Config &cfg) : cfg_(cfg) {}

  base::GridDimension getDimensions() const { return {1, 1}; }

  base::RRGraph getRRGraph() const { return base::RRGraph{1, 1}; }

  base::PnRPlacedNetlist parseNetlist(base::PnRNetlistReader) const {
    return base::PnRPlacedNetlist{};
  }

  base::PnRNetlistWriter dumpNetlist(base::PnRPlacedNetlist) const {
    return base::PnRNetlistWriter{};
  }

  base::PnRNetlistWriter dumpNetlist(base::PnRPlacedNetlist,
                                     base::RoutingState) const {
    return base::PnRNetlistWriter{};
  }

  base::PhysicalCore getPhysicalCore(base::GridPosition) const {
    return base::PhysicalCore{base::RRNode{""}, base::RRNode{""}, 0, 0};
  }

  base::PhysicalCoreSet getPhysicalCores() const {
    return base::PhysicalCoreSet{};
  }

  base::LogicalCoreCompatibilitySet
  getLogicalCoreCompatibilitySet(const base::Placement &) const {
    return base::LogicalCoreCompatibilitySet{};
  }

  base::LogicalToPhysicalCoreAvailablityTable
  getLogicalToPhysicalCoreAvailablityTable(const base::Placement &) const {
    return base::LogicalToPhysicalCoreAvailablityTable{};
  }

  bool isLegalPlacement(const base::Placement &) const { return true; }

  base::Placement getLegalPlacementWithRandomSampling(
      const base::Placement &placement, utils::RandomSeed) const {
    return placement;
  }

  virtual void generateInitialPlacement(const base::Placement &,
                                        base::Placement &,
                                        const base::RoutingNetList &,
                                        utils::RandomSeed) const {}

  void generateLegalPlacementWithBoundedRandomSwap(
      const base::Placement &, base::Placement &,
      std::vector<base::LogicalCore> &, utils::RandomSeed, int) const {}

  base::RegionPlacement extendPlacementToRegionPlacement(
      const base::Placement &, const std::vector<base::GridPositionOffset> &,
      const base::LogicalCoreSet &) const {
    return base::RegionPlacement{};
  }

  std::string visualizePlacement(const base::PnRNetlistReader &) const {
    return "digraph {}";
  }

  std::string visualizeRouting(const base::PnRNetlistReader &) const {
    return "digraph {}";
  }
};

} // namespace

TEST(ArchTypeErasureTest, CreateArchUsingStaticFactory) {
  base::Config cfg;
  auto arch = arch::Arch::create<MockArch>(cfg);

  EXPECT_NO_THROW(arch.getRRGraph());
}

TEST(ArchTypeErasureTest, ArchCanBeMoved) {
  base::Config cfg;
  auto arch1 = arch::Arch::create<MockArch>(cfg);
  auto arch2 = std::move(arch1);

  EXPECT_NO_THROW(arch2.getRRGraph());
  EXPECT_NO_THROW(arch2.getPhysicalCores());

  base::Placement placement;
  EXPECT_NO_THROW(arch2.isLegalPlacement(placement));
}

TEST(ArchTypeErasureTest, CanChainMoveOperations) {
  base::Config cfg;
  auto arch1 = arch::Arch::create<MockArch>(cfg);
  auto arch2 = std::move(arch1);
  auto arch3 = std::move(arch2);

  EXPECT_NO_THROW(arch3.getRRGraph());
}

TEST(ArchTypeErasureTest, ArchConstCorrectness) {
  base::Config cfg;
  const auto arch = arch::Arch::create<MockArch>(cfg);

  EXPECT_NO_THROW(arch.getRRGraph());

  base::Placement placement;
  EXPECT_NO_THROW(arch.isLegalPlacement(placement));
}

TEST(ArchTypeErasureTest, AllInterfaceMethodsAreCallable) {
  base::Config cfg;
  auto arch = arch::Arch::create<MockArch>(cfg);

  EXPECT_NO_THROW(arch.getRRGraph());
  EXPECT_NO_THROW(arch.getPhysicalCores());

  base::Placement placement;
  EXPECT_NO_THROW(arch.getLogicalCoreCompatibilitySet(placement));
  EXPECT_NO_THROW(arch.getLogicalToPhysicalCoreAvailablityTable(placement));
  EXPECT_TRUE(arch.isLegalPlacement(placement));

  std::vector<base::GridPositionOffset> offsets;
  base::LogicalCoreSet cores;
  EXPECT_NO_THROW(
      arch.extendPlacementToRegionPlacement(placement, offsets, cores));

  const auto reader = base::PnRNetlistReader::fromTOML("");
  EXPECT_FALSE(arch.visualizePlacement(reader).empty());
  EXPECT_FALSE(arch.visualizeRouting(reader).empty());
}

TEST(CreateArchRegistryTest, CreateNpuArchitectureWithValidConfig) {
  base::Config cfg;
  cfg.set("type", std::string("npu"));

  EXPECT_NO_THROW(arch::createArch(cfg));
}

TEST(CreateArchRegistryTest, UnsupportedArchitectureTypeThrows) {
  base::Config cfg;
  cfg.set("type", std::string("tpu"));

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)arch::createArch(cfg); }, {"tpu", "not implemented"});
}

TEST(CreateArchRegistryTest, MissingTypeFieldThrows) {
  base::Config cfg;

  EXPECT_THROW(arch::createArch(cfg), std::runtime_error);
}
