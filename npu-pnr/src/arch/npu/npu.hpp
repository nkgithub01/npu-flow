#ifndef _NPU_HPP_
#define _NPU_HPP_

/* NPU v2 device is present in Ryzen AI: Strix, Strix Halo and Krackan Point
   SOCs. It has 8 Columns and 6 Rows:

  5 CCCCCCCC
  4 CCCCCCCC
  3 CCCCCCCC
  2 CCCCCCCC
  1 MMMMMMMM
  0 DDDDDDDD
    01234567

  C represents a 'compute tile' including a processor, stream switch, local
  memories and DMAs; M represents a 'memory tile' including a stream switch,
  larger local memories and DMAs; D represents a 'Shim DMA tile' including a
  Shim DMA and stream switch.

  Reference: https://xilinx.github.io/mlir-aie/Devices
*/

#include <memory>

#include "arch/npu/route_tree.hpp"
#include "arch/npu/tile.hpp"
#include "base/abstraction.hpp"
#include "base/common.hpp"
#include "base/engine.hpp"
#include "base/placement.hpp"
#include "base/routing.hpp"
#include "base/rr_graph.hpp"
#include "utils/misc.hpp"

namespace arch {
namespace npu {

using npu_tiles::Tile;
using npu_tiles::TileType;

using base::RREdgeCapacity;
using base::RREdgeType;
using base::RRNode;

class NPU {
public:
  using Direction = npu_tiles::Direction;
  using TilePtr = std::shared_ptr<Tile>;
  using TilePos = base::GridPosition;
  using TileOffset = base::GridPositionOffset;

  const int kColumnsInNPU;
  const int kRowsInNPU;
  const int kRowsOfComputeTilesInNPU;
  const int kRowsOfMemoryTilesInNPU;
  const int kRowsOfShimTilesInNPU;

private:
  std::vector<std::vector<TilePtr>> tiles_;

  utils::Lookup<RRNode, TilePtr> rr_node_to_tile_map_;
  utils::Lookup<TileType, std::vector<TilePtr>> tile_type_to_tile_map_;
  utils::Lookup<TilePtr, base::PhysicalCore> tile_to_p_core_map_;

  base::RRGraph rr_graph_;

  inline std::optional<TilePtr> getTile(TilePos pos) const {
    const auto [row_y, col_x] = pos.getRowYColXPair();
    if ((col_x >= 0 && col_x < kColumnsInNPU) &&
        (row_y >= 0 && row_y < kRowsInNPU)) {
      return tiles_[row_y][col_x];
    }
    return std::nullopt;
  }

  inline std::optional<TilePtr> getTile(int row_y, int col_x) const {
    return getTile(TilePos{row_y, col_x});
  }

  // Helper functions
  // 1. Build NPU tiles
  void buildTiles();
  void registerNeighborTiles();
  // 2. Build RRGraph
  void buildRRGraph();
  void appendIntraTileInterconnectsToRRGraph();
  void appendNeighborSharingInterconnectsToRRGraph();
  void appendSwitchBoxNetworkToRRGraph();
  void populateRRNodeLookup();
  void setPhysicalCoreAndMemoryCapacities();
  // 3. Dump routing information
  SupportedRouteTree
  parseRouteTree(const base::RouteTree &, const RouteTreeResourceAllocator &,
                 const base::LogicalToPhysicalCoreMapping &) const;

public:
  NPU(const base::Config &cfg)
      : kColumnsInNPU(cfg.getOrDefault<int>("num_cols", 8)),
        kRowsInNPU(cfg.getOrDefault<int>("num_rows", 6)),
        kRowsOfComputeTilesInNPU(cfg.getOrDefault<int>("num_compute_rows", 4)),
        kRowsOfMemoryTilesInNPU(cfg.getOrDefault<int>("num_memory_rows", 1)),
        kRowsOfShimTilesInNPU(cfg.getOrDefault<int>("num_shim_rows", 1)),
        rr_graph_(kRowsInNPU, kColumnsInNPU)
  {
    if (kRowsInNPU != (kRowsOfComputeTilesInNPU + kRowsOfMemoryTilesInNPU +
                       kRowsOfShimTilesInNPU)) {
      throw std::runtime_error(
          "NPU row configuration mismatch in constructor: kRowsInNPU must be "
          "equal to the sum of rows of compute, memory and shim tiles");
    }
    tiles_.resize(kRowsInNPU, std::vector<TilePtr>(kColumnsInNPU, nullptr));
    buildTiles();
    registerNeighborTiles();
    buildRRGraph();
  }

  base::GridDimension getDimensions() const {
    return base::GridDimension(kRowsInNPU, kColumnsInNPU);
  }

  // Required by netlist reader and writer
  base::PnRPlacedNetlist parseNetlist(base::PnRNetlistReader rd) const;
  base::PnRNetlistWriter dumpNetlist(base::PnRPlacedNetlist p) const;
  base::PnRNetlistWriter dumpNetlist(base::PnRPlacedNetlist p,
                                     base::RoutingState r) const;

  base::RRGraph getRRGraph() const { return rr_graph_; }
  base::PhysicalCore getPhysicalCore(TilePos pos) const;
  base::PhysicalCoreSet getPhysicalCores() const;

  // Required by SA and MILP placer
  base::LogicalCoreCompatibilitySet
  getLogicalCoreCompatibilitySet(const base::Placement &) const;
  base::LogicalToPhysicalCoreAvailablityTable
  getLogicalToPhysicalCoreAvailablityTable(const base::Placement &) const;
  bool isLegalPlacement(const base::Placement &) const;
  base::Placement getLegalPlacementWithRandomSampling(const base::Placement &,
                                                      utils::RandomSeed) const;
  bool getUnplacedNeighbourLogicalCores(
    base::LogicalCore &l_core,
    const utils::Set<base::LogicalCore> &neighbor_l_cores,
    const utils::Set<base::LogicalCore> &unplaced_l_cores)
    const;
  void generateInitialPlacement(
    const base::Placement &old_placement, base::Placement &new_placement,
    const base::RoutingNetList &netlist, utils::RandomSeed seed)
    const;
  void generateLegalPlacementWithBoundedRandomSwap(
    const base::Placement &old_placement, base::Placement &new_placement,
    std::vector<base::LogicalCore> &moved_l_cores,
    utils::RandomSeed seed, const int swap_bound)
    const;
  // Required by LSMO placer
  base::RegionPlacement extendPlacementToRegionPlacement(
    const base::Placement &, const std::vector<base::GridPositionOffset> &,
    const base::LogicalCoreSet &) const;

  // Visualizations
  std::string visualizePlacement(const base::PnRNetlistReader &) const;
  std::string visualizeRouting(const base::PnRNetlistReader &) const;
};

} // namespace npu
} // namespace arch

#endif
