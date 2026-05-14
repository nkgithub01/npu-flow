#include "arch/npu/npu.hpp"

#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>
#include <iostream>

#include "arch/npu/tile.hpp"
#include "base/abstraction.hpp"
#include "base/common.hpp"
#include "base/placement.hpp"
#include "base/rr_graph.hpp"
#include "utils/misc.hpp"

namespace arch {
namespace npu {

void NPU::buildTiles() {
  for (int j = 0; j < kColumnsInNPU; ++j) {
    // Shim Tiles (first rows)
    for (int i = 0; i < kRowsOfShimTilesInNPU; ++i) {
      tiles_[i][j] = npu_tiles::create<npu_tiles::ShimTile>({i, j});
    }
    // Memory Tiles (middle rows)
    for (int offset = 0; offset < kRowsOfMemoryTilesInNPU; ++offset) {
      const int i = kRowsOfShimTilesInNPU + offset;
      tiles_[i][j] = npu_tiles::create<npu_tiles::MemoryTile>({i, j});
    }
    // Compute Tiles (last rows)
    for (int offset = 0; offset < kRowsOfComputeTilesInNPU; ++offset) {
      const int i = kRowsOfShimTilesInNPU + kRowsOfMemoryTilesInNPU + offset;
      tiles_[i][j] = npu_tiles::create<npu_tiles::ComputeTile>({i, j});
    }
  }
}

void NPU::registerNeighborTiles() {
  for (int i = 0; i < kRowsInNPU; ++i) {
    for (int j = 0; j < kColumnsInNPU; ++j) {
      if (i > 0) {
        tiles_[i][j]->registerNeighborTile(tiles_[i - 1][j], Direction::South);
      }
      if (i < kRowsInNPU - 1) {
        tiles_[i][j]->registerNeighborTile(tiles_[i + 1][j], Direction::North);
      }
      if (j > 0) {
        tiles_[i][j]->registerNeighborTile(tiles_[i][j - 1], Direction::West);
      }
      if (j < kColumnsInNPU - 1) {
        tiles_[i][j]->registerNeighborTile(tiles_[i][j + 1], Direction::East);
      }
    }
  }
}

constexpr float kIntraTileKernelLinkingCost = 0.01f;
constexpr float kNeighborSharingCost = 0.1f;
constexpr float kCircuitSwitchingCost = 1.0f;
constexpr float kPacketSwitchingCost = 1.5f;

void NPU::buildRRGraph() {
  appendIntraTileInterconnectsToRRGraph();
  appendNeighborSharingInterconnectsToRRGraph();
  appendSwitchBoxNetworkToRRGraph();
  populateRRNodeLookup();               // populate `rr_node_to_tile_map_`
  setPhysicalCoreAndMemoryCapacities(); // populate `tile_to_p_core_map_`
}

void NPU::appendIntraTileInterconnectsToRRGraph() {
  for (int i = 0; i < kRowsInNPU; ++i) {
    for (int j = 0; j < kColumnsInNPU; ++j) {
      std::shared_ptr<Tile> tile = tiles_[i][j];

      if (tile->getTileType() != TileType::Compute) {
        continue; // Only compute tiles have intra-tile kernel linking
                  // connections (only compute tiles can run kernels)
      }

      RRNode tile_src_ep(tile->getSourceEndpoint());
      RRNode tile_sink_ep(tile->getSinkEndpoint());

      rr_graph_.addNode(tile_src_ep, i, j, false);
      rr_graph_.addNode(tile_sink_ep, i, j, false);

      rr_graph_.addEdge(
          tile_src_ep, tile_sink_ep, RREdgeType::IntraTileKernelLinking,
          // TODO: change all int to size_t in the npu tiles
          (size_t)tile->getNumChannelsBetweenInterTileKernelLinking(),
          RREdgeCapacity::ConsolidationBehavior::NonConsolidated,
          kIntraTileKernelLinkingCost);
    }
  }
}

void NPU::appendNeighborSharingInterconnectsToRRGraph() {
  for (int i = 0; i < kRowsInNPU; ++i) {
    for (int j = 0; j < kColumnsInNPU; ++j) {
      std::shared_ptr<Tile> tile = tiles_[i][j];
      // Only add edges starting from the source endpoint of `tile` to the sink
      // endpoints of its neighbor tiles.
      for (const Direction &direction : {Direction::North, Direction::South,
                                         Direction::East, Direction::West}) {
        if (const auto &neighbor = tile->getNeighborTile(direction);
            neighbor.has_value() &&
            neighbor->hasNeighborSharingInterconnect()) {
          std::shared_ptr<Tile> neighbor_tile = neighbor->getTile();

          RRNode tile_src_ep(tile->getSourceEndpoint());
          RRNode nbr_sink_ep(neighbor_tile->getSinkEndpoint());

          rr_graph_.addNode(tile_src_ep, i, j, false);
          rr_graph_.addNode(nbr_sink_ep, neighbor_tile->getRowY(),
                            neighbor_tile->getColX(), false);

          // To form a net from `tile` to `neighbor_tile` using neighbor-sharing
          // interconnect, we need to add two directed edges:
          // 1. Memory shared on source side, iif `neighbor_tile` can access
          //    local memory in `tile`
          // 2. Memory shared on sink side, iif `tile` can access local memory
          //    in `neighbor_tile`
          if (neighbor->getTile()
                  ->getNeighborTile(npu_tiles::opposite(direction))
                  ->isMemoryAccessible()) {
            rr_graph_.addEdge(
                tile_src_ep, nbr_sink_ep, RREdgeType::NeighborSharingOnSource,
                (size_t)tile->getNumChannelsBetweenNeighborSharing(),
                RREdgeCapacity::ConsolidationBehavior::NonConsolidated,
                kNeighborSharingCost);
          }
          if (neighbor->isMemoryAccessible()) {
            rr_graph_.addEdge(
                tile_src_ep, nbr_sink_ep, RREdgeType::NeighborSharingOnSink,
                (size_t)tile->getNumChannelsBetweenNeighborSharing(),
                RREdgeCapacity::ConsolidationBehavior::NonConsolidated,
                kNeighborSharingCost);
          }
        }
      }
    }
  }
}

void NPU::appendSwitchBoxNetworkToRRGraph() {
  enum class EdgeType {
    DMAToSB,
    SBToDMA,
    SBToSB,
  };
  auto add_edges = [&](EdgeType type, RRNode src, RRNode sink, TilePtr src_tile,
                       TilePtr sink_tile, int num_channels) -> void {
    if (num_channels <= 0) {
      return;
    }
    size_t capacity = static_cast<size_t>(num_channels);
    if (src_tile->getTileType() == TileType::Compute &&
        sink_tile->getTileType() == TileType::Compute) {
      auto cct_edge = rr_graph_.addEdge(
          src, sink, RREdgeType::CircuitSwitching, capacity,
          RREdgeCapacity::ConsolidationBehavior::NonConsolidated,
          kCircuitSwitchingCost);
      auto pkt_edge = rr_graph_.addEdge(
          src, sink, RREdgeType::PacketSwitching, capacity,
          (type == EdgeType::SBToDMA /* packet switching cannot consolidate
                                     when sending through SB to DMA edges,
                                     due to the limitation in MLIR routing
                                     pass implementation */
               ? RREdgeCapacity::ConsolidationBehavior::NonConsolidated
               : RREdgeCapacity::ConsolidationBehavior::Consolidated),
          kPacketSwitchingCost);
      rr_graph_.setEdgeSharing({cct_edge, pkt_edge}, capacity);
    } else {
      rr_graph_.addEdge(src, sink, RREdgeType::CircuitSwitching, capacity,
                        RREdgeCapacity::ConsolidationBehavior::NonConsolidated,
                        kCircuitSwitchingCost);
    }
  };

  // Build SwitchBox network
  for (int i = 0; i < kRowsInNPU; ++i) {
    for (int j = 0; j < kColumnsInNPU; ++j) {
      std::shared_ptr<Tile> tile = tiles_[i][j];

      RRNode tile_sb(tile->getSwitchBox());
      RRNode tile_src_ep(tile->getSourceEndpoint());
      RRNode tile_sink_ep(tile->getSinkEndpoint());

      rr_graph_.addNode(tile_sb, i, j, true);
      rr_graph_.addNode(tile_src_ep, i, j, false);
      rr_graph_.addNode(tile_sink_ep, i, j, false);

      // Intra-tile SwitchBox connections
      //// 1. Tile DMA ==> SwitchBox
      add_edges(EdgeType::DMAToSB, tile_src_ep, tile_sb, tile, tile,
                tile->getNumChannelsBetweenTileDMAandSwitchBox());
      //// 2. SwitchBox ==> Tile DMA
      //// Note: this edge uses exclusive capacity since multiple packet streams
      //// cannot share the same DMA channel when sending data into the tile.
      add_edges(EdgeType::SBToDMA, tile_sb, tile_sink_ep, tile, tile,
                tile->getNumChannelsBetweenTileDMAandSwitchBox());

      // Inter-tile SwitchBox connections
      //// 1. SwitchBox ==> North-neighbor SwitchBox
      if (auto north_tile = getTile(i + 1, j); north_tile.has_value()) {
        RRNode north_sb(north_tile.value()->getSwitchBox());
        rr_graph_.addNode(north_sb, i + 1, j,
                          true); // duplicate addNodes are ok
        add_edges(EdgeType::SBToSB, tile_sb, north_sb, tile, north_tile.value(),
                  tile->getNumChannelsSwitchBoxToNorth());
      }
      //// 2. SwitchBox ==> South-neighbor SwitchBox
      if (auto south_tile = getTile(i - 1, j); south_tile.has_value()) {
        RRNode south_sb(south_tile.value()->getSwitchBox());
        rr_graph_.addNode(south_sb, i - 1, j, true);
        add_edges(EdgeType::SBToSB, tile_sb, south_sb, tile, south_tile.value(),
                  tile->getNumChannelsSwitchBoxToSouth());
      }
      //// 3. SwitchBox ==> East-neighbor SwitchBox
      if (auto east_tile = getTile(i, j + 1); east_tile.has_value()) {
        RRNode east_sb(east_tile.value()->getSwitchBox());
        rr_graph_.addNode(east_sb, i, j + 1, true);
        add_edges(EdgeType::SBToSB, tile_sb, east_sb, tile, east_tile.value(),
                  tile->getNumChannelsSwitchBoxToEast());
      }
      //// 4. SwitchBox ==> West-neighbor SwitchBox
      if (auto west_tile = getTile(i, j - 1); west_tile.has_value()) {
        RRNode west_sb(west_tile.value()->getSwitchBox());
        rr_graph_.addNode(west_sb, i, j - 1, true);
        add_edges(EdgeType::SBToSB, tile_sb, west_sb, tile, west_tile.value(),
                  tile->getNumChannelsSwitchBoxToWest());
      }
    }
  }
}

// NOTE: the order of tile_type_to_tile_map_ population must be consistent with the order of tile types in the NPU architecture,
// since it will be used in the SA placer to determine the boundary of logical core swapping.
void NPU::populateRRNodeLookup() {
  rr_node_to_tile_map_.clear();
  for (int i = 0; i < kRowsInNPU; ++i) {
    for (int j = 0; j < kColumnsInNPU; ++j) {
      std::shared_ptr<Tile> tile = tiles_[i][j];
      RRNode src_ep(tile->getSourceEndpoint());
      RRNode sink_ep(tile->getSinkEndpoint());

      rr_node_to_tile_map_[src_ep] = tile;
      rr_node_to_tile_map_[sink_ep] = tile;
      rr_node_to_tile_map_[RRNode(tile->getSwitchBox())] = tile;

      tile_type_to_tile_map_[tile->getTileType()].push_back(tile);
    }
  }

  if (rr_node_to_tile_map_.size() != (kColumnsInNPU * kRowsInNPU * 3)) {
    throw std::runtime_error("rr_node_to_tile_map_ size mismatch");
  }

  if ((tile_type_to_tile_map_.size() != 3) ||
      (tile_type_to_tile_map_.at(TileType::Compute).size() !=
       kRowsOfComputeTilesInNPU * kColumnsInNPU) ||
      (tile_type_to_tile_map_.at(TileType::Memory).size() !=
       kRowsOfMemoryTilesInNPU * kColumnsInNPU) ||
      (tile_type_to_tile_map_.at(TileType::Shim).size() !=
       kRowsOfShimTilesInNPU * kColumnsInNPU)) {
    throw std::runtime_error("tile_type_to_tile_map_ size mismatch");
  }
}

void NPU::setPhysicalCoreAndMemoryCapacities() {
  for (int i = 0; i < kRowsInNPU; ++i) {
    for (int j = 0; j < kColumnsInNPU; ++j) {
      std::shared_ptr<Tile> tile = tiles_[i][j];
      RRNode src_ep(tile->getSourceEndpoint());
      RRNode sink_ep(tile->getSinkEndpoint());
      tile_to_p_core_map_.insert_or_assign(
          tile, base::PhysicalCore(src_ep, sink_ep, tile->getMemoryCapacity(),
                                   tile->getLockCapacity()));
    }
  }
}

base::PhysicalCoreSet NPU::getPhysicalCores() const {
  base::PhysicalCoreSet p_cores;
  for (const auto p_core : std::views::values(tile_to_p_core_map_)) {
    p_cores.insert(p_core);
  }
  return p_cores;
}

base::LogicalToPhysicalCoreAvailablityTable
NPU::getLogicalToPhysicalCoreAvailablityTable(
    const base::Placement &placement) const {
  utils::Lookup<TileType, base::PhysicalCoreSet> tile_type_to_p_core_set;
  for (const auto &[tile_type, tile_list] : tile_type_to_tile_map_) {
    base::PhysicalCoreSet p_core_set;
    for (const auto &tile : tile_list) {
      p_core_set.insert(tile_to_p_core_map_.at(tile));
    }
    tile_type_to_p_core_set.insert_or_assign(tile_type, p_core_set);
  }

  base::LogicalToPhysicalCoreAvailablityTable table;
  for (const auto &[l_core, p_core] : placement.data()) {
    TilePtr tile = rr_node_to_tile_map_.at(p_core.getSource());
    table.add(l_core, tile_type_to_p_core_set.at(tile->getTileType()));
  }

  return table;
}

base::LogicalCoreCompatibilitySet
NPU::getLogicalCoreCompatibilitySet(const base::Placement &placement) const {
  utils::Lookup<TileType, base::LogicalCoreSet> tile_type_to_subsets;
  for (const auto &[l_core, p_core] : placement.data()) {
    const TilePtr tile = rr_node_to_tile_map_.at(p_core.getSource());
    tile_type_to_subsets[tile->getTileType()].insert(l_core);
  }

  std::vector<base::LogicalCoreSet> subsets;
  if (tile_type_to_subsets.contains(TileType::Shim)) {
    subsets.push_back(tile_type_to_subsets.at(TileType::Shim));
  }
  if (tile_type_to_subsets.contains(TileType::Memory)) {
    subsets.push_back(tile_type_to_subsets.at(TileType::Memory));
  }
  if (tile_type_to_subsets.contains(TileType::Compute)) {
    for (const base::LogicalCore &l_core :
         tile_type_to_subsets.at(TileType::Compute)) {
      subsets.push_back({l_core});
    }
  }
  return base::LogicalCoreCompatibilitySet{subsets};
}

base::PhysicalCore NPU::getPhysicalCore(TilePos pos) const {
  if (auto tile = getTile(pos); tile.has_value()) {
    return tile_to_p_core_map_.at(tile.value());
  }
  throw std::runtime_error(std::format("No tile found at row {}, column {}",
                                       pos.getRowY(), pos.getColX()));
}

bool NPU::isLegalPlacement(const base::Placement &placement) const {
  // Check if no two compute tiles (or logical cores) are mapped to the same
  // physical core, since compute tiles cannot be packed together.
  // TODO: considering merging this function with compatibility set concept
  base::PhysicalCoreSet occupied_p_cores;
  for (const auto &[l_core, p_core] : placement.data()) {
    TilePtr tile = rr_node_to_tile_map_.at(p_core.getSource());
    if (tile->getTileType() == TileType::Compute) {
      if (occupied_p_cores.contains(p_core)) {
        return false;
      } else {
        occupied_p_cores.insert(p_core);
      }
    }
  }
  return true;
}

// TODO: consider passing avail_table as an argument to avoid redundant
// computation when this function is called multiple times in the placer
base::Placement
NPU::getLegalPlacementWithRandomSampling(const base::Placement &p,
                                         utils::RandomSeed s) const {
  utils::Lookup<TileType, std::vector<base::LogicalCore>>
      tile_type_to_l_core_vec;
  for (const auto &[l_core, p_core] : p.data()) {
    TileType type = rr_node_to_tile_map_.at(p_core.getSource())->getTileType();
    if (!tile_type_to_l_core_vec.contains(type)) {
      tile_type_to_l_core_vec.insert_or_assign(
          type, std::vector<base::LogicalCore>{});
    }
    tile_type_to_l_core_vec.at(type).push_back(l_core);
  }
  auto avail_p_core_table = getLogicalToPhysicalCoreAvailablityTable(p);
  base::Placement new_placement;

  for (const TileType type :
       {TileType::Shim, TileType::Memory, TileType::Compute}) {
    if (tile_type_to_l_core_vec.contains(type)) {
      base::PhysicalCoreSet p_core_set =
          avail_p_core_table.at(tile_type_to_l_core_vec.at(type).front());
      std::vector<base::PhysicalCore> p_core_vec{p_core_set.begin(),
                                                 p_core_set.end()};
      int num_l_cores = tile_type_to_l_core_vec.at(type).size();
      int num_p_cores = p_core_set.size();
      if (type == TileType::Compute) {
        if (num_l_cores > num_p_cores) {
          throw std::runtime_error(
              "Not enough physical compute cores to map all logical cores");
        }
      } else {
        // Allow packing for non-compute tiles
        for (int i = num_l_cores; i > num_p_cores; i -= num_p_cores) {
          p_core_vec.insert(p_core_vec.end(), p_core_set.begin(),
                            p_core_set.end());
        }
      }

      std::vector<base::PhysicalCore> selected_p_cores;
      std::sample(p_core_vec.begin(), p_core_vec.end(),
                  std::back_inserter(selected_p_cores), num_l_cores,
                  std::mt19937{s});
      for (int i = 0; i < num_l_cores; ++i) {
        new_placement.add(tile_type_to_l_core_vec.at(type).at(i),
                          selected_p_cores.at(i));
      }
    }
  }

  return new_placement;
}

bool NPU::getUnplacedNeighbourLogicalCores(
  base::LogicalCore &l_core,
  const utils::Set<base::LogicalCore> &neighbor_l_cores,
  const utils::Set<base::LogicalCore> &unplaced_l_cores
) const {
  for (const auto &neighbor_l_core : neighbor_l_cores) {
    if (unplaced_l_cores.contains(neighbor_l_core)) {
      l_core = neighbor_l_core;
      return true;
    }
  }
  return false;
}

// Generate the initial placement.
void NPU::generateInitialPlacement(
  const base::Placement &old_placement, base::Placement &new_placement, const base::RoutingNetList &netlist, utils::RandomSeed seed)
const {
  // Get a list of logical cores in the old placement, grouped by tile type.
  utils::Set<base::LogicalCore> l_cores;
  utils::Set<base::LogicalCore> shim_l_cores;
  utils::Set<base::LogicalCore> mem_l_cores;
  utils::Set<base::LogicalCore> compute_l_cores;
  for (const auto &[l_core, p_core] : old_placement.data()) {
    TilePtr tile = rr_node_to_tile_map_.at(p_core.getSource());
    if (tile->getTileType() == TileType::Shim) {
      shim_l_cores.insert(l_core);
    } else if (tile->getTileType() == TileType::Memory) {
      mem_l_cores.insert(l_core);
    } else if (tile->getTileType() == TileType::Compute) {
      compute_l_cores.insert(l_core);
    }
    l_cores.insert(l_core);
  }

  if (compute_l_cores.size() > (kRowsOfComputeTilesInNPU * kColumnsInNPU)) {
    throw std::runtime_error(
        "Not enough physical compute cores to map all logical cores");
  }

  // Keep track of the current logical core mapping on the NPU, organized by tile position.
  std::vector<std::vector<std::vector<base::LogicalCore>>> current_l_core_mapping;
  current_l_core_mapping.resize(kRowsInNPU);
  for (auto &row : current_l_core_mapping) {
    row.resize(kColumnsInNPU);
    for (auto &cell : row) {
      cell.clear();
    }
  }

  // Keep track of the neighbour logical core for each logical core.
  utils::Lookup<base::LogicalCore, utils::Set<base::LogicalCore>> l_core_to_neighbor_l_cores;
  for (const auto &net : netlist.getNets()) {
    base::LogicalCore src_core = net.getStartCore();
    for (const auto & dst_core : net.getTargetCores()){
      l_core_to_neighbor_l_cores[src_core].insert(dst_core);
      l_core_to_neighbor_l_cores[dst_core].insert(src_core);
    }
  }

  // Start with COMPUTE tiles, then MEMORY tiles, and finally SHIM tiles.
  int placement_idx = 0;
  int y_offset = kRowsOfShimTilesInNPU + kRowsOfMemoryTilesInNPU;
  int x_offset = 0;
  int current_y = y_offset;
  int current_x = x_offset;
  base::LogicalCore l_core_to_place = compute_l_cores.empty() ? (mem_l_cores.empty() ? *shim_l_cores.begin() : *mem_l_cores.begin()) : *compute_l_cores.begin();
  new_placement.update(l_core_to_place) = getPhysicalCore({current_y, current_x});
  current_l_core_mapping[current_y][current_x].push_back(l_core_to_place);
  l_cores.erase(l_core_to_place);
  compute_l_cores.erase(l_core_to_place);

  bool placed = false;
  while(compute_l_cores.size() > 0 && placement_idx < (kRowsOfComputeTilesInNPU * kColumnsInNPU)) {
    placed = false;
    // Check if current position is valid and available. If not, move to the next position.
    if (current_y < kRowsInNPU && current_x < kColumnsInNPU && current_l_core_mapping[current_y][current_x].size() == 0) {
      // Check if bottom neighbour exists. If yes, place its neighbour logical core here.
      if (current_y - 1 >= y_offset && current_l_core_mapping[current_y - 1][current_x].size() > 0 && !placed) {
        l_core_to_place = current_l_core_mapping[current_y - 1][current_x].front();
        if (getUnplacedNeighbourLogicalCores(l_core_to_place, l_core_to_neighbor_l_cores[l_core_to_place], compute_l_cores)) {
          new_placement.update(l_core_to_place) = getPhysicalCore({current_y, current_x});
          current_l_core_mapping[current_y][current_x].push_back(l_core_to_place);
          l_cores.erase(l_core_to_place);
          compute_l_cores.erase(l_core_to_place);
          placed = true;
        }
      }
      // Check if left neighbour exists. If yes, place its neighbour logical core here.
      if (current_x - 1 >= x_offset && current_l_core_mapping[current_y][current_x - 1].size() > 0 && !placed) {
        l_core_to_place = current_l_core_mapping[current_y][current_x - 1].front();
        if (getUnplacedNeighbourLogicalCores(l_core_to_place, l_core_to_neighbor_l_cores[l_core_to_place], compute_l_cores)) {
          new_placement.update(l_core_to_place) = getPhysicalCore({current_y, current_x});
          current_l_core_mapping[current_y][current_x].push_back(l_core_to_place);
          l_cores.erase(l_core_to_place);
          compute_l_cores.erase(l_core_to_place);
          placed = true;
        }
      }
      // If no neighbour logical core to place, just place any remaining logical core here.
      if (compute_l_cores.size() > 0 && !placed) {
        l_core_to_place = *compute_l_cores.begin();
        new_placement.update(l_core_to_place) = getPhysicalCore({current_y, current_x});
        current_l_core_mapping[current_y][current_x].push_back(l_core_to_place);
        l_cores.erase(l_core_to_place);
        compute_l_cores.erase(l_core_to_place);
        placed = true;
      }
    }
    // Move to the next position vertically first, then horizontally.
    placement_idx++;
    current_y = y_offset + placement_idx % kRowsOfComputeTilesInNPU;
    current_x = x_offset + placement_idx / kRowsOfComputeTilesInNPU;
  }

  // The MEMORY and SHIM tiles are placed in the same columns as the COMPUTE tiles that they connected to.
  for (const auto &mem_l_core : mem_l_cores) {
    int y_offset = kRowsOfShimTilesInNPU;
    double centrol_x = 0;
    double num_placed_neighbors = 0;
    for (const auto &neightbour_core : l_core_to_neighbor_l_cores[mem_l_core]) {
      if (!l_cores.contains(neightbour_core)) {
        centrol_x += rr_node_to_tile_map_.at(new_placement.at(neightbour_core).getSource())->getColX();
        num_placed_neighbors++;
      }
    }
    if (num_placed_neighbors > 0) {
      centrol_x /= num_placed_neighbors;
      new_placement.update(mem_l_core) = getPhysicalCore({y_offset, static_cast<int>(centrol_x)});
      current_l_core_mapping[y_offset][static_cast<int>(centrol_x)].push_back(mem_l_core);
      l_cores.erase(mem_l_core);
    }
  }

  for (const auto &shim_l_core : shim_l_cores) {
    int y_offset = 0;
    double centrol_x = 0;
    double num_placed_neighbors = 0;
    for (const auto &neightbour_core : l_core_to_neighbor_l_cores[shim_l_core]) {
      if (!l_cores.contains(neightbour_core)) {
        centrol_x += rr_node_to_tile_map_.at(new_placement.at(neightbour_core).getSource())->getColX();
        num_placed_neighbors++;
      }
    }
    if (num_placed_neighbors > 0) {
      centrol_x /= num_placed_neighbors;
      new_placement.update(shim_l_core) = getPhysicalCore({y_offset, static_cast<int>(centrol_x)});
      current_l_core_mapping[y_offset][static_cast<int>(centrol_x)].push_back(shim_l_core);
      l_cores.erase(shim_l_core);
    }
  }

  if (l_cores.size() > 0) {
    throw std::runtime_error("Some logical cores are not placed in the initial placement");
  }
  if (!isLegalPlacement(new_placement)) {
    throw std::runtime_error("Generated initial placement is not legal");
  }
}

void NPU::generateLegalPlacementWithBoundedRandomSwap(
  const base::Placement &old_placement, base::Placement &new_placement, std::vector<base::LogicalCore> &moved_l_cores,
  utils::RandomSeed seed, const int swap_bound)
  const {

  // Get current physical core mapping for the selected logical cores, and their corresponding tile positions.
  std::vector<std::vector<std::vector<base::LogicalCore>>> current_l_core_mapping;
  current_l_core_mapping.resize(kRowsInNPU);
  for (auto &row : current_l_core_mapping) {
    row.resize(kColumnsInNPU);
  }
  for (const auto &[l_core, p_core] : old_placement.data()) {
    TilePtr tile = rr_node_to_tile_map_.at(p_core.getSource());
    const int row_y = tile->getRowY();
    const int col_x = tile->getColX();
    current_l_core_mapping[row_y][col_x].push_back(l_core);
  }

  // Randomly select logical cores to be moved/swapped.
  utils::RandomNumberGenerator rnd_gen(0, std::numeric_limits<int>::max(), seed);
  int rnd_l_core_idx = rnd_gen.get() % old_placement.size();
  base::LogicalCore l_core_to_move = old_placement.keyVec().at(rnd_l_core_idx);
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // NOTE: this relies on the assumption that tiles of the same type are placed together in the NPU architecture,             //
  // which is currently true but may need to be revisited if we want to support more flexible NPU architectures in the future.//
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // For selected logical core, randomly select another physical core from its candidate set that is within the specified swap bound to swap with.
  npu_tiles::TileType tile_type = rr_node_to_tile_map_.at(old_placement.at(l_core_to_move).getSource())->getTileType();
  int target_tile_type_y_lower_bound = tile_type_to_tile_map_.at(tile_type).front()->getRowY();
  int target_tile_type_y_upper_bound = tile_type_to_tile_map_.at(tile_type).back()->getRowY();
  int target_tile_type_x_lower_bound = tile_type_to_tile_map_.at(tile_type).front()->getColX();
  int target_tile_type_x_upper_bound = tile_type_to_tile_map_.at(tile_type).back()->getColX();
  int current_pos_y = (rr_node_to_tile_map_.at(old_placement.at(l_core_to_move).getSource()))->getRowY();
  int current_pos_x = (rr_node_to_tile_map_.at(old_placement.at(l_core_to_move).getSource()))->getColX();
  int candidate_pos_y_lower_bound = std::max(target_tile_type_y_lower_bound, current_pos_y - swap_bound);
  int candidate_pos_y_upper_bound = std::min(target_tile_type_y_upper_bound, current_pos_y + swap_bound);
  int candidate_pos_x_lower_bound = std::max(target_tile_type_x_lower_bound, current_pos_x - swap_bound);
  int candidate_pos_x_upper_bound = std::min(target_tile_type_x_upper_bound, current_pos_x + swap_bound);
  int candidate_pos_region_size_y = candidate_pos_y_upper_bound - candidate_pos_y_lower_bound + 1;
  int candidate_pos_region_size_x = candidate_pos_x_upper_bound - candidate_pos_x_lower_bound + 1;

  int rnd_swap_location_idx = rnd_gen.get() % (candidate_pos_region_size_y * candidate_pos_region_size_x); // number of candidate positions within the swap bound
  int target_pos_y = candidate_pos_y_lower_bound + rnd_swap_location_idx / candidate_pos_region_size_x;
  int target_pos_x = candidate_pos_x_lower_bound + rnd_swap_location_idx % candidate_pos_region_size_x;

  moved_l_cores.clear();
  // Check if the selected new position have space. If not, swap with the logical core currently occupying the new position.
  if (current_l_core_mapping[target_pos_y][target_pos_x].size() >= getTile({target_pos_y, target_pos_x}).value()->getLogicalCapacity()) {
    // Swap with a random logical core in the new position.
    rnd_l_core_idx = rnd_gen.get() % current_l_core_mapping[target_pos_y][target_pos_x].size();
    base::LogicalCore l_core_to_swap = current_l_core_mapping[target_pos_y][target_pos_x][rnd_l_core_idx];
    new_placement.update(l_core_to_swap) = old_placement.at(l_core_to_move);
    new_placement.update(l_core_to_move) = old_placement.at(l_core_to_swap);
    moved_l_cores.push_back(l_core_to_move);
    moved_l_cores.push_back(l_core_to_swap);
  } else {
    // If the new position has space, just move the selected logical core to the new position.
    new_placement.update(l_core_to_move) = getPhysicalCore({target_pos_y, target_pos_x});
    moved_l_cores.push_back(l_core_to_move);
  }
}

base::RegionPlacement NPU::extendPlacementToRegionPlacement(
    const base::Placement &placement,
    const std::vector<base::GridPositionOffset> &offsets,
    const base::LogicalCoreSet &l_cores_to_consider) const {
  base::RegionPlacement region_placement;

  for (const auto &[l_core, p_core] : placement.data()) {
    TilePtr tile = rr_node_to_tile_map_.at(p_core.getSource());
    // TODO: use a proper coordinate system rather mixing row/column, x/y,
    // top/buttom, and left/right
    const int center_row_y = tile->getRowY();
    const int center_col_x = tile->getColX();

    base::PhysicalCoreSet candidate_p_cores;
    if (l_cores_to_consider.contains(l_core)) {
      // If the logical core is in the set to consider, find all candidate
      // physical cores according to the given offsets.
      // Note: The offsets may or may not contain the (0,0) offset to include
      // the originally mapped physical core.
      for (const auto &offset : offsets) {
        const int candidate_row_y = center_row_y + offset.getDy();
        const int candidate_col_x = center_col_x + offset.getDx();
        if (auto candidate_tile = getTile(candidate_row_y, candidate_col_x);
            candidate_tile.has_value() &&
            candidate_tile.value()->getTileType() == tile->getTileType()) {
          candidate_p_cores.insert(
              tile_to_p_core_map_.at(candidate_tile.value()));
        }
      }
    } else {
      // If the logical core is not in the set to consider, just add its
      // originally mapped physical core as the only candidate.
      candidate_p_cores.insert(p_core);
    }

    region_placement.add(
        l_core, base::PhysicalCoreRegion{
                    std::format("{}(center={},size={})", l_core.getName(),
                                tile->getName(), candidate_p_cores.size()),
                    candidate_p_cores});
  }

  return region_placement;
}

} // namespace npu
} // namespace arch
