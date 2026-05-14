#include "arch/npu/tile.hpp"

namespace arch {
namespace npu {
namespace npu_tiles {

void TileComponent::registerTile(std::shared_ptr<Tile> tile) {
  if (!(associated_tile_ == nullptr && tile != nullptr)) {
    throw std::runtime_error(std::format(
        "Tile component {} (associated with tile {}) cannot register to tile "
        "{}",
        getName(), associated_tile_ ? associated_tile_->getName() : "null",
        tile ? tile->getName() : "null"));
  }
  associated_tile_ = tile;
  setUniqueName(std::format("{}_{}", this->getName(), tile->getName()));
}

void ComputeTile::registerNeighborTile(std::shared_ptr<Tile> neighbor,
                                       Direction dir) {
  if (neighbor->isa(TileType::Compute)) {
    // This tile can access memory in all neighboring compute tiles EXCEPT East
    // However, East neighbors can access this tile's memory (unidirectional)
    neighbor_tile.insert_or_assign(
        dir, NeighborTile{neighbor, (dir != Direction::East), true});
  } else if (neighbor->isa(TileType::Memory)) {
    // Unable to access Memory's memory through neighbor sharing
    // TODO: double check if this assumption is correct
    neighbor_tile.insert_or_assign(dir, NeighborTile{neighbor, false, false});
  } else {
    throw std::runtime_error(
        "Unreachable: Neighbor tile of Compute must be Compute or Memory");
  }
}

void MemoryTile::registerNeighborTile(std::shared_ptr<Tile> neighbor,
                                      Direction dir) {
  // Memory tile cannot access neighbor's local memory
  // TODO: double check if this assumption is correct
  neighbor_tile.insert_or_assign(dir, NeighborTile{neighbor, false, false});
}

void ShimTile::registerNeighborTile(std::shared_ptr<Tile> neighbor,
                                    Direction dir) {
  if (neighbor->isa(TileType::Compute)) {
    throw std::runtime_error(
        "Unreachable: Neighbor tile of Shim must be Shim or Memory");
  }
  // Shim tile cannot access neighbor's local memory
  // TODO: double check if this assumption is correct
  neighbor_tile.insert_or_assign(dir, NeighborTile{neighbor, false, false});
}

} // namespace npu_tiles
} // namespace npu

} // namespace arch