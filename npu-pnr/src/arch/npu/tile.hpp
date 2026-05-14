#ifndef _NPU_TILE_HPP_
#define _NPU_TILE_HPP_

#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "base/rr_graph.hpp"
#include "utils/misc.hpp"

namespace arch {
namespace npu {
namespace npu_tiles {

using base::RRNode;
using utils::NamedClass;

enum class TileType { Compute = 0, Memory, Shim };

inline std::ostream &operator<<(std::ostream &os, const TileType &type) {
  switch (type) {
  case TileType::Compute:
    os << "Compute";
    break;
  case TileType::Memory:
    os << "Memory";
    break;
  case TileType::Shim:
    os << "Shim";
    break;
  default:
    os << "Unknown";
    break;
  }
  return os;
}

enum class Direction { North = 0, South, West, East };
inline Direction opposite(Direction dir) {
  switch (dir) {
  case Direction::North:
    return Direction::South;
  case Direction::South:
    return Direction::North;
  case Direction::West:
    return Direction::East;
  case Direction::East:
    return Direction::West;
  }
  throw std::logic_error("Unreachable");
}

class Tile;

class TileComponent : public NamedClass {
protected:
  std::shared_ptr<Tile> associated_tile_;

  TileComponent(const std::string &name_prefix)
      : NamedClass(name_prefix), associated_tile_(nullptr) {}

public:
  virtual ~TileComponent() = default;

  void registerTile(std::shared_ptr<Tile> tile);

  explicit operator RRNode() {
    if (associated_tile_ == nullptr) {
      throw std::runtime_error(std::format(
          "Tile component {} is not registered to any tile", getName()));
    }
    return RRNode(getName());
  }
};

class SwitchBox : public TileComponent {
public:
  SwitchBox() : TileComponent("SB") {}
};

// TODO: converge VirtualEndpoint and SwitchBox
class VirtualEndpoint : public TileComponent {
public:
  enum class Type { Source = 0, Sink }; // TODO: consider refactoring this
  VirtualEndpoint(const Type &ep_type)
      : TileComponent(
            std::format("ep_{}", ep_type == Type::Source ? "src" : "sink")) {}
};

class NeighborTile {
private:
  std::shared_ptr<Tile> tile_;
  bool is_memory_accessible_;
  bool has_neighbor_sharing_interconnect_;

public:
  NeighborTile() = delete;
  NeighborTile(std::shared_ptr<Tile> tile, bool is_memory_accessible,
               bool has_neighbor_sharing_interconnect)
      : tile_(tile), is_memory_accessible_(is_memory_accessible),
        has_neighbor_sharing_interconnect_(has_neighbor_sharing_interconnect) {}

  std::shared_ptr<Tile> getTile() const { return tile_; }
  [[nodiscard]] bool isMemoryAccessible() const {
    return is_memory_accessible_;
  }
  [[nodiscard]] bool hasNeighborSharingInterconnect() const {
    return has_neighbor_sharing_interconnect_;
  }
};

class Tile : public NamedClass {
protected:
  std::pair<int /*row_y*/, int /*col_x*/> pos_;
  TileType type_;
  VirtualEndpoint src_ep_, sink_ep_; // TODO: use register concept

  // Interconnect (TODO: use interconnect class for abstraction)
  // 1. Switch network
  SwitchBox sb_;

  // 2. Neighbor-sharing interconnect
  //    Note: Tile cannot access local memory in East neighbor, but the East
  //    neighbor can access local memory in the tile
  std::unordered_map<Direction, NeighborTile> neighbor_tile;

  static std::string generateTileName(std::pair<int, int> pos, TileType type) {
    std::string coord = std::format("{},{}", pos.first, pos.second);
    switch (type) {
    case TileType::Compute:
      return std::format("compute[{}]", coord);
    case TileType::Memory:
      return std::format("memory[{}]", coord);
    case TileType::Shim:
      return std::format("shim[{}]", coord);
    default:
      return "unknown";
    }
  }

  // Protected constructor to ensure that Tile can only be constructed through
  // derived classes
  Tile(std::pair<int, int> pos, TileType type)
      : NamedClass(generateTileName(pos, type)), pos_(pos), type_(type),
        src_ep_(VirtualEndpoint::Type::Source),
        sink_ep_(VirtualEndpoint::Type::Sink) {}

public:
  virtual ~Tile() = default;

  virtual void registerNeighborTile(std::shared_ptr<Tile> neighbor,
                                    Direction dir) = 0;

  // Intra-tile interconnect parameters
  virtual int getNumChannelsBetweenInterTileKernelLinking() const {
    return base::RREdgeCapacity::kInfiniteCapacity;
  }

  // Neighbor sharing interconnect parameters
  virtual int getNumChannelsBetweenNeighborSharing() const {
    return base::RREdgeCapacity::kInfiniteCapacity;
  }

  // SwitchBox network parameters
  // bi-directional
  virtual int getNumChannelsBetweenTileDMAandSwitchBox() const = 0;
  // single-directional
  virtual int getNumChannelsSwitchBoxToNorth() const = 0;
  virtual int getNumChannelsSwitchBoxToSouth() const = 0;
  virtual int getNumChannelsSwitchBoxToEast() const = 0;
  virtual int getNumChannelsSwitchBoxToWest() const = 0;

  virtual size_t getMemoryCapacity() const = 0; // in bytes
  virtual size_t getLockCapacity() const = 0;
  virtual size_t getLogicalCapacity() const = 0; // in terms of how many logical cores can be placed on the tile, used for legal placement checking and random sampling of legal placements

  static void registerTileComponents(std::shared_ptr<Tile> tile) {
    if (tile == nullptr) {
      throw std::runtime_error("Cannot register components to a null tile");
    }
    tile->sb_.registerTile(tile); // Safe because tile is already managed
    tile->src_ep_.registerTile(tile);
    tile->sink_ep_.registerTile(tile);
  }

  [[nodiscard]] bool isa(TileType type) const { return type_ == type; }

  SwitchBox getSwitchBox() const { return sb_; }
  std::optional<NeighborTile> getNeighborTile(Direction dir) const {
    return neighbor_tile.contains(dir)
               ? std::make_optional(neighbor_tile.at(dir))
               : std::nullopt;
  }

  VirtualEndpoint getSourceEndpoint() const { return src_ep_; }
  VirtualEndpoint getSinkEndpoint() const { return sink_ep_; }

  TileType getTileType() const { return type_; }
  int getRowY() const { return pos_.first; }
  int getColX() const { return pos_.second; }
};

class ComputeTile : public Tile {
public:
  ComputeTile() = delete;
  ComputeTile(std::pair<int, int> pos) : Tile(pos, TileType::Compute) {}

  void registerNeighborTile(std::shared_ptr<Tile>, Direction) final;

  // https://docs.amd.com/r/en-US/am020-versal-aie-ml/AXI4-Stream-Interconnect
  int getNumChannelsBetweenTileDMAandSwitchBox() const final { return 2; }
  int getNumChannelsSwitchBoxToNorth() const final { return 6; }
  int getNumChannelsSwitchBoxToSouth() const final { return 4; }
  int getNumChannelsSwitchBoxToEast() const final { return 4; }
  int getNumChannelsSwitchBoxToWest() const final { return 4; }

  size_t getMemoryCapacity() const final {
    return 64 * 1024; // 64 KB, as per AMD documentation
  }
  size_t getLockCapacity() const final { return 16; }
  size_t getLogicalCapacity() const final { return 1; }
};

class MemoryTile : public Tile {
public:
  MemoryTile() = delete;
  MemoryTile(std::pair<int, int> pos) : Tile(pos, TileType::Memory) {}

  void registerNeighborTile(std::shared_ptr<Tile>, Direction) final;

  // https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Memory-Tile-Architecture
  int getNumChannelsBetweenTileDMAandSwitchBox() const final { return 6; }
  int getNumChannelsSwitchBoxToNorth() const final { return 6; }
  int getNumChannelsSwitchBoxToSouth() const final { return 4; }
  // TODO: MemoryTile DMA[0-3] can access local memory in East/West neighbors,
  // which should be considered when implementing memory capacity constraints
  int getNumChannelsSwitchBoxToEast() const final { return 0; }
  int getNumChannelsSwitchBoxToWest() const final { return 0; }

  size_t getMemoryCapacity() const final {
    return 512 * 1024; // 512 KB, as per AMD documentation
  }
  size_t getLockCapacity() const final { return 64; }
  size_t getLogicalCapacity() const final { return std::numeric_limits<int>::max(); }
};

class ShimTile : public Tile {
public:
  ShimTile() = delete;
  ShimTile(std::pair<int, int> pos) : Tile(pos, TileType::Shim) {}

  void registerNeighborTile(std::shared_ptr<Tile>, Direction) final;

  // https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Array-Interface-Architecture
  int getNumChannelsBetweenTileDMAandSwitchBox() const final { return 2; }
  int getNumChannelsSwitchBoxToNorth() const final { return 6; }
  int getNumChannelsSwitchBoxToSouth() const final { return 4; }
  // Shim tiles do not have East/West SwitchBox connections
  // https://riallto.ai/notebooks/3_2_Ryzenai_capabilities.html#interface-tile-properties
  int getNumChannelsSwitchBoxToEast() const final { return 4; }
  int getNumChannelsSwitchBoxToWest() const final { return 4; }

  size_t getMemoryCapacity() const final {
    return std::numeric_limits<int>::max(); // ShimTile does not have local memory, but it can access
                // external memory through its DMA; thus, we assume it has
                // infinite memory capacity
  }
  size_t getLockCapacity() const final {
    return std::numeric_limits<int>::max(); // Similarly, we assume infinite lock capacity
  }
  size_t getLogicalCapacity() const final { return std::numeric_limits<int>::max(); }
};

template <typename T>
  requires utils::either<T, ComputeTile, MemoryTile, ShimTile>
std::shared_ptr<T> create(std::pair<int, int> pos) {
  auto tile = std::shared_ptr<T>(new T(pos));
  T::registerTileComponents(tile);
  return tile;
}

} // namespace npu_tiles
} // namespace npu
} // namespace arch

#endif
