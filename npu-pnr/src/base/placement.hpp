#ifndef _BASE_PLACEMENT_HPP_
#define _BASE_PLACEMENT_HPP_

#include <ostream>

#include "base/abstraction.hpp"
#include "utils/misc.hpp"

namespace base {

// Region Placement (one logical core to multiple candidate physical cores)
class PhysicalCoreRegion : public utils::NamedClass {
private:
  PhysicalCoreSet p_cores_;

public:
  PhysicalCoreRegion() = delete;

  explicit PhysicalCoreRegion(std::string name, const PhysicalCoreSet &p_cores)
      : utils::NamedClass(name), p_cores_(p_cores) {}

  const PhysicalCoreSet &getPhysicalCores() const { return p_cores_; }

  friend std::ostream &operator<<(std::ostream &os,
                                  const PhysicalCoreRegion &region) {
    os << "PhysicalCoreRegion " << region.getName() << ": [ "
       << utils::getName(region.p_cores_) << " ]";
    return os;
  }
};

using RegionPlacement = utils::LookupWrapper<LogicalCore, PhysicalCoreRegion>;

inline LogicalToPhysicalCoreAvailablityTable
cast(const RegionPlacement &placement) {
  LogicalToPhysicalCoreAvailablityTable table;
  for (const auto &[l_core, region] : placement.data()) {
    table.add(l_core, region.getPhysicalCores());
  }
  return table;
}

// Single Placement (one logical core to one physical core)
using Placement = utils::LookupWrapper<LogicalCore, PhysicalCore>;

class PlacementInfo{
public:
  Placement placement;
  utils::LookupWrapper<PhysicalCore, utils::Set<LogicalCore>> phy_to_logical_lookup;
  
  PlacementInfo() = delete;
  explicit PlacementInfo(const Placement &p)
    : placement(p) {
    for (const auto &[l_core, p_core] : placement.data()) {
      phy_to_logical_lookup.update(p_core).insert(l_core);
    }
  }
};

template <typename PlacementType>
  requires utils::either<PlacementType, Placement, RegionPlacement>
LogicalToPhysicalCoreAvailablityTable cast(const PlacementType &placement) {
  LogicalToPhysicalCoreAvailablityTable table;
  for (const auto &[l_core, p_core] : placement.data()) {
    if constexpr (std::is_same_v<PlacementType, RegionPlacement>) {
      // TODO: consider using a genalized method such as PhysicalCoreSet{} cast
      // for RegionPlacement
      table.add(l_core, p_core.getPhysicalCores());
    } else if constexpr (std::is_same_v<PlacementType, Placement>) {
      table.add(l_core, PhysicalCoreSet{p_core});
    }
  }
  return table;
}

// Common move function for both Placement and RegionPlacement
template <typename PlacementType>
  requires utils::either<PlacementType, Placement, RegionPlacement>
void move(PlacementType &placement, const LogicalCore &l_core,
          const typename PlacementType::value_type &new_p_core) {
  placement.update(l_core) = new_p_core;
}

inline std::string serializePlacement(const Placement &placement) {
  std::stringstream ss;
  for (const auto &[l_core, p_core] : placement.data()) {
    // TODO: use more structured format (e.g., JSON, TOML) if needed
    ss << std::format("{}\t{}\n", l_core.serialize(), p_core.serialize());
  }
  return ss.str();
}

inline Placement deserializePlacement(const std::string &raw_str) {
  Placement placement;
  std::stringstream ss(raw_str);
  std::string line;
  while (std::getline(ss, line)) {
    std::vector<std::string> tokens = utils::split(line, '\t');
    if (tokens.size() != 2) {
      throw std::runtime_error(
          "Invalid placement format: each line must contain exactly two "
          "tokens separated by a tab");
    }
    placement.add(LogicalCore::deserialize(tokens[0]),
                  PhysicalCore::deserialize(tokens[1]));
  }
  return placement;
}

} // namespace base

#endif
