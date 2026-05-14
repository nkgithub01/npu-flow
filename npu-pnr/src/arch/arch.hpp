#pragma once

#include <utility>

#include "base/abstraction.hpp"
#include "base/common.hpp"
#include "base/engine.hpp"
#include "base/placement.hpp"
#include "base/routing.hpp"
#include "base/rr_graph.hpp"
#include "utils/misc.hpp"

namespace arch {

class Arch {
private:
  struct ArchConcept {
    virtual ~ArchConcept() = default;

    // Get architecture specs
    virtual base::GridDimension getDimensions() const = 0;

    // Generate RR graph
    virtual base::RRGraph getRRGraph() const = 0;

    // Parsing and dumping netlist
    virtual base::PnRPlacedNetlist
    parseNetlist(base::PnRNetlistReader rd) const = 0;
    virtual base::PnRNetlistWriter
    dumpNetlist(base::PnRPlacedNetlist p) const = 0;
    virtual base::PnRNetlistWriter dumpNetlist(base::PnRPlacedNetlist p,
                                               base::RoutingState r) const = 0;

    virtual base::PhysicalCore
    getPhysicalCore(base::GridPosition pos) const = 0;
    virtual base::PhysicalCoreSet getPhysicalCores() const = 0;
    virtual base::LogicalCoreCompatibilitySet
    getLogicalCoreCompatibilitySet(const base::Placement &p) const = 0;
    virtual base::LogicalToPhysicalCoreAvailablityTable
    getLogicalToPhysicalCoreAvailablityTable(
      const base::Placement &p) const = 0;

    virtual bool isLegalPlacement(const base::Placement &p) const = 0;
    virtual base::Placement
    getLegalPlacementWithRandomSampling(const base::Placement &p,
                                        utils::RandomSeed s) const = 0;
    virtual void generateInitialPlacement(
      const base::Placement &old_placement, base::Placement &new_placement,
      const base::RoutingNetList &netlist, utils::RandomSeed seed)
      const = 0;
    virtual void generateLegalPlacementWithBoundedRandomSwap(
      const base::Placement &old_placement, base::Placement &new_placement, std::vector<base::LogicalCore> &moved_l_cores,
      utils::RandomSeed s, const int swap_bound)
      const = 0;
    virtual base::RegionPlacement extendPlacementToRegionPlacement(
      const base::Placement &p,
      const std::vector<base::GridPositionOffset> &offsets,
      const base::LogicalCoreSet &l_cores) const = 0;

    virtual std::string
    visualizePlacement(const base::PnRNetlistReader &rd) const = 0;
    virtual std::string
    visualizeRouting(const base::PnRNetlistReader &rd) const = 0;
  };

  template <typename T> struct ArchModel : ArchConcept {
    T impl_;

    explicit ArchModel(const base::Config &cfg) : impl_(cfg) {}

    base::GridDimension getDimensions() const override {
      return impl_.getDimensions();
    }

    base::RRGraph getRRGraph() const override { return impl_.getRRGraph(); }

    base::PnRPlacedNetlist
    parseNetlist(base::PnRNetlistReader rd) const override {
      return impl_.parseNetlist(std::move(rd));
    }

    base::PnRNetlistWriter
    dumpNetlist(base::PnRPlacedNetlist p) const override {
      return impl_.dumpNetlist(std::move(p));
    }

    base::PnRNetlistWriter dumpNetlist(base::PnRPlacedNetlist p,
                                       base::RoutingState r) const override {
      return impl_.dumpNetlist(std::move(p), std::move(r));
    }

    base::PhysicalCore getPhysicalCore(base::GridPosition pos) const override {
      return impl_.getPhysicalCore(pos);
    }

    base::PhysicalCoreSet getPhysicalCores() const override {
      return impl_.getPhysicalCores();
    }

    base::LogicalCoreCompatibilitySet
    getLogicalCoreCompatibilitySet(const base::Placement &p) const override {
      return impl_.getLogicalCoreCompatibilitySet(p);
    }

    base::LogicalToPhysicalCoreAvailablityTable
    getLogicalToPhysicalCoreAvailablityTable(
      const base::Placement &p) const override {
      return impl_.getLogicalToPhysicalCoreAvailablityTable(p);
    }

    bool isLegalPlacement(const base::Placement &p) const override {
      return impl_.isLegalPlacement(p);
    }

    base::Placement
    getLegalPlacementWithRandomSampling(const base::Placement &p,
                                        utils::RandomSeed s) const override {
      return impl_.getLegalPlacementWithRandomSampling(p, s);
    }

    virtual void generateInitialPlacement(
      const base::Placement &old_placement, base::Placement &new_placement,
      const base::RoutingNetList &netlist, utils::RandomSeed seed)
    const{
      impl_.generateInitialPlacement(old_placement, new_placement, netlist, seed);
    }

    void generateLegalPlacementWithBoundedRandomSwap(
      const base::Placement &old_placement, base::Placement &new_placement, std::vector<base::LogicalCore> &moved_l_cores,
      utils::RandomSeed s, const int swap_bound)
      const override {
      impl_.generateLegalPlacementWithBoundedRandomSwap(old_placement, new_placement, moved_l_cores, s, swap_bound);
    }

    base::RegionPlacement extendPlacementToRegionPlacement(
      const base::Placement &p,
      const std::vector<base::GridPositionOffset> &offsets,
      const base::LogicalCoreSet &l_cores) const override {
      return impl_.extendPlacementToRegionPlacement(p, offsets, l_cores);
    }

    std::string
    visualizePlacement(const base::PnRNetlistReader &rd) const override {
      return impl_.visualizePlacement(rd);
    }

    std::string
    visualizeRouting(const base::PnRNetlistReader &rd) const override {
      return impl_.visualizeRouting(rd);
    }
  };

  std::unique_ptr<ArchConcept> pimpl_;

  // Private constructor
  template <typename T>
  explicit Arch(std::in_place_type_t<T>, const base::Config &cfg)
      : pimpl_(std::make_unique<ArchModel<T>>(cfg)) {}

public:
  // Static factory
  template <typename T> static Arch create(const base::Config &cfg) {
    return Arch(std::in_place_type<T>, cfg);
  }

  base::GridDimension getDimensions() const { return pimpl_->getDimensions(); }

  base::RRGraph getRRGraph() const { return pimpl_->getRRGraph(); }

  base::PnRPlacedNetlist parseNetlist(base::PnRNetlistReader rd) const {
    return pimpl_->parseNetlist(std::move(rd));
  }

  base::PnRNetlistWriter dumpNetlist(base::PnRPlacedNetlist p) const {
    return pimpl_->dumpNetlist(std::move(p));
  }

  base::PnRNetlistWriter dumpNetlist(base::PnRPlacedNetlist p,
                                     base::RoutingState r) const {
    return pimpl_->dumpNetlist(std::move(p), std::move(r));
  }

  base::PhysicalCore getPhysicalCore(base::GridPosition pos) const {
    return pimpl_->getPhysicalCore(pos);
  }

  base::PhysicalCoreSet getPhysicalCores() const {
    return pimpl_->getPhysicalCores();
  }

  base::LogicalCoreCompatibilitySet
  getLogicalCoreCompatibilitySet(const base::Placement &p) const {
    return pimpl_->getLogicalCoreCompatibilitySet(p);
  }

  base::LogicalToPhysicalCoreAvailablityTable
  getLogicalToPhysicalCoreAvailablityTable(const base::Placement &p) const {
    return pimpl_->getLogicalToPhysicalCoreAvailablityTable(p);
  }

  bool isLegalPlacement(const base::Placement &p) const {
    return pimpl_->isLegalPlacement(p);
  }

  base::Placement
  getLegalPlacementWithRandomSampling(const base::Placement &p,
                                      utils::RandomSeed s) const {
    return pimpl_->getLegalPlacementWithRandomSampling(p, s);
  }

  virtual void generateInitialPlacement(
    const base::Placement &old_placement, base::Placement &new_placement,
    const base::RoutingNetList &netlist, utils::RandomSeed seed)
  const {
    pimpl_->generateInitialPlacement(old_placement, new_placement, netlist, seed);
  }

  void generateLegalPlacementWithBoundedRandomSwap(
    const base::Placement &old_placement, base::Placement &new_placement, std::vector<base::LogicalCore> &moved_l_cores,
    utils::RandomSeed s, const int swap_bound)
    const {
    pimpl_->generateLegalPlacementWithBoundedRandomSwap(old_placement, new_placement, moved_l_cores, s, swap_bound);
  }

  base::RegionPlacement extendPlacementToRegionPlacement(
    const base::Placement &p,
    const std::vector<base::GridPositionOffset> &offsets,
    const base::LogicalCoreSet &l_cores) const {
    return pimpl_->extendPlacementToRegionPlacement(p, offsets, l_cores);
  }

  std::string visualizePlacement(const base::PnRNetlistReader &rd) const {
    return pimpl_->visualizePlacement(rd);
  }

  std::string visualizeRouting(const base::PnRNetlistReader &rd) const {
    return pimpl_->visualizeRouting(rd);
  }
};

}; // namespace arch
