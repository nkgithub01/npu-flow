#ifndef _BASE_ROUTING_MODE_HPP_
#define _BASE_ROUTING_MODE_HPP_

#include <algorithm>
#include <ostream>
#include <string>

#include "base/abstraction.hpp"
#include "base/common.hpp"
#include "base/rr_graph.hpp"

namespace base {

class RoutingMode {
public:
  enum class LoggingLevel {
    Silent,  // No logging
    Minimal, // Only log high-level information
    Normal,  // Log standard information
    Verbose, // Log detailed information (including solver output)
    Debug,   // Log detailed information for debugging
  };

  enum class AlgorithmMode {
    Default,         // Default algorithmatic mode
    HighPerformance, // Optimize for high performance (routing speed)
    HighQuality,     // Optimize for high quality (solution quality)
  };

  enum class RouteTreeReconstruction {
    Full, // Complete route tree reconstruction
    Skip, // No route tree (path) construction
  };

  enum class CongestionMapReconstruction {
    Full, // Complete congestion map reconstruction
    Skip, // No congestion map reconstruction
  };

  enum class BufferAllocReconstruction {
    Full, // Complete buffer allocation reconstruction
    Skip, // No buffer allocation reconstruction
  };

  enum class LogicalCorePlacementReconstruction {
    Full, // Full reconstruction of logical core placement solution
    Skip, // No reconstruction of logical core placement solution
  };

  enum class MulticastRouteTreeEdge {
    // Highest flexibility for multicast routing; different edges of a multicast
    // tree can use different types of interconnect (e.g., one edge/branch uses
    // circuit-switching, while another edge/branch uses neighbor-sharing).
    Heterogeneous,
    // All edges of a multicast tree must use the same *set* of types of
    // interconnect (e.g., if one edge/branch uses neighbor-sharing on the
    // source, another edge/branch can use neighbor-sharing on the sink, since
    // both types belong to neighbor-sharing interconnect set).
    SemiHeterogeneous,
    // All edge of a multicast tree must use the same type of interconnect
    // (e.g., if one edge/branch uses neighbor-sharing on the source, other
    // edges/branches must use neighbor-sharing on the source as well).
    Homogeneous,
  };

  enum class NetLink {
    // Enforce linked nets to use the interconnect type with the highest
    // flexibility (e.g., the link-from nets and link-to nets can use different
    // types of interconnect, as long as no buffer duplications are in the
    // shared tile)
    HighestFlexibility,
    // Enforce linked nets to use circuit-switching interconnect
    ForceLinkedNetsToCircuitSwitching,
    // Ignore linked nets during routing (treat them as independent nets)
    Ignored,
  };

  enum class MemoryCapacityConstraint {
    Enforced, // Enforce memory capacity constraints during routing
    Ignored,  // Ignore memory capacity constraints during routing
  };

  enum class LockCapacityConstraint {
    Enforced, // Enforce lock capacity constraints during routing
    Ignored,  // Ignore lock capacity constraints during routing
  };

  enum class LogicalCorePlacement {
    // Logical cores can be placed to any valid physical core during the MILP
    // placement, meaning that the placement of logical cores (and the nets on
    // them) can be changed by the solver to optimize placement and route
    // simultaneously). The packing of logical cores is optional (depending the
    // mode LogicalCorePacking).
    Explorable,
    // Logical cores are fixed to specific physical cores, meaning that only
    // routing of nets is optimized by the solver. Must be set for SA placer in
    // order to respect the annealing solution.
    Fixed,
  };

  enum class LogicalCorePacking {
    // Different logical cores can be or not be packed into the same core during
    // the MILP placement depending on the compatibility set. Can be used with
    // either LogicalCorePlacement::Explorable or Fixed.
    Enforced,
    // No logical core packing constraints is applied. Any two logical cores can
    // be packed into the same physical core if available table permits. Can be
    // used with either LogicalCorePlacement::Explorable or Fixed.
    Ignored,
  };

  enum class LogicalCorePlacementBlacklist {
    // Use a blacklist to prevent a certain placement solution from being
    // considered or explored by the MILP solver.
    Enforced,
    // No logical core placement blacklist is applied.
    Ignored,
  };

  enum class TimeLimit {
    Unlimited, // No time limit for routing
    Limited,   // Enforce a time limit for routing
  };

  struct HyperParameters {
    double objective_memory_capacity_scaling_factor = 0.0015; // 1/64KiB * 100%
    double objective_congestion_penalty_scaling_factor =
        1e5; // Avoid congestion by default

    explicit HyperParameters() = default;
  };

  using AllowedRREdgeTypeSubset = utils::Set<RREdgeType>;
  // Note: "OrderedSet" only means using vector to store the sets. The order of
  // sets does not actually matter.
  using AllowedRREdgeTypeOrderedSet = std::vector<AllowedRREdgeTypeSubset>;
  using LogicalToPhysicalCoreMappingBlacklist =
      std::vector<LogicalToPhysicalCoreMapping>;

private:
  LoggingLevel logging_level_;
  AlgorithmMode algorithm_mode_;
  RouteTreeReconstruction route_tree_reconstruction_;
  CongestionMapReconstruction congestion_map_reconstruction_;
  BufferAllocReconstruction buffer_alloc_reconstruction_;
  LogicalCorePlacementReconstruction logical_core_placement_reconstruction_;
  MulticastRouteTreeEdge multicast_tree_edge_;
  NetLink net_link_;
  MemoryCapacityConstraint memory_capacity_constraint_;
  LockCapacityConstraint lock_capacity_constraint_;
  LogicalCorePlacement logical_core_placement_;
  LogicalCorePacking logical_core_packing_;
  LogicalCorePlacementBlacklist logical_core_placement_blacklist_;
  TimeLimit time_limit_;

  AllowedRREdgeTypeOrderedSet allowed_multicast_rr_edge_types_;

  LogicalToPhysicalCoreAvailablityTable logical_to_physical_core_avail_table_;
  LogicalCoreCompatibilitySet logical_core_compat_set_;
  LogicalToPhysicalCoreMappingBlacklist
      logical_to_physical_core_mapping_blacklist_;
  double time_limit_in_sec_;
  HyperParameters hyper_params_;

public:
  // Use highest flexibility and most detailed routing mode by default
  RoutingMode()
      : logging_level_(LoggingLevel::Normal),
        algorithm_mode_(AlgorithmMode::Default),
        route_tree_reconstruction_(RouteTreeReconstruction::Full),
        congestion_map_reconstruction_(CongestionMapReconstruction::Full),
        buffer_alloc_reconstruction_(BufferAllocReconstruction::Full),
        logical_core_placement_reconstruction_(
            LogicalCorePlacementReconstruction::Full),
        net_link_(NetLink::ForceLinkedNetsToCircuitSwitching),
        memory_capacity_constraint_(MemoryCapacityConstraint::Enforced),
        lock_capacity_constraint_(LockCapacityConstraint::Ignored),
        hyper_params_(HyperParameters{}) {
    // TODO: enforce BufferAllocReconstruction::Skip being used together with
    // MemoryCapacityConstraint::Ignored (useful feature to add in routing mode)
    set(MulticastRouteTreeEdge::Homogeneous, {{RREdgeType::CircuitSwitching}});
    set(LogicalCorePlacement::Explorable, {});
    set(LogicalCorePacking::Ignored, {});
    set(LogicalCorePlacementBlacklist::Ignored, {});
    // TODO: refactor similar setter such that if there is no need to pass in
    // extra data, we can just use the simpler set() method.
    set(TimeLimit::Unlimited, {});
  }

  template <typename ModeType> void set(ModeType mode) {
    if constexpr (std::is_same_v<ModeType, LoggingLevel>) {
      logging_level_ = mode;
    } else if constexpr (std::is_same_v<ModeType, AlgorithmMode>) {
      algorithm_mode_ = mode;
    } else if constexpr (std::is_same_v<ModeType, RouteTreeReconstruction>) {
      route_tree_reconstruction_ = mode;
    } else if constexpr (std::is_same_v<ModeType,
                                        CongestionMapReconstruction>) {
      congestion_map_reconstruction_ = mode;
    } else if constexpr (std::is_same_v<ModeType, BufferAllocReconstruction>) {
      buffer_alloc_reconstruction_ = mode;
    } else if constexpr (std::is_same_v<ModeType,
                                        LogicalCorePlacementReconstruction>) {
      logical_core_placement_reconstruction_ = mode;
    } else if constexpr (std::is_same_v<ModeType, MulticastRouteTreeEdge>) {
      static_assert(false, "MulticastRouteTreeEdge can only be set along with "
                           "AllowedRREdgeTypeSetList.");
    } else if constexpr (std::is_same_v<ModeType, NetLink>) {
      net_link_ = mode;
    } else if constexpr (std::is_same_v<ModeType, MemoryCapacityConstraint>) {
      memory_capacity_constraint_ = mode;
    } else if constexpr (std::is_same_v<ModeType, LockCapacityConstraint>) {
      lock_capacity_constraint_ = mode;
    } else if constexpr (std::is_same_v<ModeType, LogicalCorePlacement>) {
      static_assert(false, "LogicalCorePlacement can only be set along with "
                           "LogicalToPhysicalCoreAvailablityTable.");
    } else if constexpr (std::is_same_v<ModeType, LogicalCorePacking>) {
      static_assert(false, "LogicalCorePacking can only be set along with "
                           "LogicalCoreCompatibilitySet.");
    } else if constexpr (std::is_same_v<ModeType,
                                        LogicalCorePlacementBlacklist>) {
      static_assert(false,
                    "LogicalCorePlacementBlacklist can only be set along with "
                    "LogicalToPhysicalCoreMappingBlacklist.");
    } else if constexpr (std::is_same_v<ModeType, TimeLimit>) {
      static_assert(
          false, "TimeLimit can only be set along with time limit in seconds.");
    } else if constexpr (std::is_same_v<ModeType, HyperParameters>) {
      hyper_params_ = mode;
    } else {
      static_assert(false, "Unsupported RoutingMode type for set() method.");
    }
  }

  void set(MulticastRouteTreeEdge mode,
           const AllowedRREdgeTypeOrderedSet &allowed) {
    if (allowed.empty()) {
      throw std::runtime_error("Allowed rr edge types cannot be an empty list");
    }

    if (std::ranges::any_of(allowed, [](const AllowedRREdgeTypeSubset &s) {
          return s.empty();
        })) {
      throw std::runtime_error(
          "Allowed rr edge types cannot contain empty sets");
    }

    if (mode == MulticastRouteTreeEdge::Heterogeneous) {
      if (allowed.size() != 1) {
        throw std::runtime_error(
            "Allowed rr edge types can only be set as a single set when "
            "multicast routing mode is Heterogeneous");
      }
    } else if (mode == MulticastRouteTreeEdge::Homogeneous) {
      if (std::ranges::any_of(allowed, [](const AllowedRREdgeTypeSubset &s) {
            return s.size() != 1;
          })) {
        throw std::runtime_error(
            "Allowed rr edge types can only be set as single-element sets "
            "when multicast routing mode is Homogeneous");
      }
    } else if (mode == MulticastRouteTreeEdge::SemiHeterogeneous) {
      if (allowed.size() == 1 ||
          std::ranges::all_of(allowed, [](const AllowedRREdgeTypeSubset &s) {
            return s.size() == 1;
          })) {
        throw std::runtime_error(
            "Allowed rr edge types must be set as multiple sets with at least "
            "one set containing multiple elements when multicast routing mode "
            "is SemiHeterogeneous");
      }
    } else {
      throw std::runtime_error("Unsupported multicast routing mode");
    }

    // Check that there is no duplicate rr edge type in all allowed sets
    utils::Set<RREdgeType> all_allowed_types;
    for (const auto &type_set : allowed) {
      for (const auto &type : type_set) {
        if (all_allowed_types.contains(type)) {
          throw std::runtime_error("Allowed rr edge type sets cannot contain "
                                   "duplicate rr edge types");
        }
        all_allowed_types.insert(type);
      }
    }

    multicast_tree_edge_ = mode;
    allowed_multicast_rr_edge_types_ = allowed;
  }

  void set(LogicalCorePlacement mode,
           const LogicalToPhysicalCoreAvailablityTable &avail_table) {
    if (mode == LogicalCorePlacement::Fixed) {
      if (avail_table.empty()) {
        throw std::runtime_error(
            "LogicalToPhysicalCoreAvailablityTable cannot be empty when "
            "LogicalCorePlacement is fixed");
      }
      if (std::ranges::any_of(avail_table.data(), [](const auto &entry) {
            return entry.second.size() != 1;
          })) {
        throw std::runtime_error(
            "LogicalToPhysicalCoreAvailablityTable must map each logical core "
            "to exactly one physical core when LogicalCorePlacement is fixed");
      }
      // Don't enforce packing mode here; instead, we ensure the compatibility
      // of LogicalCorePlacement and LogicalCorePacking in getter methods.
    } else if (mode == LogicalCorePlacement::Explorable) {
      if (std::ranges::any_of(avail_table.data(), [](const auto &entry) {
            return entry.second.empty();
          })) {
        throw std::runtime_error("LogicalToPhysicalCoreAvailablityTable cannot "
                                 "contain empty PhysicalCoreSet");
      }
    } else {
      throw std::runtime_error("Unsupported logical core placement mode");
    }

    logical_core_placement_ = mode;
    logical_to_physical_core_avail_table_ = avail_table;
  }

  void set(LogicalCorePacking mode,
           const LogicalCoreCompatibilitySet &compat_set) {
    if (mode == LogicalCorePacking::Enforced) {
      if (compat_set.empty())
        throw std::runtime_error("LogicalCoreCompatibilitySet cannot be empty "
                                 "when logical core packing is enforced");
    } else if (mode == LogicalCorePacking::Ignored) {
      if (!compat_set.empty())
        throw std::runtime_error("LogicalCoreCompatibilitySet must be empty "
                                 "when logical core packing is ignored");
    } else {
      throw std::runtime_error("Unsupported logical core packing mode");
    }

    logical_core_packing_ = mode;
    logical_core_compat_set_ = compat_set;
  }

  void set(LogicalCorePlacementBlacklist mode,
           const LogicalToPhysicalCoreMappingBlacklist &blacklist) {
    if (mode == LogicalCorePlacementBlacklist::Enforced) {
      if (blacklist.empty())
        throw std::runtime_error(
            "Logical core placement blacklist cannot be empty "
            "when enforced");
    } else if (mode == LogicalCorePlacementBlacklist::Ignored) {
      if (!blacklist.empty())
        throw std::runtime_error(
            "Logical core placement blacklist must be empty "
            "when ignored");
    } else {
      throw std::runtime_error(
          "Unsupported logical core placement blacklist mode");
    }

    logical_core_placement_blacklist_ = mode;
    logical_to_physical_core_mapping_blacklist_ = blacklist;
  }

  void set(TimeLimit mode, double time_limit_secs) {
    if (mode == TimeLimit::Unlimited) {
      if (time_limit_secs != 0) {
        throw std::runtime_error(
            "Time limit (in seconds) must be zero when time limit mode is "
            "Unlimited");
      }
    } else if (mode == TimeLimit::Limited) {
      if (time_limit_secs <= 0) {
        throw std::runtime_error(
            "Time limit (in seconds) must be positive when time limit mode is "
            "Limited");
      }
    }

    time_limit_ = mode;
    time_limit_in_sec_ = time_limit_secs;
  }

  template <typename ModeType> ModeType get() const {
    if constexpr (std::is_same_v<ModeType, LoggingLevel>) {
      return logging_level_;
    } else if constexpr (std::is_same_v<ModeType, AlgorithmMode>) {
      return algorithm_mode_;
    } else if constexpr (std::is_same_v<ModeType, RouteTreeReconstruction>) {
      return route_tree_reconstruction_;
    } else if constexpr (std::is_same_v<ModeType,
                                        CongestionMapReconstruction>) {
      return congestion_map_reconstruction_;
    } else if constexpr (std::is_same_v<ModeType, BufferAllocReconstruction>) {
      return buffer_alloc_reconstruction_;
    } else if constexpr (std::is_same_v<ModeType,
                                        LogicalCorePlacementReconstruction>) {
      return logical_core_placement_reconstruction_;
    } else if constexpr (std::is_same_v<ModeType, MulticastRouteTreeEdge>) {
      return multicast_tree_edge_;
    } else if constexpr (std::is_same_v<ModeType, NetLink>) {
      return net_link_;
    } else if constexpr (std::is_same_v<ModeType, MemoryCapacityConstraint>) {
      return memory_capacity_constraint_;
    } else if constexpr (std::is_same_v<ModeType, LockCapacityConstraint>) {
      return lock_capacity_constraint_;
    } else if constexpr (std::is_same_v<ModeType, LogicalCorePlacement>) {
      return logical_core_placement_;
    } else if constexpr (std::is_same_v<ModeType, LogicalCorePacking>) {
      return logical_core_packing_;
    } else if constexpr (std::is_same_v<ModeType,
                                        LogicalCorePlacementBlacklist>) {
      return logical_core_placement_blacklist_;
    } else if constexpr (std::is_same_v<ModeType, TimeLimit>) {
      return time_limit_;
    } else if constexpr (std::is_same_v<ModeType, HyperParameters>) {
      return hyper_params_;
    } else {
      static_assert(false, "Unsupported RoutingMode type for get() method");
    }
  }

  template <typename ModeType> [[nodiscard]] bool is(ModeType mode) const {
    return get<ModeType>() == mode;
  }

  void setLazyEvaluation() {
    route_tree_reconstruction_ = RouteTreeReconstruction::Skip;
    congestion_map_reconstruction_ = CongestionMapReconstruction::Skip;
    buffer_alloc_reconstruction_ = BufferAllocReconstruction::Skip;
    logical_core_placement_reconstruction_ =
        LogicalCorePlacementReconstruction::Skip;
  }

  [[nodiscard]] bool isLazyEvaluation() const {
    return route_tree_reconstruction_ == RouteTreeReconstruction::Skip &&
           congestion_map_reconstruction_ ==
               CongestionMapReconstruction::Skip &&
           buffer_alloc_reconstruction_ == BufferAllocReconstruction::Skip &&
           logical_core_placement_reconstruction_ ==
               LogicalCorePlacementReconstruction::Skip;
  }

  const AllowedRREdgeTypeOrderedSet &
  getMulticastAllowedRREdgeTypeOrderedSet() const {
    return allowed_multicast_rr_edge_types_;
  }

  std::vector<RREdgeType> getMulticastAllowedRREdgeTypes() const {
    std::vector<RREdgeType> types;
    for (const auto &type_set : allowed_multicast_rr_edge_types_) {
      types.insert(types.end(), type_set.begin(), type_set.end());
    }
    return types;
  }

  const LogicalToPhysicalCoreAvailablityTable &
  getLogicalToPhysicalCoreAvailablityTable() const {
    return logical_to_physical_core_avail_table_;
  }

  const LogicalCoreCompatibilitySet &getLogicalCoreCompatibilitySet() const {
    return logical_core_compat_set_;
  }

  const LogicalToPhysicalCoreMappingBlacklist &
  getLogicalToPhysicalCoreMappingBlacklist() const {
    return logical_to_physical_core_mapping_blacklist_;
  }

  double getTimeLimitInSeconds() const {
    if (time_limit_ == TimeLimit::Unlimited) {
      throw std::runtime_error(
          "Cannot get time limit value when time limit mode is Unlimited");
    }
    return time_limit_in_sec_;
  }

  friend std::ostream &operator<<(std::ostream &os, const RoutingMode &mode) {
    base::Config cfg;

    // AlgorithmMode
    switch (mode.get<AlgorithmMode>()) {
    case AlgorithmMode::Default:
      cfg.set<std::string>("algo_mode", "default");
      break;
    case AlgorithmMode::HighPerformance:
      cfg.set<std::string>("algo_mode", "high_perf");
      break;
    case AlgorithmMode::HighQuality:
      cfg.set<std::string>("algo_mode", "high_qual");
      break;
    };

    // isLazyEvaluation
    cfg.set<bool>("is_lazy_eval", mode.isLazyEvaluation());

    // MulticastRouteTreeEdge
    switch (mode.get<MulticastRouteTreeEdge>()) {
    case MulticastRouteTreeEdge::Heterogeneous:
      cfg.set<std::string>("multicast", "hetero");
      break;
    case MulticastRouteTreeEdge::SemiHeterogeneous:
      cfg.set<std::string>("multicast", "semi");
      break;
    case MulticastRouteTreeEdge::Homogeneous:
      cfg.set<std::string>("multicast", "homo");
      break;
    };

    // NetLink
    switch (mode.get<NetLink>()) {
    case NetLink::HighestFlexibility:
      cfg.set<std::string>("net_link", "flexible");
      break;
    case NetLink::ForceLinkedNetsToCircuitSwitching:
      cfg.set<std::string>("net_link", "force_cct");
      break;
    case NetLink::Ignored:
      cfg.set<std::string>("net_link", "ignored");
      break;
    };

    // MemoryCapacityConstraint
    switch (mode.get<MemoryCapacityConstraint>()) {
    case MemoryCapacityConstraint::Enforced:
      cfg.set<std::string>("mem_cap", "enforced");
      break;
    case MemoryCapacityConstraint::Ignored:
      cfg.set<std::string>("mem_cap", "ignored");
      break;
    };

    // LockCapacityConstraint
    switch (mode.get<LockCapacityConstraint>()) {
    case LockCapacityConstraint::Enforced:
      cfg.set<std::string>("lock_cap", "enforced");
      break;
    case LockCapacityConstraint::Ignored:
      cfg.set<std::string>("lock_cap", "ignored");
      break;
    };

    // LogicalCorePlacement
    switch (mode.get<LogicalCorePlacement>()) {
    case LogicalCorePlacement::Explorable:
      cfg.set<std::string>("placement", "explorable");
      break;
    case LogicalCorePlacement::Fixed:
      cfg.set<std::string>("placement", "fixed");
      break;
    };

    // LogicalCorePacking
    switch (mode.get<LogicalCorePacking>()) {
    case LogicalCorePacking::Enforced:
      cfg.set<std::string>("packing", "enforced");
      break;
    case LogicalCorePacking::Ignored:
      cfg.set<std::string>("packing", "ignored");
      break;
    };

    // LogicalCorePlacementBlacklist
    switch (mode.get<LogicalCorePlacementBlacklist>()) {
    case LogicalCorePlacementBlacklist::Enforced:
      cfg.set<std::string>("blacklist", "enforced");
      break;
    case LogicalCorePlacementBlacklist::Ignored:
      cfg.set<std::string>("blacklist", "ignored");
      break;
    };

    // TimeLimit
    switch (mode.get<TimeLimit>()) {
    case TimeLimit::Unlimited:
      cfg.set<double>("time_limit_sec", static_cast<double>(-1));
      break;
    case TimeLimit::Limited:
      cfg.set<double>("time_limit_sec", mode.getTimeLimitInSeconds());
      break;
    };

    // HyperParameters
    cfg.set<double>(
        "mem_cap_sf",
        mode.get<HyperParameters>().objective_memory_capacity_scaling_factor);
    cfg.set<double>("cg_sf", mode.get<HyperParameters>()
                                 .objective_congestion_penalty_scaling_factor);

    os << cfg.toJSON();
    return os;
  }
};

} // namespace base

#endif
