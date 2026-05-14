#ifndef _BASE_ROUTING_STATE_HPP_
#define _BASE_ROUTING_STATE_HPP_

#include <cstddef>
#include <optional>

#include "base/abstraction.hpp"
#include "base/routing/routing_mode.hpp"
#include "base/routing/routing_net.hpp"
#include "base/rr_graph.hpp"

namespace base {

class RoutingState;

class RouteTree {
  friend class RoutingState;

private:
  RoutingNet associated_input_net_;
  std::vector<base::RREdge> rr_edges_;

public:
  RouteTree() = delete;
  RouteTree(const RoutingNet &p) : associated_input_net_(p) {}

  // Setters
  void addEdge(const base::RREdge &edge) { rr_edges_.push_back(edge); }

  // Getters
  const std::vector<base::RREdge> &getEdges() const { return rr_edges_; }

  const RoutingNet &getAssociatedInputNet() const {
    return associated_input_net_;
  }

  friend std::ostream &operator<<(std::ostream &os, const RouteTree &tree) {
    os << std::format("RouteTree for net {}:\n{}",
                      tree.associated_input_net_.getName(),
                      utils::toString(tree.rr_edges_, "\n", "\t"));
    return os;
  }
};

// TODO: reconstruct buffer allocation for each net
class BufferAllocation {
private:
  size_t total_buffer_size_;
  // std::vector<std::pair<RoutingNet, size_t>> buffers_;

public:
  BufferAllocation() : total_buffer_size_(0) /*, buffers_()*/ {}

  // void addBuffer(const RoutingNet &allocated_to_net, size_t buffer_size) {
  //   buffers_.emplace_back(allocated_to_net, buffer_size);
  // }

  void setTotalBufferSize(size_t total_buffer_size) {
    total_buffer_size_ = total_buffer_size;
  }

  const size_t getTotalBufferSize() const {
    // size_t temp = 0;
    // for (const auto &[net, buffer_size] : buffers_) {
    //   temp += buffer_size;
    // }
    // if (temp != total_buffer_size_) {
    //   std::string msg =
    //       std::format("Inconsistent total buffer size: expected {}, got
    //       {}\n",
    //                   total_buffer_size_, temp);
    //   for (const auto &[net, buffer_size] : buffers_) {
    //     msg += std::format("\tNet {}: {} bytes\n", net.getName(),
    //     buffer_size);
    //   }
    //   throw std::runtime_error(
    //       std::format("Inconsistent total buffer size\n{}", msg));
    // }
    return total_buffer_size_;
  }

  // const std::vector<std::pair<RoutingNet, size_t>> &getBuffers() const {
  //   return buffers_;
  // }

  // size_t getAllocatedBufferSize(const RoutingNet &allocated_to_net) const {
  //   for (const auto &[net, buffer_size] : buffers_) {
  //     if (net == allocated_to_net) {
  //       return buffer_size;
  //     }
  //   }
  //   return 0;
  // }
};

using BufferAllocationMap = utils::Lookup<LogicalCore, BufferAllocation>;
// Over-utilization: 1 means 1 unit over capacity
using CongestionMap = utils::Lookup<base::RREdge, size_t>;

class RoutingState {
public:
  enum class Legality {
    Legal,     // All constraints are satisfied and no congestion occurs
    Congested, // Some resources are over-utilized but all hard constraints
               // are satisfied
    Fatal,     // Some hard constraints are violated (e.g., memory capacity or
               // some physical cores are not reachable)
  };

private:
  RoutingMode mode_;       // TODO: store extra information for DebugRouting
  RoutingNetList netlist_; // Used for debugging purpose
  Legality legality_;
  float routing_cost_;
  size_t memory_usage_;
  std::vector<RouteTree> route_trees_;
  CongestionMap congestion_map_;
  BufferAllocationMap buffer_allocation_map_;
  LogicalToPhysicalCoreMapping logical_to_physical_core_mapping_;

public:
  RoutingState() = delete;

  constexpr static float kInfiniteRoutingCost =
      std::numeric_limits<float>::max();
  constexpr static float kInvalidRoutingCost = kInfiniteRoutingCost;

  static RoutingState createInvalidRoutingState() {
    return RoutingState(RoutingMode{}, Legality::Fatal, kInvalidRoutingCost);
  }

  explicit RoutingState(RoutingMode mode, Legality legality, float routing_cost)
      : mode_(mode), netlist_(), legality_(legality),
        routing_cost_(routing_cost), memory_usage_(0), route_trees_(),
        congestion_map_(), buffer_allocation_map_(),
        logical_to_physical_core_mapping_() {}

  explicit RoutingState(
      RoutingMode mode, RoutingNetList netlist, Legality legality,
      float routing_cost, std::vector<RouteTree> route_trees,
      CongestionMap congestion_map, BufferAllocationMap buffer_alloc_map,
      LogicalToPhysicalCoreMapping logical_to_physical_core_mapping)
      : mode_(mode), netlist_(std::move(netlist)), legality_(legality),
        routing_cost_(routing_cost), memory_usage_(0),
        route_trees_(std::move(route_trees)),
        congestion_map_(std::move(congestion_map)),
        buffer_allocation_map_(std::move(buffer_alloc_map)),
        logical_to_physical_core_mapping_(
            std::move(logical_to_physical_core_mapping)) {
    // Don't care the input if reconstruction is skipped, since the getters
    // will do the validation when called
    if (mode_.is(RoutingMode::RouteTreeReconstruction::Full)) {
      if (route_trees_.empty()) {
        throw std::runtime_error(
            "Route trees cannot be empty if reconstruction is requested");
      }
      for (RouteTree &tree : route_trees_) {
        if (tree.getEdges().empty()) {
          throw std::runtime_error("Each route tree must have at least one "
                                   "edge if reconstruction is requested");
        }
      }
    }

    if (mode_.is(RoutingMode::CongestionMapReconstruction::Full)) {
      if (std::ranges::any_of(congestion_map_,
                              [](const auto &kv) { return kv.second == 0; })) {
        throw std::runtime_error(
            "Congestion map should not contain edges with zero congestion");
      }
      // TODO: consider adding more comprehensive validators for congestion map;
      // however, these is existing one already in the router.
    }

    if (mode_.is(RoutingMode::BufferAllocReconstruction::Full)) {
      if (buffer_allocation_map_.empty()) {
        throw std::runtime_error(
            "Buffer allocation map cannot be empty if reconstruction is "
            "requested");
      }
      for (const auto &buf_alloc : std::views::values(buffer_allocation_map_)) {
        memory_usage_ += buf_alloc.getTotalBufferSize();
      }
    }

    if (mode_.is(RoutingMode::LogicalCorePlacementReconstruction::Full)) {
      if (logical_to_physical_core_mapping_.empty()) {
        throw std::runtime_error(
            "Logical to physical core mapping cannot be empty if logical "
            "core placement reconstruction is requested (special case: if "
            "the "
            "logical core placement is fixed, the core mapping should be the "
            "same as the initial logical core availablity table).");
      }
      if (mode.is(RoutingMode::LogicalCorePlacement::Fixed)) {
        for (const auto &[l_core, p_core] :
             logical_to_physical_core_mapping_.data()) {
          const auto &initial_avail_p_core_set =
              mode_.getLogicalToPhysicalCoreAvailablityTable().at(l_core);
          if (initial_avail_p_core_set.size() != 1 ||
              !initial_avail_p_core_set.contains(p_core)) {
            throw std::runtime_error(std::format(
                "Logical core {} is mapped to physical core {} which is not "
                "the intended physical core according to the initial logical "
                "core availability table, or the initial availability table "
                "has none or multiple physical cores for a fixed-placement "
                "logical core.",
                l_core.getName(), p_core.getName()));
          }
        }
      }
    }
    // TODO: add a more comprehensive validator for logical to physical core
    // mapping solution
  }

  float getRoutingCost() const { return routing_cost_; }

  std::optional<size_t> getMemoryUsage() const {
    return mode_.is(RoutingMode::BufferAllocReconstruction::Skip)
               ? std::nullopt
               : std::make_optional(memory_usage_);
  }

  // // TODO: find a better way to do this and similar things elsewhere
  // std::vector<RouteTree> &getMutableRouteTrees() { return route_trees_; }

  std::optional<std::vector<RouteTree>> getRouteTrees() const {
    return mode_.is(RoutingMode::RouteTreeReconstruction::Skip)
               ? std::nullopt
               : std::make_optional(route_trees_);
  }

  std::optional<CongestionMap> getCongestionMap() const {
    return mode_.is(RoutingMode::CongestionMapReconstruction::Skip)
               ? std::nullopt
               : std::make_optional(congestion_map_);
  }

  std::optional<BufferAllocationMap> getBufferAllocationMap() const {
    return mode_.is(RoutingMode::BufferAllocReconstruction::Skip)
               ? std::nullopt
               : std::make_optional(buffer_allocation_map_);
  }

  std::optional<LogicalToPhysicalCoreMapping>
  getLogicalToPhysicalCoreMapping() const {
    return mode_.is(RoutingMode::LogicalCorePlacementReconstruction::Skip)
               ? std::nullopt
               : std::make_optional(logical_to_physical_core_mapping_);
  }

  std::optional<RoutingNetList> getRoutingNetList() const {
    return mode_.isLazyEvaluation() ? std::nullopt
                                    : std::make_optional(netlist_);
  }

  Legality getLegality() const { return legality_; }
};

// TODO: consider adding a new base::routing namespace for routing-related
// classes and helper functions
inline std::ostream &operator<<(std::ostream &os,
                                const RoutingState::Legality &legality) {
  switch (legality) {
  case RoutingState::Legality::Legal:
    os << "Legal";
    break;
  case RoutingState::Legality::Congested:
    os << "Congested";
    break;
  case RoutingState::Legality::Fatal:
    os << "Fatal";
    break;
  default:
    os << "Unknown";
    break;
  }
  return os;
}

} // namespace base

#endif
