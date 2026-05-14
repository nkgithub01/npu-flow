#ifndef _BASE_ABSTRACTION_HPP_
#define _BASE_ABSTRACTION_HPP_

#include <format>
#include <stdexcept>
#include <vector>

#include "base/rr_graph.hpp"
#include "base/tf_graph.hpp"
#include "utils/misc.hpp"

namespace base {

class LogicalCore : public utils::NamedClass {
private:
  TrafficFlowEndpoint tf_ep_;

public:
  LogicalCore() = delete;
  explicit LogicalCore(TrafficFlowEndpoint tf_ep)
      : NamedClass(std::format("l_core({})", utils::getName(tf_ep))),
        tf_ep_(tf_ep) {}

  const TrafficFlowEndpoint &getTrafficFlowEndpoint() const { return tf_ep_; }

  // Serialization
  std::string serialize() const { return tf_ep_.getID(); }
  static LogicalCore deserialize(const std::string &serialized_str) {
    return LogicalCore{TrafficFlowEndpoint{serialized_str}};
  }
};

class PhysicalCore : public utils::NamedClass {
public:
  using MemoryInBytes = size_t; // in bytes

private:
  RRNode source_;
  RRNode sink_;
  MemoryInBytes memory_capacity_;
  size_t lock_capacity_;

public:
  PhysicalCore() = delete;
  PhysicalCore(RRNode source, RRNode sink, MemoryInBytes memory_capacity,
               size_t lock_capacity)
      : NamedClass(std::format("p_core({},{},mem_cap={},lock_cap={})",
                               source.getName(), sink.getName(),
                               memory_capacity, lock_capacity)),
        source_(source), sink_(sink), memory_capacity_(memory_capacity),
        lock_capacity_(lock_capacity) {
    // Don't validate the type of RRNode to allow for maximum flexibility
  }

  const RRNode &getSource() const { return source_; }
  const RRNode &getSink() const { return sink_; }
  const MemoryInBytes &getMemoryCapacity() const { return memory_capacity_; }
  const size_t &getLockCapacity() const { return lock_capacity_; }

  // Serialization
  // TODO: maybe create another abstract class for serializable objects
  // TODO: add unit tests
  std::string serialize() const {
    return std::format("{};{};{};{}", source_.serialize(), sink_.serialize(),
                       memory_capacity_, lock_capacity_);
  }
  static PhysicalCore deserialize(const std::string &serialized_str) {
    std::vector<std::string> parts = utils::split(serialized_str, ';');
    if (parts.size() != 4) {
      throw std::runtime_error("Invalid serialized PhysicalCore format: " +
                               serialized_str);
    }
    return PhysicalCore{
        RRNode::deserialize(parts[0]), RRNode::deserialize(parts[1]),
        MemoryInBytes{std::stoull(parts[2])}, size_t{std::stoull(parts[3])}};
  }
};

using LogicalCoreSet = utils::Set<base::LogicalCore>;
using PhysicalCoreSet = utils::Set<base::PhysicalCore>;

using LogicalToPhysicalCoreAvailablityTable =
    utils::LookupWrapper<LogicalCore, PhysicalCoreSet>;

using PhysicalToLogicalCoreAvailablityTable =
    utils::LookupWrapper<PhysicalCore, LogicalCoreSet>;

inline PhysicalToLogicalCoreAvailablityTable
cast(const LogicalToPhysicalCoreAvailablityTable &l_to_p_table) {
  PhysicalToLogicalCoreAvailablityTable p_to_l_table;
  for (const auto &[l_core, p_core_set] : l_to_p_table.data()) {
    for (const auto &p_core : p_core_set) {
      if (!p_to_l_table.contains(p_core)) {
        p_to_l_table.add(p_core, LogicalCoreSet{});
      } else if (p_to_l_table.at(p_core).contains(l_core)) {
        throw std::runtime_error(std::format(
            "Unreachable: Logical core {} has duplicated mapping to "
            "physical core {} in the availablity table",
            l_core.getName(), p_core.getName()));
      }
      p_to_l_table.update(p_core).insert(l_core);
    }
  }
  return p_to_l_table;
}

class LogicalCoreCompatibilitySet {
public:
  using Subset = utils::Set<LogicalCore>;

private:
  std::vector<Subset> subsets_; // TODO: consider using set of sets

public:
  LogicalCoreCompatibilitySet() = default;
  explicit LogicalCoreCompatibilitySet(std::vector<Subset> subsets)
      : subsets_(std::move(subsets)) {
    // Check that each subset is non-empty
    for (const auto &subset : subsets_) {
      if (subset.empty()) {
        throw std::runtime_error(
            "Logical core compatibility subsets must be non-empty");
      }
    }
    // Check that subsets are disjoint
    utils::Set<LogicalCore> all_cores;
    for (const auto &subset : subsets_) {
      for (const auto &core : subset) {
        if (all_cores.contains(core)) {
          throw std::runtime_error(
              "Logical core compatibility subsets must be disjoint");
        }
        all_cores.insert(core);
      }
    }
  }

  [[nodiscard]] bool empty() const { return subsets_.empty(); }

  const std::vector<Subset> &getSubsets() const { return subsets_; }

  friend std::ostream &operator<<(std::ostream &os,
                                  const LogicalCoreCompatibilitySet &set) {
    for (size_t i = 0; i < set.subsets_.size(); ++i) {
      os << std::format("Subset {}: [ {} ]{}\n", i,
                        utils::getName(set.subsets_[i]),
                        (i == set.subsets_.size() - 1 ? "" : ","));
    }
    return os;
  }
};

// TODO: converge with Placement
using LogicalToPhysicalCoreMapping =
    utils::LookupWrapper<LogicalCore, PhysicalCore>;

} // namespace base

#endif
