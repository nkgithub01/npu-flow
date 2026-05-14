#ifndef _BASE_ROUTING_NET_HPP_
#define _BASE_ROUTING_NET_HPP_

#include <algorithm>
#include <cstddef>
#include <format>
#include <ranges>

#include "base/abstraction.hpp"
#include "base/tf_graph.hpp"
#include "utils/misc.hpp"

namespace base {

class RoutingNet : public utils::NamedClass /*TODO: need faster hashing*/ {
public:
  using ID = TrafficFlow::ID;
  using BufferDemand = TrafficFlow::BufferDemand;

private:
  // TODO: net id should explicitly equal to the associated traffic flow id
  ID id_; // different nets can have the same source and sink
  // Note: be aware of that the start and targets LogicalCores are unique, but
  // for different net the buffer demand can be different
  // TODO: use id for hashing
  LogicalCore start_;
  std::vector<LogicalCore> targets_;
  utils::Lookup<LogicalCore, BufferDemand>
      buffer_demands_; // buffer demand (in bytes) for each logical core

public:
  RoutingNet() = delete;

  explicit RoutingNet(ID id, LogicalCore start,
                      std::vector<LogicalCore> targets)
      : NamedClass(std::format("{}->{}(net_id={})", start.getName(),
                               utils::getName(targets), id)),
        id_(id), start_(start), targets_(targets) {}

  void setBufferDemand(const LogicalCore &l_core, BufferDemand demand) {
    if (!(l_core == start_ ||
          std::ranges::any_of(targets_, [l_core](const LogicalCore &target) {
            return target == l_core;
          }))) {
      throw std::runtime_error(
          std::format("Logical core {} is not part of the net {}",
                      l_core.getName(), getName()));
    }
    buffer_demands_.insert_or_assign(l_core, demand);
  }

  const LogicalCore &getStartCore() const { return start_; }
  const std::vector<LogicalCore> &getTargetCores() const { return targets_; }
  const size_t getNumTargetCores() const { return targets_.size(); }
  [[nodiscard]] bool isMulticast() const { return targets_.size() > 1; }
  const ID getId() const { return id_; }
  const BufferDemand getBufferDemand(const LogicalCore &l_core) const {
    return buffer_demands_.at(l_core);
  }

  friend std::ostream &operator<<(std::ostream &out, const RoutingNet &net) {
    out << std::format("Net {}: {{ ", net.id_);
    out << std::format("start: {}, ", net.start_.getName());
    out << std::format("targets: [ {} ], ", utils::getName(net.targets_));
    std::vector<std::string> demand_strs;
    for (const auto &[l_core, demand] : net.buffer_demands_) {
      demand_strs.emplace_back(std::format("{}: {} {} bytes", l_core.getName(),
                                           utils::toString(demand),
                                           demand.getTotalBytes()));
    }
    out << std::format("buffer_demands: [ {} ]",
                       utils::toString(demand_strs, ", "));
    out << " }";
    return out;
  }
};

class RoutingNetLink {
private:
  LogicalCore linked_core_; // the logical core where the nets are linked
  std::vector<RoutingNet> from_nets_;
  std::vector<RoutingNet> to_nets_;

public:
  RoutingNetLink(const std::vector<RoutingNet> &from_nets,
                 const std::vector<RoutingNet> &to_nets)
      : from_nets_(from_nets), to_nets_(to_nets),
        linked_core_(to_nets.at(0).getStartCore()) {
    // TODO: use assert for verification, while using exception for runtime
    // checks
    if (from_nets.empty() || to_nets.empty()) {
      throw std::runtime_error("Link-from and link-to nets must not be empty");
    }
    if ((from_nets.size() == 1 && to_nets.size() == 1 /*passthrough*/) ||
        (from_nets.size() == 1 && to_nets.size() > 1 /*distribute*/) ||
        (from_nets.size() > 1 && to_nets.size() == 1 /*join*/)) {
      const auto &linked_core = linked_core_;
      if (std::ranges::any_of(to_nets, [&linked_core](const RoutingNet &net) {
            return net.getStartCore() != linked_core;
          })) {
        throw std::runtime_error("All link-to nets must have the same start "
                                 "matching the linked core");
      }
      if (std::ranges::any_of(from_nets, [&linked_core](const RoutingNet &net) {
            return std::ranges::none_of(
                net.getTargetCores(),
                [&linked_core](const LogicalCore &target) {
                  return target == linked_core;
                });
          })) {
        throw std::runtime_error(
            "All link-from nets must have a target matching the linked core");
      }
    } else {
      throw std::runtime_error(
          "Net link must be either passthrough, distribute or join");
    }
  }

  const LogicalCore &getLinkedCore() const { return linked_core_; }
  const std::vector<RoutingNet> &getFromNets() const { return from_nets_; }
  const std::vector<RoutingNet> &getToNets() const { return to_nets_; }

  friend std::ostream &operator<<(std::ostream &out,
                                  const RoutingNetLink &net_link) {
    out << std::format("NetLink at {} : {{ ", net_link.linked_core_.getName());
    out << std::format("from_nets: [ {} ], ",
                       utils::toString(net_link.from_nets_, ", "));
    out << std::format("to_nets: [ {} ]",
                       utils::toString(net_link.to_nets_, ", "));
    out << " }";
    return out;
  }
};

// TODO: optimize this class to avoid unnecessary copies
class RoutingNetList {
private:
  utils::Lookup<LogicalCore, std::vector<RoutingNet>> l_core_to_nets_;
  utils::Lookup<RoutingNet::ID, RoutingNet> net_id_to_net_;

  // For a certain RoutingNetLink, the start logical cores of all linked-to
  // nets guarantees to be the same, thus used for lookup
  utils::Lookup<LogicalCore, std::vector<RoutingNetLink>> l_core_to_net_links_;
  utils::Set<RoutingNet> linked_nets_; // for fast lookup of whether a net is linked

  static constexpr std::vector<RoutingNet> kEmptyNets{};
  static constexpr std::vector<RoutingNetLink> kEmptyLinkedNets{};

public:
  RoutingNetList() = default;
  explicit RoutingNetList(const TrafficFlowGraph &tf_graph) {
    // Add nets
    for (const std::shared_ptr<TrafficFlow> tf : tf_graph.getEdgeViews()) {
      // TODO: change the name (source/sink) of TrafficFlowEndpoint
      LogicalCore start_core{tf->getSource()};
      std::vector<LogicalCore> target_cores;
      std::vector<RoutingNet::BufferDemand> target_core_buffer_demands;
      for (const TrafficFlowEndpoint &sink : tf->getSinks()) {
        target_cores.emplace_back(sink);
        target_core_buffer_demands.emplace_back(tf->getBufferDemand(sink));
      }

      RoutingNet net{tf->getId(), start_core, target_cores};
      net.setBufferDemand(start_core, tf->getBufferDemand(tf->getSource()));
      for (size_t i = 0; i < target_cores.size(); ++i) {
        net.setBufferDemand(target_cores[i], target_core_buffer_demands[i]);
      }
      addNet(net);
    }
    // Add net links
    for (const auto &tf_link : tf_graph.getTrafficFlowLinkings()) {
      std::vector<RoutingNet> link_from_nets;
      for (const auto &tf : tf_link.getFromTrafficFlows()) {
        const RoutingNet &net = net_id_to_net_.at(tf->getId());
        link_from_nets.push_back(net);
        linked_nets_.insert(net);
      }
      std::vector<RoutingNet> link_to_nets;
      for (const auto &tf : tf_link.getToTrafficFlows()) {
        const RoutingNet &net = net_id_to_net_.at(tf->getId());
        link_to_nets.push_back(net);
        linked_nets_.insert(net);
      }
      addNetLink(RoutingNetLink(link_from_nets, link_to_nets));
    }
  }

  void addNet(const RoutingNet &net) {
    l_core_to_nets_[net.getStartCore()].push_back(net);
    for (const auto &target : net.getTargetCores()) {
      l_core_to_nets_[target].push_back(net);
    }
    net_id_to_net_.insert_or_assign(net.getId(), net);
  }

  void addNetLink(const RoutingNetLink &net_link) {
    l_core_to_net_links_[net_link.getLinkedCore()].push_back(net_link);
  }

  const RoutingNet &getNetById(const RoutingNet::ID &net_id) const {
    return net_id_to_net_.at(net_id);
  }

  const auto getNets() const { return std::views::values(net_id_to_net_); }

  const std::vector<RoutingNet> &
  getNetsAtLogicalCore(const LogicalCore &l_core) const {
    return (!l_core_to_nets_.contains(l_core)) ? kEmptyNets
                                               : l_core_to_nets_.at(l_core);
  }

  // This function returns the nets that does not have linked center at this logical core
  // It will return a net that is linked to another net with link center on a differen core.
  std::vector<RoutingNet>
  getNetWithNoLinkCenterAtLogicalCore(const LogicalCore &l_core) const {
    if (!l_core_to_nets_.contains(l_core)) {
      return kEmptyNets;
    }
    utils::Set<RoutingNet> linked_nets;
    for (const auto &net_link : getNetLinksAtLogicalCore(l_core)) {
      linked_nets.insert(net_link.getFromNets().begin(),
                         net_link.getFromNets().end());
      linked_nets.insert(net_link.getToNets().begin(),
                         net_link.getToNets().end());
    }
    std::vector<RoutingNet> non_link_center_nets;
    for (const auto &net : l_core_to_nets_.at(l_core)) {
      if (!linked_nets.contains(net)) {
        non_link_center_nets.push_back(net);
      }
    }
    return non_link_center_nets;
  }

  std::vector<RoutingNet>
  getNonLinkedNetsConnetedToLogicalCore(const LogicalCore &l_core) const {
    if (!l_core_to_nets_.contains(l_core)) {
      return kEmptyNets;
    }
    std::vector<RoutingNet> non_linked_nets;
    for (const auto &net : l_core_to_nets_.at(l_core)) {
      if (!linked_nets_.contains(net)) {
        non_linked_nets.push_back(net);
      }
    }
    return non_linked_nets;
  }

  std::vector<RoutingNetLink> getNetLinks() const {
    std::vector<RoutingNetLink> net_links;
    for (const auto &links : std::views::values(l_core_to_net_links_)) {
      net_links.insert(net_links.end(), links.begin(), links.end());
    }
    return net_links;
  }

  utils::Set<RoutingNet> getLinkedNets() const {
    return linked_nets_;
  }

  std::vector<RoutingNet> getNonLinkedNets() const {
    std::vector<RoutingNet> non_linked_nets;
    for (const auto &net : std::views::values(net_id_to_net_)) {
      if (!linked_nets_.contains(net)) {
        non_linked_nets.push_back(net);
      }
    }
    return non_linked_nets;
  }

  const std::vector<RoutingNetLink> &
  getNetLinksAtLogicalCore(const LogicalCore &l_core) const {
    if (!l_core_to_net_links_.contains(l_core)) {
      return kEmptyLinkedNets;
    }
    return l_core_to_net_links_.at(l_core);
  }

  RoutingNet::BufferDemand
  getNetLinkBufferDemandAtLogicalCore(const RoutingNetLink &net_link,
                                      const LogicalCore &l_core) const {
    const auto &from_nets = net_link.getFromNets();
    const auto &to_nets = net_link.getToNets();
    const LogicalCore &linked_core = net_link.getLinkedCore();
    if (from_nets.size() == 1 && to_nets.size() == 1) {
      // Passthrough
      RoutingNet::BufferDemand demand =
          from_nets.at(0).getBufferDemand(linked_core);
      if (demand == to_nets.at(0).getBufferDemand(linked_core)) {
        return demand;
      } else {
        throw std::runtime_error(
            std::format("Passthrough net link {} -> {} must have the same "
                        "buffer demand at the linked logical core {}",
                        from_nets.at(0).getName(), to_nets.at(0).getName(),
                        linked_core.getName()));
      }
    } else if (from_nets.size() == 1 && to_nets.size() > 1) {
      // Distribute
      // TODO: add check for buffer demand consistency
      // Note: use the largest buffer demand between both sides no matter what
      // the actual pattern (copy/split/merge) is in one-to-many, one-to-one, or
      // many-to-one.
      // TODO: rename the pattern types to use a more consistent naming scheme.
      // Currently we have (1) [one,many]-to-[one,many], (2) passthrough, join,
      // and distribute, (3) copy/split/merge.
      return from_nets.at(0).getBufferDemand(linked_core);
    } else if (from_nets.size() > 1 && to_nets.size() == 1) {
      // Join
      // TODO: add check for buffer demand consistency
      return to_nets.at(0).getBufferDemand(linked_core);
    } else {
      throw std::runtime_error(std::format(
          "Net links must be either passthrough, distribute or join\n"
          "\tLink-from nets: {}\n"
          "\tLink-to nets: {}\n",
          utils::getName(from_nets), utils::getName(to_nets)));
    }
  }

  size_t getNumLogicalCores() const { return l_core_to_nets_.size(); }

  friend std::ostream &operator<<(std::ostream &out,
                                  const RoutingNetList &netlist) {
    out << "Netlist: { ";
    out << std::format("nets: [ {} ], ",
                       utils::toString(netlist.getNets(), ", "));
    out << std::format("net_links: [ {} ] ",
                       utils::toString(netlist.getNetLinks(), ", "));
    out << " }";
    return out;
  }
};

} // namespace base

#endif
