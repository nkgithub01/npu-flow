#ifndef _BASE_TRAFFIC_FLOW_GRAPH_HPP_
#define _BASE_TRAFFIC_FLOW_GRAPH_HPP_

#include <algorithm>
#include <format>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "utils/graph.hpp"
#include "utils/misc.hpp"

namespace base {

using utils::NamedClass;

class TrafficFlowEndpoint : public NamedClass, public utils::GraphNode {
  // TrafficFlowEndpoint is a node in the traffic flow graph. A certain
  // such node can be used multiple times in the graph, either as a source
  // of a traffic flow or as a sink.

public:
  TrafficFlowEndpoint() = delete;
  // TODO: use ID type instead of string to avoid confusion
  TrafficFlowEndpoint(const std::string &partial_name)
      : NamedClass(std::format("tf_ep_{}", partial_name)) {}

  std::string getID() const {
    return getName().substr(std::string("tf_ep_").length());
  }
};

class TrafficFlow : public NamedClass,
                    public utils::GraphEdge<TrafficFlowEndpoint,
                                            utils::GraphEdgeType::HyperEdge> {
  // TrafficFlow is a hyperedge in the traffic flow graph. It connects a source
  // node to one or more sink nodes. It is used to represent a unicast or
  // multicast traffic flow in the network.
public:
  using ID = std::string;
  struct BufferDemand {
    size_t depth;
    size_t width;
    BufferDemand() = delete;
    explicit BufferDemand(size_t buffer_depth, size_t buffer_width)
        : depth(buffer_depth), width(buffer_width) {}
    size_t getTotalBytes() const { return depth * width; }
    bool operator==(const BufferDemand &other) const {
      return depth == other.depth && width == other.width;
    }
    friend std::ostream &operator<<(std::ostream &out, const BufferDemand &bd) {
      out << std::format("bd(dep={},wid={})", bd.depth, bd.width);
      return out;
    }
  };

private:
  ID id_; // TODO: change to std::string
  std::unordered_map<TrafficFlowEndpoint, BufferDemand, utils::NamedClassHash>
      buffer_demands_; // in bytes, used for buffer allocation
  bool is_self_loop_;

public:
  TrafficFlow() = delete;
  TrafficFlow(std::shared_ptr<TrafficFlowEndpoint> source,
              std::vector<std::shared_ptr<TrafficFlowEndpoint>> sinks)
      : NamedClass("tf_unique_name_uninitialized"),
        GraphEdge<TrafficFlowEndpoint, utils::GraphEdgeType::HyperEdge>(source,
                                                                        sinks) {
    if (sinks.size() > 1 &&
        std::ranges::any_of(
            sinks, [source](const auto &sink) { return *sink == *source; })) {
      throw std::runtime_error(std::format(
          "Error: traffic flow {} has multiple destinations including its "
          "source itself ({} -> {{..., {}, ...}}). In multicast, inside-tile "
          "flow (or self-loops, kernel linking) are not supported.",
          getName(), source->getName(), source->getName()));
    }
    is_self_loop_ = bool(sinks.size() == 1 && *sinks.at(0) == *source);
  }

  void setId(const ID id) {
    id_ = id;
    setAttribute("id", id);
  }
  void setBufferDemand(const TrafficFlowEndpoint &ep,
                       const BufferDemand demand) {
    if (!hasNode(ep)) {
      throw std::runtime_error(
          std::format("Error: traffic flow {} does not have endpoint {} to set "
                      "buffer demand",
                      getName(), ep.getName()));
    }
    buffer_demands_.insert_or_assign(ep, demand);
    setAttribute(std::format("bd_{}", ep.getName()), utils::toString(demand));
  }

  TrafficFlowEndpoint getSource() const { return *getFromNodePtr(); }
  std::vector<TrafficFlowEndpoint> getSinks() const {
    std::vector<TrafficFlowEndpoint> result;
    for (const auto &ptr : getToNodePtrs()) {
      result.push_back(*ptr);
    }
    return result;
  }
  ID getId() const { return id_; }
  BufferDemand getBufferDemand(const TrafficFlowEndpoint &ep) const {
    return buffer_demands_.at(ep); // bounds-checked
  }
  [[nodiscard]] bool isSelfLoop() const { return is_self_loop_; }
};

using TrafficFlowGraphBase = utils::GraphClass<TrafficFlowEndpoint, TrafficFlow,
                                               utils::GraphTopology::Any>;

class TrafficFlowGraph : public TrafficFlowGraphBase {
  // TrafficFlowGraph is a hypergraph that contains traffic flow endpoints and
  // traffic flows (hyperedges). It is used to represent the network topology
  // and the traffic flows in the network.
  // TODO: analyze the TrafficFlowGraph to guide the router
private:
  class BufferDemands {
  private:
    std::vector<TrafficFlow::BufferDemand> buffer_demands_;

  public:
    BufferDemands() = delete;
    BufferDemands(const std::vector<TrafficFlow::BufferDemand> &buffer_demands)
        : buffer_demands_(buffer_demands) {}
    TrafficFlow::BufferDemand getSourceBufferDemand() const {
      return buffer_demands_.at(0); // bounds-checked
    }
    TrafficFlow::BufferDemand getSinkBufferDemand(const size_t index) const {
      return buffer_demands_.at(index + 1); // bounds-checked
    }
  };

  using TrafficFlowPtr = std::shared_ptr<TrafficFlow>;
  using TrafficFlowPtrList = std::vector<TrafficFlowPtr>;

  class TrafficFlowLinking {
  public:
    using ID = std::string;

  private:
    ID id_;
    TrafficFlowPtrList link_from_tfs_;
    TrafficFlowPtrList link_to_tfs_;

  public:
    TrafficFlowLinking() = delete;
    TrafficFlowLinking(ID id, const TrafficFlowPtrList &from_tfs,
                       const TrafficFlowPtrList &to_tfs)
        : id_(id), link_from_tfs_(from_tfs), link_to_tfs_(to_tfs) {}
    ID getId() const { return id_; }
    TrafficFlowPtrList getFromTrafficFlows() const { return link_from_tfs_; }
    TrafficFlowPtrList getToTrafficFlows() const { return link_to_tfs_; }
  };
  std::vector<TrafficFlowLinking> linked_tfs_;

public:
  TrafficFlowGraph() = default;

  TrafficFlowGraph(
      const std::initializer_list<
          std::tuple<TrafficFlowEndpoint, TrafficFlowEndpoint, BufferDemands>>
          &tf_ep_list) {
    for (const auto &tuple : tf_ep_list) {
      const auto &[src, sink, buffer_demands] = tuple;
      addTrafficFlow(src, std::vector{sink}, buffer_demands);
    }
  }

  TrafficFlowGraph(
      const std::initializer_list<std::tuple<
          TrafficFlowEndpoint, std::vector<TrafficFlowEndpoint>, BufferDemands>>
          &tf_ep_list) {
    for (const auto &tuple : tf_ep_list) {
      const auto &[src, sink, buffer_demands] = tuple;
      addTrafficFlow(src, sink, buffer_demands);
    }
  }

  TrafficFlowPtr addTrafficFlow(TrafficFlowEndpoint source,
                                TrafficFlowEndpoint sink,
                                BufferDemands buffer_demands) {
    // Single sink case (unicast)
    return addTrafficFlow(source, std::vector{sink}, buffer_demands);
  }

  TrafficFlowPtr addTrafficFlow(TrafficFlowEndpoint source,
                                std::vector<TrafficFlowEndpoint> sinks,
                                BufferDemands buffer_demands) {
    // HyperEdges have different ids even if the source and sink are the same
    // If not specified, the id is set to the number of edges in the graph
    return addTrafficFlow(std::to_string(getNumEdges()), source, sinks,
                          buffer_demands);
  }

  TrafficFlowPtr addTrafficFlow(TrafficFlow::ID tf_id,
                                TrafficFlowEndpoint source,
                                std::vector<TrafficFlowEndpoint> sinks,
                                BufferDemands buffer_demands) {
    std::vector<std::shared_ptr<TrafficFlowEndpoint>> sink_ptrs;
    for (const auto &sink : sinks) {
      sink_ptrs.push_back(addNode(sink));
    }

    const std::shared_ptr<TrafficFlow> edge_ptr =
        TrafficFlowGraphBase::addEdge(addNode(source), sink_ptrs);

    edge_ptr->setId(tf_id);

    edge_ptr->setUniqueName(
        std::format("tf(|{}|{}|,{})", source.getName(), utils::getName(sinks),
                    utils::concateAttributes(edge_ptr->getAttributes())));

    edge_ptr->setBufferDemand(source, buffer_demands.getSourceBufferDemand());
    for (size_t i = 0; i < sinks.size(); ++i) {
      edge_ptr->setBufferDemand(sinks.at(i),
                                buffer_demands.getSinkBufferDemand(i));
    }

    return edge_ptr;
  }

  // TODO: consider using ids to store links in RoutingNetList
  void addTrafficFlowLink(TrafficFlowLinking::ID tf_link_id,
                          const std::vector<TrafficFlow::ID> &link_from_tf_ids,
                          const std::vector<TrafficFlow::ID> &link_to_tf_ids) {
    const auto &edge_views = getEdgeViews();

    auto transform_tf_ids_to_tf_ptrs =
        [&edge_views](const std::vector<TrafficFlow::ID> &tf_ids) {
          TrafficFlowPtrList tf_ptrs;
          for (const auto &tf_id : tf_ids) {
            if (const auto it =
                    std::ranges::find_if(edge_views,
                                         [tf_id](const auto &edge_ptr) {
                                           return edge_ptr->getId() == tf_id;
                                         });
                it != edge_views.end()) {
              tf_ptrs.emplace_back(*it);
            } else {
              throw std::runtime_error(
                  std::format("Traffic flow with id {} not found", tf_id));
            }
          }
          return tf_ptrs;
        };

    linked_tfs_.emplace_back(tf_link_id,
                             transform_tf_ids_to_tf_ptrs(link_from_tf_ids),
                             transform_tf_ids_to_tf_ptrs(link_to_tf_ids));
  }

  std::vector<TrafficFlowEndpoint> getNodes() const {
    std::vector<TrafficFlowEndpoint> result;
    for (const auto &node_ptr : getNodeViews()) {
      result.push_back(*node_ptr);
    }
    return result;
  }

  TrafficFlow getTrafficFlow(const TrafficFlow::ID tf_id) const {
    // Returns the traffic flow with the given id
    for (const auto &edge_ptr : getEdgeViews()) {
      if (edge_ptr->getId() == tf_id) {
        return *edge_ptr;
      }
    }
    throw std::out_of_range(
        std::format("Traffic flow with id {} not found", tf_id));
  }

  // TODO: consider make it consistent of using either pointers or instances
  const std::vector<TrafficFlowLinking> &getTrafficFlowLinkings() const {
    return linked_tfs_;
  }
};

} // namespace base

#endif
