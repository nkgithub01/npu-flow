#ifndef _BASE_ROUTING_RESOURCE_GRAPH_HPP_
#define _BASE_ROUTING_RESOURCE_GRAPH_HPP_

#include <memory>
#include <string>
#include <vector>

#include "utils/graph.hpp"
#include "utils/misc.hpp"

namespace base {

using utils::NamedClass;

class RRNode : public NamedClass, public utils::GraphNode {
  // RRNode is a node in the routing resource graph. It represents a routing
  // resource such as a switch or a endpoint. It can be created either by name
  // or by ID. If using ID, the name will be set to the `rr_node_{ID}`.
public:
  RRNode() = delete;

  explicit RRNode(const std::string &name)
      : NamedClass(std::format(
            "rr_{}", name) /*shorter prefix for better visualization*/) {}

  // TODO: consider removing this constructor to avoid confusion between name
  // and ID
  explicit RRNode(int id)
      : NamedClass(id < 0 ? "invalid_rr_node" : std::format("rr_node_{}", id)) {
  }

  std::string getID() const {
    return getName().substr(std::string("rr_").length());
  }

  // Serialization
  std::string serialize() const { return getID(); }
  static RRNode deserialize(const std::string &serialized_str) {
    return RRNode{serialized_str};
  }
};

const RRNode kInvalidRRNode(-1);

// TODO: move this into RREdge class as RREdge::Type
enum class RREdgeType {
  DontCare = -1,
  IntraTileKernelLinking,
  NeighborSharingOnSource,
  NeighborSharingOnSink,
  CircuitSwitching,
  PacketSwitching,
};

inline std::string getName(RREdgeType type) {
  switch (type) {
  case RREdgeType::IntraTileKernelLinking:
    return "intra_tile";
  case RREdgeType::NeighborSharingOnSource:
    return "nbr_src";
  case RREdgeType::NeighborSharingOnSink:
    return "nbr_sink";
  case RREdgeType::CircuitSwitching:
    return "cct";
  case RREdgeType::PacketSwitching:
    return "pkt";
  default:
    return "none";
  }
}

class RREdgeCapacity {
public:
  enum class SharingBehavior { Shared, Exclusive };
  enum class ConsolidationBehavior { Consolidated, NonConsolidated };

private:
  size_t capacity_;
  SharingBehavior sharing_;
  ConsolidationBehavior consolidation_;

public:
  static constexpr size_t kInfiniteCapacity = 1e9f;
  RREdgeCapacity()
      : capacity_(kInfiniteCapacity), sharing_(SharingBehavior::Exclusive),
        consolidation_(ConsolidationBehavior::NonConsolidated) {}
  RREdgeCapacity(size_t capacity, SharingBehavior sharing,
                 ConsolidationBehavior consolidation)
      : capacity_(capacity), sharing_(sharing), consolidation_(consolidation) {}

  size_t getValue() const { return capacity_; }
  SharingBehavior getSharingBehavior() const { return sharing_; }
  ConsolidationBehavior getConsolidationBehavior() const {
    return consolidation_;
  }

  [[nodiscard]] bool is(SharingBehavior sharing) const {
    return sharing_ == sharing;
  }
  [[nodiscard]] bool is(ConsolidationBehavior consolidation) const {
    return consolidation_ == consolidation;
  }
  [[nodiscard]] bool isInfinite() const {
    return capacity_ == kInfiniteCapacity;
  }
};

class RREdge
    : public NamedClass,
      public utils::GraphEdge<RRNode, utils::GraphEdgeType::RegularEdge> {
  // RREdge is an edge in the routing resource graph. It connects two RRNodes
  // and has a type, capacity (the maximum number of connections that can be
  // made through this edge), and cost associated with it.
private:
  // TODO: multiple edges between two nodes
  RREdgeType type_;
  RREdgeCapacity capacity_;
  float cost_;

public:
  RREdge() = delete;
  RREdge(std::shared_ptr<RRNode> from, std::shared_ptr<RRNode> to)
      : NamedClass("rr_edge_unique_name_uninitialized"),
        GraphEdge<RRNode, utils::GraphEdgeType::RegularEdge>(from, to) {}

  void setType(RREdgeType type) {
    type_ = type;
    setAttribute("type", ::base::getName(type));
  }

  void setCapacity(RREdgeCapacity capacity) {
    capacity_ = capacity;

    std::string sharing_str;
    if (capacity.is(RREdgeCapacity::SharingBehavior::Shared)) {
      sharing_str = "shared";
    } else if (capacity.is(RREdgeCapacity::SharingBehavior::Exclusive)) {
      sharing_str = "exclusive";
    } else {
      throw std::runtime_error("Unknown sharing behavior for RREdgeCapacity");
    }

    std::string consolidation_str;
    if (capacity.is(RREdgeCapacity::ConsolidationBehavior::Consolidated)) {
      consolidation_str = "cons";
    } else if (capacity.is(
                   RREdgeCapacity::ConsolidationBehavior::NonConsolidated)) {
      consolidation_str = "non_cons";
    } else {
      throw std::runtime_error(
          "Unknown consolidation behavior for RREdgeCapacity");
    }

    setAttribute("cap", std::format("{}({},{})",
                                    (capacity.isInfinite()
                                         ? "inf"
                                         : std::to_string(capacity.getValue())),
                                    sharing_str, consolidation_str));
  }

  void setCost(float cost) {
    cost_ = cost;
    setAttribute("cost", std::format("{:.2f}", cost));
  }

  RRNode getFromNode() const { return *getFromNodePtr(); }
  RRNode getToNode() const { return *getToNodePtr(); }
  RREdgeType getType() const { return type_; }
  RREdgeCapacity getCapacity() const { return capacity_; }
  float getCost() const { return cost_; }

  friend std::ostream &operator<<(std::ostream &os, const RREdge &edge) {
    os << std::format("RREdge: from {} to {}, type={}, capacity={}, cost={}",
                      edge.getFromNode().getName(), edge.getToNode().getName(),
                      edge.getAttribute("type"), edge.getAttribute("cap"),
                      edge.getCost());
    return os;
  }
};

template <utils::GraphTopology kTopo>
using RRGraphBaseVarients = utils::GraphClass<RRNode, RREdge, kTopo>;

using GridLikeRRGraphBase = RRGraphBaseVarients<utils::GraphTopology::Grid>;

class RRGraph : public GridLikeRRGraphBase /*TODO: make this configurable*/ {
private:
  using EdgePtr = std::shared_ptr<RREdge>;
  using EdgePtrVec = std::vector<EdgePtr>;
  using NodeEdgePtrsMap = utils::Lookup<RRNode, EdgePtrVec>;
  NodeEdgePtrsMap edges_entering_node_;
  NodeEdgePtrsMap edges_leaving_node_;

  std::vector<EdgePtrVec> shared_edge_groups_;
  // TODO: make the loopup key type the pointer?
  utils::Lookup<RREdge, size_t /*index*/> shared_edge_group_lookup_;

  void ensureEdgeEnteringOrLeavingNodeLookupExists(const RRNode &node) {
    if (!edges_entering_node_.contains(node)) {
      edges_entering_node_.emplace(node, EdgePtrVec{});
    }
    if (!edges_leaving_node_.contains(node)) {
      edges_leaving_node_.emplace(node, EdgePtrVec{});
    }
  }

public:
  RRGraph() = delete;
  RRGraph(size_t num_rows, size_t num_cols)
      : GridLikeRRGraphBase(num_rows, num_cols) {}

  // TODO: consider returning node pointer as in addEdge
  void addNode(RRNode node, size_t row_y, size_t col_x,
               bool is_grid_backbone_node) {
    GridLikeRRGraphBase::addNode(node, row_y, col_x, is_grid_backbone_node);
    ensureEdgeEnteringOrLeavingNodeLookupExists(node);
  }

  EdgePtr addEdge(RRNode from, RRNode to, RREdgeType type, size_t capacity,
                  RREdgeCapacity::ConsolidationBehavior edge_consolidation,
                  float cost_per_unit_before_consolidation) {
    if (capacity == 0) {
      // TODO: consider better handling of zero-capacity edges
      throw std::runtime_error("Cannot add edge with zero capacity from " +
                               from.getName() + " to " + to.getName());
    }

    const std::shared_ptr<RREdge> edge_ptr =
        GridLikeRRGraphBase::addEdge(getNodePtr(from), getNodePtr(to));

    edge_ptr->setType(type);
    edge_ptr->setCapacity({
        capacity,
        // Set SharingBehavior to Exclusive by default; if shared is intended,
        // use setEdgeSharing after adding the edges
        RREdgeCapacity::SharingBehavior::Exclusive,
        edge_consolidation,
    });
    // Note: the cost will be applied to each unit of capacity occupied before
    // edge consolidation (e.g., multiple packet switching nets are considered
    // as one unit)
    edge_ptr->setCost(cost_per_unit_before_consolidation);

    edge_ptr->setUniqueName(
        std::format("rr_edge(|{}|{}|,{})", from.getName(), to.getName(),
                    utils::concateAttributes(edge_ptr->getAttributes())));

    ensureEdgeEnteringOrLeavingNodeLookupExists(from);
    ensureEdgeEnteringOrLeavingNodeLookupExists(to);
    edges_leaving_node_.at(from).push_back(edge_ptr);
    edges_entering_node_.at(to).push_back(edge_ptr);

    return edge_ptr;
  }

  void setEdgeSharing(EdgePtrVec shared_edges, size_t shared_capacity) {
    shared_edge_groups_.push_back(shared_edges);
    size_t loopup_index = shared_edge_groups_.size() - 1;
    for (const auto &edge_ptr : shared_edges) {
      RREdgeCapacity old_capacity = edge_ptr->getCapacity();
      if (old_capacity.is(RREdgeCapacity::SharingBehavior::Shared)) {
        throw std::runtime_error(
            "Edge " + edge_ptr->getName() +
            " is already marked as shared. Cannot set sharing again.");
      }
      RREdgeCapacity new_capacity(shared_capacity,
                                  RREdgeCapacity::SharingBehavior::Shared,
                                  old_capacity.getConsolidationBehavior());
      edge_ptr->setCapacity(new_capacity);
      // TODO: find a more elegant way to update the name after changing
      // capacity
      edge_ptr->setUniqueName(
          std::format("rr_edge(|{}|{}|,{})", edge_ptr->getFromNode().getName(),
                      edge_ptr->getToNode().getName(),
                      utils::concateAttributes(edge_ptr->getAttributes())));
      shared_edge_group_lookup_.insert_or_assign(*edge_ptr, loopup_index);
    }
  }

  auto getNodePtrs() const { return getNodeViews(); }

  auto getEdgePtrs() const { return getEdgeViews(); } // router used only

  const EdgePtrVec &getEdgesEnteringNode(const RRNode &node) const {
    return edges_entering_node_.at(node); // ok even if no such edge exists
  }

  const EdgePtrVec &getEdgesLeavingNode(const RRNode &node) const {
    return edges_leaving_node_.at(node); // ok even if no such edge exists
  }

  const std::vector<EdgePtrVec> &getSharedEdgeGroups() const {
    return shared_edge_groups_;
  }

  const EdgePtrVec &getSharedEdgeGroupAtEdge(const RREdge &edge) const {
    if (!shared_edge_group_lookup_.contains(edge)) {
      throw std::runtime_error("Edge " + edge.getName() +
                               " is not part of any shared edge group.");
    }
    return shared_edge_groups_.at(shared_edge_group_lookup_.at(edge));
  }

  friend std::ostream &operator<<(std::ostream &os, const RRGraph &graph) {
    // TODO: consider generalizing this using utils::GraphClass
    // TODO: consider using a better format (e.g., JSON)
    os << "Routing Resource Graph:\n";
    os << "  Nodes:\n";
    for (const auto &node_ptr : graph.getNodePtrs()) {
      os << std::format("\t{}\n", node_ptr->getName());
    }
    os << "  Edges:\n";
    for (const auto &edge_ptr : graph.getEdgePtrs()) {
      os << std::format("\t{}\n", edge_ptr->getName());
    }
    os << "  Shared Edge Groups:\n";
    for (const auto &edge_group : graph.getSharedEdgeGroups()) {
      os << "\tGroup:\n";
      for (const auto &edge_ptr : edge_group) {
        os << std::format("\t\t{}\n", edge_ptr->getName());
      }
    }
    return os;
  }
};

} // namespace base

#endif
