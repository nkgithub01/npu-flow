#ifndef _BASE_BUFFER_ALLOC_HPP_
#define _BASE_BUFFER_ALLOC_HPP_

#include "base/rr_graph.hpp"

namespace base {

enum class BufferType {
  Source, // Buffer allocated on source endpoint
  Sink,   // Buffer allocated on sink endpoint
};

// TODO: consider moving this into RREdge class
inline bool isBufferNeeded(BufferType type, const RREdge &edge) {
  // Returns 1 if the buffer is needed for the given edge and endpoint type
  switch (edge.getType()) {
  case RREdgeType::IntraTileKernelLinking:
    // Does matter allocate on source or sink for intra-tile linking, as long as
    // there is exactly one buffer allocated on the tile
    return (type == BufferType::Sink) ? 0 : 1;
  case RREdgeType::NeighborSharingOnSource:
    return (type == BufferType::Sink) ? 0 : 1;
  case RREdgeType::NeighborSharingOnSink:
    return (type == BufferType::Source) ? 0 : 1;
  case RREdgeType::CircuitSwitching:
    return 1;
  case RREdgeType::PacketSwitching:
    return 1;
  default:
    return 0;
  }
}

} // namespace base

#endif
