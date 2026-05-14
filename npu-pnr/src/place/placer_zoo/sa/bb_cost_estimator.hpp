#pragma once

#include <format>
#include <iostream>

#include "arch/npu/tile.hpp"
#include "base/abstraction.hpp"
#include "base/placement.hpp"
#include "base/routing/routing_net.hpp"
#include "base/routing/routing_state.hpp"
#include "utils/misc.hpp"

// TODO: this cost estimator is currently NPU-specific, so it should be refactored into the npu architecture namespace.
namespace place::placer_zoo::sa {

constexpr double kDefaultHPWLFactor = 4.0;
constexpr double kDefaultMulticastHPWLFactor = 3.0;
constexpr double kDefaultRoutingTrackCongestionPenaltyFactor = 2.0;
constexpr double kDefaultCoreCongestionPenaltyFactor = 10.0;

using arch::npu::npu_tiles::TileType;
using base::LogicalCore;
using base::PhysicalCore;
using base::Placement;
using base::RoutingState;

enum class Quadrant : size_t {
  NorthWest = 0,
  NorthEast = 1,
  SouthWest = 2,
  SouthEast = 3
};

struct BoundingBox {
  int min_row;
  int max_row;
  int min_col;
  int max_col;
  Quadrant direction_from_src_to_tgt;
};

struct netBBInfo {
  std::pair<int, int> src_yx;
  std::vector<std::pair<int, int>> tgt_yxs;
  BoundingBox overall_bb;
  std::vector<BoundingBox> src_tgt_pair_bb;
  bool is_multicast;
  std::vector<int> num_targets_in_each_quadrant = {0, 0, 0, 0};
};

enum class trackOrientation { North, South, West, East };
class routingTrackInfo {
public:
  int col_x_start, row_y_start, col_x_end, row_y_end; // endpoints of the routing track
  trackOrientation orientation;
  int capacity = 0;
  std::string probability_distribution;
  double utilization_expected_value = 0.0;

  routingTrackInfo() = default;
  routingTrackInfo(int y_start, int x_start, int y_end, int x_end, std::string prob_dist)
      : probability_distribution(prob_dist) {
    row_y_start = y_start;
    col_x_start = x_start;
    row_y_end = y_end;
    col_x_end = x_end;

    if (row_y_start == row_y_end){
      if (col_x_start < col_x_end){
        orientation = trackOrientation::East;
      } else if (col_x_start > col_x_end){
        orientation = trackOrientation::West;
      } else {
        throw std::runtime_error("Error: Invalid routing track endpoints. "
                                 "Routing tracks must not have the start "
                                 "and end point the same.");
      }
    } else if (col_x_start == col_x_end){
      if (row_y_start < row_y_end){
        orientation = trackOrientation::North;
      } else if (row_y_start > row_y_end){
        orientation = trackOrientation::South;
      } else {
        throw std::runtime_error("Error: Invalid routing track endpoints. "
                                 "Routing tracks must not have the start "
                                 "and end point the same.");
      }
    } else {
      throw std::runtime_error("Error: Invalid routing track endpoints. "
                                 "Routing tracks must be either horizontal "
                                 "or vertical.");
    }

    switch (orientation){
      case trackOrientation::North:
        capacity = 6;
        break;
      case trackOrientation::South:
        capacity = 4;
        break;
      case trackOrientation::West:
        capacity = 4;
        break;
      case trackOrientation::East:
        capacity = 4;
        break;
      default:
        throw std::runtime_error("Error: Invalid routing track orientation.");
    }
  }

  bool inBoundingBox(const BoundingBox &bb) const {
    return (bb.min_col <= col_x_start && col_x_start <= bb.max_col &&
            bb.min_col <= col_x_end   && col_x_end   <= bb.max_col &&
            bb.min_row <= row_y_start && row_y_start <= bb.max_row &&
            bb.min_row <= row_y_end   && row_y_end   <= bb.max_row);
  }
  
  // Calculate the utilization probability of each track given the
  // the shortest path from source to target within the bounding box is
  // uniformly likely to be used
  double getPathUniformUtilizationProbability(const BoundingBox &bb,
                            const std::vector<std::vector<double>>
                                &binomial_coefficient_lookup_table) {
    // Total number of paths from A to B is Comb(dX + dY, dX)
    int dX_AB = bb.max_col - bb.min_col;
    int dY_AB = bb.max_row - bb.min_row;
    double total_paths =
        binomial_coefficient_lookup_table[dX_AB + dY_AB][dX_AB];
    // Number of paths that go through this routing track P->Q is
    // Path(A->P)
    // * Path(Q->B), given that Ax <= Px <= Qx <= Bx and Ay <= Py <= Qy <=
    // By
    int dX_AP = col_x_start - bb.min_col;
    int dY_AP = row_y_start - bb.min_row;
    int dX_QB = bb.max_col - col_x_end;
    int dY_QB = bb.max_row - row_y_end;
    double num_path_A_to_P =
        binomial_coefficient_lookup_table[dX_AP + dY_AP][dX_AP];
    double num_path_Q_to_B =
        binomial_coefficient_lookup_table[dX_QB + dY_QB][dX_QB];
    double paths_through_track = num_path_A_to_P * num_path_Q_to_B;
    // Utilization probability is paths_through_track / total_paths
    return paths_through_track / total_paths;
  }

  // Calculate the utilization probability of this routing track
  // assuming each routing track in the bounding box has equal probability of being used
  double getChannelUniformUtilizationProbability(const BoundingBox &bb) {
    // Assume each routing channel has equal probability of being used
    // if the track is in the bounding box
    int dX_AB = bb.max_col - bb.min_col;
    int dY_AB = bb.max_row - bb.min_row;
    int total_distance = dX_AB + dY_AB;
    int num_routing_channel_in_bb = dX_AB*(dY_AB + 1) + dY_AB*(dX_AB + 1);
    return  static_cast<double>(total_distance) / static_cast<double>(num_routing_channel_in_bb);
  }

  // Calculate utilization probability of this routing track asuming the
  // track is in the bounding box bb
  double getUtilizationProbability(const BoundingBox &bb,
                                   const std::vector<std::vector<double>>
                                         &binomial_coefficient_lookup_table) {
    if (probability_distribution == "path_uniform") {
      return getPathUniformUtilizationProbability(bb, binomial_coefficient_lookup_table);
    } else if (probability_distribution == "channel_uniform") {
      return getChannelUniformUtilizationProbability(bb);
    } else {
      throw std::runtime_error(std::format(
        "Error: Unknown probability distribution model '{}'.",
        probability_distribution));
    }
  }

  void updateExpectedValue(const BoundingBox &bb,
                           const std::vector<std::vector<double>>
                                 &binomial_coefficient_lookup_table,
                           double factor = 1.0) {
    utilization_expected_value += factor * getUtilizationProbability(bb, binomial_coefficient_lookup_table);
  }

  double getExpectedValueOfCongestion() const {
    return utilization_expected_value;
  }

  bool isTrackCongested() const {
    return utilization_expected_value > capacity;
  }

  double getCongestionPenalty() const {
    if (isTrackCongested()) {
      return utilization_expected_value - capacity;
    } else {
      return 0.0;
    }
  }

};

class coreInfo {
public:
  int row_y;
  int col_x;
  arch::npu::npu_tiles::TileType core_type;
  int in_channel_usage = 0;
  int out_channel_usage = 0;
  int buffer_usage = 0;
  int lock_usage = 0;
  size_t channel_capacity;
  size_t buffer_capacity;
  size_t lock_capacity;

  coreInfo() = delete;
  coreInfo(int row, int col){
    row_y = row;
    col_x = col;
    if (row_y == 0) {
      arch::npu::npu_tiles::ShimTile core_tile(std::make_pair(row_y, col_x));
      core_type = core_tile.getTileType();
      channel_capacity = core_tile.getNumChannelsBetweenTileDMAandSwitchBox();
      buffer_capacity = core_tile.getMemoryCapacity();
      lock_capacity = core_tile.getLockCapacity();
    } else if (row_y == 1) {
      arch::npu::npu_tiles::MemoryTile core_tile(std::make_pair(row_y, col_x));
      core_type = core_tile.getTileType();
      channel_capacity = core_tile.getNumChannelsBetweenTileDMAandSwitchBox();
      buffer_capacity = core_tile.getMemoryCapacity();
      lock_capacity = core_tile.getLockCapacity();
    } else {
      arch::npu::npu_tiles::ComputeTile core_tile(std::make_pair(row_y, col_x));
      core_type = core_tile.getTileType();
      channel_capacity = core_tile.getNumChannelsBetweenTileDMAandSwitchBox();
      buffer_capacity = core_tile.getMemoryCapacity();
      lock_capacity = core_tile.getLockCapacity();
    }
    resetUsage();
  }

  void updateInputChannelUsage(int usage) {
    in_channel_usage += usage;
    if (in_channel_usage < 0) {
      throw std::runtime_error(std::format("Error: Core row:{} col:{} in_channel_usage cannot be negative.", row_y, col_x));
    }
  }

  void updateOutputChannelUsage(int usage) {
    out_channel_usage += usage;
    if (out_channel_usage < 0) {
      throw std::runtime_error(std::format("Error: Core row:{} col:{} out_channel_usage cannot be negative.", row_y, col_x));
    }
  }

  void updateBufferUsage(int usage) {
    buffer_usage += usage;
    if (buffer_usage < 0) {
      throw std::runtime_error(std::format("Error: Core row:{} col:{} buffer_usage cannot be negative.", row_y, col_x));
    }
  }

  void updateLockUsage(int usage) {
    lock_usage += usage;
    if (lock_usage < 0) {
      throw std::runtime_error(std::format("Error: Core row:{} col:{} lock_usage cannot be negative.", row_y, col_x));
    }
  }

  bool isResourceOverused() const {
    bool channel_overused = (in_channel_usage > channel_capacity) || (out_channel_usage > channel_capacity);
    bool buffer_overused = (buffer_usage > buffer_capacity);
    bool lock_overused = (lock_usage > lock_capacity);
    return channel_overused || buffer_overused || lock_overused;
  }

  int getResourceOverusageLevel() const {
    int resource_overusage_level = 0;
    if (in_channel_usage > channel_capacity) {
      resource_overusage_level += in_channel_usage - channel_capacity;
    }
    if (out_channel_usage > channel_capacity) {
      resource_overusage_level += out_channel_usage - channel_capacity;
    }
    if (buffer_usage > buffer_capacity) {
      resource_overusage_level += static_cast<int>(static_cast<double>(buffer_usage)/static_cast<double>(buffer_capacity));
    }
    if (lock_usage > lock_capacity) {
      resource_overusage_level += lock_usage - lock_capacity;
    }
    return resource_overusage_level;
  }

  void resetUsage() {
    in_channel_usage = 0;
    out_channel_usage = 0;
    buffer_usage = 0;
    lock_usage = 0;
  }
};

class BoundingBoxBasedCostEstimator {
private:

  void initializeLogicalCoreToXYCoordMapping(const base::Placement &placement) {
    current_l_core_to_x_y.clear();
    recorded_l_core_to_x_y.clear();
    for (const auto &[l_core, p_core] : placement.data()) {
      auto [row, col] = p_core_to_x_y.at(p_core);
      current_l_core_to_x_y[l_core] = {row, col};
      recorded_l_core_to_x_y[l_core] = {row, col};
    }
  }

  void populateBBInfoForNet(const base::RoutingNet &net, netBBInfo &net_bb_info){
    net_bb_info.is_multicast = net.isMulticast();
    // Find overall bounding box
    auto [src_row, src_col] = current_l_core_to_x_y.at(net.getStartCore());
    net_bb_info.overall_bb.min_row = src_row;
    net_bb_info.overall_bb.max_row = src_row;
    net_bb_info.overall_bb.min_col = src_col;
    net_bb_info.overall_bb.max_col = src_col;
    net_bb_info.src_yx = {src_row, src_col};
    // Source-target-pair bounding boxes
    for (const auto &target_core : net.getTargetCores()) {
      auto [tgt_row, tgt_col] = current_l_core_to_x_y.at(target_core);
      // Update overall bounding box
      net_bb_info.overall_bb.min_row =
          std::min(net_bb_info.overall_bb.min_row, tgt_row);
      net_bb_info.overall_bb.max_row =
          std::max(net_bb_info.overall_bb.max_row, tgt_row);
      net_bb_info.overall_bb.min_col =
          std::min(net_bb_info.overall_bb.min_col, tgt_col);
      net_bb_info.overall_bb.max_col =
          std::max(net_bb_info.overall_bb.max_col, tgt_col);
      net_bb_info.tgt_yxs.push_back({tgt_row, tgt_col});
      // Update source-target-pair bounding box
      BoundingBox st_bb;
      st_bb.min_row = std::min(src_row, tgt_row);
      st_bb.max_row = std::max(src_row, tgt_row);
      st_bb.min_col = std::min(src_col, tgt_col);
      st_bb.max_col = std::max(src_col, tgt_col);
      // Update number of targets in each quadrant
      if (tgt_row <= src_row && tgt_col <= src_col) {
        st_bb.direction_from_src_to_tgt = Quadrant::SouthWest;
        net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::SouthWest)]++;
      } else if (tgt_row <= src_row && tgt_col > src_col) {
        st_bb.direction_from_src_to_tgt = Quadrant::SouthEast;
        net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::SouthEast)]++;
      } else if (tgt_row > src_row && tgt_col <= src_col) {
        st_bb.direction_from_src_to_tgt = Quadrant::NorthWest;
        net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::NorthWest)]++;
      } else if (tgt_row > src_row && tgt_col > src_col) {
        st_bb.direction_from_src_to_tgt = Quadrant::NorthEast;
        net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::NorthEast)]++;
      }
      net_bb_info.src_tgt_pair_bb.push_back(st_bb);
    }
  }

  void initializeBoundingBoxInfoForNets(const base::RoutingNetList &netlist) {
    netBoundingBoxes.clear();
    for (const auto &net : netlist.getNets()) {
      netBBInfo net_bb_info;
      populateBBInfoForNet(net, net_bb_info);
      netBoundingBoxes.insert({net.getId(), net_bb_info});
    }
  }

  void initializeBinomialCoefficientLookupTable() {
    binomial_coefficient_lookup_table.clear();
    binomial_coefficient_lookup_table.reserve(kNumColumns + kNumRows);
    for (int n = 0; n < kNumColumns + kNumRows; n++) {
      std::vector<double> row;
      for (int k = 0; k <= n; k++) {
        if (k == 0 || k == n) {
          row.push_back(1.0);
        } else {
          double value = binomial_coefficient_lookup_table[n - 1][k - 1] +
                         binomial_coefficient_lookup_table[n - 1][k];
          row.push_back(value);
        }
      }
      binomial_coefficient_lookup_table.push_back(row);
    }
  }

  void populateRoutingTrackInfoForQuadrant(
    Quadrant direction,
    const BoundingBox &bb,
    const int num_targets_in_region,
    const double factor = 1.0
  ) {
    int start_row, start_col, end_row, end_col;
    int delta_row, delta_col;
    switch (direction) {
      case Quadrant::NorthEast:
        start_row = bb.min_row;
        start_col = bb.min_col;
        end_row = bb.max_row;
        end_col = bb.max_col;
        delta_row = 1;
        delta_col = 1;
      case Quadrant::NorthWest:
        start_row = bb.min_row;
        start_col = bb.max_col;
        end_row = bb.max_row;
        end_col = bb.min_col;
        delta_row = 1;
        delta_col = -1;
      case Quadrant::SouthWest:
        start_row = bb.max_row;
        start_col = bb.max_col;
        end_row = bb.min_row;
        end_col = bb.min_col;
        delta_row = -1;
        delta_col = -1;
      case Quadrant::SouthEast:
        start_row = bb.max_row;
        start_col = bb.min_col;
        end_row = bb.min_row;
        end_col = bb.max_col;
        delta_row = -1;
        delta_col = 1;
    }

    // Horizontal tracks
    for (int row = start_row; row != end_row+delta_row; row += delta_row) {
      for (int col = start_col; col != end_col; col += delta_col) {
        routing_tracks_info
          .at({row, col, row, col+delta_col})
          .updateExpectedValue(
            bb,
            binomial_coefficient_lookup_table,
            factor * calculateHPWLCorrectionFactor(num_targets_in_region)
          );
      }
    }
    // Vertical tracks
    for (int row = start_row; row != end_row; row += delta_row) {
      for (int col = start_col; col != end_col+delta_col; col += delta_col) {
        routing_tracks_info
          .at({row, col, row+delta_row, col})
          .updateExpectedValue(
            bb,
            binomial_coefficient_lookup_table,
            factor * calculateHPWLCorrectionFactor(num_targets_in_region)
          );
      }
    }
  }

  void populateRoutingTrackInfoForNet(const netBBInfo &net_bb_info, const double factor = 1.0) {
    if (net_bb_info.is_multicast) {
      BoundingBox quadrant_bb;
      // For multicast net, we calculate the utilization probability for each routing track
      // based on the bounding box of the source and targets in each quadrant.
      // For North East quadrant
      if (net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::NorthEast)] > 0) {
        quadrant_bb.min_row = net_bb_info.src_yx.first;
        quadrant_bb.min_col = net_bb_info.src_yx.second;
        quadrant_bb.max_row = net_bb_info.overall_bb.max_row;
        quadrant_bb.max_col = net_bb_info.overall_bb.max_col;
        populateRoutingTrackInfoForQuadrant(
          Quadrant::NorthEast,
          quadrant_bb,
          net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::NorthEast)],
          factor
        );
      }
      // For North West quadrant
      if (net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::NorthWest)] > 0) {
        quadrant_bb.min_row = net_bb_info.src_yx.first;
        quadrant_bb.min_col = net_bb_info.overall_bb.min_col;
        quadrant_bb.max_row = net_bb_info.overall_bb.max_row;
        quadrant_bb.max_col = net_bb_info.src_yx.second;
        populateRoutingTrackInfoForQuadrant(
          Quadrant::NorthWest,
          quadrant_bb,
          net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::NorthWest)],
          factor
        );
      }
      // For South West quadrant
      if (net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::SouthWest)] > 0) {
        quadrant_bb.min_row = net_bb_info.overall_bb.min_row;
        quadrant_bb.min_col = net_bb_info.overall_bb.min_col;
        quadrant_bb.max_row = net_bb_info.src_yx.first;
        quadrant_bb.max_col = net_bb_info.src_yx.second;
        populateRoutingTrackInfoForQuadrant(
          Quadrant::SouthWest,
          quadrant_bb,
          net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::SouthWest)],
          factor
        );
      }
      // For South East quadrant
      if (net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::SouthEast)] > 0) {
        quadrant_bb.min_row = net_bb_info.overall_bb.min_row;
        quadrant_bb.min_col = net_bb_info.src_yx.second;
        quadrant_bb.max_row = net_bb_info.src_yx.first;
        quadrant_bb.max_col = net_bb_info.overall_bb.max_col;
        populateRoutingTrackInfoForQuadrant(
          Quadrant::SouthEast,
          quadrant_bb,
          net_bb_info.num_targets_in_each_quadrant[static_cast<size_t>(Quadrant::SouthEast)],
          factor
        );
      }
    } else {
      // For unicast net, we calculate the utilization probability for each routing track
      // based on the bounding box of the source-target pair.
      populateRoutingTrackInfoForQuadrant(
        net_bb_info.src_tgt_pair_bb[0].direction_from_src_to_tgt,
        net_bb_info.src_tgt_pair_bb[0],
        1,
        factor
      );
    }
  }

  void initializeRoutingTrackInfo() {
    routing_tracks_info.clear();
    // Initialize horizontal routing track info
    for (int row_y = 0; row_y < kNumRows; row_y++) {
      for (int col_x = 0; col_x < kNumColumns - 1; col_x++) {
      routing_tracks_info[{row_y, col_x+1, row_y, col_x  }] = routingTrackInfo(row_y, col_x+1, row_y, col_x, cost_estimator_probability_distribution); // to West
      routing_tracks_info[{row_y, col_x,   row_y, col_x+1}] = routingTrackInfo(row_y, col_x, row_y, col_x+1, cost_estimator_probability_distribution); // to East
      }
    }
    // Initialize vertical routing track info
    for (int row_y = 0; row_y < kNumRows - 1; row_y++) {
      for (int col_x = 0; col_x < kNumColumns; col_x++) {
      routing_tracks_info[{row_y,   col_x, row_y+1, col_x}] = routingTrackInfo(row_y, col_x, row_y+1, col_x, cost_estimator_probability_distribution); // to North
      routing_tracks_info[{row_y+1, col_x, row_y,   col_x}] = routingTrackInfo(row_y+1, col_x, row_y, col_x, cost_estimator_probability_distribution); // to South
      }
    }

    // Update usage counts
    for (const auto &[net_id, net_bb_info] : netBoundingBoxes) {
      populateRoutingTrackInfoForNet(net_bb_info);
    }
  }

  void populateCoreInfoPerNet(
    const base::RoutingNet &net,
    const std::unordered_map<LogicalCore, std::pair<int, int>, utils::NamedClassHash> &l_core_to_x_y,
    int factor = 1 // This factor is used to determine whether we are adding (1) or subtracting (-1) the resource usage for the cores.
  ) {
    auto src_l_core = net.getStartCore();
    auto [src_row, src_col] = l_core_to_x_y.at(src_l_core);
    bool use_dma = false;
    for (const auto &target_core : net.getTargetCores()) {
      auto [tgt_row, tgt_col] = l_core_to_x_y.at(target_core);
      // Check if this net is using DMA based on whether the source and target cores are close to each other and whether they are both COMP tiles.
      if (std::abs(src_col - tgt_col) + std::abs(src_row - tgt_row) > 1 || 
        core_infos[src_row][src_col].core_type != arch::npu::npu_tiles::TileType::Compute || 
        core_infos[tgt_row][tgt_col].core_type != arch::npu::npu_tiles::TileType::Compute
      ) {
        core_infos[tgt_row][tgt_col].updateInputChannelUsage(factor);
        core_infos[tgt_row][tgt_col].updateBufferUsage(factor * net.getBufferDemand(target_core).getTotalBytes());
        core_infos[tgt_row][tgt_col].updateLockUsage(factor * 2 * net.getBufferDemand(target_core).depth);
        use_dma = true;
      }
      // TODO: need better model for buffer usage when using neighboring routing tracks. 
      // The buffer on src or dest choice is unknown at placement stage. 
      // Currently assume all buffers are on source for simplicity, 
      // which may overestimate the penalty for cores that are more likely to have buffers on the destination side.
    }
    // Update channel usage if at least one of the target cores is not using neighboring routing tracks
    if (use_dma) {
      core_infos[src_row][src_col].updateOutputChannelUsage(factor);
    }

    core_infos[src_row][src_col].updateBufferUsage(factor * net.getBufferDemand(src_l_core).getTotalBytes());
    core_infos[src_row][src_col].updateLockUsage(factor * 2 * net.getBufferDemand(src_l_core).depth);
  }


  void initializeCoreInfo(const base::RoutingNetList &netlist) {
    core_infos.clear();
    core_infos.reserve(kNumRows);
    for (int row = 0; row < kNumRows; row++) {
      std::vector<coreInfo> core_info_row;
      core_info_row.reserve(kNumColumns);
      for (int col = 0; col < kNumColumns; col++) {
        coreInfo c_info(row, col);
        core_info_row.push_back(c_info);
      }
      core_infos.push_back(core_info_row);
    }

    // Update core resource usage based on the linked nets first
    for (const auto &link : netlist.getNetLinks()){
      // Check the from nets and update the resource usage for the source cores. 
      for (const auto &net : link.getFromNets()) {
        populateCoreInfoPerNet(net, current_l_core_to_x_y);
      }

      // Check the to nets and update the resource usage for the target cores.
      // For net linking, the intermediate cores (which is the src core here) 
      // should not be double counted for buffer and lock usage, but should be counted for channel usage.
      for (const auto &net : link.getToNets()) {
        auto src_l_core = net.getStartCore();
        auto [src_row, src_col] = current_l_core_to_x_y.at(src_l_core);
        populateCoreInfoPerNet(net, current_l_core_to_x_y);
        // Since the source core is also a target core for the linked net,
        // we need to subtract the buffer and lock usage for the source core to avoid double counting
        core_infos[src_row][src_col].updateBufferUsage(-1 * net.getBufferDemand(src_l_core).getTotalBytes());
        core_infos[src_row][src_col].updateLockUsage(-1 * net.getBufferDemand(src_l_core).depth * 2);
      }
    }

    // Update core resource usage based on the non-linked nets
    for (const auto &net : netlist.getNonLinkedNets()) {
      populateCoreInfoPerNet(net, current_l_core_to_x_y);
    }
  }

  void initializeDataStructures(
    const base::RoutingNetList &netlist,
    const base::Placement &placement
  ) {
    initializeLogicalCoreToXYCoordMapping(placement);
    initializeBoundingBoxInfoForNets(netlist);
    if (enable_probabilistic_routing_congestion_estimation) {
      initializeBinomialCoefficientLookupTable();
      initializeRoutingTrackInfo();
    }
    if (enable_cost_legality_estimator) {
      initializeCoreInfo(netlist);
    }
  }

public:
  bool enable_probabilistic_routing_congestion_estimation;
  std::string cost_estimator_probability_distribution;
  bool enable_cost_legality_estimator;
  // The placement corresponding to the current state of the cost estimator.
  // This get updated when the cost estimator is updated with a new placement state, 
  // and is used for incremental update of the cost estimator when there are changes in the placement state.
  base::Placement recorded_placement;
  // Map the placement to coordinates
  std::unordered_map<base::PhysicalCore, std::pair<int, int>, utils::NamedClassHash> p_core_to_x_y; // p_core -> <row_y, col_x>
  mutable std::unordered_map<LogicalCore, std::pair<int, int>, utils::NamedClassHash> recorded_l_core_to_x_y; // l_core -> <row_y, col_x>
  mutable std::unordered_map<LogicalCore, std::pair<int, int>, utils::NamedClassHash> current_l_core_to_x_y; // l_core -> <row_y, col_x>
  mutable std::unordered_map<base::RoutingNet::ID, netBBInfo> netBoundingBoxes;
  // This is used to store the outdated bounding box info for the nets that are affected by the moved logical cores,
  // which will be used for incremental update of the routing_tracks_info. Since we need to subtract the old expected value
  // based on the outdated bounding box info before adding the new expected value based on the updated bounding box info for the affected nets,
  // we need to keep track of the outdated bounding box info for those nets.
  mutable std::unordered_map<base::RoutingNet::ID, netBBInfo> outdatedNetBoundingBoxes;
  // Will use heatmap-based route utilization probability model
  // Precompute the binomial coefficient lookup tables for combinatorial path
  // counting on grids The number of paths on a rectangular grid from a starting
  // point S = (x1,y1) to an ending point T = (x2,y2), using only right (R) and
  // up (U) steps (or right and down, depending on orientation), can be
  // calculated using binomial coefficients (combinations). Path(S->T) = C(dX +
  // dY, dX) = C(dX + dY, dY) = (dX + dY)! / (dX! * dY!) The number of path that
  // go through a specific edge Ph = (x,y) to Qh = (x+1,y) (horizontal edge) or
  // Pv = (x,y) to Qv = (x,y+1) (vertical edge) can be calculated as: Path(P->Q)
  // = Path (A->P) * Path (Q->B)
  std::vector<std::vector<double>> binomial_coefficient_lookup_table;
  mutable std::map<std::tuple<int/*y_start*/, int/*x_start*/, int/*y_end*/, int/*x_end*/>, routingTrackInfo> routing_tracks_info;
  mutable std::vector<std::vector<coreInfo>> core_infos; // 2D vector to store core info for each physical core, indexed by [row][col]

  int kNumColumns;
  int kNumRows;

  BoundingBoxBasedCostEstimator() = default;

  BoundingBoxBasedCostEstimator(
    const int num_columns, const int num_rows,
    bool enable_probabilistic_routing_congestion_estimation,
    std::string cost_estimator_probability_distribution,
    bool enable_cost_legality_estimator,
    const std::unordered_map<base::PhysicalCore, std::pair<int, int>, utils::NamedClassHash> &p_core_to_x_y_lookup,
    const base::RoutingNetList &netlist,
    const base::Placement &placement
  ) : kNumColumns(num_columns), kNumRows(num_rows),
      enable_probabilistic_routing_congestion_estimation(enable_probabilistic_routing_congestion_estimation),
      cost_estimator_probability_distribution(cost_estimator_probability_distribution),
      enable_cost_legality_estimator(enable_cost_legality_estimator),
      recorded_placement(placement),
      p_core_to_x_y(p_core_to_x_y_lookup)
  {
    initializeDataStructures(netlist, placement);
  }

  // For debugging and analysis purposes, print the binomial coefficient lookup table
  void printBinomialCoefficientLookupTable() const {
    if (!enable_probabilistic_routing_congestion_estimation) {
      std::cout << "Probabilistic routing congestion estimation is disabled, binomial coefficient lookup table is not initialized." << std::endl;
      return;
    }

    std::cout << "[SA] Binomial Coefficient Lookup Table:\n";
    for (int n = 0; n < kNumColumns + kNumRows; n++) {
      std::cout << "    ";
      for (int k = 0; k <= n; k++) {
        std::cout << std::format("Comb({:4}, {:4}) = {:.4f} ", n, k,
                                 binomial_coefficient_lookup_table[n][k]);
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }

  // Get the logical cores that are moved in the new placement compared to the recorded placement,
  // which will be used for incremental update of the cost estimator.
  void getMovedLogicalCores(
    const base::Placement &new_placement,
    std::vector<LogicalCore> &moved_l_cores
  ) {
    moved_l_cores.clear();
    for (const auto &[l_core, p_core] : new_placement.data()) {
      if (recorded_placement.at(l_core) != p_core) {
        moved_l_cores.push_back(l_core);
      }
    }
  }

  // Get affected nets due to the moved logical cores, which will be used for incremental update of the cost estimator.
  // Use set to avoid duplicate nets when multiple logical cores of the same net are moved.
  void getAffectedNets(
    const base::RoutingNetList &netlist,
    const std::vector<LogicalCore> &moved_l_cores,
    utils::Set<base::RoutingNet> &affected_nets
  ) {
    for (const auto &l_core : moved_l_cores) {
      auto nets = netlist.getNetsAtLogicalCore(l_core);
      affected_nets.insert(nets.begin(), nets.end());
    }
  }

  // Update the current logical core to XY mapping for the moved logical cores based on the new placement
  void updateCurrentLogicalCoreToXYMapping(
    const base::Placement &new_placement,
    const std::vector<LogicalCore> &moved_l_cores
  ) {
    for (const auto &l_core : moved_l_cores) {
      auto p_core = new_placement.at(l_core);
      auto [row, col] = p_core_to_x_y.at(p_core);
      current_l_core_to_x_y[l_core] = {row, col};
    }
  }

  // Update the bounding box info for the nets that are affected by the moved logical cores
  void updateBoundingBoxInfoForNets(
    const base::RoutingNetList &netlist,
    const utils::Set<base::RoutingNet> &affected_nets
  ) {
    // Update the bounding box info for the affected nets
    for (const auto &net : affected_nets) {
      netBBInfo net_bb_info;
      populateBBInfoForNet(net, net_bb_info);
      netBoundingBoxes[net.getId()] = net_bb_info;
    }
  }

  // For multi-cast nets, give a factor to HPWL if number of target > 3,
  // and the factor increase linearly as the number of targets increases,
  // which is based on the observation from the real netlist that the
  // WL of multi-cast nets tend to increase as the number of targets increases.
  double calculateHPWLCorrectionFactor(int num_targets) const {
    if (num_targets <= 3) {
      return 1.0;
    } else {
      return 1.0 + static_cast<double>(num_targets - 3) * 
             (kDefaultMulticastHPWLFactor - 1.0) / static_cast<double>(kNumRows * kNumColumns);
    }
  }

  // Calculate the HPWL cost for all nets based on the bounding box info
  double calculateHPWLCost() const {
    double HPWLCost = 0.0;
    for (const auto &[net_id, net_bb_info] : netBoundingBoxes) {
      double chan_x = 4.0; // each routing channel has 4 tracks in horizontal direction
      double chan_y_S2N = 6.0; // each routing channel has 6 tracks in vertical direction (from south to north targets)
      double chan_y_N2S = 4.0; // each routing channel has 4 tracks in vertical direction (from north to south targets)
      double hpwl =
          (net_bb_info.overall_bb.max_row - net_bb_info.src_yx.first) / chan_y_N2S +
          (net_bb_info.src_yx.first - net_bb_info.overall_bb.min_row) / chan_y_S2N +
          (net_bb_info.overall_bb.max_col - net_bb_info.overall_bb.min_col) / chan_x;
      int num_targets = net_bb_info.src_tgt_pair_bb.size();
      // For multi-cast nets, give a factor to HPWL if number of target > 3
      double net_cost = kDefaultHPWLFactor * hpwl * calculateHPWLCorrectionFactor(num_targets);
      HPWLCost += net_cost;
    }
    return HPWLCost;
  }

  // Get the outdated bounding box info for the nets that are affected by the moved logical cores,
  // which will be used for incremental update of the routing track info.
  void getOutdatedNetBoundingBoxes(const utils::Set<base::RoutingNet> &affected_nets) {
    outdatedNetBoundingBoxes.clear();
    for (const auto &net : affected_nets) {
      outdatedNetBoundingBoxes.insert({net.getId(), netBoundingBoxes.at(net.getId())});
    }
  }

  // Update the routing track info based on the updated bounding box info for the affected nets.
  void updateRoutingTrackInfo(const utils::Set<base::RoutingNet> &affected_nets) {
    for (const auto &net : affected_nets) {
      const auto &old_net_bb_info = outdatedNetBoundingBoxes.at(net.getId());
      populateRoutingTrackInfoForNet(old_net_bb_info, -1.0); // Subtract the expected value contributed by the old bounding box info for the net.
      const auto &new_net_bb_info = netBoundingBoxes.at(net.getId());
      populateRoutingTrackInfoForNet(new_net_bb_info); // Update the expected value with the new bounding box info for the net.
    }
  }

  // Calculate the routing congestion cost based on the expected congestion level for each routing track.
  double calculateRoutingCongestionCost() const {
    double congestion_cost = 0.0;
    for (const auto &[track_endpoints, track_info] : routing_tracks_info) {
      if (track_info.isTrackCongested()) {
        congestion_cost += track_info.getCongestionPenalty();
      }
    }
    return congestion_cost;
  }

  // Update the core info based on the updated logical core to XY mapping for the moved logical cores and the netlist.
  void updateCoreInfo(
    const base::RoutingNetList &netlist,
    const std::vector<LogicalCore> &moved_l_cores
  ) {
    // Get the affected links due to the moved logical cores.
    // For linked nets the intermediate cores should not be double counted for buffer and lock usage, 
    // but should be counted for channel usage. They need special handling in the core info update to avoid double counting.
    std::vector<base::RoutingNetLink> affected_links;
    for (const auto &l_core : moved_l_cores) {
      auto links = netlist.getNetLinksAtLogicalCore(l_core);
      affected_links.insert(affected_links.end(), links.begin(), links.end());
    }

    // Get the remaining affected nets that either not linked or link center is not at these moved logical cores.
    // They can be updated in the normal way without special handling.
    std::set<base::RoutingNet> affected_non_linked_nets;
    for (const auto &l_core : moved_l_cores) {
      auto nets = netlist.getNetWithNoLinkCenterAtLogicalCore(l_core);
      affected_non_linked_nets.insert(nets.begin(), nets.end());
    }

    // Update core resource usage based on the linked nets first
    for (const auto &link : affected_links){
      // Check the from nets and update the resource usage for the source cores. 
      for (const auto &net : link.getFromNets()) {
        populateCoreInfoPerNet(net, recorded_l_core_to_x_y, -1); // Subtract the resource usage in old placement.
        populateCoreInfoPerNet(net, current_l_core_to_x_y); // Update the new resource usage in current placement.

        affected_non_linked_nets.erase(net); // Remove from non-linked nets since they are already updated as linked nets.
      }

      // Check the to nets and update the resource usage for the target cores.
      // For net linking, the intermediate cores (which is the src core here) 
      // should not be double counted for buffer and lock usage, but should be counted for channel usage.
      for (const auto &net : link.getToNets()) {
        auto src_l_core = net.getStartCore();
        auto [old_src_row, old_src_col] = recorded_l_core_to_x_y.at(src_l_core);
        auto [new_src_row, new_src_col] = current_l_core_to_x_y.at(src_l_core);
        // Since the source core is also a target core for the linked net,
        // we need to add the buffer and lock usage for the source core to avoid double counting.
        // To avoid negative usage when the net is moved from a more congested core to a less congested core,
        // we add the resource usage for the source core before subtracting the resource usage for the target cores.
        core_infos[old_src_row][old_src_col].updateBufferUsage(net.getBufferDemand(src_l_core).getTotalBytes());
        core_infos[old_src_row][old_src_col].updateLockUsage(net.getBufferDemand(src_l_core).depth * 2);
        populateCoreInfoPerNet(net, recorded_l_core_to_x_y, -1); // Subtract the resource usage in old placement.
        
        populateCoreInfoPerNet(net, current_l_core_to_x_y); // Update the new resource usage in current placement.
        // Since the source core is also a target core for the linked net,
        // we need to subtract the buffer and lock usage for the source core to avoid double counting
        core_infos[new_src_row][new_src_col].updateBufferUsage(-1 * net.getBufferDemand(src_l_core).getTotalBytes());
        core_infos[new_src_row][new_src_col].updateLockUsage(-1 * net.getBufferDemand(src_l_core).depth * 2);

        affected_non_linked_nets.erase(net); // Remove from non-linked nets since they are already updated as linked nets.
      }
    }

    // Update core resource usage based on the non-linked nets
    for (const auto &net : affected_non_linked_nets) {
      populateCoreInfoPerNet(net, recorded_l_core_to_x_y, -1); // Subtract the resource usage in old placement.
      populateCoreInfoPerNet(net, current_l_core_to_x_y); // Update the new resource usage in current placement.
    }
  }

  // Calculate the core resource overusage cost based on the core info.
  double calculateCoreResourceOverusageCost() const {
    double overusage_cost = 0.0;
    for (int row = 0; row < kNumRows; row++) {
      for (int col = 0; col < kNumColumns; col++) {
        if (core_infos[row][col].isResourceOverused()) {
          overusage_cost += core_infos[row][col].getResourceOverusageLevel();
        }
      }
    }
    return overusage_cost;
  }

  // Update the cost estimator state based on the new placement with delta change provided
  void updateEstimatorStateWithPlacementDeltaChange(
      const base::RoutingNetList &netlist, const base::Placement &new_placement, const std::vector<base::LogicalCore> &moved_l_cores
  ) {
    // Update the internal data structures of the cost estimator based on the new placement.
    utils::Set<base::RoutingNet> affected_nets;
    getAffectedNets(netlist, moved_l_cores, affected_nets);
    updateCurrentLogicalCoreToXYMapping(new_placement, moved_l_cores);
    
    // Get the outdated bounding box info for the affected nets before updating the bounding box info for those nets,
    if (enable_probabilistic_routing_congestion_estimation) {
      getOutdatedNetBoundingBoxes(affected_nets);
    }
    // Update the bounding box info for the nets that are affected by the moved logical cores
    updateBoundingBoxInfoForNets(netlist, affected_nets);

    // Update congestion penalty related info based on the updated bounding box info for the affected nets.
    if (enable_probabilistic_routing_congestion_estimation) {
      updateRoutingTrackInfo(affected_nets);
    }

    // Update core resource usage info
    if (enable_cost_legality_estimator) {
      updateCoreInfo(netlist, moved_l_cores);
    }

    // Update the recorded placement and logical core to xy mapping at the end
    for (const auto &l_core : moved_l_cores) {
      recorded_placement.update(l_core) = new_placement.at(l_core);
      recorded_l_core_to_x_y[l_core] = current_l_core_to_x_y[l_core];
    }
  }

  // A faster, less accurate estimation of placement cost for SA moves with delta change provided
  // to avoid iterating through all the placed logical cores.
  std::pair<float, base::RoutingState::Legality> estimatePlacementCostWithPlacementDeltaChange(
      const base::RoutingNetList &netlist, const base::Placement &new_placement, const std::vector<base::LogicalCore> &moved_l_cores
  ) {
    // Update the estimator state
    updateEstimatorStateWithPlacementDeltaChange(netlist, new_placement, moved_l_cores);
    
    // Calculate the placement cost
    // HPWL cost
    double total_cost = calculateHPWLCost();

    // Congestion cost
    if (enable_probabilistic_routing_congestion_estimation) {
      double routing_congestion_cost = calculateRoutingCongestionCost();
      total_cost += routing_congestion_cost * kDefaultRoutingTrackCongestionPenaltyFactor;
    }

    // Estimate legality
    base::RoutingState::Legality legality = base::RoutingState::Legality::Legal;
    if (enable_cost_legality_estimator) {
      // Check for congested cores
      double core_resource_overusage_cost = calculateCoreResourceOverusageCost();
      if (core_resource_overusage_cost > 0) {
        legality = base::RoutingState::Legality::Congested;
      }
      total_cost += core_resource_overusage_cost * kDefaultCoreCongestionPenaltyFactor;
    }

    return {static_cast<float>(total_cost), legality};
  }

  // A faster, less accurate estimation of placement cost for SA moves
  std::pair<float, base::RoutingState::Legality> estimatePlacementCost(
      const base::RoutingNetList &netlist, const base::Placement &new_placement
  ) {
    // Update the internal data structures of the cost estimator based on the new placement.
    std::vector<LogicalCore> moved_l_cores;
    getMovedLogicalCores(new_placement, moved_l_cores);
    return estimatePlacementCostWithPlacementDeltaChange(netlist, new_placement, moved_l_cores);
  }
};

} // namespace place::placer_zoo::sa
