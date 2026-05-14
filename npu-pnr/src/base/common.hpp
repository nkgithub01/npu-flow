#ifndef _BASE_COMMON_HPP_
#define _BASE_COMMON_HPP_

#include <any>
#include <format>
#include <ostream>
#include <unordered_map>
#include <utility>

#include "nlohmann/json.hpp"

#include "utils/misc.hpp"

namespace base {

class GridPosition {
private:
  std::pair<int, int> y_x_;

public:
  GridPosition() = delete;
  GridPosition(int row_y, int col_x) : y_x_{row_y, col_x} {}
  GridPosition(const std::string &row_y_col_x) {
    size_t comma_pos = row_y_col_x.find(',');
    if (comma_pos == std::string::npos) {
      throw std::runtime_error("Invalid grid position format: " + row_y_col_x);
    }
    int row_y = std::stoi(utils::trim(row_y_col_x.substr(0, comma_pos)));
    int col_x = std::stoi(utils::trim(row_y_col_x.substr(comma_pos + 1)));
    y_x_ = std::make_pair(row_y, col_x);
  }

  int getRowY() const { return y_x_.first; }
  int getColX() const { return y_x_.second; }
  std::pair<int, int> getRowYColXPair() const { return y_x_; }
  // TODO: consider a more consistant API
  std::string toString() const {
    return std::format("{},{}", getRowY(), getColX());
  }

  std::string toDetailedString() const {
    return std::format("(row_y={},col_x={})", getRowY(), getColX());
  }

  auto operator<=>(const GridPosition &) const = default;

  friend std::ostream &operator<<(std::ostream &os, const GridPosition &pos) {
    os << pos.toString();
    return os;
  }
};

class GridPositionOffset {
private:
  std::pair<int, int> dy_dx_;

public:
  GridPositionOffset() = delete;
  GridPositionOffset(int dy, int dx) : dy_dx_{dy, dx} {}

  int getDy() const { return dy_dx_.first; }
  int getDx() const { return dy_dx_.second; }
  std::pair<int, int> getDyDxPair() const { return dy_dx_; }

  auto operator<=>(const GridPositionOffset &) const = default;

  friend std::ostream &operator<<(std::ostream &os,
                                  const GridPositionOffset &offset) {
    os << std::format("(dy={},dx={})", offset.getDy(), offset.getDx());
    return os;
  }
};

class GridDimension {
private:
  std::pair<int, int> row_y_col_x_;

public:
  GridDimension() = delete;
  GridDimension(int num_rows, int num_cols)
      : row_y_col_x_{num_rows, num_cols} {}
  int getNumRows() const { return row_y_col_x_.first; }
  int getNumCols() const { return row_y_col_x_.second; }
  std::pair<int, int> getNumRowsColsPair() const { return row_y_col_x_; }
  std::string toString() const {
    return std::format("{},{}", getNumRows(), getNumCols());
  }
  auto operator<=>(const GridDimension &) const = default;
  friend std::ostream &operator<<(std::ostream &os, const GridDimension &dim) {
    os << dim.toString();
    return os;
  }
};

class Config {
private:
  std::unordered_map<std::string, std::any> params_;

public:
  Config() = default;

  // Set a config parameter. Overwrites existing value if key already exists.
  // Note: If using set with concrete types, make sure to use the exact same
  // type when getting the value later. Othewise, the type with be inferred
  // as int for integer literals, double for floating point literals, etc.
  template <typename T> Config &set(const std::string &key, const T &value) {
    params_[key] = value;
    return *this;
  }

  // Get a config parameter with type checking. Must use the exact same type
  // as when setting the value. Throws exception if key doesn't exist or type
  // mismatch occurs.
  template <typename T> T get(const std::string &key) const {
    auto it = params_.find(key);
    if (it == params_.end()) {
      throw std::runtime_error(std::format("Config key '{}' not found", key));
    }
    try {
      return std::any_cast<T>(it->second);
    } catch (const std::bad_any_cast &) {
      throw std::runtime_error(
          std::format("Type mismatch for config key '{}'", key));
    }
  }

  // Get a config parameter with default value if key doesn't exist. Throw
  // exception on type mismatch (no implicit conversion, e.g., int to size_t,
  // due to std::any_cast).
  // Note: If using getOrDefault with concrete types, make sure to use the exact
  // same type when setting the value. Otherwise, the type will be inferred as
  // int for integer literals, double for floating point literals, etc. based on
  // the default_value parameter.
  template <typename T>
  T getOrDefault(const std::string &key, const T &default_value) const {
    auto it = params_.find(key);
    if (it == params_.end()) {
      return default_value;
    }
    try {
      return std::any_cast<T>(it->second);
    } catch (const std::bad_any_cast &) {
      throw std::runtime_error(
          std::format("Type mismatch for config key '{}'", key));
    }
  }

  bool has(const std::string &key) const { return params_.contains(key); }

  void remove(const std::string &key) { params_.erase(key); }

  void clear() { params_.clear(); }

  // Compact JSON representation for telemetry storage
  std::string toJSON() const {
    nlohmann::json json;
    for (const auto &[key, value] : params_) {
      if (value.type() == typeid(int)) {
        json[key] = std::any_cast<int>(value);
      } else if (value.type() == typeid(double)) {
        json[key] = std::any_cast<double>(value);
      } else if (value.type() == typeid(bool)) {
        json[key] = std::any_cast<bool>(value);
      } else if (value.type() == typeid(std::string)) {
        json[key] = std::any_cast<std::string>(value);
      } else {
        throw std::runtime_error(
            "Unsupported config value type for JSON serialization");
      }
    }
    return json.is_null() ? "{}" : json.dump(-1 /*compact representation*/);
  }
};

enum class Verbosity {
  Silent = -2,  // no output at all
  Minimal = -1, // only critical stats at the end
  Normal = 0,   // print critical stats periodically
  Verbose = 1,  // print critical stats more frequently
  Debug = 2     // print detailed debug info as frequently as possible
};

inline Verbosity getVerbosityFromString(const std::string &v) {
  if (v == "silent") {
    return Verbosity::Silent;
  } else if (v == "minimal") {
    return Verbosity::Minimal;
  } else if (v == "normal") {
    return Verbosity::Normal;
  } else if (v == "verbose") {
    return Verbosity::Verbose;
  } else if (v == "debug") {
    return Verbosity::Debug;
  } else {
    throw std::invalid_argument("Unknown verbosity level: " + v);
  }
}

} // namespace base

#endif
