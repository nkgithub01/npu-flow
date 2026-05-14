#ifndef _UTILS_MISC_HPP_
#define _UTILS_MISC_HPP_

#include <algorithm>
#include <cctype>
#include <chrono>
#include <concepts>
#include <format>
#include <fstream>
#include <map>
#include <random>
#include <ranges>
#include <set>
#include <string>
#include <string_view>

namespace utils {

template <typename T, typename... U>
concept either = (std::same_as<T, U> || ...);

class NamedClass {
  friend class NamedClassHash;

private:
  std::string name_;

public:
  NamedClass() = delete;
  explicit constexpr NamedClass(const std::string &name) : name_(name) {}
  explicit NamedClass(int id)
      : name_(std::format("(unknown_type)id_{}", std::to_string(id))) {}
  virtual ~NamedClass() = default;

  void setUniqueName(const std::string &name) { name_ = name; }
  constexpr const std::string &getName() const { return name_; }

  auto operator<=>(const NamedClass &other) const = default;
};

// TODO: add specialization for std::format

class NamedClassHash {
public:
  size_t operator()(const NamedClass &c) const {
    return std::hash<std::string>()(c.name_);
  }
};

template <typename T>
  requires std::derived_from<T, NamedClass>
std::string getName(const T &obj) {
  return obj.getName();
}

template <std::ranges::range R>
  requires std::derived_from<std::ranges::range_value_t<R>, NamedClass>
std::string getName(const R &range, const std::string &delimiter = ",") {
  std::string result;
  for (const auto &obj : range) {
    if (!result.empty()) {
      result += delimiter;
    }
    result += obj.getName();
  }
  return result;
}

// TODO: investigate if using unordered_map/set would cause undeterministic
// behavior, and if they have huge performance benefits compared to map/set.

// template <typename KeyType, typename ValueType>
// using Lookup = std::unordered_map<
//     KeyType, ValueType,
//     std::conditional_t<std::derived_from<KeyType, NamedClass>,
//     NamedClassHash,
//                        std::hash<KeyType>>>;
template <typename KeyType, typename ValueType>
using Lookup = std::map<KeyType, ValueType>;

// template <typename ValueType>
// using Set = std::unordered_set<
//     ValueType, std::conditional_t<std::derived_from<ValueType, NamedClass>,
//                                   NamedClassHash, std::hash<ValueType>>>;
template <typename ValueType> using Set = std::set<ValueType>;

template <typename ValueType> using Vec = std::vector<ValueType>;

// TODO: make this a generic cast function using concepts
template <typename ValueType>
inline Vec<ValueType> castSetToVec(const Set<ValueType> &set) {
  utils::Vec<ValueType> vec;
  for (const auto &elem : set) {
    vec.push_back(elem);
  }
  return vec;
}

template <typename T>
  requires(std::convertible_to<T, std::string> || !std::ranges::range<T>)
std::string toString(const T &obj) {
  std::stringstream ss;
  ss << obj;
  return ss.str();
}

template <std::ranges::range R>
// R must not be a string to avoid ambiguity
  requires(!std::same_as<R, std::string>)
std::string toString(const R &range, const std::string &delimiter = ",",
                     const std::string &prefix = "",
                     const std::string &suffix = "") {
  std::string result;
  for (const auto &obj : range) {
    if (!result.empty()) {
      result += delimiter;
    }
    result += prefix + toString(obj) + suffix;
  }
  return result;
}

template <typename KeyType, typename ValueType>
  requires requires(KeyType key, ValueType val) {
    { utils::getName(key) } -> std::same_as<std::string>;
    { utils::getName(val) } -> std::same_as<std::string>;
  }
class LookupWrapper {
private:
  utils::Lookup<KeyType, ValueType> lookup_;

public:
  using key_type = KeyType;
  using value_type = ValueType;

  LookupWrapper() = default;
  LookupWrapper(const utils::Lookup<KeyType, ValueType> &lookup)
      : lookup_(lookup) {}

  void add(const KeyType &key, const ValueType &val) {
    if (lookup_.contains(key) && lookup_.at(key) != val) {
      throw std::runtime_error(std::format(
          "{} is already mapped to {}, cannot remap to {}", utils::getName(key),
          utils::getName(lookup_.at(key)), utils::getName(val)));
    }
    lookup_.insert_or_assign(key, val);
  }

  const ValueType &at(const KeyType &key) const { return lookup_.at(key); }

  ValueType &update(const KeyType &key) { return lookup_.at(key); }

  const auto &data() const { return lookup_; }

  [[nodiscard]] bool contains(const KeyType &key) const {
    return lookup_.contains(key);
  }

  [[nodiscard]] size_t size() const { return lookup_.size(); }

  [[nodiscard]] bool empty() const { return lookup_.empty(); }

  const utils::Set<KeyType> filter(const ValueType &val) const {
    utils::Set<KeyType> keys;
    for (const auto &[key, map_to_val] : lookup_) {
      if (map_to_val == val) {
        keys.insert(key);
      }
    }
    return keys;
  }

  const utils::Set<KeyType> keys() const {
    utils::Set<KeyType> keys;
    for (const auto &key : std::views::keys(lookup_)) {
      keys.insert(key);
    }
    return keys;
  }

  const utils::Vec<KeyType> keyVec() const {
    utils::Vec<KeyType> keys;
    for (const auto &key : std::views::keys(lookup_)) {
      keys.push_back(key);
    }
    return keys;
  }

  const utils::Set<ValueType> values() const {
    utils::Set<ValueType> values;
    for (const auto &val : std::views::values(lookup_)) {
      values.insert(val);
    }
    return values;
  }

  friend std::ostream &
  operator<<(std::ostream &os,
             const LookupWrapper<KeyType, ValueType> &lookup_wrapper) {
    for (const auto &[key, map_to_val] : lookup_wrapper.lookup_) {
      if constexpr (std::ranges::range<ValueType>) {
        os << std::format("{} -> [ {} ] (size={})\n", utils::getName(key),
                          utils::getName(map_to_val), map_to_val.size());
      } else {
        os << std::format("{} -> {}\n", utils::getName(key),
                          utils::getName(map_to_val));
      }
    }
    return os;
  }

  // Equality operator comparing two unordered maps (ignoring order)
  bool operator==(const LookupWrapper<KeyType, ValueType> &other) const {
    // Compare sizes first for quick rejection
    if (lookup_.size() != other.lookup_.size()) {
      return false;
    }
    // Check if all pairs in this lookup exist in the other lookup
    for (const auto &[key, val] : lookup_) {
      if (!other.lookup_.contains(key) || other.lookup_.at(key) != val) {
        return false;
      }
    }
    // Check if all pairs in the other lookup exist in this lookup
    for (const auto &[key, val] : other.lookup_) {
      if (!lookup_.contains(key) || lookup_.at(key) != val) {
        return false;
      }
    }
    // All pairs match
    return true;
  }
};

// Attributes can be printed as records in the DOT format for visualization
// TODO: use reserved attributes, e.g., __vis_color, __vis_weight, for styling
// TODO: use record-based nodes https://graphviz.org/doc/info/shapes.html#record
using AttributeMap = std::map<std::string, std::string>;

inline std::string concateAttributes(const AttributeMap &attrs) {
  std::string result;
  for (const auto &[key, value] : attrs) {
    if (key.starts_with("__vis_")) {
      // Skip reserved attributes for visualization
      // TODO: apply styling based on these attributes
      continue;
    }
    if (!result.empty()) {
      result += ",";
    }
    result += std::format("{}={}", key, value);
  }
  return result;
};

class AttributableClass {
private:
  AttributeMap attributes_;

public:
  AttributableClass() = default;
  virtual ~AttributableClass() = default;

  void setAttribute(const std::string &key, const std::string &value) {
    attributes_[key] = value; // overwrite if exists
  }
  std::string getAttribute(const std::string &key) const {
    return attributes_.at(key);
  }
  const AttributeMap &getAttributes() const { return attributes_; }
};

using RandomSeed = unsigned int;

class RandomNumberGenerator {
private:
  std::mt19937 gen;
  std::uniform_int_distribution<> distr;

public:
  RandomNumberGenerator(RandomSeed seed = std::random_device{}())
      : gen(seed), distr(std::numeric_limits<int>::min(),
                         std::numeric_limits<int>::max()) {}

  RandomNumberGenerator(int closed_interval_min, int close_interval_max,
                        RandomSeed seed)
      : gen(seed), distr(closed_interval_min, close_interval_max) {}

  int get() { return distr(gen); }

  std::pair<int, int> getClosedInterval() const {
    return {distr.a(), distr.b()};
  }
  int getMin() const { return distr.a(); }
  int getMax() const { return distr.b(); }

  void setSeed(RandomSeed seed) { gen.seed(seed); }
};

template <typename Resolution = std::chrono::nanoseconds>
class CumulativeTimer {
private:
  std::chrono::steady_clock::time_point begin_time_point_;
  Resolution cumulative_time_;

public:
  CumulativeTimer()
      : begin_time_point_(std::chrono::steady_clock::now()),
        cumulative_time_(0) {}

  void start() { begin_time_point_ = std::chrono::steady_clock::now(); }

  // TODO: considering renaming to accumulate() to avoid confusion which may
  // lead to misuse of the method causing double counting
  void pause() {
    cumulative_time_ += std::chrono::duration_cast<Resolution>(
        std::chrono::steady_clock::now() - begin_time_point_);
  }

  void reset() {
    cumulative_time_ = Resolution(0);
    begin_time_point_ = std::chrono::steady_clock::now();
  }

  double getSeconds() const {
    // std::chrono::duration<float> cast to seconds by default
    return std::chrono::duration<double>(cumulative_time_).count();
  }

  double getMilliSeconds() const {
    // std::chrono::duration<float> cast to seconds by default
    return std::chrono::duration<double, std::milli>(cumulative_time_).count();
  }

  double getMicroSeconds() const {
    // std::chrono::duration<float> cast to seconds by default
    return std::chrono::duration<double, std::micro>(cumulative_time_).count();
  }
};

// TODO: add concepts to restrict Timer type
template <typename Timer> void start(Timer &timer) { timer.start(); }
template <typename Timer> void pause(Timer &timer) { timer.pause(); }
template <typename Timer> void reset(Timer &timer) { timer.reset(); }
template <typename Timer, typename... Args>
void start(Timer &timer, Args &...args) {
  timer.start();
  start(args...);
}
template <typename Timer, typename... Args>
void pause(Timer &timer, Args &...args) {
  timer.pause();
  pause(args...);
}
template <typename Timer, typename... Args>
void reset(Timer &timer, Args &...args) {
  timer.reset();
  reset(args...);
}

inline std::string readFileToString(std::ifstream file) {
  if (!file.is_open()) {
    throw std::runtime_error("File stream is not open");
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

inline std::string readFileToString(const std::string &file_path) {
  return readFileToString(std::move(std::ifstream(file_path)));
}

// Trim from the start (in place)
inline void trimLeftInPlace(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
}

// Trim from the end (in place)
inline void trimRightInPlace(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

// Trim from both ends (in place)
inline void trimInPlace(std::string &s) {
  trimLeftInPlace(s);
  trimRightInPlace(s);
}

// Trim from both ends (copying)
inline std::string trim(const std::string &s) {
  auto ws_front = std::find_if_not(
      s.begin(), s.end(), [](unsigned char ch) { return std::isspace(ch); });
  auto ws_back = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char ch) {
                   return std::isspace(ch);
                 }).base();
  return (ws_back <= ws_front ? std::string() : std::string(ws_front, ws_back));
}

// Get the raw command-line string
inline std::string getCommandLineString(int argc, char **argv) {
  std::vector<std::string> all_args(argv, argv + argc);
  return utils::toString(all_args, " ");
}

// Get current timestamp with microsecond precision
// TODO: make the precision configurable
inline std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    now.time_since_epoch()) %
                1000000;

  std::tm tm_buf;
  localtime_r(&now_time_t, &tm_buf);

  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0')
      << std::setw(6) << now_us.count();
  return oss.str();
}

// Split a string by a delimiter character
inline std::vector<std::string> split(std::string_view str, char delimiter) {
  std::vector<std::string> tokens;
  size_t start = 0;
  size_t end = str.find(delimiter);

  while (end != std::string_view::npos) {
    tokens.emplace_back(str.substr(start, end - start));
    start = end + 1;
    end = str.find(delimiter, start);
  }

  tokens.emplace_back(str.substr(start));
  return tokens;
}

// Check if a floating-point number is mathematically an integer
template <typename T>
  requires std::floating_point<T>
[[nodiscard]] inline bool isFloatingPointAnInteger(T value) {
  T integer_part;
  return std::modf(value, &integer_part) == 0.0;
}

template <std::ranges::range R1, std::ranges::range R2>
inline void concateWithMove(R1 &dest, R2 &src) {
  dest.insert(dest.end(), std::make_move_iterator(src.begin()),
              std::make_move_iterator(src.end()));
}

} // namespace utils

#endif
