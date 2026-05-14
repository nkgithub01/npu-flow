#ifndef _UTILS_VISUALIZATION_HPP_
#define _UTILS_VISUALIZATION_HPP_

#include <string>
#include <unordered_map>
#include <vector>

namespace utils {

struct ColorScheme {
  std::string fill, stroke;
  ColorScheme(std::string f, std::string s) : fill(f), stroke(s) {}
};

// Color schemes
const std::unordered_map<std::string, ColorScheme> highlighted_color_scheme = {
    // Borrowed from http://draw.io/
    {"red", {"#F8CECC", "#B85450"}},
    {"blue", {"#DAE8FC", "#6C8EBF"}},
    {"green", {"#D5E8D4", "#82B366"}},
    {"orange", {"#FFE6CC", "#D79B00"}},
    {"yellow", {"#FFF2CC", "#D6B656"}},
    {"purple", {"#E1D5E7", "#9673A6"}},
    {"grey", {"#F5F5F5", "#666666"}},
    // Custom colors
    {"steelblue", {"#E8F2FF", "steelblue3"}},
};

inline const std::string getStrokeColorLoop(const size_t index) {
  // Paul Tol's color blind friendly palette
  const std::vector<std::string> palette{
      "#EE6677", // red
      "#228833", // green
      "#4477AA", // blue
      "#66CCEE", // cyan
      "#AA3377", // purple
      "#EE99AA", // light red/pink
      "#77AADD", // light blue
      "#99DDFF", // light cyan
      "#44BB99", // teal
      "#99CCFF", // sky blue
      "#77CC77"  // light green
  };
  return palette.at(index % palette.size());
}

} // namespace utils

#endif