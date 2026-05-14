#pragma once

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>

#include <gtest/gtest.h>

namespace gtest_helpers {

inline void ExpectStringContains(std::string_view actual,
                                 std::string_view expected_substring) {
  EXPECT_NE(actual.find(expected_substring), std::string_view::npos)
      << "Expected string to contain '" << expected_substring << "' but got: "
      << actual;
}

inline void ExpectStringContainsAll(
    std::string_view actual,
    std::initializer_list<std::string_view> expected_substrings) {
  for (const std::string_view expected_substring : expected_substrings) {
    ExpectStringContains(actual, expected_substring);
  }
}

template <typename Exception = std::runtime_error, typename Fn>
void ExpectThrowsWithSubstrings(
    Fn &&fn,
    std::initializer_list<std::string_view> expected_substrings) {
  try {
    std::forward<Fn>(fn)();
    ADD_FAILURE() << "Expected exception of type '" << typeid(Exception).name()
                  << "' to be thrown";
    return;
  } catch (const Exception &ex) {
    ExpectStringContainsAll(ex.what(), expected_substrings);
  } catch (const std::exception &ex) {
    ADD_FAILURE() << "Expected exception of type '" << typeid(Exception).name()
                  << "' but caught std::exception: " << ex.what();
  } catch (...) {
    ADD_FAILURE() << "Expected exception of type '" << typeid(Exception).name()
                  << "' but caught a non-std::exception type";
  }
}

} // namespace gtest_helpers
