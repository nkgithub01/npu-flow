#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "ortools/math_opt/cpp/math_opt.h"

namespace math_opt = ::operations_research::math_opt;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

} // namespace

TEST(MathOptMilpTest, SolvesIntegerProgrammingExample) {
  math_opt::Model model("Integer programming example");

  const math_opt::Variable x = model.AddIntegerVariable(0.0, kInf, "x");
  const math_opt::Variable y = model.AddIntegerVariable(0.0, kInf, "y");

  model.AddLinearConstraint(x + 7 * y <= 17.5, "c1");
  model.AddLinearConstraint(x <= 3.5, "c2");
  model.Maximize(x + 10 * y);

  const absl::StatusOr<math_opt::SolveResult> result =
      Solve(model, math_opt::SolverType::kGscip);

  ASSERT_TRUE(result.status().ok());
  EXPECT_TRUE(result->termination.EnsureIsOptimalOrFeasible().ok());
  EXPECT_TRUE(result->termination.EnsureIsOptimal().ok());

  EXPECT_DOUBLE_EQ(result->objective_value(), 23.0);
  EXPECT_EQ(std::llround(result->variable_values().at(x)), 3);
  EXPECT_EQ(std::llround(result->variable_values().at(y)), 2);
}
