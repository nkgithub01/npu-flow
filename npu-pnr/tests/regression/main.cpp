#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "regression/test_utils.hpp"
#include "toml++/toml.hpp"

namespace {

struct RegressionConfig {
  std::string placer;
  std::string router;
  int max_placement_iterations;
};

struct RegressionCase {
  std::string name;
  std::string sanitized_name;
  std::filesystem::path path;
  RegressionConfig config;
  std::size_t expected_placement_size;
  double expected_routing_cost;
  double routing_cost_epsilon;
};

std::filesystem::path resolvePathRelativeToManifest(
    const std::filesystem::path &manifest_path,
    const std::filesystem::path &case_path) {
  if (case_path.is_absolute()) {
    return case_path.lexically_normal();
  }
  return (manifest_path.parent_path() / case_path).lexically_normal();
}

std::string sanitizeTestName(std::string_view raw_name) {
  std::string sanitized;
  bool previous_was_underscore = false;

  for (const unsigned char ch : raw_name) {
    if (std::isalnum(ch) != 0) {
      sanitized.push_back(static_cast<char>(ch));
      previous_was_underscore = false;
    } else if (!previous_was_underscore) {
      sanitized.push_back('_');
      previous_was_underscore = true;
    }
  }

  while (!sanitized.empty() && sanitized.back() == '_') {
    sanitized.pop_back();
  }

  if (sanitized.empty()) {
    sanitized = "Case";
  }
  if (std::isdigit(static_cast<unsigned char>(sanitized.front())) != 0) {
    sanitized = std::string("Case_") + sanitized;
  }

  return sanitized;
}

std::string getRequiredString(const toml::table &table, std::string_view key,
                              std::string_view context) {
  if (const auto value = table[key].value<std::string>()) {
    return *value;
  }
  throw std::runtime_error(
      std::format("{} must define a string '{}'", context, key));
}

int64_t getRequiredInteger(const toml::table &table, std::string_view key,
                           std::string_view context) {
  if (const auto value = table[key].value<int64_t>()) {
    return *value;
  }
  throw std::runtime_error(
      std::format("{} must define an integer '{}'", context, key));
}

double getRequiredDouble(const toml::table &table, std::string_view key,
                         std::string_view context) {
  if (const auto value = table[key].value<double>()) {
    return *value;
  }
  if (const auto value = table[key].value<int64_t>()) {
    return static_cast<double>(*value);
  }
  throw std::runtime_error(
      std::format("{} must define a numeric '{}'", context, key));
}

const toml::table &getRequiredTable(const toml::table &table,
                                    std::string_view key,
                                    std::string_view context) {
  if (const auto *child = table[key].as_table()) {
    return *child;
  }
  throw std::runtime_error(
      std::format("{} must define a table '{}'", context, key));
}

void validateRegressionConfig(const RegressionConfig &config,
                              std::string_view context) {
  static const std::vector<std::string> kSupportedPlacers = {
      "sa", "ls", "milp", "noop"};
  static const std::vector<std::string> kSupportedRouters = {"milp"};

  if (std::ranges::find(kSupportedPlacers, config.placer) ==
      kSupportedPlacers.end()) {
    throw std::runtime_error(std::format(
        "{} uses unsupported placer '{}'", context, config.placer));
  }
  if (std::ranges::find(kSupportedRouters, config.router) ==
      kSupportedRouters.end()) {
    throw std::runtime_error(std::format(
        "{} uses unsupported router '{}'", context, config.router));
  }
  if (config.max_placement_iterations < 0) {
    throw std::runtime_error(std::format(
        "{} must use a non-negative max_placement_iterations", context));
  }
}

RegressionCase parseRegressionCase(const toml::table &case_table,
                                   std::size_t case_index,
                                   const std::filesystem::path &manifest_path) {
  const std::string context = std::format("case[{}]", case_index);
  const std::string name = getRequiredString(case_table, "name", context);
  if (name.empty()) {
    throw std::runtime_error(std::format("{} must use a non-empty name",
                                         context));
  }

  const std::filesystem::path resolved_path = resolvePathRelativeToManifest(
      manifest_path, getRequiredString(case_table, "path", context));
  if (!std::filesystem::exists(resolved_path)) {
    throw std::runtime_error(std::format("{} path does not exist: {}", context,
                                         resolved_path.string()));
  }
  if (!std::filesystem::is_regular_file(resolved_path)) {
    throw std::runtime_error(std::format("{} path is not a regular file: {}",
                                         context, resolved_path.string()));
  }

  const auto &config_table = getRequiredTable(case_table, "config", context);
  RegressionConfig config{
      .placer = getRequiredString(config_table, "placer",
                                  std::format("{}.config", context)),
      .router = getRequiredString(config_table, "router",
                                  std::format("{}.config", context)),
      .max_placement_iterations =
          static_cast<int>(getRequiredInteger(config_table,
                                              "max_placement_iterations",
                                              std::format("{}.config", context))),
  };
  validateRegressionConfig(config, context);

  const int64_t expected_placement_size =
      getRequiredInteger(case_table, "expected_placement_size", context);
  if (expected_placement_size < 0) {
    throw std::runtime_error(std::format(
        "{} must use a non-negative expected_placement_size", context));
  }

  const double routing_cost_epsilon =
      getRequiredDouble(case_table, "routing_cost_epsilon", context);
  if (routing_cost_epsilon < 0.0) {
    throw std::runtime_error(
        std::format("{} must use a non-negative routing_cost_epsilon",
                    context));
  }

  return RegressionCase{
      .name = name,
      .sanitized_name = sanitizeTestName(name),
      .path = resolved_path,
      .config = std::move(config),
      .expected_placement_size =
          static_cast<std::size_t>(expected_placement_size),
      .expected_routing_cost =
          getRequiredDouble(case_table, "expected_routing_cost", context),
      .routing_cost_epsilon = routing_cost_epsilon,
  };
}

std::vector<RegressionCase>
loadRegressionCases(const std::filesystem::path &manifest_path) {
  if (!std::filesystem::exists(manifest_path)) {
    throw std::runtime_error(
        std::format("Regression manifest does not exist: {}",
                    manifest_path.string()));
  }

  const toml::table manifest = toml::parse_file(manifest_path.string());
  const auto *case_array = manifest["case"].as_array();
  if (case_array == nullptr) {
    throw std::runtime_error(std::format(
        "Regression manifest must define at least one [[case]] entry: {}",
        manifest_path.string()));
  }
  if (case_array->empty()) {
    throw std::runtime_error(std::format(
        "Regression manifest must not contain an empty [[case]] list: {}",
        manifest_path.string()));
  }

  std::vector<RegressionCase> cases;
  cases.reserve(case_array->size());

  for (std::size_t index = 0; index < case_array->size(); ++index) {
    const auto *case_table = (*case_array)[index].as_table();
    if (case_table == nullptr) {
      throw std::runtime_error(
          std::format("case[{}] must be a TOML table", index));
    }
    cases.push_back(parseRegressionCase(*case_table, index, manifest_path));
  }

  std::unordered_map<std::string, std::string> sanitized_name_to_original_name;
  for (const auto &regression_case : cases) {
    const auto [it, inserted] = sanitized_name_to_original_name.emplace(
        regression_case.sanitized_name, regression_case.name);
    if (!inserted) {
      throw std::runtime_error(std::format(
          "Duplicate regression case names after sanitization: '{}' and '{}' "
          "both map to '{}'",
          it->second, regression_case.name, regression_case.sanitized_name));
    }
  }

  return cases;
}

std::filesystem::path parseManifestPathFromCommandLine(
    int argc, char **argv, std::vector<char *> &filtered_argv) {
  std::filesystem::path manifest_path =
      std::filesystem::path(NPU_PNR_DEFAULT_REGRESSION_MANIFEST);
  filtered_argv.reserve(static_cast<std::size_t>(argc));
  filtered_argv.push_back(argv[0]);

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    constexpr std::string_view kFlag = "--regression_file";
    constexpr std::string_view kFlagWithEquals = "--regression_file=";

    if (arg == kFlag) {
      if (i + 1 >= argc) {
        throw std::runtime_error(
            "--regression_file requires a path argument");
      }
      manifest_path = argv[++i];
      continue;
    }
    if (arg.starts_with(kFlagWithEquals)) {
      const std::string_view value = arg.substr(kFlagWithEquals.size());
      if (value.empty()) {
        throw std::runtime_error(
            "--regression_file= requires a non-empty path value");
      }
      manifest_path = value;
      continue;
    }
    filtered_argv.push_back(argv[i]);
  }

  return std::filesystem::absolute(manifest_path).lexically_normal();
}

class RegressionCaseTest : public ::testing::Test {
public:
  explicit RegressionCaseTest(RegressionCase regression_case)
      : regression_case_(std::move(regression_case)) {}

protected:
  void TestBody() override {
    SCOPED_TRACE(std::format(
        "case='{}', path='{}', placer='{}', router='{}', max_iters={}",
        regression_case_.name, regression_case_.path.string(),
        regression_case_.config.placer, regression_case_.config.router,
        regression_case_.config.max_placement_iterations));

    test_utils::EndToEndRunner runner({
        .placer = regression_case_.config.placer,
        .router = regression_case_.config.router,
        .max_placement_iterations =
            regression_case_.config.max_placement_iterations,
    });
    const auto result = runner.run(regression_case_.path);

    EXPECT_EQ(result.placement.size(), regression_case_.expected_placement_size);
    EXPECT_EQ(result.routing.getLegality(),
              base::RoutingState::Legality::Legal);
    EXPECT_NEAR(result.routing.getRoutingCost(),
                regression_case_.expected_routing_cost,
                regression_case_.routing_cost_epsilon);
  }

private:
  RegressionCase regression_case_;
};

void registerRegressionTests(const std::vector<RegressionCase> &cases) {
  for (const auto &regression_case : cases) {
    ::testing::RegisterTest(
        "Regression", regression_case.sanitized_name.c_str(), nullptr,
        nullptr, __FILE__, __LINE__,
        [regression_case]() -> RegressionCaseTest * {
          return new RegressionCaseTest(regression_case);
        });
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::vector<char *> filtered_argv;
    const std::filesystem::path manifest_path =
        parseManifestPathFromCommandLine(argc, argv, filtered_argv);

    int filtered_argc = static_cast<int>(filtered_argv.size());
    ::testing::InitGoogleTest(&filtered_argc, filtered_argv.data());

    const auto cases = loadRegressionCases(manifest_path);
    registerRegressionTests(cases);
    return RUN_ALL_TESTS();
  } catch (const std::exception &ex) {
    std::cerr << "Failed to initialize regression tests: " << ex.what()
              << std::endl;
    return 1;
  }
}
