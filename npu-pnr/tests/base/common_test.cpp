#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "base/common.hpp"
#include "gtest_helpers.hpp"

namespace {

struct CustomSettings {
  int id;
  std::string name;
  double value;

  bool operator==(const CustomSettings &other) const {
    return id == other.id && name == other.name &&
           std::abs(value - other.value) < 1e-9;
  }
};

} // namespace

TEST(ConfigTest, SettingAndGettingBasicTypes) {
  base::Config config;

  config.set("int_key", 42);
  config.set("double_key", 3.14);
  config.set("bool_key", true);
  config.set("string_key", std::string("hello"));

  EXPECT_EQ(config.get<int>("int_key"), 42);
  EXPECT_DOUBLE_EQ(config.get<double>("double_key"), 3.14);
  EXPECT_TRUE(config.get<bool>("bool_key"));
  EXPECT_EQ(config.get<std::string>("string_key"), "hello");
}

TEST(ConfigTest, OverwritingExistingValues) {
  base::Config config;

  config.set("key", 10);
  EXPECT_EQ(config.get<int>("key"), 10);

  config.set("key", 20);
  EXPECT_EQ(config.get<int>("key"), 20);
}

TEST(ConfigTest, ExactTypeMatchingOnGetters) {
  base::Config config;
  config.set("int_key", 10);

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)config.get<size_t>("int_key"); },
      {"Type mismatch for config key 'int_key'"});
  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)config.getOrDefault<long>("int_key", 0); },
      {"Type mismatch for config key 'int_key'"});

  EXPECT_EQ(config.get<int>("int_key"), 10);
  EXPECT_EQ(config.getOrDefault("int_key", 0), 10);
}

TEST(ConfigTest, ErrorHandling) {
  base::Config config;

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)config.get<int>("nonexistent"); },
      {"Config key 'nonexistent' not found"});

  config.set("key", 42);
  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)config.get<double>("key"); },
      {"Type mismatch for config key 'key'"});
}

TEST(ConfigTest, GetOrDefaultBehavior) {
  base::Config config;

  EXPECT_EQ(config.getOrDefault("nonexistent", 100), 100);

  config.set("key", 42);
  EXPECT_EQ(config.getOrDefault("key", 100), 42);
  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)config.getOrDefault<double>("key", 3.14); },
      {"Type mismatch for config key 'key'"});
}

TEST(ConfigTest, KeyOperations) {
  base::Config config;

  EXPECT_FALSE(config.has("key"));
  config.set("key", 42);
  EXPECT_TRUE(config.has("key"));

  config.remove("key");
  EXPECT_FALSE(config.has("key"));

  config.set("k1", 1);
  config.set("k2", 2);
  config.clear();
  EXPECT_FALSE(config.has("k1"));
  EXPECT_FALSE(config.has("k2"));
}

TEST(ConfigTest, JsonSerialization) {
  base::Config config;
  config.set("number", 42);
  config.set("flag", true);
  config.set("name", std::string("test"));

  const std::string json = config.toJSON();
  EXPECT_NE(json.find("\"number\":42"), std::string::npos);
  EXPECT_NE(json.find("\"flag\":true"), std::string::npos);
  EXPECT_NE(json.find("\"name\":\"test\""), std::string::npos);

  base::Config empty;
  EXPECT_EQ(empty.toJSON(), "{}");

  config.set("unsupported", 'c');
  EXPECT_THROW(config.toJSON(), std::runtime_error);
}

TEST(ConfigTest, CustomStructStorage) {
  base::Config config;
  const CustomSettings settings{42, "test", 3.14};
  config.set("custom", settings);

  EXPECT_EQ(config.get<CustomSettings>("custom"), settings);

  const CustomSettings default_settings{0, "default", 0.0};
  EXPECT_EQ(config.getOrDefault("missing", default_settings),
            default_settings);
}

TEST(ConfigTest, NestedConfigSingleLevel) {
  base::Config config;
  base::Config inner;
  inner.set("inner_int", 100);
  inner.set("inner_string", std::string("nested"));

  config.set("nested", inner);

  const auto retrieved = config.get<base::Config>("nested");
  EXPECT_EQ(retrieved.get<int>("inner_int"), 100);
  EXPECT_EQ(retrieved.get<std::string>("inner_string"), "nested");
}

TEST(ConfigTest, NestedConfigMultipleLevels) {
  base::Config config;
  base::Config level3;
  level3.set("deep", 999);

  base::Config level2;
  level2.set("mid", 42);
  level2.set("level3", level3);

  config.set("level2", level2);

  const auto l2 = config.get<base::Config>("level2");
  EXPECT_EQ(l2.get<int>("mid"), 42);

  const auto l3 = l2.get<base::Config>("level3");
  EXPECT_EQ(l3.get<int>("deep"), 999);
}

TEST(ConfigTest, NestedConfigWithMixedTypes) {
  base::Config config;
  base::Config inner;
  inner.set("param", 123);

  const CustomSettings custom{7, "inner_custom", 7.89};
  inner.set("custom", custom);

  config.set("outer_int", 999);
  config.set("inner_config", inner);

  EXPECT_EQ(config.get<int>("outer_int"), 999);
  const auto nested = config.get<base::Config>("inner_config");
  EXPECT_EQ(nested.get<int>("param"), 123);
  EXPECT_EQ(nested.get<CustomSettings>("custom"), custom);
}
