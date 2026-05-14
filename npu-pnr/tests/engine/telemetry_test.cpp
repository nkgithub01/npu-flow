#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "nlohmann/json.hpp"
#include "sqlite3.h"

#include "base/common.hpp"
#include "engine/telemetry.hpp"
#include "gtest_helpers.hpp"

namespace {

bool isDatabaseValid(const std::string &db_path) {
  return std::filesystem::exists(db_path);
}

int countEventsInDatabase(const std::string &db_path) {
  sqlite3 *db = nullptr;
  if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) !=
      SQLITE_OK) {
    return -1;
  }

  const char *sql = "SELECT COUNT(*) FROM events;";
  sqlite3_stmt *stmt = nullptr;
  int count = -1;

  if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
    if (sqlite3_step(stmt) == SQLITE_ROW) {
      count = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
  }

  sqlite3_close(db);
  return count;
}

bool hasEventInDatabase(const std::string &db_path,
                        const std::string &event_name) {
  sqlite3 *db = nullptr;
  if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) !=
      SQLITE_OK) {
    return false;
  }

  const char *sql = "SELECT COUNT(*) FROM events WHERE event_name = ?;";
  sqlite3_stmt *stmt = nullptr;
  bool found = false;

  if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
    sqlite3_bind_text(stmt, 1, event_name.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
      found = sqlite3_column_int(stmt, 0) > 0;
    }
    sqlite3_finalize(stmt);
  }

  sqlite3_close(db);
  return found;
}

struct EventProperties {
  std::string start_properties;
  std::string end_properties;
  bool found = false;
};

EventProperties getEventProperties(const std::string &db_path,
                                   const std::string &event_name) {
  EventProperties result;
  sqlite3 *db = nullptr;

  if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) !=
      SQLITE_OK) {
    return result;
  }

  const char *sql =
      "SELECT start_properties, end_properties FROM events WHERE event_name = "
      "? ORDER BY id DESC LIMIT 1;";
  sqlite3_stmt *stmt = nullptr;

  if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
    sqlite3_bind_text(stmt, 1, event_name.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
      result.found = true;

      const char *start_props =
          reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
      if (start_props) {
        result.start_properties = start_props;
      }

      const char *end_props =
          reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
      if (end_props) {
        result.end_properties = end_props;
      }
    }
    sqlite3_finalize(stmt);
  }

  sqlite3_close(db);
  return result;
}

class DatabaseCleanup {
private:
  std::string path_;

public:
  explicit DatabaseCleanup(std::string path) : path_(std::move(path)) {}

  ~DatabaseCleanup() {
    if (std::filesystem::exists(path_)) {
      std::filesystem::remove(path_);
    }
  }
};

base::Config makeTelemetryConfig(const std::string &db_path) {
  base::Config cfg;
  cfg.set("save_path", db_path);
  return cfg;
}

} // namespace

TEST(SQLiteBasedTelemetryDatabaseCreationTest,
     CreatesDatabaseWithConfiguredPath) {
  const std::string db_path = "test_telemetry_default.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    EXPECT_TRUE(isDatabaseValid(db_path));
  }
}

TEST(SQLiteBasedTelemetryDatabaseCreationTest, ThrowsOnInvalidDatabasePath) {
  base::Config cfg;
  cfg.set("save_path", std::string("/invalid/path/to/db.db"));

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { (void)engine::SQLiteBasedTelemetry(cfg); },
      {"Failed"});
}

TEST(SQLiteBasedTelemetryDatabaseCreationTest,
     CreatesEventsTableOnInitialization) {
  const std::string db_path = "test_telemetry_table.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  { engine::SQLiteBasedTelemetry telemetry(cfg); }

  EXPECT_EQ(countEventsInDatabase(db_path), 0);
}

TEST(SQLiteBasedTelemetryEventTrackingTest, SingleEventIsTracked) {
  const std::string db_path = "test_telemetry_events.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    telemetry.startEvent("test_event");
    telemetry.endEvent("test_event");
  }

  EXPECT_EQ(countEventsInDatabase(db_path), 1);
  EXPECT_TRUE(hasEventInDatabase(db_path, "test_event"));
}

TEST(SQLiteBasedTelemetryEventTrackingTest, MultipleEventsAreTracked) {
  const std::string db_path = "test_telemetry_events.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    telemetry.startEvent("event1");
    telemetry.endEvent("event1");
    telemetry.startEvent("event2");
    telemetry.endEvent("event2");
    telemetry.startEvent("event3");
    telemetry.endEvent("event3");
  }

  EXPECT_EQ(countEventsInDatabase(db_path), 3);
}

TEST(SQLiteBasedTelemetryEventTrackingTest, NestedEventsAreTracked) {
  const std::string db_path = "test_telemetry_events.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    telemetry.startEvent("outer");
    telemetry.startEvent("inner");
    telemetry.endEvent("inner");
    telemetry.endEvent("outer");
  }

  EXPECT_EQ(countEventsInDatabase(db_path), 2);
  EXPECT_TRUE(hasEventInDatabase(db_path, "outer"));
  EXPECT_TRUE(hasEventInDatabase(db_path, "inner"));
}

TEST(SQLiteBasedTelemetryEventTrackingTest,
     SameEventNameCanBeUsedMultipleTimes) {
  const std::string db_path = "test_telemetry_events.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    telemetry.startEvent("repeat");
    telemetry.endEvent("repeat");
    telemetry.startEvent("repeat");
    telemetry.endEvent("repeat");
  }

  EXPECT_EQ(countEventsInDatabase(db_path), 2);
}

TEST(SQLiteBasedTelemetryEventTrackingTest,
     EventsWithPropertiesAreTracked) {
  const std::string db_path = "test_telemetry_events.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    base::Config start_props;
    start_props.set("iteration", 1);
    base::Config end_props;
    end_props.set("result", std::string("success"));

    telemetry.startEvent("event_with_props", start_props);
    telemetry.endEvent("event_with_props", end_props);
  }

  EXPECT_EQ(countEventsInDatabase(db_path), 1);
}

TEST(SQLiteBasedTelemetryErrorHandlingTest, ThrowsOnMismatchedEventEnd) {
  const std::string db_path = "test_telemetry_errors.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  engine::SQLiteBasedTelemetry telemetry(cfg);
  telemetry.startEvent("event1");

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { telemetry.endEvent("event2"); }, {"Mismatched"});
}

TEST(SQLiteBasedTelemetryErrorHandlingTest,
     ThrowsWhenEndingEventThatWasNeverStarted) {
  const std::string db_path = "test_telemetry_errors.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  engine::SQLiteBasedTelemetry telemetry(cfg);

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { telemetry.endEvent("nonexistent"); }, {"Mismatched"});
}

TEST(SQLiteBasedTelemetryErrorHandlingTest, ThrowsOnWrongNestingOrder) {
  const std::string db_path = "test_telemetry_errors.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  engine::SQLiteBasedTelemetry telemetry(cfg);
  telemetry.startEvent("outer");
  telemetry.startEvent("inner");

  gtest_helpers::ExpectThrowsWithSubstrings<std::runtime_error>(
      [&]() { telemetry.endEvent("outer"); }, {"Mismatched"});
}

TEST(SQLiteBasedTelemetryErrorHandlingTest,
     ProperNestingMustBeMaintained) {
  const std::string db_path = "test_telemetry_errors.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  engine::SQLiteBasedTelemetry telemetry(cfg);
  telemetry.startEvent("event1");
  telemetry.startEvent("event2");
  telemetry.startEvent("event3");

  EXPECT_NO_THROW(telemetry.endEvent("event3"));
  EXPECT_NO_THROW(telemetry.endEvent("event2"));
  EXPECT_NO_THROW(telemetry.endEvent("event1"));
}

TEST(TelemetryTypeErasureTest,
     CreateTelemetryUsingStaticFactoryWithNoOpTelemetry) {
  base::Config cfg;
  auto telemetry = engine::Telemetry::create<engine::NoOpTelemetry>(cfg);

  EXPECT_NO_THROW(telemetry.startEvent("test"));
  EXPECT_NO_THROW(telemetry.endEvent("test"));
}

TEST(TelemetryTypeErasureTest,
     CreateTelemetryUsingStaticFactoryWithSQLiteBasedTelemetry) {
  const std::string db_path = "test_type_erasure.db";
  DatabaseCleanup cleanup(db_path);
  auto cfg = makeTelemetryConfig(db_path);

  auto telemetry = engine::Telemetry::create<engine::SQLiteBasedTelemetry>(cfg);
  EXPECT_NO_THROW(telemetry.startEvent("test"));
  EXPECT_NO_THROW(telemetry.endEvent("test"));
}

TEST(TelemetryTypeErasureTest, TelemetryCanBeMoved) {
  base::Config cfg;
  auto telemetry1 = engine::Telemetry::create<engine::NoOpTelemetry>(cfg);
  auto telemetry2 = std::move(telemetry1);

  EXPECT_NO_THROW(telemetry2.startEvent("test"));
  EXPECT_NO_THROW(telemetry2.endEvent("test"));
}

TEST(TelemetryTypeErasureTest, CanChainMoveOperations) {
  base::Config cfg;
  auto telemetry1 = engine::Telemetry::create<engine::NoOpTelemetry>(cfg);
  auto telemetry2 = std::move(telemetry1);
  auto telemetry3 = std::move(telemetry2);

  EXPECT_NO_THROW(telemetry3.startEvent("test"));
  EXPECT_NO_THROW(telemetry3.endEvent("test"));
}

TEST(TelemetryTypeErasureTest,
     MultipleTelemetryInstancesCanCoexistIndependently) {
  const std::string db_path1 = "test_multi1.db";
  const std::string db_path2 = "test_multi2.db";
  DatabaseCleanup cleanup1(db_path1);
  DatabaseCleanup cleanup2(db_path2);

  auto cfg1 = makeTelemetryConfig(db_path1);
  auto cfg2 = makeTelemetryConfig(db_path2);

  auto telemetry1 = engine::Telemetry::create<engine::SQLiteBasedTelemetry>(cfg1);
  auto telemetry2 = engine::Telemetry::create<engine::SQLiteBasedTelemetry>(cfg2);

  EXPECT_NO_THROW(telemetry1.startEvent("event1"));
  EXPECT_NO_THROW(telemetry2.startEvent("event2"));
  EXPECT_NO_THROW(telemetry1.endEvent("event1"));
  EXPECT_NO_THROW(telemetry2.endEvent("event2"));
}

TEST(TelemetryTypeErasureTest, DifferentTelemetryTypesCanCoexist) {
  const std::string db_path = "test_mixed.db";
  DatabaseCleanup cleanup(db_path);

  base::Config noop_cfg;
  auto sqlite_cfg = makeTelemetryConfig(db_path);

  auto noop_telemetry = engine::Telemetry::create<engine::NoOpTelemetry>(noop_cfg);
  auto sqlite_telemetry =
      engine::Telemetry::create<engine::SQLiteBasedTelemetry>(sqlite_cfg);

  EXPECT_NO_THROW(noop_telemetry.startEvent("noop_event"));
  EXPECT_NO_THROW(sqlite_telemetry.startEvent("sqlite_event"));
  EXPECT_NO_THROW(noop_telemetry.endEvent("noop_event"));
  EXPECT_NO_THROW(sqlite_telemetry.endEvent("sqlite_event"));
}

TEST(CreateTelemetryFactoryTest, CreatesNoOpTelemetryWhenDisabled) {
  base::Config cfg;
  cfg.set("enable", false);

  auto telemetry = engine::createTelemetry(cfg);
  EXPECT_NO_THROW(telemetry.startEvent("test"));
  EXPECT_NO_THROW(telemetry.endEvent("different_test"));
}

TEST(CreateTelemetryFactoryTest, CreatesNoOpTelemetryWhenEnableIsNotSet) {
  base::Config cfg;

  auto telemetry = engine::createTelemetry(cfg);
  EXPECT_NO_THROW(telemetry.startEvent("test"));
  EXPECT_NO_THROW(telemetry.endEvent("test"));
}

TEST(CreateTelemetryFactoryTest, CreatesSQLiteBasedTelemetryWhenEnabled) {
  const std::string db_path = "test_factory_enabled.db";
  DatabaseCleanup cleanup(db_path);

  base::Config cfg;
  cfg.set("enable", true);
  cfg.set("save_path", db_path);

  {
    auto telemetry = engine::createTelemetry(cfg);
    telemetry.startEvent("factory_test");
    telemetry.endEvent("factory_test");
  }

  EXPECT_TRUE(hasEventInDatabase(db_path, "factory_test"));
}

TEST(CreateTelemetryFactoryTest, FactoryCreatedTelemetryCanBeMoved) {
  base::Config cfg;
  cfg.set("enable", false);

  auto telemetry1 = engine::createTelemetry(cfg);
  auto telemetry2 = std::move(telemetry1);

  EXPECT_NO_THROW(telemetry2.startEvent("test"));
  EXPECT_NO_THROW(telemetry2.endEvent("test"));
}

TEST(CreateTelemetryFactoryTest,
     MultipleFactoryCreatedInstancesAreIndependent) {
  const std::string db_path1 = "test_factory1.db";
  const std::string db_path2 = "test_factory2.db";
  DatabaseCleanup cleanup1(db_path1);
  DatabaseCleanup cleanup2(db_path2);

  base::Config cfg1;
  cfg1.set("enable", true);
  cfg1.set("save_path", db_path1);

  base::Config cfg2;
  cfg2.set("enable", true);
  cfg2.set("save_path", db_path2);

  {
    auto telemetry1 = engine::createTelemetry(cfg1);
    auto telemetry2 = engine::createTelemetry(cfg2);

    telemetry1.startEvent("event_in_db1");
    telemetry2.startEvent("event_in_db2");
    telemetry1.endEvent("event_in_db1");
    telemetry2.endEvent("event_in_db2");
  }

  EXPECT_TRUE(hasEventInDatabase(db_path1, "event_in_db1"));
  EXPECT_FALSE(hasEventInDatabase(db_path1, "event_in_db2"));
  EXPECT_TRUE(hasEventInDatabase(db_path2, "event_in_db2"));
  EXPECT_FALSE(hasEventInDatabase(db_path2, "event_in_db1"));
}

TEST(SQLiteBasedTelemetryPersistenceTest,
     EventsPersistAfterTelemetryObjectDestruction) {
  const std::string db_path = "test_persistence.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    telemetry.startEvent("persistent_event");
    telemetry.endEvent("persistent_event");
  }

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    telemetry.startEvent("new_event");
    telemetry.endEvent("new_event");
  }

  EXPECT_EQ(countEventsInDatabase(db_path), 2);
  EXPECT_TRUE(hasEventInDatabase(db_path, "persistent_event"));
  EXPECT_TRUE(hasEventInDatabase(db_path, "new_event"));
}

TEST(TelemetryPropertiesTest, EventsWithEmptyProperties) {
  const std::string db_path = "test_properties.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  engine::SQLiteBasedTelemetry telemetry(cfg);
  base::Config empty_props;

  EXPECT_NO_THROW(telemetry.startEvent("event", empty_props));
  EXPECT_NO_THROW(telemetry.endEvent("event", empty_props));
}

TEST(TelemetryPropertiesTest, EventsWithDifferentPropertyTypes) {
  const std::string db_path = "test_properties.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  engine::SQLiteBasedTelemetry telemetry(cfg);
  base::Config props;
  props.set("string_prop", std::string("value"));
  props.set("int_prop", 42);
  props.set("bool_prop", true);

  EXPECT_NO_THROW(telemetry.startEvent("event", props));
  EXPECT_NO_THROW(telemetry.endEvent("event", props));
}

TEST(TelemetryPropertiesTest, DifferentPropertiesForStartAndEnd) {
  const std::string db_path = "test_properties.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  engine::SQLiteBasedTelemetry telemetry(cfg);
  base::Config start_props;
  start_props.set("start_time", 0);

  base::Config end_props;
  end_props.set("end_time", 100);
  end_props.set("duration", 100);

  EXPECT_NO_THROW(telemetry.startEvent("event", start_props));
  EXPECT_NO_THROW(telemetry.endEvent("event", end_props));
}

TEST(TelemetryPropertiesVerificationInDatabaseTest,
     VerifyIntegerPropertiesAreStoredCorrectly) {
  const std::string db_path = "test_properties_content.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    base::Config start_props;
    start_props.set("iteration", 42);

    base::Config end_props;
    end_props.set("final_value", 100);

    telemetry.startEvent("int_event", start_props);
    telemetry.endEvent("int_event", end_props);
  }

  const auto props = getEventProperties(db_path, "int_event");
  ASSERT_TRUE(props.found);
  const auto start_json = nlohmann::json::parse(props.start_properties);
  const auto end_json = nlohmann::json::parse(props.end_properties);
  EXPECT_EQ(start_json["iteration"], 42);
  EXPECT_EQ(end_json["final_value"], 100);
}

TEST(TelemetryPropertiesVerificationInDatabaseTest,
     VerifyStringPropertiesAreStoredCorrectly) {
  const std::string db_path = "test_properties_content.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    base::Config start_props;
    start_props.set("stage", std::string("initialization"));

    base::Config end_props;
    end_props.set("status", std::string("completed"));

    telemetry.startEvent("string_event", start_props);
    telemetry.endEvent("string_event", end_props);
  }

  const auto props = getEventProperties(db_path, "string_event");
  ASSERT_TRUE(props.found);
  const auto start_json = nlohmann::json::parse(props.start_properties);
  const auto end_json = nlohmann::json::parse(props.end_properties);
  EXPECT_EQ(start_json["stage"], "initialization");
  EXPECT_EQ(end_json["status"], "completed");
}

TEST(TelemetryPropertiesVerificationInDatabaseTest,
     VerifyBooleanPropertiesAreStoredCorrectly) {
  const std::string db_path = "test_properties_content.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    base::Config start_props;
    start_props.set("enabled", true);

    base::Config end_props;
    end_props.set("success", false);

    telemetry.startEvent("bool_event", start_props);
    telemetry.endEvent("bool_event", end_props);
  }

  const auto props = getEventProperties(db_path, "bool_event");
  ASSERT_TRUE(props.found);
  const auto start_json = nlohmann::json::parse(props.start_properties);
  const auto end_json = nlohmann::json::parse(props.end_properties);
  EXPECT_EQ(start_json["enabled"], true);
  EXPECT_EQ(end_json["success"], false);
}

TEST(TelemetryPropertiesVerificationInDatabaseTest,
     VerifyMixedPropertyTypesAreStoredCorrectly) {
  const std::string db_path = "test_properties_content.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    base::Config start_props;
    start_props.set("name", std::string("test_run"));
    start_props.set("iteration", 1);
    start_props.set("debug_mode", true);

    base::Config end_props;
    end_props.set("result", std::string("success"));
    end_props.set("count", 42);
    end_props.set("valid", true);

    telemetry.startEvent("mixed_event", start_props);
    telemetry.endEvent("mixed_event", end_props);
  }

  const auto props = getEventProperties(db_path, "mixed_event");
  ASSERT_TRUE(props.found);
  const auto start_json = nlohmann::json::parse(props.start_properties);
  const auto end_json = nlohmann::json::parse(props.end_properties);

  EXPECT_EQ(start_json["name"], "test_run");
  EXPECT_EQ(start_json["iteration"], 1);
  EXPECT_EQ(start_json["debug_mode"], true);

  EXPECT_EQ(end_json["result"], "success");
  EXPECT_EQ(end_json["count"], 42);
  EXPECT_EQ(end_json["valid"], true);
}

TEST(TelemetryPropertiesVerificationInDatabaseTest,
     VerifyEmptyPropertiesResultInEmptyJson) {
  const std::string db_path = "test_properties_content.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    base::Config empty_props;

    telemetry.startEvent("empty_props_event", empty_props);
    telemetry.endEvent("empty_props_event", empty_props);
  }

  const auto props = getEventProperties(db_path, "empty_props_event");
  ASSERT_TRUE(props.found);
  EXPECT_EQ(props.start_properties, "{}");
  EXPECT_EQ(props.end_properties, "{}");
}

TEST(TelemetryPropertiesVerificationInDatabaseTest,
     VerifyPropertiesForLatestEventWhenNamesRepeat) {
  const std::string db_path = "test_properties_content.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);

    base::Config first_start;
    first_start.set("run", 1);
    base::Config first_end;
    first_end.set("result", std::string("first"));

    telemetry.startEvent("repeated", first_start);
    telemetry.endEvent("repeated", first_end);

    base::Config second_start;
    second_start.set("run", 2);
    base::Config second_end;
    second_end.set("result", std::string("second"));

    telemetry.startEvent("repeated", second_start);
    telemetry.endEvent("repeated", second_end);
  }

  const auto props = getEventProperties(db_path, "repeated");
  ASSERT_TRUE(props.found);
  const auto start_json = nlohmann::json::parse(props.start_properties);
  const auto end_json = nlohmann::json::parse(props.end_properties);
  EXPECT_EQ(start_json["run"], 2);
  EXPECT_EQ(end_json["result"], "second");
}

TEST(TelemetryPropertiesVerificationInDatabaseTest,
     VerifyOnlyStartPropertiesWhenEventIsNotEnded) {
  const std::string db_path = "test_properties_content.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);
    base::Config start_props;
    start_props.set("started", true);

    telemetry.startEvent("not_ended", start_props);
  }

  const auto props = getEventProperties(db_path, "not_ended");
  ASSERT_TRUE(props.found);
  const auto start_json = nlohmann::json::parse(props.start_properties);
  EXPECT_EQ(start_json["started"], true);
  EXPECT_TRUE(props.end_properties.empty());
}

TEST(TelemetryPropertiesVerificationInDatabaseTest,
     VerifyPropertiesForNestedEvents) {
  const std::string db_path = "test_properties_content.db";
  DatabaseCleanup cleanup(db_path);
  const auto cfg = makeTelemetryConfig(db_path);

  {
    engine::SQLiteBasedTelemetry telemetry(cfg);

    base::Config outer_start;
    outer_start.set("level", std::string("outer"));
    base::Config inner_start;
    inner_start.set("level", std::string("inner"));

    base::Config outer_end;
    outer_end.set("completed", std::string("outer"));
    base::Config inner_end;
    inner_end.set("completed", std::string("inner"));

    telemetry.startEvent("outer", outer_start);
    telemetry.startEvent("inner", inner_start);
    telemetry.endEvent("inner", inner_end);
    telemetry.endEvent("outer", outer_end);
  }

  const auto outer_props = getEventProperties(db_path, "outer");
  ASSERT_TRUE(outer_props.found);
  const auto outer_start_json = nlohmann::json::parse(outer_props.start_properties);
  const auto outer_end_json = nlohmann::json::parse(outer_props.end_properties);
  EXPECT_EQ(outer_start_json["level"], "outer");
  EXPECT_EQ(outer_end_json["completed"], "outer");

  const auto inner_props = getEventProperties(db_path, "inner");
  ASSERT_TRUE(inner_props.found);
  const auto inner_start_json = nlohmann::json::parse(inner_props.start_properties);
  const auto inner_end_json = nlohmann::json::parse(inner_props.end_properties);
  EXPECT_EQ(inner_start_json["level"], "inner");
  EXPECT_EQ(inner_end_json["completed"], "inner");
}
