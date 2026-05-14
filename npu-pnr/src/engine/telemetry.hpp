#pragma once

#include <stack>
#include <string>
#include <utility>

#include "sqlite3.h"

#include "base/common.hpp"

namespace engine {

class Telemetry {
private:
  struct TelemetryConcept {
    virtual ~TelemetryConcept() = default;
    virtual void
    startEvent(const std::string &event_name,
               const base::Config &properties = base::Config()) = 0;
    virtual void endEvent(const std::string &event_name,
                          const base::Config &properties = base::Config()) = 0;
  };

  template <typename T> struct TelemetryModel : TelemetryConcept {
    T impl_;

    explicit TelemetryModel(const base::Config &cfg) : impl_(cfg) {}

    void startEvent(const std::string &event_name,
                    const base::Config &properties = base::Config()) override {
      impl_.startEvent(event_name, properties);
    }
    void endEvent(const std::string &event_name,
                  const base::Config &properties = base::Config()) override {
      impl_.endEvent(event_name, properties);
    }
  };

  std::unique_ptr<TelemetryConcept> pimpl_;

  // Private constructor
  template <typename T>
  explicit Telemetry(std::in_place_type_t<T>, const base::Config &cfg)
      : pimpl_(std::make_unique<TelemetryModel<T>>(cfg)) {}

public:
  // Static factory
  template <typename T> static Telemetry create(const base::Config &cfg) {
    return Telemetry(std::in_place_type<T>, cfg);
  }

  void startEvent(const std::string &event_name,
                  const base::Config &properties = base::Config()) {
    pimpl_->startEvent(event_name, properties);
  }
  void endEvent(const std::string &event_name,
                const base::Config &properties = base::Config()) {
    pimpl_->endEvent(event_name, properties);
  }
};

class SQLiteBasedTelemetry {
private:
  sqlite3 *db_;
  // To track nested events and ensure proper pairing
  std::stack<std::pair<std::string, int>> event_stack_;

public:
  SQLiteBasedTelemetry(const base::Config &cfg) : db_(nullptr) {
    std::string db_path =
        cfg.getOrDefault<std::string>("save_path", "telemetry.db");

    if (auto rc = sqlite3_open_v2(db_path.c_str(), &db_,
                                  SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                                  nullptr);
        rc != SQLITE_OK) {
      throw std::runtime_error("Failed to open database: " +
                               std::string(sqlite3_errmsg(db_)));
    }

    char *err_msg = nullptr;
    if (auto rc = sqlite3_exec(db_,
                               "CREATE TABLE IF NOT EXISTS events ("
                               "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                               "event_name TEXT NOT NULL, "
                               "start_time TEXT, "
                               "end_time TEXT, "
                               "start_properties TEXT, "
                               "end_properties TEXT"
                               ");",
                               nullptr, nullptr, &err_msg);
        rc != SQLITE_OK) {
      std::string error = err_msg;
      sqlite3_free(err_msg);
      sqlite3_close(db_);
      throw std::runtime_error("Failed to create table: " + error);
    }
  }

  ~SQLiteBasedTelemetry() {
    if (db_) {
      sqlite3_close(db_);
    }
  }

  // TODO: consider adding scope-based event tracking
  void startEvent(const std::string &event_name,
                  const base::Config &properties = base::Config()) {
    const char *sql = "INSERT INTO events (event_name, start_time, "
                      "start_properties) VALUES (?, ?, ?);";
    sqlite3_stmt *stmt = nullptr;

    if (auto rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        rc != SQLITE_OK) {
      throw std::runtime_error("Failed to prepare statement: " +
                               std::string(sqlite3_errmsg(db_)));
    }

    std::string start_time = utils::getCurrentTimestamp();
    std::string props_json = properties.toJSON();
    sqlite3_bind_text(stmt, 1, event_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, start_time.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, props_json.c_str(), -1, SQLITE_TRANSIENT);

    if (auto rc = sqlite3_step(stmt); rc != SQLITE_DONE) {
      sqlite3_finalize(stmt);
      throw std::runtime_error("Failed to execute insert: " +
                               std::string(sqlite3_errmsg(db_)));
    }

    int64_t row_id = sqlite3_last_insert_rowid(db_);
    sqlite3_finalize(stmt);

    event_stack_.push({event_name, row_id});
  }

  void endEvent(const std::string &event_name,
                const base::Config &properties = base::Config()) {
    if (event_stack_.empty() || event_stack_.top().first != event_name) {
      throw std::runtime_error("Mismatched event end: " + event_name);
    }

    const char *sql = "UPDATE events SET end_time = ?, end_properties = ? "
                      "WHERE id = ? AND end_time IS NULL;";
    sqlite3_stmt *stmt = nullptr;

    if (auto rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        rc != SQLITE_OK) {
      throw std::runtime_error("Failed to prepare statement: " +
                               std::string(sqlite3_errmsg(db_)));
    }

    std::string end_time = utils::getCurrentTimestamp();
    std::string props_json = properties.toJSON();
    sqlite3_bind_text(stmt, 1, end_time.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, props_json.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 3, event_stack_.top().second);

    if (auto rc = sqlite3_step(stmt); rc != SQLITE_DONE) {
      sqlite3_finalize(stmt);
      throw std::runtime_error("Failed to execute update: " +
                               std::string(sqlite3_errmsg(db_)));
    }

    sqlite3_finalize(stmt);

    // TODO: maybe double check if the affected row has the correct event_name?
    event_stack_.pop();
  }
};

class NoOpTelemetry {
public:
  NoOpTelemetry(const base::Config &) {}
  void startEvent(const std::string &, const base::Config & = base::Config()) {}
  void endEvent(const std::string &, const base::Config & = base::Config()) {}
};

inline Telemetry createTelemetry(const base::Config &cfg) {
  if (cfg.getOrDefault<bool>("enable", false)) {
    return Telemetry::create<SQLiteBasedTelemetry>(cfg);
  } else {
    return Telemetry::create<NoOpTelemetry>(cfg);
  }
}

} // namespace engine
