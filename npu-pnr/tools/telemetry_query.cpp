#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "argparse/argparse.hpp"
#include "nlohmann/json.hpp"
#include "sqlite3.h"

#include "arch/registry.hpp"
#include "base/common.hpp"
#include "base/engine.hpp"
#include "base/placement.hpp"
#include "engine/engine.hpp"

using json = nlohmann::json;

// The telemetry database contains events with the following schema:
//   - id: INTEGER PRIMARY KEY AUTOINCREMENT
//   - event_name: TEXT NOT NULL
//   - start_time: TEXT
//   - end_time: TEXT
//   - start_properties: TEXT (JSON)
//   - end_properties: TEXT (JSON)
//
// This tool supports two query modes:
//   1. Move Query: Extracts SA move events from "sa_move" events.
//   2. Iteration Query: Extracts SA iteration events from "sa_iter" events.
//   3. Netlist Iteration Query: Reconstructs netlist state at each SA move by
//      combining initial netlist with placement from each SA move; can
//      optionally run MILP full routing to get routed netlist state.
//   4. Engine Run Query: Extracts the only initial engine run events from
//      "engine_run" event (assuming only one exists in the database).
//
// The results are output in JSON format for easy parsing.

class TelemetryQuery {
private:
  sqlite3 *db_;
  std::string db_path_;

  base::Config arch_cfg_;

  // Helper struct to hold query results
  struct QueryResult {
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
  };

  // Execute a SQL query and return results as a QueryResult struct.
  // Handles statement preparation, execution, and cleanup.
  QueryResult executeQuery(const std::string &sql) {
    QueryResult result;
    sqlite3_stmt *stmt = nullptr;

    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to prepare SQL statement: " +
                               std::string(sqlite3_errmsg(db_)));
    }

    // Get column names
    int col_count = sqlite3_column_count(stmt);
    for (int i = 0; i < col_count; ++i) {
      result.columns.push_back(sqlite3_column_name(stmt, i));
    }

    // Get rows
    while (sqlite3_step(stmt) == SQLITE_ROW) {
      std::vector<std::string> row;
      for (int i = 0; i < col_count; ++i) {
        const char *text =
            reinterpret_cast<const char *>(sqlite3_column_text(stmt, i));
        row.push_back(text ? text : "");
      }
      result.rows.push_back(row);
    }

    sqlite3_finalize(stmt);
    return result;
  }

  std::string queryGenericEvent(std::string event_name, int index = -1) {
    // Index parameter is used to filter specific event by its sequential
    // position (0-based) among all events.
    // When index >= 0: returns only the event at that index
    // When index == -1: returns all events (no filter)

    std::string sql = std::format(
        "SELECT id, start_time, end_time, start_properties, end_properties "
        "FROM events WHERE event_name = '{}' ORDER BY id",
        event_name);

    // Filter by index using LIMIT/OFFSET if a specific index is requested
    // This selects the (index+1)th event (0-based indexing)
    if (index >= 0) {
      sql += " LIMIT 1 OFFSET " + std::to_string(index);
    }

    QueryResult result = executeQuery(sql);

    if (result.rows.empty()) {
      if (index >= 0) {
        throw std::runtime_error(std::format(
            "No '{}' event found at index {}.\n", event_name, index));
      } else {
        throw std::runtime_error(
            std::format("No '{}' events found in the database.\n", event_name));
      }
    }

    json result_json = json::array();
    for (const auto &row : result.rows) {
      std::string id = row[0];
      std::string start_time = row[1];
      std::string end_time = row[2];
      std::string start_properties_str = row[3];
      std::string end_properties_str = row[4];

      json event_obj = {
          {"event_id", id}, {"start_time", start_time}, {"end_time", end_time}};

      if (start_properties_str.empty()) {
        throw std::runtime_error(
            "Error: start_properties is empty in event ID " + id);
      }
      event_obj.update(json::parse(start_properties_str), true);

      if (end_properties_str.empty()) {
        throw std::runtime_error("Error: end_properties is empty in event ID " +
                                 id);
      }
      event_obj.update(json::parse(end_properties_str), true);

      result_json.push_back(event_obj);
    }

    return result_json.dump(4);
  }

public:
  TelemetryQuery(const std::string &db_path, const base::Config &arch_cfg)
      : db_(nullptr), db_path_(db_path), arch_cfg_(arch_cfg) {
    if (sqlite3_open_v2(db_path.c_str(), &db_, SQLITE_OPEN_READONLY, nullptr) !=
        SQLITE_OK) {
      throw std::runtime_error("Failed to open database: " +
                               std::string(sqlite3_errmsg(db_)));
    }
  }

  ~TelemetryQuery() {
    if (db_) {
      sqlite3_close(db_);
    }
  }

  // Disable copy operations to prevent double-close
  TelemetryQuery(const TelemetryQuery &) = delete;
  TelemetryQuery &operator=(const TelemetryQuery &) = delete;

  std::string queryMoveEvent(int index = -1) {
    return queryGenericEvent("sa_move", index);
  }

  std::string queryIterationEvent(int index = -1) {
    return queryGenericEvent("sa_iter", index);
  }

  std::string queryNetlistIterationEvent(bool run_router = false,
                                         int index = -1) {
    // Step 1: Get the initial netlist from engine_run event
    // The engine_run event contains the input_netlist field in
    // start_properties, which stores the original TOML netlist that was used to
    // start the PnR run.
    // Note: assume only one engine_run event exists.
    std::string engine_run_sql =
        "SELECT start_properties FROM events WHERE event_name = 'engine_run' "
        "ORDER BY id LIMIT 1";
    QueryResult engine_run_result = executeQuery(engine_run_sql);

    if (engine_run_result.rows.empty()) {
      throw std::runtime_error(
          "No 'engine_run' event found in the database. Cannot reconstruct "
          "netlist without initial configuration.");
    }

    // Parse the engine config to get the input netlist
    std::string engine_start_props_str = engine_run_result.rows[0][0];
    json engine_props = json::parse(engine_start_props_str);
    std::string initial_netlist_toml =
        engine_props["input_netlist"].get<std::string>();

    // Parse the initial netlist using PnRNetlistReader
    auto arch = arch::createArch(arch_cfg_);
    auto placed_netlist = arch.parseNetlist(
        base::PnRNetlistReader::fromTOML(initial_netlist_toml));

    // Step 2: Query sa_move events to get placement at each event
    // Each sa_move event contains the placement state in the
    // end_properties.placement field (serialized format).
    std::string eval_sql =
        "SELECT id, start_properties, end_properties "
        "FROM events WHERE event_name = 'sa_move' ORDER BY id";

    // Filter by index using LIMIT/OFFSET if a specific index is requested
    if (index >= 0) {
      eval_sql += " LIMIT 1 OFFSET " + std::to_string(index);
    }

    QueryResult eval_result = executeQuery(eval_sql);

    if (eval_result.rows.empty()) {
      throw std::runtime_error(
          std::format("No 'sa_move' event found at index {}.\n", index));
    }

    // Step 3: For each SA move, reconstruct the netlist with placement
    // The placement maps logical cores to physical cores. By combining this
    // with the initial netlist, we can see the complete state at each move.
    auto run_pnr_engine = [](const base::Config &arch_cfg,
                             const base::PnRPlacedNetlist &netlist) {
      engine::PnREngine pnr_engine(
          base::Config()
              .set<base::Config>("arch", arch_cfg)
              .set("placer", base::Config().set<std::string>("type", "noop"))
              .set("router", base::Config().set<std::string>("type", "milp"))
              .set("logger",
                   base::Config().set<std::string>("verbose", "silent"))
              .set("telemetry", base::Config().set<bool>("enable", false)));
      return pnr_engine.write(pnr_engine.run(netlist)).toTOML();
    };
    json result_json = json::array();
    for (const auto &row : eval_result.rows) {
      std::string event_id = row[0];
      std::string start_props_str = row[1];
      std::string end_props_str = row[2];

      json start_props = json::parse(start_props_str);
      int iter = start_props["iter"].get<int>();
      int move_in_iter = start_props["move_in_iter"].get<int>();

      json end_props = json::parse(end_props_str);
      bool acceptance = end_props["acceptance"].get<bool>();
      std::string placement_str = end_props["placement"].get<std::string>();
      base::Placement placement = base::deserializePlacement(placement_str);
      base::PnRPlacedNetlist current_netlist{
          .tf_graph = placed_netlist.tf_graph, .placement = placement};

      std::string netlist_toml =
          run_router ? run_pnr_engine(arch_cfg_, current_netlist)
                     : arch.dumpNetlist(current_netlist).toTOML();

      result_json.push_back({{"event_id", event_id},
                             {"iter", iter},
                             {"move_in_iter", move_in_iter},
                             {"acceptance", acceptance},
                             {"netlist", netlist_toml}});
    }

    return result_json.dump(4);
  }

  // Assume only one engine_run event exists
  std::string queryEngineRun() { return queryGenericEvent("engine_run", 0); }
};

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("telemetry_query");

  program.add_description(
      "Query telemetry data from PnR SQLite databases.\n\n"
      "Supported query modes:\n"
      "  move    - Extract SA move events\n"
      "  iter    - Extract SA iteration events\n"
      "  netlist - Reconstruct netlist state at each SA move event\n"
      "  routed_netlist - Apply MILP full routing to `netlist` query results\n"
      "  engine_run - Extract the only (one) initial engine run event\n");

  program.add_argument("database")
      .default_value(std::string{"telemetry.db"})
      .help("path to the SQLite telemetry database file (e.g., telemetry.db)");

  program.add_argument("-m", "--mode")
      .default_value(std::string{"move"})
      .help("query mode [move|iter|netlist|routed_netlist|engine_run]");

  program.add_argument("-o", "--output")
      .default_value(std::string{""})
      .help("output file path (default: stdout)");

  program.add_argument("-i", "--index")
      .default_value(-1)
      .scan<'i', int>()
      .help("the specific index of the event to query (default: all)");

  // Architecture parameters
  program.add_argument("--num_cols")
      .help("number of columns in the NPU architecture")
      .default_value(8)
      .scan<'i', int>();
  program.add_argument("--num_rows")
      .help("number of rows in the NPU architecture")
      .default_value(6)
      .scan<'i', int>();
  program.add_argument("--num_compute_rows")
      .help("number of compute tile rows in the NPU")
      .default_value(4)
      .scan<'i', int>();
  program.add_argument("--num_memory_rows")
      .help("number of memory tile rows in the NPU")
      .default_value(1)
      .scan<'i', int>();
  program.add_argument("--num_shim_rows")
      .help("number of shim tile rows in the NPU")
      .default_value(1)
      .scan<'i', int>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }

  std::string db_path = program.get<std::string>("database");
  std::string mode = program.get<std::string>("--mode");
  std::string output_path = program.get<std::string>("--output");
  int index = program.get<int>("--index");

  base::Config arch_cfg;
  arch_cfg.set<std::string>("type", "npu");
  arch_cfg.set<int>("num_cols", program.get<int>("--num_cols"));
  arch_cfg.set<int>("num_rows", program.get<int>("--num_rows"));
  arch_cfg.set<int>("num_compute_rows", program.get<int>("--num_compute_rows"));
  arch_cfg.set<int>("num_memory_rows", program.get<int>("--num_memory_rows"));
  arch_cfg.set<int>("num_shim_rows", program.get<int>("--num_shim_rows"));

  TelemetryQuery query(db_path, arch_cfg);
  std::string result;

  if (mode == "move") {
    result = query.queryMoveEvent(index);
  } else if (mode == "iter") {
    result = query.queryIterationEvent(index);
  } else if (mode == "netlist") {
    result = query.queryNetlistIterationEvent(false, index);
  } else if (mode == "routed_netlist") {
    result = query.queryNetlistIterationEvent(true, index);
  } else if (mode == "engine_run") {
    result = query.queryEngineRun();
  } else {
    throw std::runtime_error("Unknown query mode: " + mode);
  }

  if (!output_path.empty()) {
    std::ofstream out_file(output_path);
    if (!out_file.is_open()) {
      throw std::runtime_error("Failed to open output file: " + output_path);
    }
    out_file << result;
    std::cout << "Results written to: " << output_path << "\n";
  } else {
    std::cout << result;
  }

  return EXIT_SUCCESS;
}
