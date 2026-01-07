#!/usr/bin/env python3
import yaml
import os
import shutil
import argparse
import time
import traceback
import tarfile
import shlex
import json

from enum import Enum
from multiprocessing import Pool
from subprocess import Popen, PIPE, TimeoutExpired
from dataclasses import dataclass

# ==============================================================================
# Utility functions
# ==============================================================================

def read_text_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


def write_text_file(file_path: str, content: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)

@dataclass
class CommandResult:
    cmd: str
    cwd: str
    env: dict
    stdout: str
    stderr: str
    returncode: int

    def ok(self):
        return self.returncode == 0

    def check(self):
        if not self.ok():
            raise RuntimeError(
                f"Command '{self.cmd}' (on directory '{self.cwd}' with env '{self.env}') "
                f"failed with return code {self.returncode}.\n\n"
                f"Stdout:\n{self.stdout}\n\nStderr:\n{self.stderr}"
            )

    def __repr__(self):
        return (
            f"CommandResult(cmd={self.cmd}, cwd={self.cwd}, returncode={self.returncode}, "
            f"stdout={self.stdout},\nstderr={self.stderr},\nenv={self.env})"
        )

def synch_run_cmd(cmd, cwd=None, env=None, timeout=None):
    assert isinstance(cmd, str), "Command must be a string"
    process = Popen(cmd, cwd=cwd, shell=True, stdout=PIPE, stderr=PIPE, env=env)

    stdout, stderr = process.communicate(timeout=timeout)
    return CommandResult(
        cmd=cmd,
        cwd=cwd,
        env=env,
        stdout=stdout.decode(),
        stderr=stderr.decode(),
        returncode=process.returncode,
    )

class BuildStage(Enum):
    NONE = 0
    CLEAN = 1
    BUILD = 2
    COMPILE = 3
    RUN = 4

class BenchmarkTask:
    """
    Represents a single benchmark task defined in a YAML config.
    """

    def __init__(
        self,
        config_path: str,
        task_name: str,
        output_dir: str,
        args,
    ):
        # --- Load and validate configuration ---
        assert os.path.exists(config_path), f"Config file {config_path} does not exist"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.src_dir = os.path.dirname(os.path.abspath(config_path))
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Basic metadata ---
        self.benchmark_name = config.get("name", "undefined")
        self.task_name = task_name
        
        self.last_built_stage = BuildStage.NONE if not args.run_only else BuildStage.COMPILE

        # --- Command definitions ---
        self.clean_cmd = config.get("clean")
        self.build_cmd = config.get("build")
        self.compile_cmd = config.get("compile")
        self.run_cmd = config.get("run")
        self.tasks = config.get("tasks", [])

        # Validate required commands from yaml
        for cmd_name, cmd_value in {
            "clean": self.clean_cmd,
            "build": self.build_cmd,
            "compile": self.compile_cmd,
            "run": self.run_cmd,
        }.items():
            assert cmd_value, f"No {cmd_name} command specified in {config_path}"

        # --- Task-specific setup ---
        self.env_vars, self.output_mlir = self._prepare_env(self.tasks, args)

    # -------------------------------------------------------------------------
    # Internal setup helpers
    # -------------------------------------------------------------------------
    def _prepare_env(self, tasks, args):
        for task in tasks:
            if task.get("name") == self.task_name:
                env = os.environ.copy()
                for k, v in task.get("params", {}).items():
                    env[k] = str(v)

                env["placer"] = args.iron_placer
                env["use_placed"] = "1" if args.iron_placer == "hand_placed" else "0"
                if args.iron_placer == "sa_placer":
                    env["pnr_args"] = args.pnr_args_string or ""
                env["aiecc_extra_args"] = (
                    "--use-pnr-routing" if args.router_type == "pnr" else
                    "--packet-sw-objFifos" if args.router_aie_use_pkt_routing else ""
                ) 
                env["netlist_only"] = "1" if args.netlist_only else "0"
                env["src_dir"] = self.src_dir
                env["output_dir"] = self.output_dir
                env["build_dir"] = os.path.join(self.output_dir, "build")
                if args.telemetry_enable:
                    env["pnr_args"] += " " + shlex.quote("--telemetry.enable")
                    env["pnr_args"] += " " + shlex.quote(
                        f"--telemetry.save_path={env['build_dir']}/telemetry.db"
                    )
                output_mlir = task.get("output")
                assert output_mlir, f"No output MLIR specified for task {self.task_name}"
                return env, output_mlir

        raise ValueError(f"Task {self.task_name} not found in benchmark {self.benchmark_name}")

    def _get_output_file(self, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        assert os.path.exists(path), f"Expected file '{filename}' not found in {self.output_dir}"
        return os.path.abspath(path)

    def log(self, msg: str) -> None:
        print(f"[{self.benchmark_name}/{self.task_name}] {msg}")
      
    # -------------------------------------------------------------------------
    # Build pipeline methods
    # -------------------------------------------------------------------------

    def clean_build_task(self) -> None:
        synch_run_cmd(self.clean_cmd, cwd=self.src_dir).check()
        self.last_built_stage = BuildStage.CLEAN

    def build_task(self) -> str:
        start_time = time.perf_counter()
        result = synch_run_cmd(self.build_cmd, cwd=self.src_dir, env=self.env_vars)
        result.check()
        end_time = time.perf_counter()
        build_time = end_time - start_time
        self.last_built_stage = BuildStage.BUILD

        output_mlir = self._get_output_file(self.output_mlir)
        summary_files = {}

        return f"{result}\nBuild took: {build_time:.2f} seconds"

    def compile_task(self) -> str:
        assert self.last_built_stage == BuildStage.BUILD, "Build before compile"

        start_time = time.perf_counter()
        result = synch_run_cmd(self.compile_cmd, cwd=self.src_dir, env=self.env_vars, timeout=300)
        result.check()
        end_time = time.perf_counter()
        compile_time = end_time - start_time
        self.last_built_stage = BuildStage.COMPILE

        return f"{result}\nCompile took: {compile_time:.2f} seconds"

    def run_task(self) -> str:
        assert self.last_built_stage == BuildStage.COMPILE, "Compile before run"

        result = synch_run_cmd(self.run_cmd, cwd=self.src_dir, env=self.env_vars)
        result.check()
        self.last_built_stage = BuildStage.RUN

        return str(result)

    def custom_task(self, custom_cmd: str) -> str:
        """Run an arbitrary custom command with the configured environment."""
        result = synch_run_cmd(custom_cmd, cwd=self.src_dir, env=self.env_vars)
        result.check()
        return str(result)

    def __repr__(self) -> str:
        return f"BenchmarkTask({self.benchmark_name}/{self.task_name})"

def import_pnr_results(
    task: BenchmarkTask,
    import_root: str,
    iteration: int,
) -> str:
    if iteration == -1:
        os.makedirs(os.path.join(task.output_dir, "build"), exist_ok=True)
        import_dir = os.path.join(import_root, task.benchmark_name, task.task_name)
        imported_netlist = os.path.join(import_dir, "build", "pnr_placed_netlist.json")
        if not os.path.exists(imported_netlist):
            raise FileNotFoundError(f"Imported netlist not found: {imported_netlist}")
        dest_netlist = os.path.join(task.output_dir, "build", "pnr_placed_netlist.json")
        shutil.copyfile(imported_netlist, dest_netlist)
        return f"Copied PnR placed netlist from {imported_netlist} to {dest_netlist}\n"

    if iteration is None:
        raise ValueError("Importing PnR results requires a valid iteration number (--import-pnr-iter)")

    import_dir = os.path.join(import_root, task.benchmark_name, task.task_name)
    telemetry_db = os.path.join(import_dir, "build", "telemetry.db")

    if not os.path.exists(telemetry_db):
        raise FileNotFoundError(f"Telemetry DB not found: {telemetry_db}")

    task.log(f"Importing precomputed PnR results (iteration {iteration}) ...")

    bin_dir = os.path.expandvars("$NPU_PNR_BIN_DIR/tools")
    query_bin = os.path.join(bin_dir, "telemetry_query")
    translator_bin = os.path.join(bin_dir, "netlist_translator")

    for path in (query_bin, translator_bin):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required tool not found: {path}")

    os.makedirs(os.path.join(task.output_dir, "build"), exist_ok=True)
    query_json = os.path.join(task.output_dir, "build", "query.json")
    routed_netlist = os.path.join(task.output_dir, "build", "pnr_placed_netlist.json")
    temp_toml = os.path.join(task.output_dir, "build", "netlist.toml")

    query_cmd = (
        f"{query_bin} {shlex.quote(telemetry_db)} "
        f"-m routed_netlist "
        f"-i {iteration} "
        f"-o {shlex.quote(query_json)}"
    )
    query_res = synch_run_cmd(query_cmd)
    query_res.check()

    with open(query_json, "r") as f:
        query_data = json.load(f)
        # Assuming we only query one iteration
        toml_str = query_data[0].get("netlist")
        if toml_str is None:
            raise RuntimeError(f"No 'netlist' field found in telemetry query output")
        write_text_file(temp_toml, toml_str)

    translate_cmd = (
        f"{translator_bin} "
        f"{shlex.quote(temp_toml)} "
        f"-o {shlex.quote(routed_netlist)}"
    )
    json_res = synch_run_cmd(translate_cmd)
    json_res.check()

    return (
        f"Imported PnR results from iteration {iteration}\n"
        f"{query_res}\n"
        f"{json_res}\n"
    )

def log_error(task: BenchmarkTask, e: Exception, err_file: str, verbose: bool) -> None:
    err_path = os.path.join(task.output_dir, err_file)
    write_text_file(err_path, traceback.format_exc())
    task.log(f" Error: {e}")
    if verbose:
        traceback.print_exc()

def build_single_task(task: BenchmarkTask, args) -> tuple[BenchmarkTask, bool]:
    import_log = ""
    try:
        task.log("Cleaning ...")
        task.clean_build_task()

        task.log("Building ...")
        if args.import_pnr_results:
            import_log = import_pnr_results(task, args.import_pnr_results, args.import_pnr_iter)
        build_res = task.build_task() 
        if args.debug and not args.netlist_only:
            write_text_file(
                os.path.join(task.output_dir, f"{task.task_name}.build.log"), 
                import_log + build_res,
            )
        return task, True
    except Exception as e:
        if args.netlist_only:
            task.log(f"Netlist-only mode: ignoring exception ...")
            return task, True
        log_error(task, e, f"{task.task_name}.build.error.log", args.verbose)
        return task, False

def compile_benchmark_tasks(tasks: list[tuple[BenchmarkTask, bool]], args) -> tuple[BenchmarkTask, bool]:
    compile_results = []
    for task, ok in tasks:
        if not ok:
            task.log("Skipping compile due to build stage errors.")
            compile_results.append((task, False))
            continue
        try:
            task.log("Compiling ...")
            compile_res = task.compile_task()
            if args.debug:
                write_text_file(os.path.join(task.output_dir, f"{task.task_name}.compile.log"), compile_res)
            compile_results.append((task, True))
        except Exception as e:
            log_error(task, e, f"{task.task_name}.compile.error.log", args.verbose)
            compile_results.append((task, False))
    return compile_results

def create_netlist_zip(root_path: str) -> None:
    staging_dir = os.path.join(root_path, "_netlists_tmp")

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    netlist_paths = []
    for dir_path, _, files in os.walk(root_path):
        if "netlist.json" in files:
            netlist_paths.append(os.path.join(dir_path, "netlist.json"))
    
    for netlist_path in netlist_paths:
        # Path looks like: root_path / benchmark_name / task_name / build / netlist.json
        build_dir = os.path.dirname(netlist_path)
        task_dir = os.path.dirname(build_dir)
        benchmark_dir = os.path.dirname(task_dir)

        task_name = os.path.basename(task_dir)
        benchmark_name = os.path.basename(benchmark_dir)
        out_name = f"{benchmark_name}_{task_name}.toml"
        dest_path = os.path.join(staging_dir, out_name)

        translator_bin = os.path.expandvars("$NPU_PNR_BIN_DIR/tools/netlist_translator")
        translate_cmd = (
            f"{translator_bin} "
            f"{shlex.quote(netlist_path)} "
            f"-o {shlex.quote(dest_path)} "
        )
        translate_res = synch_run_cmd(translate_cmd)
        translate_res.check()

    tar_path = os.path.join(root_path, "netlists.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for filename in os.listdir(staging_dir):
            tar.add(os.path.join(staging_dir, filename), arcname=filename)
    shutil.rmtree(staging_dir)

    print(f"Created netlist archive at {tar_path} containing {len(netlist_paths)} netlists.")

def run_benchmark_tasks(tasks: list[tuple[BenchmarkTask, bool]], args) -> None:
    tasks_to_run = []
    if not args.run_only:
        for task, ok in tasks:
            if not ok:
                task.log("Skipping run due to previous errors.")
                continue
            tasks_to_run.append(task)
    else:
        for task in tasks:
            err_path = os.path.join(task.output_dir, f"{task.task_name}.error.log")
            if os.path.exists(err_path):
                task.log("Skipping run due to previous errors.")
            else:
                tasks_to_run.append(task)

    for task in tasks_to_run:
        try:
            task.log("Running ...")
            run_res = task.run_task()
            if args.debug:
                write_text_file(os.path.join(task.output_dir, f"{task.task_name}.run.log"), run_res)
        except Exception as e:
            task.log(f" Error: {e}")
            write_text_file(
                os.path.join(task.output_dir, f"{task.task_name}.run.error.log"),
                traceback.format_exc(),
            )
            if args.verbose:
                traceback.print_exc()

def parse_tasklist_entry(entry: str, args) -> BenchmarkTask:
    parts = entry.split("/")
    assert len(parts) == 2, f"Invalid task entry '{entry}'. Expected format: benchmark/task"

    benchmark_name, task_name = parts
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "benchmarks", benchmark_name, "config.yml")
    output_dir = os.path.join(args.output_dir, benchmark_name, task_name)

    return BenchmarkTask(config_path, task_name, output_dir, args)

# TODO: Remove once PnR args are first-class CLI options
def create_pnr_args_string(args) -> str:
    pnr_args_string = ""
    if args.placer_pnr_args:
        pnr_args_string += args.placer_pnr_args
    pnr_args_string += f" --placer.type {args.placer_pnr_type}"
    pnr_args_string += f" --router.type {args.router_pnr_type}"
    return pnr_args_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build and optionally run benchmarks")
    parser.add_argument("tasklists", type=str, nargs="*",
                        help="Path to the task list file or benchmark/task pairs (e.g. vector_scalar_add/default)")
    parser.add_argument("--output-dir", type=str, default="build",
                        help="Directory to store build outputs")
    # -----------------------------------------------------------------------------
    # Pipeline stages
    # -----------------------------------------------------------------------------
    parser.add_argument(
        "--clean-all",
        action="store_true",
        default=False,
        help="Remove all generated build artifacts in --output-dir before running",
    )

    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build stage creates the MLIR file from IRON, optionally runs PnR placer/router if requested",
    )

    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile stage compiles the MLIR file to an executable binary. To use AIE router, compile must be True",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Run the benchmark on NPU after compilation",
    )

    # -----------------------------------------------------------------------------
    # Placer Options
    # -----------------------------------------------------------------------------
    parser.add_argument(
        "--placer.type",
        dest="placer_type",
        choices=["aie", "pnr", "hand_placed"],
        default="aie",
        help="Select placement strategy",
    )
    # PnR Placer options
    parser.add_argument(
        "--placer.pnr.type",
        dest="placer_pnr_type",
        choices=["sa", "milp", "ls", "noop"],
        default="sa",
        help="PnR placer to use",
    )

    # NOTE: For now, pass-through string for PnR tool arguments.
    # TODO: Expose these as first-class CLI flags once stable.
    parser.add_argument(
        "--placer.pnr.args",
        dest="placer_pnr_args",
        type=str,
        default=None,
        help="Additional arguments forwarded directly to the PnR tool",
    )

    # -----------------------------------------------------------------------------
    # Router selection
    # -----------------------------------------------------------------------------
    parser.add_argument(
        "--router.type",
        dest="router_type",
        choices=["aie", "pnr"],
        default="aie",
        help="Select routing strategy",
    )

    # PnR Router options
    parser.add_argument(
        "--router.pnr.type",
        dest="router_pnr_type",
        choices=["milp", "lp"],
        default="milp",
        help="PnR router backend to use",
    )

    # AIE Router options
    parser.add_argument(
        "--router.aie.use-pkt-routing", 
        dest="router_aie_use_pkt_routing", 
        action=argparse.BooleanOptionalAction, 
        default=False,
        help="Use packet-switched routing in AIE router"
    )

    # Development/debugging options
    parser.add_argument("-j", type=int, default=None)
    parser.add_argument("--import-pnr-results", type=str, default=None)
    parser.add_argument("--import-pnr-iter", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--netlist-only", action="store_true", default=False)
    parser.add_argument("--run-only", action="store_true", default=False,
                        help="Run task without building/compiling (assumes --output-dir was previously built/compiled, skips tasks with built/compile error log)")
    parser.add_argument("--hook", type=str, default=None)
    parser.add_argument("--telemetry.enable", dest="telemetry_enable", action="store_true", default=False)

    args = parser.parse_args()
    args.pnr_args_string = create_pnr_args_string(args)

    if args.placer_type == "hand_placed":
        args.iron_placer = "hand_placed"
    elif args.placer_type == "pnr":
        args.iron_placer = "sa_placer"
    else:
        args.iron_placer = "sequential_placer"
    
    if args.placer_type != "pnr" and args.router_type == "pnr":
        raise ValueError("Cannot use PnR router with AIE placer. Please set --placer.type to 'pnr'.")

    if args.iron_placer == "hand_placed" and args.router_type == "pnr":
        raise ValueError("Cannot use PnR router with hand-placed designs.")

    if args.iron_placer == "sa_placer":
        pnr_bin = os.path.expandvars("$NPU_PNR_BIN_DIR/apps/pnr")
        if not os.path.exists(pnr_bin):
            raise FileNotFoundError(f"PnR binary not found at {pnr_bin}")

    if args.import_pnr_results and args.placer_type != "pnr":
        raise ValueError("Importing PnR results requires --placer.type to be 'pnr'.")

    if args.run_only:
        args.iron_placer = "sa_placer"
        args.build = False
        args.compile = False
        args.run = True
    if args.netlist_only:
        args.iron_placer = "sa_placer"
        args.build = True
        args.compile = False
        args.run = False
    os.makedirs(args.output_dir, exist_ok=True)

    task_entries = []
    for arg_item in args.tasklists:
        if os.path.isfile(arg_item):
            with open(arg_item, "r") as f:
                for task in yaml.safe_load(f):
                    task_entries.append(task)
        else:
            task_entries.append(arg_item)
    benchmark_tasks = [
        parse_tasklist_entry(entry, args) for entry in task_entries
    ]

    if args.clean_all:
        for bt in benchmark_tasks:
            bt.clean_build_task()
        print("Cleaned all benchmarks.")
        exit(0)
    
    results = []
    # --- Parallel build ---
    if args.build:
        n_parallel = args.j or os.cpu_count()
        if n_parallel > 1:
            print(f"Building benchmarks in parallel with {n_parallel} processes ...")
            with Pool(processes=n_parallel) as pool:
                results = pool.starmap(build_single_task, [(bt, args) for bt in benchmark_tasks])
        else:
            results = [build_single_task(bt, args) for bt in benchmark_tasks]
    if args.netlist_only:
        create_netlist_zip(args.output_dir)
        exit(0)
    # --- Serial compile ---
    if args.compile:
        results = compile_benchmark_tasks(results, args)
    # --- Serial run ---
    if args.run:
        tasks_to_run = results if args.build else benchmark_tasks
        run_benchmark_tasks(tasks_to_run, args)
    

