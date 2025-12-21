#!/usr/bin/env python3
import yaml
import os
import shutil
import argparse
import time
import traceback
import tarfile

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
        placer: str,
        output_dir: str,
        pnr_args: str = None,
        aie_pkt_routing: bool = False,
        run_only: bool = False,
        netlist_only: bool = False,
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
        self.placer = placer
        self.last_built_stage = BuildStage.NONE if not run_only else BuildStage.COMPILE

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
        self.env_vars, self.output_mlir = self._prepare_env(self.tasks, pnr_args, aie_pkt_routing, netlist_only)

    # -------------------------------------------------------------------------
    # Internal setup helpers
    # -------------------------------------------------------------------------

    def _prepare_env(self, tasks, pnr_args, aie_pkt_routing, netlist_only):
        for task in tasks:
            if task.get("name") == self.task_name:
                env = os.environ.copy()

                for k, v in task.get("params", {}).items():
                    env[k] = str(v)

                env["placer"] = self.placer
                env["use_placed"] = "1" if self.placer == "hand_placed" else "0"
                if self.placer == "sa_placer":
                    env["pnr_args"] = pnr_args or ""
                env["aiecc_extra_args"] = (
                    "--use-pnr-routing" if self.placer == "sa_placer" else
                    "--packet-sw-objFifos" if aie_pkt_routing else ""
                ) 
                env["netlist_only"] = "1" if netlist_only else "0"
                env["src_dir"] = self.src_dir
                env["output_dir"] = self.output_dir
                env["build_dir"] = os.path.join(self.output_dir, "build")
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

    def build_task(self, import_log: str = ""):
        if self.placer == "sa_placer":
            pnr_bin = os.path.expandvars("$NPU_PNR_BIN_DIR/placer")
            assert os.path.exists(pnr_bin), f"PnR binary not found at {pnr_bin}"

        start_time = time.perf_counter()
        result = synch_run_cmd(self.build_cmd, cwd=self.src_dir, env=self.env_vars)
        result.check()
        end_time = time.perf_counter()
        build_time = end_time - start_time
        self.last_built_stage = BuildStage.BUILD

        output_mlir = self._get_output_file(self.output_mlir)
        summary_files = {}

        return f"{import_log}\n{result}\nBuild took: {build_time:.2f} seconds"

    def compile_task(self):
        assert self.last_built_stage == BuildStage.BUILD, "Build before compile"

        start_time = time.perf_counter()
        result = synch_run_cmd(self.compile_cmd, cwd=self.src_dir, env=self.env_vars, timeout=300)
        result.check()
        end_time = time.perf_counter()
        compile_time = end_time - start_time
        self.last_built_stage = BuildStage.COMPILE

        return f"{result}\nCompile took: {compile_time:.2f} seconds"

    def run_task(self):
        assert self.last_built_stage == BuildStage.COMPILE, "Compile before run"

        result = synch_run_cmd(self.run_cmd, cwd=self.src_dir, env=self.env_vars)
        result.check()
        self.last_built_stage = BuildStage.RUN

        return str(result)

    def custom_task(self, custom_cmd: str):
        """Run an arbitrary custom command with the configured environment."""
        result = synch_run_cmd(custom_cmd, cwd=self.src_dir, env=self.env_vars)
        result.check()
        return str(result)

    def __repr__(self) -> str:
        return f"BenchmarkTask({self.benchmark_name}/{self.task_name})"

def import_pnr_results(task: BenchmarkTask, import_dir: str, args) -> str:
    if not import_dir:
        return ""
    import_dir = os.path.join(import_dir, task.benchmark_name, task.task_name)
    task.log("Importing precomputed PnR results ...")
    required = [
        f"build/pnr_placed_netlist{args.import_pnr_results_suffix}.json",
        f"build/pnr_route_summary{args.import_pnr_results_suffix}.json",
    ]
    for f in required:
        assert os.path.exists(os.path.join(import_dir, f)), f"Missing PnR file: {os.path.join(import_dir, f)}"
    os.makedirs(os.path.join(task.output_dir, "build"), exist_ok=True)
    dest_placed = os.path.join(task.output_dir, "build/pnr_placed_netlist.json")
    dest_summary = os.path.join(task.output_dir, "build/pnr_route_summary.json")
    shutil.copyfile(os.path.join(import_dir, required[0]), dest_placed)
    shutil.copyfile(os.path.join(import_dir, required[1]), dest_summary)

    if args.debug:
        log_path = os.path.join(import_dir, f"{task.task_name}.build.log")
        log_text = read_text_file(log_path) if os.path.exists(log_path) else "(no log found)"
        return (
            f"Imported PnR results:\n"
            f"  - Placed: {dest_placed}\n"
            f"  - Route summary: {dest_summary}\n"
            f"  - Log: {log_path}\n\n"
            f"--- Start of Original PnR Log ---\n{log_text}\n"
            f"--- End of Original PnR Log ---\n\n"
        )
    return ""

def log_error(task: BenchmarkTask, e: Exception, err_file: str, verbose: bool) -> None:
    err_path = os.path.join(task.output_dir, err_file)
    write_text_file(err_path, traceback.format_exc())
    task.log(f" Error: {e}")
    if verbose:
        traceback.print_exc()

def build_single_task(task: BenchmarkTask, args) -> tuple[BenchmarkTask, bool]:
    try:
        task.log("Cleaning ...")
        task.clean_build_task()

        task.log("Building ...")
        import_log = import_pnr_results(task, args.import_pnr_results, args)
        build_res = task.build_task(import_log) 
        if args.netlist_only:
            return task, True
        if args.debug:
            write_text_file(os.path.join(task.output_dir, f"{task.task_name}.build.log"), build_res)
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

        out_name = f"{benchmark_name}_{task_name}.json"
        dest_path = os.path.join(staging_dir, out_name)

        shutil.copy2(netlist_path, dest_path)

    tar_path = os.path.join(root, "netlists.tar.gz")
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

def parse_tasklist_entry(
    entry, 
    placer, 
    output_dir, 
    pnr_args, 
    aie_pkt_routing, 
    run_only, 
    netlist_only
) -> BenchmarkTask:
    parts = entry.split("/")
    assert len(parts) == 2, f"Invalid task entry '{entry}'. Expected format: benchmark/task"

    benchmark_name, task_name = parts
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "benchmarks", benchmark_name, "config.yml")
    output_dir = os.path.join(output_dir, benchmark_name, task_name)

    return BenchmarkTask(
        config_path=config_path,
        task_name=task_name,
        placer=placer,
        output_dir=output_dir,
        pnr_args=pnr_args,
        aie_pkt_routing=aie_pkt_routing,
        run_only=run_only,
        netlist_only=netlist_only,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build and optionally run benchmarks")
    parser.add_argument("tasklists", type=str, nargs="*",
                        help="Path to the task list file or benchmark/task pairs (e.g. vector_scalar_add/default)")
    parser.add_argument("--output-dir", type=str, default="build",
                        help="Directory to store build outputs")
    parser.add_argument("--clean-all", action="store_true", default=False)
    parser.add_argument("--build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--placer", type=str, default="sequential_placer",
                        choices=["sequential_placer", "sa_placer", "hand_placed"])
    parser.add_argument("--aie-pkt-routing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pnr-args", type=str, default=None)
    parser.add_argument("--import-pnr-results", type=str, default=None)
    parser.add_argument("--import-pnr-results-suffix", type=str, default="")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run", action="store_true", default=False)
    parser.add_argument("--run-only", action="store_true", default=False,
                        help="Run task without building/compiling (assumes --output-dir was previously built/compiled, skips tasks with built/compile error log)")
    parser.add_argument("--netlist-only", action="store_true", default=False)
    parser.add_argument("--hook", type=str, default=None)
    parser.add_argument("-j", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=True)

    args = parser.parse_args()

    if args.placer == "sa_placer" and args.pnr_args is None:
        args.pnr_args = "-n 10"
    if args.placer == "sa_placer" and args.aie_pkt_routing:
        raise ValueError("Cannot use AIE packet routing when using PnR placer+router")
    if args.run_only:
        args.build = False
        args.compile = False
        args.run = True
    if args.netlist_only:
        args.placer = "sa_placer"
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
        parse_tasklist_entry(
            entry, 
            args.placer, 
            args.output_dir, 
            args.pnr_args, 
            args.aie_pkt_routing, 
            args.run_only, 
            args.netlist_only
        )
        for entry in task_entries
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
    

