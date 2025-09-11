import yaml
import os
import shutil
import argparse
from multiprocessing import Pool
from subprocess import Popen, PIPE, TimeoutExpired
from dataclasses import dataclass
import traceback


@dataclass
class CommandResult:
    cmd: str
    cwd: str
    env: dict
    stdout: str
    stderr: str
    returncode: int
    is_timed_out: bool

    def ok(self):
        return self.returncode == 0 and not self.is_timed_out

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
            f"is_timed_out={self.is_timed_out},\nstdout={self.stdout},\nstderr={self.stderr},\nenv={self.env})"
        )


def read_text_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


def write_text_file(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)


def subprocess_run_cmd(cmd, cwd=None, env=None, timeout_sec=36000):
    assert isinstance(cmd, str), "Command must be a string"
    process = Popen(cmd, cwd=cwd, shell=True, stdout=PIPE, stderr=PIPE, env=env)

    is_timed_out = False

    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
    except TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        is_timed_out = True
        print(f"Command '{cmd}' timed out after {timeout_sec} seconds.")

    return CommandResult(
        cmd=cmd,
        cwd=cwd,
        env=env,
        stdout=stdout.decode(),
        stderr=stderr.decode(),
        returncode=process.returncode,
        is_timed_out=is_timed_out,
    )


class Benchmark:
    def __init__(self, config_path, use_placed, iron_placer):
        assert os.path.exists(config_path), f"Config file {config_path} does not exist"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.root_dir = os.path.dirname(os.path.abspath(config_path))
        self.name = config.get("name", "undefined")
        self.clean_cmd = config.get("clean", None)
        self.build_cmd = config.get("build", None)
        self.run_cmd = config.get("run", None)
        self.tasks = config.get("tasks", [])
        self.last_built_task = None
        self.use_placed = use_placed
        self.iron_placer = iron_placer

        assert self.clean_cmd, "No clean command specified"
        assert self.build_cmd, "No build command specified"
        assert self.run_cmd, "No run command specified"

    def __get_local_file(self, filename):
        path = os.path.abspath(os.path.join(self.root_dir, filename))
        assert os.path.exists(
            path
        ), f"File {filename} not found in benchmark {self.name}"
        return path

    def __get_task(self, task_name):
        for task in self.tasks:
            if task.get("name") == task_name:
                params = task.get("params", {})
                env_vars = os.environ.copy()
                for key, value in params.items():
                    env_vars[key] = str(value)
                if self.use_placed:
                    env_vars["use_placed"] = "1"
                else:
                    env_vars["use_placed"] = "0"
                    env_vars["placer"] = self.iron_placer
                output_mlir = task.get("output", None)
                assert output_mlir, f"No output MLIR specified for task {task_name}"
                return env_vars, output_mlir
        assert False, f"Task {task_name} not found in benchmark {self.name}"

    def clean_task_build(self):
        subprocess_run_cmd(cmd=self.clean_cmd, cwd=self.root_dir).check()
        self.last_built_task = None

    def build_task(self, task_name):
        param_envs, output_mlir = self.__get_task(task_name)

        build = subprocess_run_cmd(
            cmd=self.build_cmd, cwd=self.root_dir, env=param_envs
        )
        build.check()

        output_mlir = self.__get_local_file(output_mlir)
        extract_fifo_cmd = f"aie-opt {output_mlir} --aie-extract-fifo"
        extract_fifo = subprocess_run_cmd(cmd=extract_fifo_cmd, cwd=self.root_dir)
        extract_fifo.check()

        self.last_built_task = task_name

        return (
            read_text_file(output_mlir),
            read_text_file(self.__get_local_file("netlist.json")),
            # TODO: merge pnr format file and route summary file
            read_text_file(self.__get_local_file("./build/route_summary.json")),
            f"{build}\n{extract_fifo}",  # for debugging
        )

    def place_and_route_task(
        self, task_name, pnr_args, imported_pnr_file, imported_route_summary_file
    ):
        _, output_mlir = self.__get_task(task_name)
        output_mlir = self.__get_local_file(output_mlir)

        assert self.last_built_task == task_name, (
            f"Task {task_name} has not been built yet. "
            f"Current built task: {self.last_built_task}"
        )

        if (imported_pnr_file is not None) and (
            imported_route_summary_file is not None
        ):
            shutil.copyfile(
                imported_pnr_file, os.path.join(self.root_dir, "placed.json")
            )
            shutil.copyfile(
                imported_route_summary_file,
                os.path.join(self.root_dir, "route_summary.json"),
            )
            pnr = (
                f"Imported PnR results:\n"
                "    - Placed file:   {imported_pnr_file}\n"
                "    - Route summary: {imported_route_summary_file}\n"
            )
        else:
            pnr_bin = os.path.expandvars("$NPU_PNR_BIN_DIR/placer")
            assert os.path.exists(pnr_bin), f"PnR binary not found at {pnr_bin}"
            pnr_cmd = str(
                f"{pnr_bin} netlist.json "
                "--output=placed.json "
                "--route-summary=route_summary.json"
            )
            if pnr_args is not None:
                pnr_cmd += f" {pnr_args}"

            pnr = subprocess_run_cmd(cmd=pnr_cmd, cwd=self.root_dir)
            pnr.check()

        rename_cmd = "mv -f placed.json netlist.json"
        subprocess_run_cmd(cmd=rename_cmd, cwd=self.root_dir).check()

        place_fifo_cmd = f"aie-opt {output_mlir} --aie-place-tiles"
        place_fifo = subprocess_run_cmd(cmd=place_fifo_cmd, cwd=self.root_dir)
        place_fifo.check()
        write_text_file(output_mlir, place_fifo.stdout)

        return (
            place_fifo.stdout,
            read_text_file(self.__get_local_file("netlist.json")),
            # Use PnR generated route summary
            # TODO: consider switching to standard flow generated route summary
            read_text_file(self.__get_local_file("route_summary.json")),
            str(pnr),
        )

    def run_task(self, task_name):
        assert self.last_built_task == task_name, (
            f"Task {task_name} has not been built yet. "
            f"Current built task: {self.last_built_task}"
        )
        param_envs, _ = self.__get_task(task_name)
        result = subprocess_run_cmd(cmd=self.run_cmd, cwd=self.root_dir, env=param_envs)
        result.check()
        return result.stdout, result.stderr, str(result)

    def custom_task(self, task_name, custom_cmd):
        param_envs, _ = self.__get_task(task_name)
        custom = subprocess_run_cmd(cmd=custom_cmd, cwd=self.root_dir, env=param_envs)
        custom.check()
        return str(custom)

    def __repr__(self):
        return (
            f"Benchmark(name={self.name}, tasks={[x.get("name") for x in self.tasks]})"
        )


def main_routine(
    benchmark_root,
    benchmark_name,
    task_name,
    build,
    use_placed,
    iron_placer,
    pnr_after_build,
    pnr_args,
    imported_pnr_result_dir,
    run_after_build,
    hook_script,
    output_dir,
    verbose,
):
    def log(msg):
        print(f"[{benchmark_name}/{task_name}] {msg}", flush=True)

    output_dir = os.path.join(os.path.abspath(output_dir), benchmark_name)
    os.makedirs(output_dir, exist_ok=True)

    imported_dir = None
    if imported_pnr_result_dir is not None:
        imported_dir = os.path.join(
            os.path.abspath(imported_pnr_result_dir), benchmark_name
        )

    config_path = os.path.join(benchmark_root, benchmark_name, "config.yml")
    assert os.path.exists(config_path)

    benchmark = Benchmark(config_path=config_path, use_placed=use_placed, iron_placer=iron_placer)

    log("Cleaning ...")
    benchmark.clean_task_build()

    if build:
        log("Building ...")
        mlir, netlist, std_route, build_log = benchmark.build_task(task_name)

        write_text_file(
            os.path.join(output_dir, f"{task_name}.build.mlir"),
            mlir,
        )

        write_text_file(
            os.path.join(output_dir, f"{task_name}.build.json"),
            netlist,
        )

        write_text_file(
            os.path.join(output_dir, f"{task_name}.route_summary.build.json"),
            std_route,
        )

        write_text_file(
            os.path.join(output_dir, f"{task_name}.build.log"),
            build_log,
        )

    if pnr_after_build:
        log("Placing and routing ...")

        if imported_dir is not None:
            imported_pnr_file = os.path.join(imported_dir, f"{task_name}.pnr.json")
            imported_route_summary_file = os.path.join(
                imported_dir, f"{task_name}.route_summary.pnr.json"
            )
            if not os.path.exists(imported_pnr_file):
                imported_pnr_file = None
            if not os.path.exists(imported_route_summary_file):
                imported_route_summary_file = None
        else:
            imported_pnr_file = None
            imported_route_summary_file = None

        pnr_mlir, pnr_netlist, pnr_route, pnr_log = benchmark.place_and_route_task(
            task_name, pnr_args, imported_pnr_file, imported_route_summary_file
        )

        write_text_file(
            os.path.join(output_dir, f"{task_name}.pnr.mlir"),
            pnr_mlir,
        )

        write_text_file(
            os.path.join(output_dir, f"{task_name}.pnr.json"),
            pnr_netlist,
        )

        write_text_file(
            os.path.join(output_dir, f"{task_name}.route_summary.pnr.json"),
            pnr_route,
        )

        write_text_file(
            os.path.join(output_dir, f"{task_name}.pnr.log"),
            pnr_log,
        )

    if run_after_build:
        log("Running ...")
        run_stdout, run_stderr, run_log = benchmark.run_task(task_name)
        if verbose:
            log(f"Run stdout: {run_stdout}")
            log(f"Run stderr: {run_stderr}")

        write_text_file(
            os.path.join(output_dir, f"{task_name}.stdout.run.log"),
            run_stdout,
        )

        write_text_file(
            os.path.join(output_dir, f"{task_name}.stderr.run.log"),
            run_stderr,
        )

        write_text_file(
            os.path.join(output_dir, f"{task_name}.run.log"),
            run_log,
        )

    if hook_script is not None:
        log(f"Executing hook `{hook_script}` ...")

        hook_script = hook_script.split(" ")
        if os.path.exists(hook_script[0]):
            # First argument is a file path
            hook_script[0] = os.path.abspath(hook_script[0])
        else:
            # First argument is a command, try to find it in PATH
            resolved_path = shutil.which(hook_script[0])
            assert (
                resolved_path is not None
            ), f"Hook command or hook script file {hook_script[0]} does not exist"
            hook_script[0] = resolved_path
        hook_script = " ".join(hook_script)

        exec_log = benchmark.custom_task(task_name, hook_script)
        write_text_file(
            os.path.join(output_dir, f"{task_name}.hook.log"),
            exec_log,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build and optionally run benchmarks")
    parser.add_argument(
        "tasklists",
        type=str,
        nargs="*",
        help="Path to the task list file or strings of benchmark and task name pairs (e.g., `vector_scalar_add/default`)",
    )
    parser.add_argument(
        "--benchmark-root",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/../benchmarks",
        help="Root directory of benchmarks (default: /path/to/npu-flow/benchmarks)",
        required=False,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="build",
        help="Directory to store build outputs (default: ./build)",
        required=False,
    )
    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the benchmark (default: True)",
        required=False,
    )
    parser.add_argument(
        "--placed-iron",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the placed version of the IRON benchmark (default: Placed IRON)",
    )
    parser.add_argument(
        "--iron-placer",
        type=str,
        default="sequential_placer",
        choices=["sequential_placer", "null_placer"],
        help="Placer to use for the benchmark (default: sequential_placer)",
        required=False,
    )
    parser.add_argument(
        "--pnr",
        action="store_true",
        default=False,
        help="Run place-and-route after building (default: False)",
        required=False,
    )
    parser.add_argument(
        "--pnr-args",
        type=str,
        default=None,
        help="Additional arguments to pass to the place-and-route tool (default: None)",
        required=False,
    )
    parser.add_argument(
        "--import-pnr-results",
        type=str,
        default=None,
        help="""Path to an existing directory with PnR results to import
        instead of running PnR as part of the stage in this script.\nThe
        directory structure should be the same as the output directory of
        this script, with files named as <task_name>.pnr.json located in
        each benchmark subdirectory (Default: None)""",
        required=False,
    )
    parser.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Run the benchmark after the build/PnR stage (default: False)",
        required=False,
    )
    parser.add_argument(
        "--hook",
        type=str,
        default=None,
        help="""Custom processing command, or path to a custom processing
        script (python/bash/etc.) to be executed for each task (default: None)""",
        required=False,
    )
    parser.add_argument(
        "-j",
        type=int,
        default=1,
        help="""Number of parallel benchmarks to process (default: 1); within the same
        benchmark, tasks are processed sequentially due to the task clean method""",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging (default: False)",
        required=False,
    )

    args = parser.parse_args()

    if args.tasklists:
        os.makedirs(args.output_dir, exist_ok=True)

        def handle_task(task_entry):
            benchmark_name, task_name_list = task_entry
            for task_name in task_name_list:
                try:
                    main_routine(
                        benchmark_root=args.benchmark_root,
                        benchmark_name=benchmark_name,
                        task_name=task_name,
                        build=args.build,
                        use_placed=args.placed_iron,
                        iron_placer=args.iron_placer,
                        pnr_after_build=args.pnr,
                        pnr_args=args.pnr_args,
                        imported_pnr_result_dir=args.import_pnr_results,
                        run_after_build=args.run,
                        hook_script=args.hook,
                        output_dir=args.output_dir,
                        verbose=args.verbose,
                    )
                except Exception as e:
                    print(f"[{benchmark_name}/{task_name}] Error: {e}", flush=True)
                    if args.verbose:
                        traceback.print_exc()

        parallel_benchmarks = {}

        def store_task_entry(entry):
            if "/" in entry:
                benchmark_name, task_name = entry.split("/", 1)
                if benchmark_name not in parallel_benchmarks:
                    parallel_benchmarks[benchmark_name] = []
                parallel_benchmarks[benchmark_name].append(task_name)
            else:
                raise ValueError(
                    f"Invalid task entry '{entry}'. Expected format: 'benchmark_name/task_name'"
                )

        for arg_item in args.tasklists:
            if os.path.isfile(arg_item):
                with open(arg_item, "r") as f:
                    for task in yaml.safe_load(f):
                        store_task_entry(task)
            else:
                store_task_entry(arg_item)

        with Pool(args.j) as pool:
            job_list = []
            for benchmark_name, task_name_list in parallel_benchmarks.items():
                job_list.append((benchmark_name, task_name_list))
            pool.map(handle_task, job_list)
