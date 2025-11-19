# NPU-Flow
## 1. Initial Setup

```bash
git clone https://github.com/ueqri/npu-flow.git
cd npu-flow
git submodule update --init --recursive
```

### 1.1 Setup MLIR-AIE

#### Setup Virtual Environment

```bash
cd mlir-aie
python3 -m venv ironenv
source ironenv/bin/activate
python3 -m pip install --upgrade pip
```

#### Install Python requirements

```bash
python3 -m pip install -r python/requirements.txt

# This installs the pre-commit hooks defined in .pre-commit-config.yaml
pre-commit install

# Install MLIR Python Extras
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt

# Install Torch for ML examples
python3 -m pip install -r python/requirements_ml.txt
python3 -m pip install opencv-python
```

#### Build Peano (llvm-aie) from Wheels

```bash
python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

#### Build llvm/mlir-aie from Source

```bash
utils/clone-llvm.sh
utils/build-llvm-local.sh

# Build mlir-aie using local llvm
utils/build-mlir-aie.sh llvm/build
```

#### Setup Environment and Add to PATHs

```bash
source utils/betzgrp_setup.sh
```

#### Quick Test for MLIR-AIE Setup

```bash
cd npu-flow/benchmarks/vector_scalar_mul
./run.sh
```

### 1.2 Setup NPU-PnR

Prerequisite: [Google OR-Tools for C++](https://developers.google.com/optimization/install/cpp)

```bash
cd npu-pnr
mkdir build && cd build
cmake ..
make -j
```
### 2. Activate Environment

```bash
cd npu-flow
source mlir-aie/utils/betzgrp_setup.sh
source mlir-aie/ironenv/bin/activate
export NPU_PNR_BIN_DIR=$PWD/npu-pnr/build/apps
```

### 3. Running Benchmark Tests
```
Usage: utils/build_benchmarks.py tasklist
```
#### Positional Arguments
| Argument | Description | Example |
|----------|-------------|---------|
| `tasklist` |  Path to task list yaml or specific benchmark_name/task_name pairs | `benchmarks/tasklist.yml` or  `vector_scalar_add/default`

#### Optional Arguments
| Flag | Type | Default | Description
|----------|-------------|---------|---------|
| `--output-dir` | `str` | `build` | Directory to store all build and run outputs. |
| `--clean-all` | `store_true` | `False` | Cleans the entire output directory before starting. |
| `--placer` | `str` | `sequential_placer` | See choices below. |
| `--pnr-args` | `str` | `None` | Arguments to pass directly to the Placement and Routing (PnR) tool. |
| `--run` | `store_true` | `False` | Run after build/compile. |
| `--run-only` | `store_true` | `False` | Run task without building or compiling. Assumes the `--output-dir` was previously built/compiled. Skips tasks with build/compile errors. |
| `--netlist-only` | `store_true` | `False` | Builds only the netlists for the specified task list. A tar file output will be generated in the output directory. |
| `--import-pnr-results` | `str`| `None`| Path to import pre-existing PnR results from. |
| `--j` | `int` | `None` | Number of parallel jobs to run. Defaults to system CPU count. |

#### Placer Choices (`--placer`)
| Value | Description |
| --- | --- |
| `sequential_placer`| MLIR-AIE's sequential placement.
| `sa_placer`| PnR Placer.
|`hand_placed`| Manual placement.

#### Example Usage
This command runs all tasks defined in `benchmarks/tasklist.yml`, using `sa_placer` with 100 sa iterations.
```
utils/build_benchmarks.py benchmarks/tasklist.yml --placer="sa_placer" --pnr-args="-n 100" --run --output-dir=./build
```
 The output directory will look as follows:
```
output_dir/
└── benchmark_name/
    └── task_name/
        ├── build/
        │   ├── netlist.json
        │   ├── pnr_placed_netlist.json
        │   ├── pnr_route_summary.json
        │   └── post_compile_routing_summary.json
        ├── task_name.build.log
        ├── task_name.compile.log
        └── task_name.run.log
```

### 4.0 [TODO: Update Parsing Instructions] Parse Results and Generate CSV Table
```bash
# If PnR placement and routing was used
python3 utils/parse_results.py --variant=pnr \
--input-dir=path/to/build/folder --output-csv=path/to/csv

# If MLIR-AIE standard routing was used
python3 utils/parse_results.py --variant=std \
--input-dir=path/to/build/folder --output-csv=path/to/csv
```

### Parse CSV Table and Generate Excels
Simple Excel generation script that takes in at least 1 CSV file. It treats the first csv as the baseline.
For each input CSV file, the script generate 4 sheets:
- Original Data
- Filtered out failed testcases which have runtime == -1.0 in that CSV file
- Filtered out testcases that fails in any of the CSV file given
- Normalized the results with respect to the baseline (first CSV file)
The script will add an overall summary sheet as well.
```bash
python3 utils/python utils/compare_results.py \
-f path/to/csv1 path/to/csv2 path/to/csv3 \
-l sheet_name_for_csv1 sheet_name_for_csv2 sheet_name_for_csv3
```

## PnR

### PnR Arguments

```
Usage: placer [--help] [--verbose VAR] [--random-seed VAR] [--max-iterations VAR] [--output VAR] [--route-summary VAR] [--enable-packing] [--use-random-initial-placement] [--initial-placement-randomness VAR] [--log-interval VAR] [--start-temperatures VAR...] [--cooling-factors VAR...] [--num-greedy-iterations VAR] netlist
```

| Argument | Description | Default | Notes |
|----------|-------------|---------|-------|
| `netlist` | The path to the input netlist file with the initial solution | - | Positional argument |
| `-h, --help` | Shows help message and exits | - | |
| `--verbose` | The verbosity level of the placer output [silent\|minimal\|normal\|verbose\|debug] | `"verbose"` | nargs: 0 or 1 |
| `-s, --random-seed` | The random seed to use for the placer (-1 means using std::random_device) | `24` | nargs: 0 or 1 |
| `-n, --max-iterations` | The maximum number of iterations to perform during placement (the SA temperature scheduling will respect this setting) | ULLONG_MAX (2^64−1) | nargs: 0 or 1 |
| `-o, --output` | The output path where the placement solution will be saved | `"placed.json"` | nargs: 0 or 1 |
| `--route-summary` | The output path where the route summary will be saved | `"route_summary.json"` | nargs: 0 or 1 |
| `--enable-packing` | Whether to enable packing of memory/shim tiles | - | Flag |
| `-r, --use-random-initial-placement` | Whether to perform a random initial placement before starting the simulated annealing process | - | Flag |
| `--initial-placement-randomness` | The randomness factor when generating random initial placement. Higher value means more randomness. Only used when `--use-random-initial-placement` is set. | `100` | nargs: 0 or 1 |
| `--log-interval` | The number of iterations between each log printout in >=normal verbosity levels | `100` | nargs: 0 or 1 |
| `-T, --start-temperatures` | The start temperatures for different stages of the SA process. The length of this list determines how many stages will be performed (the last stage is always a greedy search with temperature 0). Example: `--start-temperatures` 1000.0 0.5 0.001, together with the example cooling factors in `--cooling-factors` below, which means the first stage (fast cooling) will start at 1000.0, the second stage (slow cooling) will start at 0.5, and the last stage (greedy) will start at 0.001 | `1000.0 0.5 0.001` | nargs: 0 or more |
| `-c, --cooling-factors` | the cooling factors for different stages of the SA process. The length of this list must be equal to the length of the `--start-temperatures` list. The last stage is always a greedy search with cooling factor 0. Example: `--cooling-factors` 0.95 0.99 0, which means the first stage (fast cooling) will use cooling factor 0.95, the second stage (slow cooling) will use cooling factor 0.99, and the last stage (greedy) will use cooling factor 0 | `0.95 0.99 0` | nargs: 0 or more |
| `-g, --num-greedy-iterations` | The number of iterations to perform in the greedy stage (last stage) of the SA process | `200` | nargs: 0 or 1 |

#### Temperature Scheduling

`--start-temperatures` and `--cooling-factors` can be used to configure the SA temperature scheduling. The length of these two lists must be equal, which indicates how many stages the SA process will have. Each stage will start at the corresponding temperature in `--start-temperatures` and use the corresponding cooling factor in `--cooling-factors` to cool down. The last stage is always a greedy search with start temperature (slightly) larger than 0 and cooling factor 0. The number of iterations in the greedy stage can be configured using `--num-greedy-iterations`.

Example schedule `--start-temperatures 1000.0 0.5 0.001 --cooling-factors 0.95 0.99 0`:
- First stage (1000.0 to 0.5): fast cooling (factor=0.95)
- Second stage (0.5 to 0.001): slow cooling (factor=0.99)
- Last stage (after 0.001): greedy (factor=0,T=0)

Note: The SA process will stop when either the maximum number of iterations set by `--max-iterations` is reached, or when all stages scheduled (including the last greedy stage) are completed.