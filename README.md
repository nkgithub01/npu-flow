# NPU Flow

## Setup

```bash
git clone https://github.com/ueqri/npu-flow.git
cd npu-flow
git submodule update --init --recursive
```

### MLIR-AIE

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

### NPU-PnR

Prerequisite: [Google OR-Tools for C++](https://developers.google.com/optimization/install/cpp)

```bash
cd npu-pnr
mkdir build && cd build
cmake ..
make -j
```

## Build Benchmarks

### Activate Environment

```bash
cd npu-flow
source mlir-aie/utils/betzgrp_setup.sh
source mlir-aie/ironenv/bin/activate
export NPU_PNR_BIN_DIR=$PWD/npu-pnr/build/apps
```

### Build Placed IRON

```bash
# Build benchmarks and run PnR (set -n <num_sa_iters> to determine # SA iters to run; 0 means doing only routing on the initial placement)
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
--build --placed-iron \
--pnr --pnr-args="-n 0" \
--output-dir=./build/placed --verbose

# Parse results and generate CSV table for both std (standard flow) and pnr (PnR flow) variants
python3 utils/parse_results.py --variant=std \
--input-dir=./build/placed --output-csv=placed_std_results.csv
python3 utils/parse_results.py --variant=pnr \
--input-dir=./build/placed --output-csv=placed_pnr_results.csv
```

### Build Unplaced IRON w/ Sequential Placer

```bash
# Build benchmarks and run PnR
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
--build --no-placed-iron --iron-placer="sequential_placer" \
--pnr --pnr-args="-n 0" \
--output-dir=./build/unplaced_seq --verbose

# Parse results and generate CSV table
python3 utils/parse_results.py --variant=std \
--input-dir=./build/unplaced_seq --output-csv=unplaced_seq_std_results.csv
python3 utils/parse_results.py --variant=pnr \
--input-dir=./build/unplaced_seq --output-csv=unplaced_seq_pnr_results.csv
```

### Build Unplaced IRON w/ Null Placer

```bash
# Build benchmarks and run PnR
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
--build --no-placed-iron --iron-placer="null_placer" \
--pnr --pnr-args="-n 0" \
--output-dir=./build/unplaced_null --verbose

# Parse results and generate CSV table
python3 utils/parse_results.py --variant=std \
--input-dir=./build/unplaced_null --output-csv=unplaced_null_std_results.csv
python3 utils/parse_results.py --variant=pnr \
--input-dir=./build/unplaced_null --output-csv=unplaced_null_pnr_results.csv
```

## Run Benchmarks

Simply adding `--run -j 1` to the build commands in the "Build Placed IRON" sections is enough. For example:

```bash
# Run benchmarks with standard flow
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
--build <--placed-iron or --no-placed-iron> \
--run -j 1\
--output-dir=/path/to/build --verbose
```

```bash
# Run benchmarks with PnR flow
python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
--build <--placed-iron or --no-placed-iron> \
--pnr --pnr-args="-n 0" \
--run \
--output-dir=/path/to/build --verbose
```

The same parser script can be used to process device run metrics. For example:

```bash
# Parse results and generate CSV table
python3 utils/parse_results.py --variant=std \
--input-dir=/path/to/build --output-csv=std_result.csv
python3 utils/parse_results.py --variant=pnr \
--input-dir=/path/to/build --output-csv=pnr_result.csv
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

### Parallel Run PnR Tool

```bash
# Generate "./build/<some benchmark>/<some task>.pnr.*" files
python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --pnr --pnr-args="-n 0" --verbose --output-dir=./build

# Load "./build/<some benchmark>/<some task>.pnr.*" files and run PnR in parallel
python3 utils/parallel_pnr.py ./build -o ./tmp-parallel-pnr --pnr-args="-n 100000"

# Run PnR-ed benchmarks on device with "./tmp-parallel-pnr" imported
# Note: --output-dir directory can be changed
python3 utils/build_benchmarks.py benchmarks/tasklist.yml --build --pnr --import-pnr-results ./tmp-parallel-pnr --run -j 1 --verbose --output-dir=./build
```