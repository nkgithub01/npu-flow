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
export NPU_PNR_BIN_DIR=$PWD/npu-pnr/build
```

### 3. Running Benchmark Tests
```
Usage: utils/build_benchmarks.py tasklist
```
#### Positional Arguments
| Argument | Description | Example |
|----------|-------------|---------|
| `tasklist` |  Path to task list yaml or specific benchmark_name/task_name pairs | `benchmarks/tasklist.yml` or  `edge_detection/col_1`

#### Optional Arguments
| Category | Flag | Default | Description |
|--------|------|---------|-------------|
| Output | `--output-dir` | `build` | Directory to store build outputs |
| Pipeline | `--clean-all` | `False` | Remove all generated build artifacts in `--output-dir` before running |
| Pipeline | `--build / --no-build` | `True` | Build stage: creates MLIR from IRON and optionally runs PnR |
| Pipeline | `--compile / --no-compile` | `True` | Compile MLIR to executable binary (required for AIE router) |
| Pipeline | `--run` | `False` | Run the benchmark on NPU after compilation |
| Placer | `--placer.type` | `aie` | Select placement strategy (`aie`, `pnr`, `hand_placed`) |
| PnR Placer | `--placer.pnr.type` | `sa` | PnR placer backend to use |
| PnR Placer | `--placer.pnr.args` | `None` | Extra arguments forwarded directly to the PnR tool |
| Router | `--router.type` | `aie` | Select routing strategy (`aie`, `pnr`) |
| PnR Router | `--router.pnr.type` | `milp` | PnR router backend to use |
| AIE Router | `--router.aie.use-pkt-routing / --no-router.aie.use-pkt-routing` | `False` | Use packet-switched routing in AIE router |
| Debug | `-j` | `None` | Parallelism level (implementation-defined) |
| Debug | `--import-pnr-results` | `None` | Import placement/routing results from previous PnR run |
| Debug | `--import-pnr-iter` | `None` | Import specific PnR iteration |
| Debug | `--verbose` | `False` | Enable verbose logging |
| Debug | `--debug` | `True` | Enable debug mode |
| Debug | `--netlist-only` | `False` | Generate netlist only and exit |
| Debug | `--run-only` | `False` | Run without build/compile (assumes prior build exists) |
| Debug | `--hook` | `None` | Custom hook for development or debugging |

#### Example Usage
This command runs all tasks defined in `benchmarks/tasklist.yml`, using PnR's SA placer and run 100 SA iterations, and uses PnR's milp router, then runs results on the NPU.
```
utils/build_benchmarks.py benchmarks/tasklist.yml --placer.type pnr --placer.pnr.type sa --placer.pnr.args="-n 100" --router.type pnr --router.pnr.type milp --run --output-dir=./build
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

## Docker Setup

To build and run the Docker container that replicates the testing environment, follow these steps:

### Build the Docker Image

Navigate to the root of the project directory and run the following command to build the Docker image:

```bash
git clone https://github.com/ueqri/npu-flow.git
cd npu-flow # Recommended to build from clean project root
git submodule update --init --recursive
docker build -t npu-flow .
```

Note: To distinguish between different builds, you can tag the image with a specific version or commit hash using the `-t` option, e.g., `-t npu-flow:latest` or `-t npu-flow:<commit-hash>`. Please refer to the [Tagging images|Docker Docs](https://docs.docker.com/get-started/docker-concepts/building-images/build-tag-and-publish-an-image/#tagging-images) for more details.

### Run the Docker Container

Once the image is built, you can run the container using:

```bash
docker run -it --rm npu-flow
# /workspace should contain all project files
```

This command will start a new container from the `npu-flow` image and drop you into a bash shell where you can execute commands in the configured environment.

## Apptainer Setup

Clone a fresh copy of **npu-flow** (recommended to build from a clean project root) and build image:

```bash
git clone https://github.com/ueqri/npu-flow.git
cd npu-flow 
git submodule update --init --recursive
sudo apptainer build image_name.sif apptainer.def
```

## Run Commands with Apptainer

To run commands using the container:

```bash
apptainer run image_name.sif <your command>
```

It is recommended to bind-mount a directory **outside the repository** for all build or benchmark outputs.  
This prevents generated files from appearing inside the repo and avoids accidentally committing them.

```bash
mkdir -p /path/to/folder/outside/repo

apptainer run \
    --bind /path/to/folder/outside/repo:/build_output \
    image_name.sif \
    python3 utils/build_benchmarks.py benchmarks/tasklist.yml \
        --placer="sa_placer" \
        --pnr-args="-n 10" \
        --output-dir=./build_output
```
In this case `build_output` will refer to the external folder.

## Interactive Shell (Like Docker's -it)

To drop into an interactive bash shell, you need to then manually activate environments.
```bash
apptainer shell image_name.sif
source /workspace/mlir-aie/ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh
```

## File Transfer to Compute Canada
```bash
scp image_name.sif username@hostname.computecanada.ca:~/projects/def-vaughn/npu-flow-images
```
