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
