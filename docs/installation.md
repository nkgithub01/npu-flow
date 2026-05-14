# Installation

This guide explains how to install NPU Flow from the project root referenced by the top-level README.

## Hardware

NPU Flow targets AMD Ryzen AI NPUs with AIE-v2 arrays. The tested reference system uses:

- Beelink SER9 Pro Mini PC
- AMD Ryzen AI 9 HX 370 processor
- 8-column x 6-row XDNA NPU
- 32 GB LPDDR5x-7500 RAM

A compatible Ryzen AI system should work, but the configuration above is the validated setup. An NPU device is required to collect hardware runtime measurements. Without an NPU, you can still build the tool flow and collect compile/PnR metrics with Docker.

## Software prerequisites

The tested software environment uses:

- Ubuntu 25.04, Linux kernel 6.14.0
- G++ 14.20
- CMake 3.31.6
- SQLite 3.46.1
- Google OR-Tools 9.12 with the SCIP solver
- XRT 2.20.0 and matching XDNA/NPU driver
- The MLIR-AIE fork included with the project

Install the common Ubuntu packages:

```bash
sudo apt update
sudo apt install build-essential cmake sqlite3 libsqlite3-dev catch2 libgtest-dev lsb-release
```

Newer package versions may work, but they have not been tested.

## Install with an NPU device

Run these commands from a clean checkout of the project repository.

### 1. Build MLIR-AIE

```bash
cd <repo-root>/mlir-aie
python3 -m venv ironenv
source ironenv/bin/activate
python3 -m pip install --upgrade pip

python3 -m pip install -r python/betzgrp_requirements.txt
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
python3 -m pip install -r python/requirements_ml.txt
python3 -m pip install opencv-python

bash utils/build-mlir-aie-from-wheels.sh
source utils/betzgrp_setup.sh
```

### 2. Check the MLIR-AIE setup

```bash
cd <repo-root>/benchmarks/vector_scalar_mul
./run.sh
```

### 3. Build NPU-PnR

```bash
cd <repo-root>
wget -O or-tools.tar.gz "https://github.com/google/or-tools/releases/download/v9.12/or-tools_amd64_ubuntu-24.04_cpp_v9.12.4544.tar.gz"
mkdir -p "$PWD/or-tools"
tar -zxf or-tools.tar.gz -C "$PWD/or-tools" --strip-components=1
export OR_TOOLS_DIR="$PWD/or-tools"

cd npu-pnr
mkdir -p build
cd build
cmake .. -DCMAKE_CXX_STANDARD=20 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$OR_TOOLS_DIR"
make -j8

cd <repo-root>
export NPU_PNR_BIN_DIR="$PWD/npu-pnr/build"
```

### 4. Verify the NPU runtime environment

Check that XRT can see the NPU:

```bash
xrt-smi examine
```

The tested setup used XRT 2.20.0 with NPU firmware 1.0.18.3. Driver and firmware mismatches can cause runtime failures even if the software build succeeds.

## Install without an NPU device

Use the Docker flow when no compatible NPU device is available. This supports compile-only and PnR metric collection, but not NPU runtime measurements.

```bash
cd <repo-root>
docker build -t npu-flow-image .
docker run -it --rm npu-flow-image
```

## Next steps

After installation, use the benchmark scripts under `benchmarks/` or the collection scripts from the project root. The mini collection is the quickest end-to-end smoke test; the largest sweeps can take several days on the tested machine.
