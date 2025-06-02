## NPU Flow

#### Initial Setup

```bash
git clone https://github.com/ueqri/npu-flow.git
cd npu-flow
git submodule update --init --recursive
```
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

#### Run Example
```bash
cd npu-flow/benchmarks/vector_scalar_mul
./run.sh
```