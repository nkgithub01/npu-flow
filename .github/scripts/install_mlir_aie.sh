sudo apt update
sudo apt install -y build-essential clang clang-14 lld lld-14 cmake ninja-build python3-venv python3-pip

cd mlir-aie
python3 -m venv ironenv
source ironenv/bin/activate
python3 -m pip install --upgrade pip

python3 -m pip install -r python/requirements.txt
pre-commit install
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
python3 -m pip install -r python/requirements_ml.txt
python3 -m pip install opencv-python
python3 -m pip install toml openpyxl

python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

utils/clone-llvm.sh
utils/build-llvm-local.sh

utils/build-mlir-aie.sh llvm/build

export MLIR_AIE_INSTALL_DIR=$(realpath ./install)
export PEANO_INSTALL_DIR=$(realpath ./ironenv/lib/python3.12/site-packages/llvm-aie)
export PATH="${MLIR_AIE_INSTALL_DIR}/bin:${PATH}"
export PYTHONPATH="${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}"
export LD_LIBRARY_PATH="${MLIR_AIE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
export PEANO_CLANG="${PEANO_INSTALL_DIR}/bin/clang++"

source /opt/xilinx/xrt/setup.sh
