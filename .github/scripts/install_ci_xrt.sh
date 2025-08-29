#!/bin/bash

set -e

VERBOSE=1

run_build() {
    if [[ $VERBOSE -eq 1 ]]; then
        "$@"
    else
        "$@" > /dev/null 2>&1
    fi
}

echo "Cloning the XDNA driver repository..."
# Clone the XDNA driver repository and initialize submodules
XDNA_SHA=0e6d303b2cc2b3fe1cf10aba0acbf57a422588fb
git clone https://github.com/amd/xdna-driver.git
export XDNA_SRC_DIR=$(realpath xdna-driver)
cd xdna-driver
echo "Checking out commit $XDNA_SHA..."
git checkout "$XDNA_SHA"
git submodule update --init --recursive

echo "Installing XRT dependencies..."
cd "$XDNA_SRC_DIR"
./tools/amdxdna_deps.sh

echo "Building XRT..."
cd "$XDNA_SRC_DIR/xrt/build"
run_build ./build.sh -npu -opt

echo "Detecting Ubuntu version..."
UBUNTU_VERSION=$(lsb_release -rs)
echo "Ubuntu version detected: $UBUNTU_VERSION"

echo "Installing new XRT packages..."
cd "$XDNA_SRC_DIR/xrt/build/Release"
# Only Ubuntu 24.04 is supported for now
apt install ./xrt_202520.2.20.0_24.04-amd64-base.deb
apt install ./xrt_202520.2.20.0_24.04-amd64-base-dev.deb
