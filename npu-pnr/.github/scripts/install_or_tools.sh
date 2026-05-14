#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <ubuntu_version, e.g., 24.04 or 24.10>"
  exit 1
fi

UBUNTU_VERSION=$1

sudo apt install -y build-essential wget tar

# TODO: make or-tools version configurable
wget -O or-tools.tar.gz "https://github.com/google/or-tools/releases/download/v9.12/or-tools_amd64_ubuntu-${UBUNTU_VERSION}_cpp_v9.12.4544.tar.gz"
mkdir -p or-tools
tar -xvf or-tools.tar.gz --strip-components 1 -C or-tools

export ortools_DIR=$(realpath ./or-tools)
export LD_LIBRARY_PATH="${ortools_DIR}/lib:${LD_LIBRARY_PATH}"
export CPATH="${ortools_DIR}/include:${CPATH}"
export PATH="${ortools_DIR}/bin:${PATH}"