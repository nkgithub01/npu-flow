apt update
apt install -y build-essential cmake lsb-release

# TODO: make or-tools version configurable
wget -O or-tools.tar.gz https://github.com/google/or-tools/releases/download/v9.12/or-tools_amd64_ubuntu-24.04_cpp_v9.12.4544.tar.gz
mkdir or-tools
tar -xvf or-tools.tar.gz --strip-components 1 -C or-tools

export OR_TOOLS_DIR=$(realpath ./or-tools)
export LD_LIBRARY_PATH="${OR_TOOLS_DIR}/lib:${LD_LIBRARY_PATH}"
export CPATH="${OR_TOOLS_DIR}/include:${CPATH}"
export PATH="${OR_TOOLS_DIR}/bin:${PATH}"
