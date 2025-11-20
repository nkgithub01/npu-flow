FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update apt and install required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    catch2 \
    git \
    ca-certificates \
    wget \
    sudo

# Set the working directory
WORKDIR /workspace

# Copy the entire repository into the container
COPY . /workspace/

# Make the scripts executable
RUN chmod +x .github/scripts/install_ci_xrt.sh \
    && chmod +x .github/scripts/install_mlir_aie.sh \
    && chmod +x .github/scripts/install_or_tools.sh

# Install XRT
RUN bash -c "source .github/scripts/install_ci_xrt.sh"

# Install MLIR-AIE
RUN bash -c "source .github/scripts/install_mlir_aie.sh"
ENV MLIR_AIE_DIR=/workspace/mlir-aie
ENV MLIR_AIE_INSTALL_DIR=$MLIR_AIE_DIR/install
ENV PEANO_INSTALL_DIR=$MLIR_AIE_DIR/ironenv/lib/python3.12/site-packages/llvm-aie
ENV PATH=$MLIR_AIE_INSTALL_DIR/bin:$PATH
ENV PYTHONPATH=$MLIR_AIE_INSTALL_DIR/python
ENV LD_LIBRARY_PATH=$MLIR_AIE_INSTALL_DIR/lib
ENV PEANO_CLANG=$PEANO_INSTALL_DIR/bin/clang++
ENV NPU2=1

# Install OR-Tools
RUN bash -c "source .github/scripts/install_or_tools.sh 24.04"
ENV OR_TOOLS_DIR=/workspace/or-tools
ENV LD_LIBRARY_PATH=$OR_TOOLS_DIR/lib:$LD_LIBRARY_PATH
ENV CPATH=$OR_TOOLS_DIR/include
ENV PATH=$OR_TOOLS_DIR/bin:$PATH

# Install PnR Tool
ENV NPU_PNR_BUILD_DIR=/workspace/npu-pnr/build
RUN mkdir -p $NPU_PNR_BUILD_DIR && \
    cd $NPU_PNR_BUILD_DIR && \
    cmake .. -DCMAKE_PREFIX_PATH=$OR_TOOLS_DIR -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make test
ENV NPU_PNR_BIN_DIR=$NPU_PNR_BUILD_DIR/apps
ENV PATH=$NPU_PNR_BUILD_DIR:$PATH

# Add environment setups to .bashrc
RUN echo "source $MLIR_AIE_DIR/ironenv/bin/activate" >> ~/.bashrc
RUN echo "source /opt/xilinx/xrt/setup.sh" >> ~/.bashrc

# Set the entrypoint
ENTRYPOINT ["/bin/bash"]
