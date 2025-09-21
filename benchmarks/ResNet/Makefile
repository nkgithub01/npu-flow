##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

device_name := npu2
target_name := resnet
aie_py_src=${target_name}.py

use_placed ?= 1
ifeq (${use_placed}, 0)
aie_py_src=${target_name}.py
else ifeq (${use_placed}, 1)
aie_py_src=${target_name}_placed.py
endif

placer ?= sequential_placer

VERBOSITY := 2
TRACE_SIZE := 16384
NUM_ITERATION := 100
WARMUP_ITERATION := 50
mlir_target?=build/${target_name}.mlir
trace_mlir_target?=build/${target_name}_trace.mlir
xclbin_target?=build/${target_name}.xclbin
trace_xclbin_target?=build/${target_name}_trace.xclbin
insts_target?=build/${target_name}_insts.bin

aieargs := --dev ${device_name}
runargs := --iters ${NUM_ITERATION} --warmup ${WARMUP_ITERATION}

ifeq (${use_placed}, 0)
aieargs+=--placer ${placer}
endif

ifeq (${placer}, sa_placer)
aieargs+=--pnr-args "${pnr_args}"
endif

build: ${mlir_target} build/conv2dk1_i8.o build/conv2dk1_skip_init.o build/conv2dk3.o build/conv2dk1_skip.o build/conv2dk1_ui8.o
aiecc: ${xclbin_target}

all: ${xclbin_target}

${mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@

build/conv2dk1_i8.o: ${AIE_KERNELS_DIR}/conv2dk1.cc
	mkdir -p ${@D}
ifeq (${device_name}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DINT8_ACT -c $< -o ${@F}
else ifeq (${device_name}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DINT8_ACT -c $< -o ${@F}
endif

build/conv2dk3.o: ${AIE_KERNELS_DIR}/conv2dk3.cc
	mkdir -p ${@D}
ifeq (${device_name}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DUINT8_ACT -c $< -o ${@F}
else ifeq (${device_name}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DUINT8_ACT -c $< -o ${@F}
endif

build/conv2dk1_skip_init.o: ${AIE_KERNELS_DIR}/conv2dk1_skip_init.cc
	mkdir -p ${@D}
ifeq (${device_name}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DINT8_ACT -c $< -o ${@F}
else ifeq (${device_name}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DINT8_ACT -c $< -o ${@F}
endif

build/conv2dk1_ui8.o: ${AIE_KERNELS_DIR}/conv2dk1.cc
	mkdir -p ${@D}
ifeq (${device_name}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DUINT8_ACT -c $< -o ${@F}
else ifeq (${device_name}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DUINT8_ACT -c $< -o ${@F}
endif

build/conv2dk1_skip.o: ${AIE_KERNELS_DIR}/conv2dk1_skip.cc
	mkdir -p ${@D}
ifeq (${device_name}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DUINT8_ACT -c $< -o ${@F}
else ifeq (${device_name}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DUINT8_ACT -c $< -o ${@F}
endif

${xclbin_target}: ${mlir_target} build/conv2dk1_i8.o build/conv2dk1_skip_init.o build/conv2dk3.o build/conv2dk1_skip.o build/conv2dk1_ui8.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary ${aiecc_extra_args} \
				--xclbin-name=${@F} --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

.PHONY: run
run: ${xclbin_target}
	python3 ${srcdir}/test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE ${runargs} 

.PHONY: clean_trace
clean_trace:
	rm -rf trace

.PHONY: clean
clean: clean_trace
	rm -rf build output log
