srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

device_name := npu2
dtype_in := i8
dtype_out := i16

target_name := GEMM
aie_py_src := ${target_name}.py

use_placed ?= 1
ifeq (${use_placed}, 0)
aie_py_src=${target_name}.py
else ifeq (${use_placed}, 1)
aie_py_src=${target_name}_placed.py
endif

placer ?= sequential_placer

# AIE kernels directory for Matrix Multiplication is in different location
AIE_KERNELS_DIR := ${MLIR_AIE_DIR}/aie_kernels/aie2p

# M, K, N: final GEMM dimensions
M:=512
K:=512
N:=512

# m, k, n: single AIE core GEMM dimensions
m:=32
k:=32
n:=32

n_aie_cols?=8
b_col_maj?=0

kernels:=mm_${m}x${k}x${n}
aieargs:=-m $m -k $k -n $n --n_aie_cols ${n_aie_cols} --b_col_maj ${b_col_maj}
runargs:=--b_col_maj ${b_col_maj}
target_suffix:=${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c

# set this flag to 1 for linear buffer allocation
# else, 0 for bank-aware
use_linear_buf_alloc?=1
ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

# Trace size
VERBOSITY := 2
TRACE_SIZE := 1048576
NUM_ITERATION := 30
WARMUP_ITERATION := 20
mlir_target?=build/aie_${target_suffix}.mlir
trace_mlir_target?=build/aie_trace_${target_suffix}.mlir
xclbin_target?=build/GEMM_${target_suffix}.xclbin
trace_xclbin_target?=build/GEMM_trace_${target_suffix}.xclbin
insts_target?=build/insts_${target_suffix}.bin

aieargs+=--dev ${device_name} -M $M -K $K -N $N --dtype_in ${dtype_in} --dtype_out ${dtype_out}
runargs+=-M $M -K $K -N $N -dtype_in=${dtype_in} -dtype_out=${dtype_out} -v ${VERBOSITY} --iters ${NUM_ITERATION} --warmup ${WARMUP_ITERATION}

ifeq (${use_placed}, 0)
aieargs+=--placer ${placer}
endif

.PHONY: all
all: ${xclbin_target}

build/%.cc.o: ${AIE_KERNELS_DIR}/%.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

${mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@

${trace_mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --trace_size ${TRACE_SIZE} > $@

build/mm_b_row_maj_${m}x${k}x${n}.o: ${AIE_KERNELS_DIR}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}

build/mm_b_col_maj_${m}x${k}x${n}.o: ${AIE_KERNELS_DIR}/mm_b_col_maj.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}

ifneq (${b_col_maj}, 1)
build/mm_${m}x${k}x${n}.o: build/mm_b_row_maj_${m}x${k}x${n}.o 
	mkdir -p ${@D}
	ar rvs $@ $^
else
build/mm_${m}x${k}x${n}.o: build/mm_b_col_maj_${m}x${k}x${n}.o
	mkdir -p ${@D}
	ar rvs $@ $^
endif

${xclbin_target}: ${mlir_target} ${kernels:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=${buffer_aloc_flag} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary \
				--xclbin-name=${@F} --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

${trace_xclbin_target}: ${trace_mlir_target} ${kernels:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=${buffer_aloc_flag} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary \
				--xclbin-name=${@F} --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

.PHONY: run
run: ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1
	python3 ${srcdir}/test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE ${runargs}

.PHONY: run_trace
run_trace: ${trace_xclbin_target}
	@mkdir -p trace
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1
	python3 ${srcdir}/test.py -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE --iters 1 ${runargs} \
			-t ${TRACE_SIZE} --trace_file trace/trace.txt
	${AIE_PARSE_TRACE_DIR}/parse_trace.py --input trace/trace.txt --mlir ${trace_mlir_target} --output trace/trace.json
	python3 ${srcdir}/../../utils/get_cycle_summary.py --input trace/trace.json --output trace/summary.txt

.PHONY: clean_trace
clean_trace:
	rm -rf trace

.PHONY: clean
clean: clean_trace
	rm -rf build output

