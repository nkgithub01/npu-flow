src_dir ?= $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
build_dir ?= $(src_dir)/build
include ${src_dir}/../makefile-common

device_name := npu2
dtype_in := i16
dtype_out := i16

target_name := cascade_GEMM
aie_py_src := ${target_name}.py

use_placed ?= 1
ifeq (${use_placed}, 0)
aie_py_src=${target_name}.py
else ifeq (${use_placed}, 1)
aie_py_src=${target_name}_placed.py
endif

# M, K, N: final GEMM dimensions
M:=512
K:=512
N:=512

# m, k, n: single AIE core GEMM dimensions
m:=32
k:=32
n:=32

n_aie_cols?=8

kernels:=mm_${m}x${k}x${n}
aieargs:=-m $m -k $k -n $n --n_aie_cols ${n_aie_cols}
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
NUM_ITERATION := 100
WARMUP_ITERATION := 50
mlir_target?=${build_dir}/aie_${target_suffix}.mlir
trace_mlir_target?=${build_dir}/aie_trace_${target_suffix}.mlir
xclbin_target?=${build_dir}/GEMM_${target_suffix}.xclbin
trace_xclbin_target?=${build_dir}/GEMM_trace_${target_suffix}.xclbin
insts_target?=${build_dir}/insts_${target_suffix}.bin
runargs+=-M $M -K $K -N $N -dtype_in=${dtype_in} -dtype_out=${dtype_out} -v ${VERBOSITY} --iters ${NUM_ITERATION} --warmup ${WARMUP_ITERATION}
aieargs+=--dev ${device_name} -M $M -K $K -N $N --dtype_in ${dtype_in} --dtype_out ${dtype_out}

build: ${mlir_target} ${build_dir}/mm_${m}x${k}x${n}.o
aiecc: ${xclbin_target}

.PHONY: all
all: ${xclbin_target}

${mlir_target}: ${src_dir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@

${trace_mlir_target}: ${src_dir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --trace_size ${TRACE_SIZE} > $@

${build_dir}/mm_${m}x${k}x${n}.o: ${AIE_KERNELS_DIR}/cascade_mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -fno-unroll-loops -DBIT_WIDTH=8 -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}

${xclbin_target}: ${mlir_target} ${kernels:%=${build_dir}/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=${buffer_aloc_flag} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary ${aiecc_extra_args} \
				--xclbin-name=${@F} --npu-insts-name=$(notdir $(insts_target)) $(notdir $(<))

${trace_xclbin_target}: ${trace_mlir_target} ${kernels:%=${build_dir}/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=${buffer_aloc_flag} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary ${aiecc_extra_args} \
				--xclbin-name=${@F} --npu-insts-name=$(notdir $(insts_target)) $(notdir $(<))

.PHONY: run
run: ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1
	python3 ${src_dir}/test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE ${runargs} 

.PHONY: run_trace
run_trace: ${trace_xclbin_target}
	@mkdir -p trace
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1
	python3 ${src_dir}/test.py -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE --iters 1 ${runargs} \
			-t ${TRACE_SIZE} --trace_file trace/trace.txt
	${AIE_PARSE_TRACE_DIR}/parse_trace.py --input trace/trace.txt --mlir ${trace_mlir_target} --output trace/trace.json
	python3 ${src_dir}/../../utils/get_cycle_summary.py --input trace/trace.json --output trace/summary.txt

.PHONY: clean_trace
clean_trace:
	rm -rf trace

.PHONY: clean
clean: clean_trace
	rm -rf ${build_dir}

