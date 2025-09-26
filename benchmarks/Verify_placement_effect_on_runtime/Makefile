srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
NETLIST_FILE_DIR := ${srcdir}

include ${srcdir}/../makefile-common
AIE_KERNELS_DIR := ${srcdir}/kernels

device_name := npu2
dtype := i32
inout_size ?= 1
# options for placement: 
# - regular
# - 1-hop
# - 2-hop
# - random
placement ?= regular
placement_seed ?= 0
enable_feedback ?= 0
expand_rate ?= 1000000

target_name := verify
aie_py_src := ${target_name}.py
use_placed ?= 1
ifeq (${use_placed}, 0)
aie_py_src=${target_name}.py
else ifeq (${use_placed}, 1)
aie_py_src=${target_name}_placed.py
endif

placer ?= sequential_placer

num_rows ?= 6
num_cols ?= 8

# set this flag to 1 for linear buffer allocation
# else, 0 for bank-aware
use_linear_buf_alloc?=1
ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

VERBOSITY := 2
TRACE_SIZE := 16384
NUM_ITERATION := 100
WARMUP_ITERATION := 100
mlir_target?=build/${target_name}.mlir
trace_mlir_target?=build/${target_name}_trace.mlir
xclbin_target?=build/${target_name}.xclbin
trace_xclbin_target?=build/${target_name}_trace.xclbin
insts_target?=build/${target_name}_insts.bin

aieargs+=--dev ${device_name} --dtype=${dtype} --inout_size=${inout_size} --placement=${placement} --placement_seed=${placement_seed} --enable_feedback=${enable_feedback} --expand_rate=${expand_rate}
runargs+= -v ${VERBOSITY} --dtype=${dtype} --iters ${NUM_ITERATION} --warmup ${WARMUP_ITERATION} --inout_size=${inout_size}
kernals := accumulate

ifeq (${use_placed}, 0)
aieargs+=--placer ${placer}
endif

build: ${mlir_target}
aiecc: ${xclbin_target}

.PHONY: all
all: ${xclbin_target}

build/%.o: ${AIE_KERNELS_DIR}/%.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

${mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@

${trace_mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --trace_size ${TRACE_SIZE} > $@

${xclbin_target}: ${mlir_target} ${kernals:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=${buffer_aloc_flag} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary ${aiecc_extra_args} \
				--xclbin-name=${@F} --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

${trace_xclbin_target}: ${trace_mlir_target} ${kernals:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=${buffer_aloc_flag} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary ${aiecc_extra_args} \
				--xclbin-name=${@F} --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

.PHONY: run
run: ${xclbin_target}
	python3 ${srcdir}/test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE ${runargs}

.PHONY: run_trace
run_trace: ${trace_xclbin_target}
	@mkdir -p trace
	python3 ${srcdir}/test.py -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE ${runargs} \
			-t ${TRACE_SIZE} --trace_file trace/trace.txt
	${AIE_PARSE_TRACE_DIR}/parse_trace.py --input trace/trace.txt --mlir ${trace_mlir_target} --output trace/trace.json
	python3 ${srcdir}/../../utils/get_cycle_summary.py --input trace/trace.json --output trace/summary.txt

.PHONY: clean_trace
clean_trace:
	rm -rf trace

.PHONY: clean
clean: clean_trace
	rm -rf build output

