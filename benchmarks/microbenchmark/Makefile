src_dir ?= $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
build_dir ?= ${src_dir}/build
NETLIST_FILE_DIR := ${build_dir}/output

include ${src_dir}/../makefile-common
AIE_KERNELS_DIR := ${src_dir}/kernels

device_name := npu2
dtype := i32

target_name := microbenchmark
aie_py_src := ${target_name}.py
use_placed ?= 1
ifeq (${use_placed}, 0)
aie_py_src=${target_name}.py
else ifeq (${use_placed}, 1)
aie_py_src=${target_name}_placed.py
endif

# options for placer:
# - sequential_placer
# - sa_placer
placer ?= sequential_placer

# available auto gen topologies (default output netlist file: netlist.json)
# - single_multicast
# - 2d_mesh
# - 3d_mesh
# - line
# - tree
# - cnn
netlist_file := ${NETLIST_FILE_DIR}/auto_gen_netlist.json
netlist_topology ?= tree
num_rows ?= 6
num_cols ?= 8
obj_fifo_depth ?= 5
tree_branching_factor ?= 2
tree_use_multicast ?= 0

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
WARMUP_ITERATION := 50
mlir_target?=${build_dir}/${target_name}.mlir
trace_mlir_target?=${build_dir}/trace_${target_name}.mlir
xclbin_target?=${build_dir}/${target_name}.xclbin
trace_xclbin_target?=${build_dir}/trace_${target_name}.xclbin
insts_target?=${build_dir}/insts_${target_name}.bin

pnr_args ?= -n 1 -r
aieargs+=--dev ${device_name} -nl ${netlist_file} --dtype=${dtype}
runargs+=-nl ${netlist_file} -v ${VERBOSITY} --dtype=${dtype} --iters ${NUM_ITERATION} --warmup ${WARMUP_ITERATION}
kernels := accumulate

ifeq (${use_placed}, 0)
aieargs+=--placer ${placer}
ifeq (${placer}, sa_placer)
aieargs+=--pnr-args "${pnr_args}"
endif
endif

build: ${mlir_target} ${kernels:%=${build_dir}/%.o}
aiecc: ${xclbin_target}

.PHONY: all
all: ${xclbin_target}

${netlist_file}:
	mkdir -p ${@D}
ifeq (${use_auto_gen_netlist}, 0)
	@echo "Using custom netlist file: ${netlist_file}"
else
	@echo "Generating netlist file: ${netlist_file} with topology: ${netlist_topology}"
	python3 ${src_dir}/auto_gen_netlist.py --netlist ${netlist_file} --topology ${netlist_topology} --num_rows ${num_rows} --num_cols ${num_cols} \
		--obj_fifo_depth ${obj_fifo_depth} --branching_factor ${tree_branching_factor} --tree_use_multicast ${tree_use_multicast}
endif

${build_dir}/%.o: ${AIE_KERNELS_DIR}/%.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

${mlir_target}: ${src_dir}/${aie_py_src} ${netlist_file}
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@

${trace_mlir_target}: ${src_dir}/${aie_py_src} ${netlist_file}
	mkdir -p ${@D}
	python3 $< ${aieargs} --trace_size ${TRACE_SIZE} > $@

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
	python3 ${src_dir}/test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE ${runargs}

.PHONY: run_trace
run_trace: ${trace_xclbin_target}
	@mkdir -p trace
	python3 ${src_dir}/test.py -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE ${runargs} \
			-t ${TRACE_SIZE} --trace_file trace/trace.txt
	${AIE_PARSE_TRACE_DIR}/parse_trace.py --input trace/trace.txt --mlir ${trace_mlir_target} --output trace/trace.json
	python3 ${src_dir}/../../utils/get_cycle_summary.py --input trace/trace.json --output trace/summary.txt

.PHONY: clean_trace
clean_trace:
	rm -rf trace

.PHONY: clean
clean: clean_trace
	rm -rf ${build_dir}

