srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
NETLIST_FILE_DIR := ${srcdir}/AIE_data_flow_netlists

include ${srcdir}/../makefile-common

device_name := npu2
dtype := i32

target_name := microbenchmark
aie_py_src := ${target_name}.py
use_placed := 1
ifeq (${use_placed}, 0)
aie_py_src=${target_name}.py
else ifeq (${use_placed}, 1)
aie_py_src=${target_name}_placed.py
endif

# available custom netlist topologies
# - 1_col_4_core_1_cross_6_core.json
# available auto gen topologies (default output netlist file: netlist.json)
# - mesh
# - tree
# - vertical_line
# - horizontal_line
use_auto_gen_netlist?=1
ifeq (${use_auto_gen_netlist}, 1)
netlist_file := ${NETLIST_FILE_DIR}/netlist.json
netlist_topology := tree
else ifeq (${use_auto_gen_netlist}, 0)
netlist_file := ${NETLIST_FILE_DIR}/1_col_4_core_1_cross_6_core.json
endif

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
NUM_ITERATION := 10
mlir_target?=build/${target_name}.mlir
trace_mlir_target?=build/${target_name}_trace.mlir
xclbin_target?=build/${target_name}.xclbin
trace_xclbin_target?=build/${target_name}_trace.xclbin
insts_target?=build/${target_name}_insts.bin

aieargs+=--dev ${device_name} -nl ${netlist_file} --dtype=${dtype}
runargs+=-nl ${netlist_file} -v ${VERBOSITY} --dtype=${dtype}

.PHONY: all
all: ${xclbin_target}

.PHONY: gen_netlist
gen_netlist:
ifeq (${use_auto_gen_netlist}, 0)
	@echo "Using custom netlist file: ${netlist_file}"
else
	@echo "Generating netlist file: ${netlist_file} with topology: ${netlist_topology}"
	python3 ${srcdir}/auto_gen_netlist.py --netlist ${netlist_file} --topology ${netlist_topology}
endif

build/%.cc.o: ${AIE_KERNELS_DIR}/%.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

${mlir_target}: ${srcdir}/${aie_py_src} gen_netlist
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@

${trace_mlir_target}: ${srcdir}/${aie_py_src} gen_netlist
	mkdir -p ${@D}
	python3 $< ${aieargs} --trace_size ${TRACE_SIZE} > $@

${xclbin_target}: ${mlir_target}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=${buffer_aloc_flag} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--xclbin-name=${@F} --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

${trace_xclbin_target}: ${trace_mlir_target}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=${buffer_aloc_flag} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--xclbin-name=${@F} --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

.PHONY: run
run: ${xclbin_target}
	python3 ${srcdir}/test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE --iters ${NUM_ITERATION} ${runargs} 

.PHONY: run_trace
run_trace: ${trace_xclbin_target}
	@mkdir -p trace
	python3 ${srcdir}/test.py -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE --iters ${NUM_ITERATION} ${runargs} \
			-t ${TRACE_SIZE} --trace_file trace/trace.txt
	${AIE_PARSE_TRACE_DIR}/parse_trace.py --input trace/trace.txt --mlir ${trace_mlir_target} --output trace/trace.json
	python3 ${srcdir}/../../utils/get_cycle_summary.py --input trace/trace.json --output trace/summary.txt

.PHONY: clean_trace
clean_trace:
	rm -rf trace

.PHONY: clean
clean: clean_trace
	rm -rf build output

