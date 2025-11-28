src_dir ?= $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
build_dir ?= ${src_dir}/build

include ${src_dir}/../makefile-common

target_name := vector_scalar_mul
aie_py_src=${target_name}.py
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

int_bit_width = 16
ifeq (${int_bit_width}, 16)
in1_size = 8192 # in bytes
out_size = 8192 # in bytes, should always be equal to in1_size
else # assume int_bit_width == 32
in1_size = 16384 # in bytes
out_size = 16384 # in bytes, should always be equal to in1_size
endif
in2_size = 4 # in bytes, should always be 4 (1x int32)
trace_size = 131072
data_size = in1_size
NUM_ITERATION := 100
WARMUP_ITERATION := 50
mlir_target?=${build_dir}/${target_name}.mlir
trace_mlir_target?=${build_dir}/trace_${target_name}.mlir
xclbin_target?=${build_dir}/${target_name}.xclbin
trace_xclbin_target?=${build_dir}/trace_${target_name}.xclbin
insts_target?=${build_dir}/insts_${target_name}.bin
pnr_args ?= -n 1 -r
aieargs+=-i1s ${in1_size} -i2s ${in2_size} -os ${out_size} -bw ${int_bit_width}
runargs+=--iters ${NUM_ITERATION} --warmup ${WARMUP_ITERATION}

ifeq (${use_placed}, 0)
aieargs+=--placer ${placer}
ifeq (${placer}, sa_placer)
aieargs+=--pnr-args="${pnr_args}"
endif
endif

.PHONY: build aiecc
build: ${mlir_target} ${build_dir}/scale.o
aiecc: ${xclbin_target}

${build_dir}/%.o: ${AIE_KERNELS_DIR}/%.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=${int_bit_width} -c $< -o ${@F}; 

${mlir_target}: ${src_dir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@

${trace_mlir_target}: ${src_dir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} -t ${trace_size} > $@

${xclbin_target}: ${mlir_target} ${build_dir}/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary ${aiecc_extra_args} \
				--xclbin-name=${@F} --npu-insts-name=$(notdir $(insts_target)) $(notdir $(<))
				
${trace_xclbin_target}: ${trace_mlir_target} ${build_dir}/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary ${aiecc_extra_args} \
				--xclbin-name=${@F} --npu-insts-name=$(notdir $(insts_target)) $(notdir $(<))

run: ${xclbin_target}
	python3 ${src_dir}/test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE \
			-i1s ${in1_size} -i2s ${in2_size} -os ${out_size} ${runargs}

run_trace: ${trace_xclbin_target}
	@mkdir -p trace
	python3 ${src_dir}/test.py -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE -t ${trace_size} \
		-i1s ${in1_size} -i2s ${in2_size} -os ${out_size} ${runargs}\
		--trace_file trace/trace.txt
	${AIE_PARSE_TRACE_DIR}/parse_trace.py --input trace/trace.txt --mlir build/aie_trace_${data_size}.mlir --output trace/trace.json
	python3 ${src_dir}/../../utils/get_cycle_summary.py --input trace/trace.json --output trace/summary.txt

.PHONY: clean_trace
clean_trace:
	rm -rf trace

.PHONY: clean
clean: clean_trace
	rm -rf ${build_dir}
