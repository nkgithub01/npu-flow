src_dir ?= $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
build_dir ?= $(src_dir)/build
include ${src_dir}/../makefile-common

target_name = edge_detection
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

NUM_COMPUTE_COLUMNS ?= 1
EDGEDETECT_WIDTH = 1920
EDGEDETECT_HEIGHT = 1080
TRACE_SIZE = 134217728
NUM_ITERATION ?= 100
WARMUP_ITERATION ?= 50
mlir_target?=${build_dir}/${target_name}.mlir
trace_mlir_target?=${build_dir}/trace_${target_name}.mlir
xclbin_target?=${build_dir}/${target_name}.xclbin
trace_xclbin_target?=${build_dir}/trace_${target_name}.xclbin
insts_target?=${build_dir}/insts_${target_name}.bin

pnr_args ?= -n 1 -r
aieargs+=-iwd ${EDGEDETECT_WIDTH} -iht ${EDGEDETECT_HEIGHT} -nc ${NUM_COMPUTE_COLUMNS}
runargs+=-iwd ${EDGEDETECT_WIDTH} -iht ${EDGEDETECT_HEIGHT} -nc ${NUM_COMPUTE_COLUMNS} --image "${src_dir}/input_image.jpg" --iters ${NUM_ITERATION} --warmup ${WARMUP_ITERATION}

ifeq (${use_placed}, 0)
aieargs+=--placer ${placer}
ifeq (${placer}, sa_placer)
aieargs+=--pnr-args "${pnr_args}"
endif
endif

.PHONY: build aiecc
build: ${mlir_target} ${build_dir}/rgba2gray.cc.o ${build_dir}/gray2rgba.cc.o ${build_dir}/filter2d.cc.o ${build_dir}/threshold.cc.o ${build_dir}/addWeighted.cc.o ${build_dir}/combined_gray2rgba_addWeighted.a
aiecc: ${xclbin_target}

${build_dir}/%.cc.o: ${AIE_KERNELS_DIR}/%.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

${build_dir}/combined_gray2rgba_addWeighted.a: ${build_dir}/gray2rgba.cc.o ${build_dir}/addWeighted.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^)

${mlir_target}: ${src_dir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@


${xclbin_target}: ${mlir_target} ${build_dir}/rgba2gray.cc.o ${build_dir}/gray2rgba.cc.o ${build_dir}/filter2d.cc.o ${build_dir}/threshold.cc.o ${build_dir}/addWeighted.cc.o ${build_dir}/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary ${aiecc_extra_args} \
				--xclbin-name=${@F} --npu-insts-name=$(notdir $(insts_target)) $(notdir $(<))

run: ${xclbin_target}
	python3 ${src_dir}/test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE ${runargs}

${trace_mlir_target}: ${src_dir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} -t ${TRACE_SIZE} > $@

${trace_xclbin_target}: ${trace_mlir_target} ${build_dir}/rgba2gray.cc.o ${build_dir}/gray2rgba.cc.o ${build_dir}/filter2d.cc.o ${build_dir}/threshold.cc.o ${build_dir}/addWeighted.cc.o ${build_dir}/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
		cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
					--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary ${aiecc_extra_args} \
					--xclbin-name=${@F} --npu-insts-name=$(notdir $(insts_target)) $(notdir $(<))
					
run_trace: ${trace_xclbin_target}
	@mkdir -p trace
	python3 ${src_dir}/test.py -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE ${runargs} -t ${TRACE_SIZE} --trace_file trace/trace.txt
	${AIE_PARSE_TRACE_DIR}/parse_trace.py --input trace/trace.txt --mlir ${trace_mlir_target} --output trace/trace.json
	python3 ${src_dir}/../../utils/get_cycle_summary.py --input trace/trace.json --output trace/summary.txt

.PHONY: clean_trace
clean_trace:
	rm -rf trace

.PHONY: clean
clean: clean_trace
	rm -rf ${build_dir} ${target_name}Out_test.jpg output
