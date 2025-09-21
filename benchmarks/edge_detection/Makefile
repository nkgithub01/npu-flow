srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

targetname = edge_detect
aie_py_src=${targetname}.py

use_placed ?= 1
ifeq (${use_placed}, 0)
aie_py_src=${targetname}.py
else ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

placer ?= sequential_placer

NUM_COMPUTE_COLUMNS ?= 1
EDGEDETECT_WIDTH = 1920
EDGEDETECT_HEIGHT = 1080
TRACE_SIZE = 134217728
NUM_ITERATION ?= 100
WARMUP_ITERATION ?= 50

aieargs+=-iwd ${EDGEDETECT_WIDTH} -iht ${EDGEDETECT_HEIGHT} -nc ${NUM_COMPUTE_COLUMNS}
runargs+=-iwd ${EDGEDETECT_WIDTH} -iht ${EDGEDETECT_HEIGHT} -nc ${NUM_COMPUTE_COLUMNS} --image "${srcdir}/input_image.jpg" --iters ${NUM_ITERATION} --warmup ${WARMUP_ITERATION}

ifeq (${use_placed}, 0)
aieargs+=--placer ${placer}
endif

ifeq (${placer}, sa_placer)
aieargs+=--pnr-args "${pnr_args}"
endif

build: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir build/rgba2gray.cc.o build/gray2rgba.cc.o build/filter2d.cc.o build/threshold.cc.o build/addWeighted.cc.o build/combined_gray2rgba_addWeighted.a
aiecc: build/final_${EDGEDETECT_WIDTH}.xclbin

all: build/final_${EDGEDETECT_WIDTH}.xclbin

build_trace: build/final_trace_${EDGEDETECT_WIDTH}.xclbin

mlir: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir

build/%.cc.o: ${AIE_KERNELS_DIR}/%.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

build/combined_gray2rgba_addWeighted.a: build/gray2rgba.cc.o build/addWeighted.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^)

build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@

build/aie2_trace_lineBased_8b_${EDGEDETECT_WIDTH}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} -t ${TRACE_SIZE} > $@

build/final_${EDGEDETECT_WIDTH}.xclbin: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir build/rgba2gray.cc.o build/gray2rgba.cc.o build/filter2d.cc.o build/threshold.cc.o build/addWeighted.cc.o build/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
				--no-xchesscc --no-xbridge --get-routing-summary ${aiecc_extra_args} -v \
				--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

build/final_trace_${EDGEDETECT_WIDTH}.xclbin: build/aie2_trace_lineBased_8b_${EDGEDETECT_WIDTH}.mlir build/rgba2gray.cc.o build/gray2rgba.cc.o build/filter2d.cc.o build/threshold.cc.o build/addWeighted.cc.o build/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
				--no-xchesscc --no-xbridge --get-routing-summary ${aiecc_extra_args} \
				--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

run: build/final_${EDGEDETECT_WIDTH}.xclbin
	python3 ${srcdir}/test.py -x build/final_${EDGEDETECT_WIDTH}.xclbin -i build/insts.bin -k MLIR_AIE ${runargs}

run_trace: build/final_trace_${EDGEDETECT_WIDTH}.xclbin
	@mkdir -p trace
	python3 ${srcdir}/test.py -x build/final_trace_${EDGEDETECT_WIDTH}.xclbin -i build/insts.bin -k MLIR_AIE ${runargs} -t ${TRACE_SIZE} --trace_file trace/trace.txt
	${AIE_PARSE_TRACE_DIR}/parse_trace.py --input trace/trace.txt --mlir build/aie2_trace_lineBased_8b_${EDGEDETECT_WIDTH}.mlir --output trace/trace.json
	python3 ${srcdir}/../../utils/get_cycle_summary.py --input trace/trace.json --output trace/summary.txt

clean_trace:
	rm -rf trace

clean: clean_trace
	rm -rf build ${targetname}Out_test.jpg output
