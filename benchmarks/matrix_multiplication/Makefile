srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

targetname = matrix_multiplication
aie_py_src=${targetname}.py

use_placed?=1
ifeq (${use_placed}, 0)
aie_py_src=${targetname}.py
else ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

# M, K, N: final GEMM dimensions
M?=512
K?=512
N?=512

# m, k, n: single AIE core GEMM dimensions
m?=32
k?=32
n?=32

n_aie_cols?=4
b_col_maj?=0

kernels=mm_${m}x${k}x${n}
aieargs+=-m $m -k $k -n $n --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj}
runargs+=--b_col_maj ${b_col_maj}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c
use_placed?=0
use_iron?=0

# set this flag to 1 for linear buffer allocation
# else, 0 for bank-aware
use_linear_buf_alloc?=1

ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
ifeq (${use_iron}, 1)
$(error Cannot specify both placed design and IRON)
endif
endif

ifeq (${use_iron}, 1)
aie_py_src=${targetname}_iron.py
endif

include ${srcdir}/../makefile-common

ifeq ($(devicename),npu2)
kernels_dir=${srcdir}/../../../../aie_kernels/aie2p
else 
kernels_dir=${srcdir}/../../../../aie_kernels/aie2
endif

build/mm_b_row_maj_${m}x${k}x${n}.o: ${kernels_dir}/mm.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}
else ifeq (${devicename}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}
endif

build/mm_b_col_maj_${m}x${k}x${n}.o: ${kernels_dir}/mm_b_col_maj.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}
else ifeq (${devicename}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}
endif

ifneq (${b_col_maj}, 1)
build/mm_${m}x${k}x${n}.o: build/mm_b_row_maj_${m}x${k}x${n}.o 
	mkdir -p ${@D}
	ar rvs $@ $^
else
build/mm_${m}x${k}x${n}.o: build/mm_b_col_maj_${m}x${k}x${n}.o
	mkdir -p ${@D}
	ar rvs $@ $^
endif








# Trace size
TRACE_SIZE = 16384
NUM_ITERATION = 10

all: build/final_${EDGEDETECT_WIDTH}.xclbin

mlir: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir

build/%.cc.o: ${AIE_KERNELS_DIR}/%.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

build/combined_gray2rgba_addWeighted.a: build/gray2rgba.cc.o build/addWeighted.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^)

build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -iwd ${EDGEDETECT_WIDTH} -iht ${EDGEDETECT_HEIGHT} -nc ${NUM_COMPUTE_COLUMNS} > $@

build/aie2_trace_lineBased_8b_${EDGEDETECT_WIDTH}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -iwd ${EDGEDETECT_WIDTH} -iht ${EDGEDETECT_HEIGHT} -nc ${NUM_COMPUTE_COLUMNS} -t ${TRACE_SIZE}> $@

build/final_${EDGEDETECT_WIDTH}.xclbin: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir build/rgba2gray.cc.o build/gray2rgba.cc.o build/filter2d.cc.o build/threshold.cc.o build/addWeighted.cc.o build/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
				--no-xchesscc --no-xbridge \
				--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

build/final_trace_${EDGEDETECT_WIDTH}.xclbin: build/aie2_trace_lineBased_8b_${EDGEDETECT_WIDTH}.mlir build/rgba2gray.cc.o build/gray2rgba.cc.o build/filter2d.cc.o build/threshold.cc.o build/addWeighted.cc.o build/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
				--no-xchesscc --no-xbridge \
				--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

run: build/final_${EDGEDETECT_WIDTH}.xclbin
	python3 ${srcdir}/test.py -x build/final_${EDGEDETECT_WIDTH}.xclbin -i build/insts.bin -k MLIR_AIE --iters ${NUM_ITERATION} \
			-iwd ${EDGEDETECT_WIDTH} -iht ${EDGEDETECT_HEIGHT} -nc ${NUM_COMPUTE_COLUMNS} --image "${srcdir}/input_image.jpg" 

run_trace: build/final_trace_${EDGEDETECT_WIDTH}.xclbin
	@mkdir -p trace
	python3 ${srcdir}/test.py -x build/final_trace_${EDGEDETECT_WIDTH}.xclbin -i build/insts.bin -k MLIR_AIE --iters ${NUM_ITERATION} \
			-iwd ${EDGEDETECT_WIDTH} -iht ${EDGEDETECT_HEIGHT} -nc ${NUM_COMPUTE_COLUMNS} --image "${srcdir}/input_image.jpg" \
			-t ${TRACE_SIZE} --trace_file trace/trace.txt
	${AIE_PARSE_TRACE_DIR}/parse_trace.py --input trace/trace.txt --mlir build/aie2_trace_lineBased_8b_${EDGEDETECT_WIDTH}.mlir --output trace/trace.json
	python3 ${srcdir}/../../utils/get_cycle_summary.py --input trace/trace.json --output trace/summary.txt

clean_trace:
	rm -rf trace

clean: clean_trace
	rm -rf build ${targetname}Out_test.jpg output
