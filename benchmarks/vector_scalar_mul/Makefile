srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

target_name := vector_scalar_mul
aie_py_src=${target_name}.py

use_placed ?= 1
ifeq (${use_placed}, 0)
aie_py_src=${target_name}.py
else ifeq (${use_placed}, 1)
aie_py_src=${target_name}_placed.py
endif

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
NUM_ITERATION := 30
WARMUP_ITERATION := 20

runargs+=--iters ${NUM_ITERATION} --warmup ${WARMUP_ITERATION}

all: build/final_${data_size}.xclbin 

build/%.o: ${AIE_KERNELS_DIR}/%.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=${int_bit_width} -c $< -o ${@F}; 

build/aie_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -i1s ${in1_size} -i2s ${in2_size} -os ${out_size} -bw ${int_bit_width} > $@

build/aie_trace_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -i1s ${in1_size} -i2s ${in2_size} -os ${out_size} -bw ${int_bit_width}  -t ${trace_size} > $@

build/final_${data_size}.xclbin: build/aie_${data_size}.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    	  		--no-xchesscc --no-xbridge --aie-generate-npu-insts --get-routing-summary ${aiecc_extra_args} \
				--npu-insts-name=insts.bin $(<:%=../%)

build/final_trace_${data_size}.xclbin: build/aie_trace_${data_size}.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    			--no-xchesscc --no-xbridge --aie-generate-npu-insts --get-routing-summary ${aiecc_extra_args} \
				--npu-insts-name=insts.bin $(<:%=../%)

run: build/final_${data_size}.xclbin 
	python3 ${srcdir}/test.py -x build/final_${data_size}.xclbin -i build/insts.bin -k MLIR_AIE \
			-i1s ${in1_size} -i2s ${in2_size} -os ${out_size} ${runargs}

run_trace: build/final_trace_${data_size}.xclbin 
	@mkdir -p trace
	python3 ${srcdir}/test.py -x build/final_trace_${data_size}.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size} \
		-i1s ${in1_size} -i2s ${in2_size} -os ${out_size} ${runargs}\
		--trace_file trace/trace.txt
	${AIE_PARSE_TRACE_DIR}/parse_trace.py --input trace/trace.txt --mlir build/aie_trace_${data_size}.mlir --output trace/trace.json
	python3 ${srcdir}/../../utils/get_cycle_summary.py --input trace/trace.json --output trace/summary.txt

clean_trace:
	rm -rf trace

clean: clean_trace
	rm -rf build output 
