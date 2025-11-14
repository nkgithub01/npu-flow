src_dir ?= $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
build_dir ?= $(src_dir)/build

include ${src_dir}/../makefile-common	

target_name := llama_mha
aie_py_src := ${target_name}.py

use_placed ?= 1
ifeq (${use_placed}, 0)
aie_py_src=${target_name}.py
else ifeq (${use_placed}, 1)
aie_py_src=${target_name}_placed.py
endif

embed_dim := 288
head_dim := 48
tile_dim := 96
seq_len := 16

NUM_ITERATION := 1
WARMUP_ITERATION := 0
runargs+=--iters ${NUM_ITERATION} --warmup ${WARMUP_ITERATION}
mlir_target?=${build_dir}/${target_name}.mlir
trace_mlir_target?=${build_dir}/trace_${target_name}.mlir
xclbin_target?=${build_dir}/${target_name}.xclbin
trace_xclbin_target?=${build_dir}/trace_${target_name}.xclbin
insts_target?=${build_dir}/insts_${target_name}.bin

ARGS := --embed-dim $(embed_dim) \
        --head-dim $(head_dim) \
        --tile-dim $(tile_dim) \
        --seq-len $(seq_len) \

build: ${mlir_target}
aiecc: ${xclbin_target}

${build_dir}/%.cc.o: %.cc 
	mkdir -p ${@D}
	${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o $@

${mlir_target}: ${src_dir}/${aie_py_src} ${build_dir}/mha.cc.o 
	mkdir -p ${@D}
	python3 $< $(ARGS) > $@

${xclbin_target}: ${mlir_target} 
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
				--no-xchesscc --no-xbridge -v --get-routing-summary ${aiecc_extra_args} \
				--xclbin-name=${@F} --npu-insts-name=$(notdir $(insts_target)) $(notdir $(<))

run: ${xclbin_target}
	python3 ${src_dir}/test.py $(ARGS) -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE ${runargs}

clean: 
	rm -rf ${build_dir} 