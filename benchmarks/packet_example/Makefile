src_dir ?= $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
build_dir ?= ${src_dir}/build
include ${src_dir}/../makefile-common

target_name := packet
aie_py_src=${target_name}_placed.py
ifeq ($(origin use_placed), undefined)
use_placed := 1
endif

placer ?= sa_placer
NUM_ITERATION := 1
WARMUP_ITERATION := 0

mlir_target?=${build_dir}/${target_name}_example.mlir
xclbin_target?=${build_dir}/${target_name}.xclbin
insts_target?=${build_dir}/${target_name}_insts.bin
num_repeat_cores ?= 1

# Intermediate PnR files
netlist_json := ${build_dir}/netlist.json
netlist_toml := ${build_dir}/netlist.toml
solution_toml := ${build_dir}/solution.toml
placed_json  := ${build_dir}/placed.json
tmp_mlir      := ${build_dir}/${target_name}_tmp.mlir

pnr_args ?= -n 1 -r
runargs+=--iters ${NUM_ITERATION} --warmup ${WARMUP_ITERATION}

.PHONY: build aiecc
build: ${mlir_target}
aiecc: ${xclbin_target}


ifeq ($(use_placed),0)
${mlir_target}: ${src_dir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< --num-repeat-cores ${num_repeat_cores} > $@
	aie-opt $@ --pnr-extract-netlist=netlist-file=${netlist_json} -o /dev/null
	${NPU_PNR_BIN_DIR}/tools/netlist_translator ${netlist_json} -o ${netlist_toml}
	${NPU_PNR_BIN_DIR}/apps/pnr ${netlist_toml} ${pnr_args} --output=${solution_toml}
	${NPU_PNR_BIN_DIR}/tools/netlist_translator ${solution_toml} -o ${placed_json}
	aie-opt $@ --pnr-place-tiles=netlist-file=${placed_json} -o ${tmp_mlir}
	mv ${tmp_mlir} $@
else
${mlir_target}: ${src_dir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< --num-repeat-cores ${num_repeat_cores} > $@
endif

${xclbin_target}: ${mlir_target}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --get-routing-summary ${aiecc_extra_args} \
				--xclbin-name=${@F} --npu-insts-name=$(notdir $(insts_target)) $(notdir $(<))

run: ${xclbin_target}
	python3 ${src_dir}/test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE ${runargs}

.PHONY: clean
clean: 
	rm -rf ${build_dir}
