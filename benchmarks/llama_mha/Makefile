srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common	

embed_dim := 288
head_dim := 48
tile_dim := 96
seq_len := 16

ARGS := --embed-dim $(embed_dim) \
        --head-dim $(head_dim) \
        --tile-dim $(tile_dim) \
        --seq-len $(seq_len) \

all: build/mha.xclbin

build/%.cc.o: %.cc 
	mkdir -p ${@D}
	${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o $@

build/mha.mlir: ${srcdir}/mha.py build/mha.cc.o 
	mkdir -p ${@D}
	python3 $< $(ARGS) > $@

build/mha.xclbin: build/mha.mlir 
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
				--no-xchesscc --no-xbridge -v --get-routing-summary \
				--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

run: build/mha.xclbin
	python3 ${srcdir}/test.py $(ARGS) -x build/mha.xclbin -i build/insts.bin -k MLIR_AIE

clean: 
	rm -rf build 
