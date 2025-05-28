srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

device = npu2
EDGEDETECT_WIDTH = 1920
EDGEDETECT_HEIGHT = 1080

targetname = edge_detect

aie_py_src=${targetname}.py
use_placed?=1

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

all: build/final_${EDGEDETECT_WIDTH}.xclbin

mlir: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir

build/%.cc.o: ${AIE_KERNELS_DIR}/%.cc
	mkdir -p ${@D}
ifeq ($(device),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else
	echo "Device type not supported"
endif

build/combined_gray2rgba_addWeighted.a: build/gray2rgba.cc.o build/addWeighted.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^)

build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${EDGEDETECT_WIDTH} ${EDGEDETECT_HEIGHT} > $@

build/final_${EDGEDETECT_WIDTH}.xclbin: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir build/rgba2gray.cc.o build/gray2rgba.cc.o build/filter2d.cc.o build/threshold.cc.o build/addWeighted.cc.o build/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

run_py: build/final_${EDGEDETECT_WIDTH}.xclbin
	python3 ${srcdir}/test.py -x build/final_${EDGEDETECT_WIDTH}.xclbin -i build/insts.bin -k MLIR_AIE --width=${EDGEDETECT_WIDTH} --height=${EDGEDETECT_HEIGHT} --image="${srcdir}/input_image.jpg"

clean:
	rm -rf build _build ${targetname}.exe ${targetname}Out_test.jpg
