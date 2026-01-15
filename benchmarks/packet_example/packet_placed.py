import numpy as np
import sys
import argparse

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

def packet(num_repeat_cores):
    @device(AIEDevice.npu2)
    def device_body():
        in_ty = np.ndarray[(16,), np.dtype[np.int32]]
        out_ty = np.ndarray[(16,), np.dtype[np.int32]]

        # Tile declarations
        in_shim = tile(0, 0)
        out_shim = tile(7, 0)
        in_mem = tile(0, 1)
        out_mem = tile(7, 1)

        repeat_tiles = []
        start_col = 0
        start_row = 2
        rows_per_col = 4
        for i in range(num_repeat_cores - 1):
            col = start_col + (i // rows_per_col)
            row = start_row + (i % rows_per_col)
            repeat_tiles.append(tile(col, row))
        
        repeat_tiles.append(tile(3, 3)) # Last repeat core at (3,3)
        split_tiles = []
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dc == 0 and dr == 0:
                    continue
                split_tiles.append(tile(3 + dc, 3 + dr))

        # ObjectFifo declarations
        inOF_L3toL2 = object_fifo("inOF_L3toL2", in_shim, in_mem, 1, in_ty)
        inOF_L2toL1 = object_fifo("inOF_L2toL1", in_mem, repeat_tiles[0], 1, in_ty) 
        object_fifo_link(inOF_L3toL2, inOF_L2toL1)

        outOF_L2toL3 = object_fifo("outOF_L2toL3", out_mem, out_shim, 1, out_ty)
        outOF_L1toL2 = object_fifo("outOF_L1toL2", split_tiles[0], out_mem, 1, out_ty)
        object_fifo_link(outOF_L1toL2, outOF_L2toL3)

        repeat_OFs = []
        for i in range(num_repeat_cores - 1):
            repeat_OFs.append(object_fifo(f"OF_repeat_{i}", repeat_tiles[i], repeat_tiles[i + 1], 1, in_ty))
        
        split_OFs = []
        for i in range(8):
            split_OFs.append(object_fifo(f"OF_1toS_{i}", repeat_tiles[-1], split_tiles[i], 1, in_ty))
        
        @core(repeat_tiles[0])
        def core_body():
            elem_in = inOF_L2toL1.acquire(ObjectFifoPort.Consume, 1)
            elem_out = repeat_OFs[0].acquire(ObjectFifoPort.Produce, 1)
            for j in range_(16):
                elem_out[j] = elem_in[j]
            inOF_L2toL1.release(ObjectFifoPort.Consume, 1)
            repeat_OFs[0].release(ObjectFifoPort.Produce, 1)

        for i in range(1, num_repeat_cores - 1):
            @core(repeat_tiles[i])
            def core_body():  
                elem_in = repeat_OFs[i - 1].acquire(ObjectFifoPort.Consume, 1)
                elem_out = repeat_OFs[i].acquire(ObjectFifoPort.Produce, 1)
                for j in range_(16):
                    elem_out[j] = elem_in[j]
                repeat_OFs[i - 1].release(ObjectFifoPort.Consume, 1)
                repeat_OFs[i].release(ObjectFifoPort.Produce, 1)

        @core(repeat_tiles[-1])
        def core_body():
            if num_repeat_cores > 1:
                elem_in = repeat_OFs[-1].acquire(ObjectFifoPort.Consume, 1)
            else:
                elem_in = inOF_L2toL1.acquire(ObjectFifoPort.Consume, 1)

            for f in split_OFs:
                elem_out = f.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(16):
                    elem_out[i] = elem_in[i] + 1
                f.release(ObjectFifoPort.Produce, 1)

            if num_repeat_cores > 1:
                repeat_OFs[-1].release(ObjectFifoPort.Consume, 1)
            else:
                inOF_L2toL1.release(ObjectFifoPort.Consume, 1)
        
        @core(split_tiles[0])
        def core_body():
            elem_in = split_OFs[0].acquire(ObjectFifoPort.Consume, 1)
            elem_out = outOF_L1toL2.acquire(ObjectFifoPort.Produce, 1)
            for j in range_(16):
                elem_out[j] = elem_in[j]
            split_OFs[0].release(ObjectFifoPort.Consume, 1)
            outOF_L1toL2.release(ObjectFifoPort.Produce, 1)
        
        @runtime_sequence(in_ty, in_ty, out_ty)
        def sequence(A, B, C):
            in_task = shim_dma_single_bd_task(inOF_L3toL2, A, 0, sizes=[1, 1, 1, 16], issue_token=True)
            out_task = shim_dma_single_bd_task(outOF_L2toL3, C, 0, sizes=[1, 1, 1, 16], issue_token=True)
            dma_start_task(in_task, out_task)
            dma_await_task(out_task)
            dma_free_task(in_task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Packet Example with Repeat Cores")
    parser.add_argument(
        "-nrc",
        "--num-repeat-cores",
        type=int,
        required=False,
        dest="num_repeat_cores",
        default=1,
        help="Number of repeat cores to use in the design",
    )
    args = parser.parse_args()

    with mlir_mod_ctx() as ctx:
        packet(args.num_repeat_cores)
        print(ctx.module)




        
        
