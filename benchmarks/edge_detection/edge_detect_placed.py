#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.
import numpy as np
import sys
import argparse

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent, ShimTileEvent, MemTileEvent

def edge_detect(width, height, trace_size):
    heightMinus1 = height - 1
    lineWidth = width
    lineWidthInBytes = width * 4
    tensorSize = width * height * 4  # 4 channels

    numCol = 3

    @device(AIEDevice.npu2)
    def device_body():
        line_bytes_ty = np.ndarray[(lineWidthInBytes,), np.dtype[np.uint8]]
        line_ty = np.ndarray[(lineWidth,), np.dtype[np.uint8]]
        tensor_3x3_ty = np.ndarray[(3, 3), np.dtype[np.int16]]

        i_tensor_ty = np.ndarray[(tensorSize,), np.dtype[np.uint8]]
        o_tensor_ty = np.ndarray[(tensorSize * numCol,), np.dtype[np.uint8]]

        # AIE Core Function declarations
        rgba2gray_line = external_func(
            "rgba2grayLine", inputs=[line_bytes_ty, line_ty, np.int32]
        )
        filter2d_line = external_func(
            "filter2dLine",
            inputs=[line_ty, line_ty, line_ty, line_ty, np.int32, tensor_3x3_ty],
        )
        threshold_line = external_func(
            "thresholdLine",
            inputs=[line_ty, line_ty, np.int32, np.int16, np.int16, np.int8],
        )
        gray2rgba_line = external_func(
            "gray2rgbaLine", inputs=[line_ty, line_bytes_ty, np.int32]
        )
        add_weighted_line = external_func(
            "addWeightedLine",
            inputs=[
                line_bytes_ty,
                line_bytes_ty,
                line_bytes_ty,
                np.int32,
                np.int16,
                np.int16,
                np.int8,
            ],
        )

        # Tile declarations
        ShimTiles = []
        MemTiles = []
        ComputeTile2s = []
        ComputeTile3s = []
        ComputeTile4s = []
        ComputeTile5s = []
        for col in range(numCol):
            ShimTiles.append(tile(col, 0))
            MemTiles.append(tile(col, 1))
            ComputeTile2s.append(tile(col, 2))
            ComputeTile3s.append(tile(col, 3))
            ComputeTile4s.append(tile(col, 4))
            ComputeTile5s.append(tile(col, 5))


        # AIE-array data movement with object fifos
        # Input
        inOF_L3L2s = []
        inOF_L2L1s = []
        for col in range(numCol):
            inOF_L3L2s.append(object_fifo(
                f"col_{col}_inOF_L3L2",
                ShimTiles[col],
                [ComputeTile2s[col], MemTiles[col]],
                [2, 2, 7],
                line_bytes_ty,
            ))
            inOF_L2L1s.append(object_fifo(
                f"col_{col}_inOF_L2L1",
                MemTiles[col],
                ComputeTile5s[col],
                7,
                line_bytes_ty,
            ))
            object_fifo_link(inOF_L3L2s[-1], inOF_L2L1s[-1])

        # Output
        outOF_L2L3s = []
        outOF_L1L2s = []
        for col in range(numCol):
            outOF_L2L3s.append(object_fifo(
                f"col_{col}_outOF_L2L3",
                MemTiles[col],
                ShimTiles[col],
                2,
                line_bytes_ty,
            ))
            outOF_L1L2s.append(object_fifo(
                f"col_{col}_outOF_L1L2",
                ComputeTile5s[col],
                MemTiles[col],
                2,
                line_bytes_ty,
            ))
            object_fifo_link(outOF_L1L2s[-1], outOF_L2L3s[-1])

        # Intermediate
        OF_2to3s = []
        OF_3to4s = []
        OF_4to5s = []
        OF_5to5s = []
        for col in range(numCol):
            OF_2to3s.append(object_fifo(
                f"col_{col}_OF_2to3",
                ComputeTile2s[col],
                ComputeTile3s[col],
                4,
                line_ty,
            ))
            OF_3to4s.append(object_fifo(
                f"col_{col}_OF_3to4",
                ComputeTile3s[col],
                ComputeTile4s[col],
                2,
                line_ty,
            ))
            OF_4to5s.append(object_fifo(
                f"col_{col}_OF_4to5",
                ComputeTile4s[col],
                ComputeTile5s[col],
                2,
                line_ty,
            ))
            OF_5to5s.append(object_fifo(
                f"col_{col}_OF_5to5",
                ComputeTile5s[col],
                ComputeTile5s[col],
                1,
                line_bytes_ty,
            ))

        # Set up compute tiles

        # Compute tile 2
        for col in range(numCol):
            @core(ComputeTile2s[col], "rgba2gray.cc.o")
            def core_body():
                for _ in range_(sys.maxsize):
                    elem_in = inOF_L3L2s[col].acquire(ObjectFifoPort.Consume, 1)
                    elem_out = OF_2to3s[col].acquire(ObjectFifoPort.Produce, 1)
    
                    rgba2gray_line(elem_in, elem_out, lineWidth)
    
                    inOF_L3L2s[col].release(ObjectFifoPort.Consume, 1)
                    OF_2to3s[col].release(ObjectFifoPort.Produce, 1)
    
            # Compute tile 3
            @core(ComputeTile3s[col], "filter2d.cc.o")
            def core_body():
                v0 = 0
                v1 = 4096
                v_minus4 = -16384
                initial_value = np.array(
                    [[v0, v1, v0], [v1, v_minus4, v1], [v0, v1, v0]], dtype=np.int16
                )
                kernel = buffer(
                    ComputeTile3s[col],
                    np.ndarray[(3, 3), np.dtype[np.int16]],
                    f"col_{col}_kernel",
                    initial_value=initial_value,
                )
    
                for _ in range_(sys.maxsize):
                    # Preamble : Top Border
                    elems_in_pre = OF_2to3s[col].acquire(ObjectFifoPort.Consume, 2)
                    elem_pre_out = OF_3to4s[col].acquire(ObjectFifoPort.Produce, 1)
                    filter2d_line(
                        elems_in_pre[0],
                        elems_in_pre[0],
                        elems_in_pre[1],
                        elem_pre_out,
                        lineWidth,
                        kernel,
                    )
                    OF_3to4s[col].release(ObjectFifoPort.Produce, 1)
    
                    # Steady State : Middle
                    for _ in range_(1, heightMinus1):
                        elems_in = OF_2to3s[col].acquire(ObjectFifoPort.Consume, 3)
                        elem_out = OF_3to4s[col].acquire(ObjectFifoPort.Produce, 1)
                        filter2d_line(
                            elems_in[0],
                            elems_in[1],
                            elems_in[2],
                            elem_out,
                            lineWidth,
                            kernel,
                        )
                        OF_2to3s[col].release(ObjectFifoPort.Consume, 1)
                        OF_3to4s[col].release(ObjectFifoPort.Produce, 1)
    
                    # Postamble : Bottom Border
                    elems_in_post = OF_2to3s[col].acquire(ObjectFifoPort.Consume, 2)
                    elem_post_out = OF_3to4s[col].acquire(ObjectFifoPort.Produce, 1)
                    filter2d_line(
                        elems_in_post[0],
                        elems_in_post[1],
                        elems_in_post[1],
                        elem_post_out,
                        lineWidth,
                        kernel,
                    )
                    OF_2to3s[col].release(ObjectFifoPort.Consume, 2)
                    OF_3to4s[col].release(ObjectFifoPort.Produce, 1)
    
            # Compute tile 4
            @core(ComputeTile4s[col], "threshold.cc.o")
            def core_body():
                v_thr = 10
                v_max = 255
                v_typ = 0
    
                for _ in range_(sys.maxsize):
                    elem_in = OF_3to4s[col].acquire(ObjectFifoPort.Consume, 1)
                    elem_out = OF_4to5s[col].acquire(ObjectFifoPort.Produce, 1)
    
                    threshold_line(elem_in, elem_out, lineWidth, v_thr, v_max, v_typ)
    
                    OF_3to4s[col].release(ObjectFifoPort.Consume, 1)
                    OF_4to5s[col].release(ObjectFifoPort.Produce, 1)
    
            # Compute tile 5
            @core(ComputeTile5s[col], "combined_gray2rgba_addWeighted.a")
            def core_body():
                for _ in range_(sys.maxsize):
                    elem_in = OF_4to5s[col].acquire(ObjectFifoPort.Consume, 1)
                    elem_out = OF_5to5s[col].acquire(ObjectFifoPort.Produce, 1)
    
                    gray2rgba_line(elem_in, elem_out, lineWidth)
    
                    OF_4to5s[col].release(ObjectFifoPort.Consume, 1)
                    OF_5to5s[col].release(ObjectFifoPort.Produce, 1)
    
                    elem_in1 = OF_5to5s[col].acquire(ObjectFifoPort.Consume, 1)
                    elem_in2 = inOF_L2L1s[col].acquire(ObjectFifoPort.Consume, 1)
                    elem_out2 = outOF_L1L2s[col].acquire(ObjectFifoPort.Produce, 1)
    
                    alpha = 16384
                    beta = 16384
                    gamma = 0
    
                    add_weighted_line(
                        elem_in1,
                        elem_in2,
                        elem_out2,
                        lineWidthInBytes,
                        alpha,
                        beta,
                        gamma,
                    )
    
                    OF_5to5s[col].release(ObjectFifoPort.Consume, 1)
                    inOF_L2L1s[col].release(ObjectFifoPort.Consume, 1)
                    outOF_L1L2s[col].release(ObjectFifoPort.Produce, 1)

        # Set up a packet-switched flow from core/mem to shim for tracing information
        # Max can only trace 31 tiles
        tiles_to_trace = ShimTiles
        # traceShim = tile(3, 0)
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTiles[2])

        # To/from AIE-array data movement
        @runtime_sequence(i_tensor_ty, i_tensor_ty, o_tensor_ty)
        def sequence(I1, I2, O):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTiles[2],
                    trace_size=trace_size,
                    coretile_events=[
                        CoreEvent.INSTR_EVENT_0,
                        CoreEvent.INSTR_EVENT_1,
                        PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
                        PortEvent(CoreEvent.PORT_RUNNING_1, 1, False),  # slave(1)
                    ],
                    shimtile_events=[
                        ShimTileEvent.DMA_S2MM_0_START_TASK,
                        ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
                        ShimTileEvent.DMA_MM2S_0_START_TASK,
                        ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
                    ]
                )
            in_tasks = []
            in_tasks.append(shim_dma_single_bd_task(inOF_L3L2s[0], I1, sizes=[1, 1, 1, tensorSize]))
            in_tasks.append(shim_dma_single_bd_task(inOF_L3L2s[1], I2, sizes=[1, 1, 1, tensorSize]))
            in_tasks.append(shim_dma_single_bd_task(inOF_L3L2s[2], I2, sizes=[1, 1, 1, tensorSize]))

            out_tasks = []
            out_tasks.append(shim_dma_single_bd_task(outOF_L2L3s[0], O, offset = 0*tensorSize, sizes=[1, 1, 1, tensorSize], issue_token=True))
            out_tasks.append(shim_dma_single_bd_task(outOF_L2L3s[1], O, offset = 1*tensorSize, sizes=[1, 1, 1, tensorSize], issue_token=True))
            out_tasks.append(shim_dma_single_bd_task(outOF_L2L3s[2], O, offset = 2*tensorSize, sizes=[1, 1, 1, tensorSize], issue_token=True))

            dma_start_task(*in_tasks, *out_tasks)
            dma_await_task(*out_tasks)
            dma_free_task(*in_tasks)

            trace_utils.gen_trace_done_aie2(ShimTiles[2])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-wd", 
        "--width", 
        required=False,
        dest="width",
        default=36,
        help="Image width",
    )
    p.add_argument(
        "-ht", 
        "--height", 
        required=False,
        dest="height",
        default=64,
        help="Image height",
    )
    p.add_argument(
        "-t",
        "--trace_size",
        required=False,
        dest="trace_size",
        default=0,
        help="Trace buffer size",
    )
    opts = p.parse_args(sys.argv[1:])
    
    with mlir_mod_ctx() as ctx:
        edge_detect(int(opts.width), int(opts.height), int(opts.trace_size))
        print(ctx.module)
