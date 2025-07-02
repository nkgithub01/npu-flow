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
from aie.utils.trace import PortEvent, ShimTilePortEvent
from aie.utils.trace_events_enum import CoreEvent, ShimTileEvent, MemTileEvent, MemEvent

def edge_detect(image_width, image_height, num_compute_flow_column, trace_size):
    height_minus1 = image_height - 1
    line_width = image_width
    line_width_in_bytes = image_width * 4
    tensor_size = image_width * image_height * 4  # 4 channels (RGBA)

    @device(AIEDevice.npu2)
    def device_body():
        line_bytes_ty = np.ndarray[(line_width_in_bytes,), np.dtype[np.uint8]]
        line_ty = np.ndarray[(line_width,), np.dtype[np.uint8]]
        tensor_3x3_ty = np.ndarray[(3, 3), np.dtype[np.int16]]

        i_tensor_ty = np.ndarray[(tensor_size * num_compute_flow_column,), np.dtype[np.uint8]]
        o_tensor_ty = np.ndarray[(tensor_size * num_compute_flow_column,), np.dtype[np.uint8]]

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
        shim_tiles = []
        mem_tiles = []
        compute_tiles_row_2 = []
        compute_tiles_row_3 = []
        compute_tiles_row_4 = []
        compute_tiles_row_5 = []
        for col_idx in range(num_compute_flow_column):
            shim_tiles.append(tile(col_idx, 0))
            mem_tiles.append(tile(col_idx, 1))
            compute_tiles_row_2.append(tile(col_idx, 2))
            compute_tiles_row_3.append(tile(col_idx, 3))
            compute_tiles_row_4.append(tile(col_idx, 4))
            compute_tiles_row_5.append(tile(col_idx, 5))

        # AIE-array data movement with object fifos
        # Input
        inOF_L3L2s = []
        inOF_L2L1s = []
        for col_idx in range(num_compute_flow_column):
            inOF_L3L2s.append(object_fifo(
                f"col_{col_idx}_inOF_L3L2",
                shim_tiles[col_idx],
                [compute_tiles_row_2[col_idx], mem_tiles[col_idx]],
                [2, 2, 7],
                line_bytes_ty,
            ))
            inOF_L2L1s.append(object_fifo(
                f"col_{col_idx}_inOF_L2L1",
                mem_tiles[col_idx],
                compute_tiles_row_5[col_idx],
                7,
                line_bytes_ty,
            ))
            object_fifo_link(inOF_L3L2s[-1], inOF_L2L1s[-1])

        # Output
        outOF_L2L3s = []
        outOF_L1L2s = []
        for col_idx in range(num_compute_flow_column):
            outOF_L2L3s.append(object_fifo(
                f"col_{col_idx}_outOF_L2L3",
                mem_tiles[col_idx],
                shim_tiles[col_idx],
                2,
                line_bytes_ty,
            ))
            outOF_L1L2s.append(object_fifo(
                f"col_{col_idx}_outOF_L1L2",
                compute_tiles_row_5[col_idx],
                mem_tiles[col_idx],
                2,
                line_bytes_ty,
            ))
            object_fifo_link(outOF_L1L2s[-1], outOF_L2L3s[-1])

        # Intermediate
        OF_2to3s = []
        OF_3to4s = []
        OF_4to5s = []
        OF_5to5s = []
        for col_idx in range(num_compute_flow_column):
            OF_2to3s.append(object_fifo(
                f"col_{col_idx}_OF_2to3",
                compute_tiles_row_2[col_idx],
                compute_tiles_row_3[col_idx],
                4,
                line_ty,
            ))
            OF_3to4s.append(object_fifo(
                f"col_{col_idx}_OF_3to4",
                compute_tiles_row_3[col_idx],
                compute_tiles_row_4[col_idx],
                2,
                line_ty,
            ))
            OF_4to5s.append(object_fifo(
                f"col_{col_idx}_OF_4to5",
                compute_tiles_row_4[col_idx],
                compute_tiles_row_5[col_idx],
                2,
                line_ty,
            ))
            OF_5to5s.append(object_fifo(
                f"col_{col_idx}_OF_5to5",
                compute_tiles_row_5[col_idx],
                compute_tiles_row_5[col_idx],
                1,
                line_bytes_ty,
            ))

        # Set up compute tiles
        for col_idx in range(num_compute_flow_column):
            # Compute tile 2
            @core(compute_tiles_row_2[col_idx], "rgba2gray.cc.o")
            def core_body():
                for _ in range_(sys.maxsize):
                    elem_in = inOF_L3L2s[col_idx].acquire(ObjectFifoPort.Consume, 1)
                    elem_out = OF_2to3s[col_idx].acquire(ObjectFifoPort.Produce, 1)
    
                    rgba2gray_line(elem_in, elem_out, line_width)
    
                    inOF_L3L2s[col_idx].release(ObjectFifoPort.Consume, 1)
                    OF_2to3s[col_idx].release(ObjectFifoPort.Produce, 1)
    
            # Compute tile 3
            @core(compute_tiles_row_3[col_idx], "filter2d.cc.o")
            def core_body():
                v0 = 0
                v1 = 4096
                v_minus4 = -16384
                initial_value = np.array(
                    [[v0, v1, v0], [v1, v_minus4, v1], [v0, v1, v0]], dtype=np.int16
                )
                kernel = buffer(
                    compute_tiles_row_3[col_idx],
                    np.ndarray[(3, 3), np.dtype[np.int16]],
                    f"col_{col_idx}_kernel",
                    initial_value=initial_value,
                )
    
                for _ in range_(sys.maxsize):
                    # Preamble : Top Border
                    elems_in_pre = OF_2to3s[col_idx].acquire(ObjectFifoPort.Consume, 2)
                    elem_pre_out = OF_3to4s[col_idx].acquire(ObjectFifoPort.Produce, 1)
                    filter2d_line(
                        elems_in_pre[0],
                        elems_in_pre[0],
                        elems_in_pre[1],
                        elem_pre_out,
                        line_width,
                        kernel,
                    )
                    
                    OF_3to4s[col_idx].release(ObjectFifoPort.Produce, 1)
    
                    # Steady State : Middle
                    for _ in range_(1, height_minus1):
                        elems_in = OF_2to3s[col_idx].acquire(ObjectFifoPort.Consume, 3)
                        elem_out = OF_3to4s[col_idx].acquire(ObjectFifoPort.Produce, 1)
                        filter2d_line(
                            elems_in[0],
                            elems_in[1],
                            elems_in[2],
                            elem_out,
                            line_width,
                            kernel,
                        )
                        OF_2to3s[col_idx].release(ObjectFifoPort.Consume, 1)
                        OF_3to4s[col_idx].release(ObjectFifoPort.Produce, 1)
    
                    # Postamble : Bottom Border
                    elems_in_post = OF_2to3s[col_idx].acquire(ObjectFifoPort.Consume, 2)
                    elem_post_out = OF_3to4s[col_idx].acquire(ObjectFifoPort.Produce, 1)
                    filter2d_line(
                        elems_in_post[0],
                        elems_in_post[1],
                        elems_in_post[1],
                        elem_post_out,
                        line_width,
                        kernel,
                    )
                    OF_2to3s[col_idx].release(ObjectFifoPort.Consume, 2)
                    OF_3to4s[col_idx].release(ObjectFifoPort.Produce, 1)
    
            # Compute tile 4
            @core(compute_tiles_row_4[col_idx], "threshold.cc.o")
            def core_body():
                v_thr = 10
                v_max = 255
                v_typ = 0
    
                for _ in range_(sys.maxsize):
                    elem_in = OF_3to4s[col_idx].acquire(ObjectFifoPort.Consume, 1)
                    elem_out = OF_4to5s[col_idx].acquire(ObjectFifoPort.Produce, 1)
    
                    threshold_line(elem_in, elem_out, line_width, v_thr, v_max, v_typ)
    
                    OF_3to4s[col_idx].release(ObjectFifoPort.Consume, 1)
                    OF_4to5s[col_idx].release(ObjectFifoPort.Produce, 1)
    
            # Compute tile 5
            @core(compute_tiles_row_5[col_idx], "combined_gray2rgba_addWeighted.a")
            def core_body():
                for _ in range_(sys.maxsize):
                    elem_in = OF_4to5s[col_idx].acquire(ObjectFifoPort.Consume, 1)
                    elem_out = OF_5to5s[col_idx].acquire(ObjectFifoPort.Produce, 1)
    
                    gray2rgba_line(elem_in, elem_out, line_width)
    
                    OF_4to5s[col_idx].release(ObjectFifoPort.Consume, 1)
                    OF_5to5s[col_idx].release(ObjectFifoPort.Produce, 1)
    
                    elem_in1 = OF_5to5s[col_idx].acquire(ObjectFifoPort.Consume, 1)
                    elem_in2 = inOF_L2L1s[col_idx].acquire(ObjectFifoPort.Consume, 1)
                    elem_out2 = outOF_L1L2s[col_idx].acquire(ObjectFifoPort.Produce, 1)
    
                    alpha = 16384
                    beta = 16384
                    gamma = 0
    
                    add_weighted_line(
                        elem_in1,
                        elem_in2,
                        elem_out2,
                        line_width_in_bytes,
                        alpha,
                        beta,
                        gamma,
                    )
    
                    OF_5to5s[col_idx].release(ObjectFifoPort.Consume, 1)
                    inOF_L2L1s[col_idx].release(ObjectFifoPort.Consume, 1)
                    outOF_L1L2s[col_idx].release(ObjectFifoPort.Produce, 1)

        tiles_to_trace = shim_tiles
        if trace_size > 0: 
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tiles[0])

        # To/from AIE-array data movement
        @runtime_sequence(i_tensor_ty, o_tensor_ty)
        def sequence(I, O):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=shim_tiles[0],
                    trace_size=trace_size,
                    shimtile_events=[
                        ShimTileEvent.DMA_MM2S_0_START_TASK,
                        ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
                        ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION, #dummy event to get enough trace
                    ],
                )
            in_tasks = []
            out_tasks = []
            for col_idx in range(num_compute_flow_column):
                in_tasks.append(shim_dma_single_bd_task(inOF_L3L2s[col_idx], I, offset = col_idx*tensor_size, sizes=[1, 1, 1, tensor_size], issue_token=True))
                out_tasks.append(shim_dma_single_bd_task(outOF_L2L3s[col_idx], O, offset = col_idx*tensor_size, sizes=[1, 1, 1, tensor_size], issue_token=True))

            dma_start_task(*in_tasks, *out_tasks)
            dma_await_task(*out_tasks)
            dma_free_task(*in_tasks)

            trace_utils.gen_trace_done_aie2(shim_tiles[0])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-iwd", 
        "--image_width", 
        type=int,
        required=False,
        dest="image_width",
        default=1920,
        help="Image width",
    )
    p.add_argument(
        "-iht", 
        "--image_height", 
        type=int,
        required=False,
        dest="image_height",
        default=1080,
        help="Image height",
    )
    p.add_argument(
        "-nc", 
        "--num_compute_flow_column", 
        type=int,
        required=False,
        dest="num_compute_flow_column",
        default=4,
        help="Number of compute flow columns on the AIE array that will be used in parallel",
    )
    p.add_argument(
        "-t",
        "--trace_size",
        type=int,
        required=False,
        dest="trace_size",
        default=0,
        help="Trace buffer size",
    )
    opts = p.parse_args(sys.argv[1:])
    
    with mlir_mod_ctx() as ctx:
        edge_detect(int(opts.image_width), int(opts.image_height), int(opts.num_compute_flow_column), int(opts.trace_size))
        print(ctx.module)
