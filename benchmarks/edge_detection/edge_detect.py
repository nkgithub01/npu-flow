#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

import numpy as np
import sys
import argparse

from aie.iron import LocalBuffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_


def edge_detect(image_width, image_height):
    num_compute_flow_column = 3
    
    height_minus1 = image_height - 1
    line_width = image_width
    line_width_in_bytes = image_width * 4
    tensor_size = image_width * image_height * 4  # 4 channels (RGBA)

    # Type definitions
    line_bytes_ty = np.ndarray[(line_width_in_bytes,), np.dtype[np.uint8]]
    line_ty = np.ndarray[(line_width,), np.dtype[np.uint8]]
    tensor_3x3_ty = np.ndarray[(3, 3), np.dtype[np.int16]]

    i_tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int8]]
    o_tensor_ty = np.ndarray[(tensor_size * num_compute_flow_column,), np.dtype[np.int8]]

    # AIE Core Function declarations
    rgba2gray_line_kernel = Kernel(
        "rgba2grayLine", "rgba2gray.cc.o", [line_bytes_ty, line_ty, np.int32]
    )
    filter2d_line_kernel = Kernel(
        "filter2dLine",
        "filter2d.cc.o",
        [line_ty, line_ty, line_ty, line_ty, np.int32, tensor_3x3_ty],
    )
    threshold_line_kernel = Kernel(
        "thresholdLine",
        "threshold.cc.o",
        [line_ty, line_ty, np.int32, np.int16, np.int16, np.int8],
    )
    gray2rgba_line_kernel = Kernel(
        "gray2rgbaLine",
        "combined_gray2rgba_addWeighted.a",
        [line_ty, line_bytes_ty, np.int32],
    )
    add_weighted_line_kernel = Kernel(
        "addWeightedLine",
        "combined_gray2rgba_addWeighted.a",
        [
            line_bytes_ty,
            line_bytes_ty,
            line_bytes_ty,
            np.int32,
            np.int16,
            np.int16,
            np.int8,
        ],
    )

    # AIE-array data movement with object fifos
    # Input
    inOF_L3L2s = []
    inOF_L2L1s = []
    for j in range(num_compute_flow_column):
        inOF_L3L2s.append(ObjectFifo(line_bytes_ty, name=f"inOF_L3L2_{j}"))
        inOF_L2L1s.append(inOF_L3L2s[-1].cons(7).forward(depth=7, name=f"inOF_L2L1_{j}"))

    # Output
    outOF_L1L2s = []
    outOF_L2L3s = []
    for j in range(num_compute_flow_column):
        outOF_L1L2s.append(ObjectFifo(line_bytes_ty, name=f"outOF_L1L2_{j}"))
        outOF_L2L3s.append(outOF_L1L2s[-1].cons().forward(name=f"outOF_L2L3_{j}"))

    # Intermediate
    depths = [4, 2, 2]
    of_intermediates = []
    for j in range(num_compute_flow_column):
        of_intermediates.append([
            ObjectFifo(line_ty, default_depth=depths[i], name=f"OF_{i + 2}to{i + 3}_{j}")
            for i in range(3)
        ])
    of_locals = []
    for j in range(num_compute_flow_column):
        of_locals.append(ObjectFifo(line_bytes_ty, default_depth=1, name=f"OF_local_{j}"))

    workers = []

    # Task for the core to perform
    def rgba2gray_fn(of_in, of_out, rgba2gray_line):
        # inOF_L3L2
        # OF_2to3 -> of_intermediates[0]
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        rgba2gray_line(elem_in, elem_out, line_width)
        of_in.release(1)
        of_out.release(1)

    # Worker to run the task
    for j in range(num_compute_flow_column):
        workers.append(
            Worker(
                rgba2gray_fn,
                [inOF_L3L2s[j].cons(), of_intermediates[j][0].prod(), rgba2gray_line_kernel],
            )
        )

    # Task for the core to perform
    def filter_fn(of_in, of_out, filter2d_line):
        # OF_2to3 -> intermediates[0]
        # OF_3to4 -> intermediates[1]

        # Define filter kernel
        v0 = 0
        v1 = 4096
        v_minus4 = -16384
        kernel = LocalBuffer(
            np.ndarray[(3, 3), np.dtype[np.int16]],
            initial_value=np.array(
                [[v0, v1, v0], [v1, v_minus4, v1], [v0, v1, v0]], dtype=np.int16
            ),
        )

        for _ in range_(sys.maxsize):
            # Preamble : Top Border
            elems_in_pre = of_in.acquire(2)
            elem_pre_out = of_out.acquire(1)
            filter2d_line(
                elems_in_pre[0],
                elems_in_pre[0],
                elems_in_pre[1],
                elem_pre_out,
                line_width,
                kernel,
            )
            of_out.release(1)

            # Steady State : Middle
            for _ in range_(1, height_minus1):
                elems_in = of_in.acquire(3)
                elem_out = of_out.acquire(1)
                filter2d_line(
                    elems_in[0],
                    elems_in[1],
                    elems_in[2],
                    elem_out,
                    line_width,
                    kernel,
                )
                of_in.release(1)
                of_out.release(1)

            # Postamble : Bottom Border
            elems_in_post = of_in.acquire(2)
            elem_post_out = of_out.acquire(1)
            filter2d_line(
                elems_in_post[0],
                elems_in_post[1],
                elems_in_post[1],
                elem_post_out,
                line_width,
                kernel,
            )
            of_in.release(2)
            of_out.release(1)

    # Worker to run the task
    for j in range(num_compute_flow_column):
        workers.append(
            Worker(
                filter_fn,
                [
                    of_intermediates[j][0].cons(),
                    of_intermediates[j][1].prod(),
                    filter2d_line_kernel,
                ],
                while_true=False,
            )
        )

    # Task for the core to perform
    def threshold_fn(of_in, of_out, threshold_line):
        v_thr = 10
        v_max = 255
        v_typ = 0

        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        threshold_line(elem_in, elem_out, line_width, v_thr, v_max, v_typ)
        of_in.release(1)
        of_out.release(1)

    # Worker to run the task
    for j in range(num_compute_flow_column):
        workers.append(
            Worker(
                threshold_fn,
                [
                    of_intermediates[j][1].cons(),
                    of_intermediates[j][2].prod(),
                    threshold_line_kernel,
                ],
            )
        )

    # Task for the core to perform
    def gray2rgba_addWeight_fn(
        of_in,
        of_in2,
        if_out_self,
        of_in_self,
        of_out,
        gray2rgba_line,
        add_weighted_line,
    ):
        elem_in = of_in.acquire(1)
        elem_out = if_out_self.acquire(1)

        gray2rgba_line(elem_in, elem_out, line_width)

        of_in.release(1)
        if_out_self.release(1)

        elem_in1 = of_in_self.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out2 = of_out.acquire(1)

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

        of_in_self.release(1)
        of_in2.release(1)
        of_out.release(1)

    # Worker to run the task
    for j in range(num_compute_flow_column):
        workers.append(
            Worker(
                gray2rgba_addWeight_fn,
                [
                    of_intermediates[j][2].cons(),
                    inOF_L2L1s[j].cons(),
                    of_locals[j].prod(),
                    of_locals[j].cons(),
                    outOF_L1L2s[j].prod(),
                    gray2rgba_line_kernel,
                    add_weighted_line_kernel,
                ],
            )
        )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(i_tensor_ty, i_tensor_ty, o_tensor_ty) as (I1, I2, O):
        rt.start(*workers)
        shim = Tile(0,0)
        rt.fill(inOF_L3L2s[0].prod(), I1, placement=shim)
        rt.drain(outOF_L2L3s[0].cons(), O, wait=True, placement=shim)
        shim = Tile(1,0)
        rt.fill(inOF_L3L2s[1].prod(), I2, placement=shim)
        rt.drain(outOF_L2L3s[1].cons(), O, wait=True, placement=shim)
        shim = Tile(2,0)
        rt.fill(inOF_L3L2s[2].prod(), I2, placement=shim)
        rt.drain(outOF_L2L3s[2].cons(), O, wait=True, placement=shim)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(NPU2, rt).resolve_program(SequentialPlacer())


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
    
    opts = p.parse_args(sys.argv[1:])

    module = edge_detect(opts.image_width, opts.image_height)
    print(module)
