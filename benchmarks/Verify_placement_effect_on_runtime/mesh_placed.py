import os
import sys
import json
import argparse
import random
import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessSequence

import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent, ShimTileEvent, MemTileEvent

dtype_map = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
}

def main(opts):
    random.seed(opts.placement_seed)

    with mlir_mod_ctx() as ctx:
        my_benchmark(opts)
        # Print the python-to-mlir conversion to stdout
        print(ctx.module)

def my_benchmark(opts):
    dev = opts.dev
    dtype_str = opts.dtype_str
    inout_size = opts.inout_size
    dtype = dtype_map[dtype_str]
    num_rows = 6
    num_cols = 8
    if dev == "npu2":
        dev_ty = AIEDevice.npu2
    else:
        raise AssertionError("Invalid device type: only NPU2 (Strix/Strix Halo/Krackan) is supported")

    @device(dev_ty)
    def device_body():
        # parse the netlist file and get the tiles and object_fifos and link the object_fifos
        in_data_dtype = np.ndarray[(inout_size,), np.dtype[dtype]]
        out_data_dtype = np.ndarray[(inout_size,), np.dtype[dtype]]
        intermediate_data_dtype = np.ndarray[(1,), np.dtype[dtype]]

        # kernal function declarations
        zero_i32 = external_func(
            "zero_int32_t_1", inputs=[intermediate_data_dtype]
        )
        accumulate_i32 = external_func(
            "accumulate_int32_t_int32_t_1", inputs=[intermediate_data_dtype, np.int32, intermediate_data_dtype]
        )

        def accumulate(in_items, range, out_items, zero_kernel, accumulate_kernel):
            for out_item in out_items:
                zero_kernel(out_item)
                for in_item in in_items:
                    accumulate_kernel(in_item, range, out_item)

        def do_kernel(in_obj_fifo_ids, out_obj_fifo_ids, obj_fifos, zero_kernel, accumulate_kernel):
            in_items = []
            out_items = []
            for output_fifo_id in out_obj_fifo_ids:
                out_items.append(obj_fifos[output_fifo_id].acquire(ObjectFifoPort.Produce, 1))
            for input_fifo_id in in_obj_fifo_ids:
                in_items.append(obj_fifos[input_fifo_id].acquire(ObjectFifoPort.Consume, 1))
            accumulate(in_items, 0, out_items, zero_kernel, accumulate_kernel)
            for input_fifo_id in in_obj_fifo_ids:
                obj_fifos[input_fifo_id].release(ObjectFifoPort.Consume, 1)
            for output_fifo_id in out_obj_fifo_ids:
                obj_fifos[output_fifo_id].release(ObjectFifoPort.Produce, 1)

        # Tile declarations as tile[row][col]
        tiles = [
            [tile(col, row) for col in range(0, num_cols)] for row in range(0, num_rows)
        ]

        fifo_depth = 2
        obj_fifo_lookup = dict()
        obj_fifos = []

        for row in range(2,num_rows):
            for col in range(0,num_cols):
                obj_fifo_lookup[(row,col)] = {"in":[], "out":[], "initial_in":[]}

        # object fifos between tiles all node connect to the top and right nodes
        for row in range(2,num_rows):
            for col in range(0,num_cols):
                curr_node = (row, col)
                top_node = (row+1, col) if row < num_rows-1 else (2, col)
                right_node = (row, col+1) if col < num_cols-1 else (row, 0)

                # Create object fifos to top nodes
                to_top_obj_fifo_id = len(obj_fifos)
                obj_fifos.append(object_fifo(
                    f"obj_fifo_{curr_node[0]}_{curr_node[1]}_to_{top_node[0]}_{top_node[1]}",
                    tiles[curr_node[0]][curr_node[1]],
                    tiles[top_node[0]][top_node[1]],
                    fifo_depth,
                    intermediate_data_dtype,
                ))
                obj_fifo_lookup[(curr_node[0], curr_node[1])]['out'].append(to_top_obj_fifo_id)
                obj_fifo_lookup[(top_node[0], top_node[1])]['in'].append(to_top_obj_fifo_id)
                # To avoid deadlock, we need to initialize some object fifos with data
                if col == 0 and row > 2:
                    obj_fifo_lookup[(top_node[0], top_node[1])]['initial_in'].append(to_top_obj_fifo_id)

                # Create object fifos to right nodes
                to_right_obj_fifo_id = len(obj_fifos)
                obj_fifos.append(object_fifo(
                    f"obj_fifo_{curr_node[0]}_{curr_node[1]}_to_{right_node[0]}_{right_node[1]}",
                    tiles[curr_node[0]][curr_node[1]],
                    tiles[right_node[0]][right_node[1]],
                    fifo_depth,
                    intermediate_data_dtype,
                ))
                obj_fifo_lookup[(curr_node[0], curr_node[1])]['out'].append(to_right_obj_fifo_id)
                obj_fifo_lookup[(right_node[0], right_node[1])]['in'].append(to_right_obj_fifo_id)
                # To avoid deadlock, we need to initialize some object fifos with data
                if row == 3 and col < num_cols-1:
                    obj_fifo_lookup[(right_node[0], right_node[1])]['initial_in'].append(to_right_obj_fifo_id)

        in_shim_to_mem_obj_fifo_id = len(obj_fifos)
        obj_fifos.append(object_fifo(
            f"obj_fifo_0_0_to_1_0",
            tiles[0][0],
            tiles[1][0],
            fifo_depth,
            intermediate_data_dtype,
        ))
        in_mem_to_core_obj_fifo_id = len(obj_fifos)
        obj_fifos.append(object_fifo(
            f"obj_fifo_1_0_to_3_0",
            tiles[1][0],
            tiles[3][0],
            fifo_depth,
            intermediate_data_dtype,
        ))
        object_fifo_link(obj_fifos[in_shim_to_mem_obj_fifo_id], obj_fifos[in_mem_to_core_obj_fifo_id])
        out_core_to_mem_obj_fifo_id = len(obj_fifos)
        obj_fifos.append(object_fifo(
            f"obj_fifo_2_{num_cols-1}_to_1_{num_cols-1}",
            tiles[2][num_cols-1],
            tiles[1][num_cols-1],
            fifo_depth,
            intermediate_data_dtype,
        ))
        out_mem_to_shim_obj_fifo_id = len(obj_fifos)
        obj_fifos.append(object_fifo(
            f"obj_fifo_1_{num_cols-1}_to_0_{num_cols-1}",
            tiles[1][num_cols-1],
            tiles[0][num_cols-1],
            fifo_depth,
            intermediate_data_dtype,
        ))
        object_fifo_link(obj_fifos[out_core_to_mem_obj_fifo_id], obj_fifos[out_mem_to_shim_obj_fifo_id])

        # Core function declarations
        for row in range(2,num_rows):
            for col in range(0,num_cols):
                # Input core that takes input from outside AIE-array then repeatedly sends to the next cores
                if (row == 3 and col == 0):
                    @core(tiles[row][col], "accumulate.o",stack_size=0xFF0)
                    def core_body():
                        # Initial accumulation to avoid deadlock
                        do_kernel([in_mem_to_core_obj_fifo_id], obj_fifo_lookup[(row, col)]["out"], obj_fifos, zero_i32, accumulate_i32)
                        for _ in range_(sys.maxsize):
                            do_kernel([in_mem_to_core_obj_fifo_id] + obj_fifo_lookup[(row, col)]["in"], obj_fifo_lookup[(row, col)]["out"], obj_fifos, zero_i32, accumulate_i32)

                # Output core that takes input from the previous cores then repeatedly sends to outside AIE-array
                elif (row == 2 and col == num_cols-1):
                    @core(tiles[row][col], "accumulate.o",stack_size=0xFF0)
                    def core_body():
                        for _ in range_(sys.maxsize):
                            do_kernel(obj_fifo_lookup[(row, col)]["in"], obj_fifo_lookup[(row, col)]["out"] + [out_core_to_mem_obj_fifo_id], obj_fifos, zero_i32, accumulate_i32)
                            
                # Intermediate cores
                elif row == 3 or col == 0:
                    @core(tiles[row][col], "accumulate.o",stack_size=0xFF0)
                    def core_body():
                        # Initial accumulation to avoid deadlock
                        do_kernel(obj_fifo_lookup[(row, col)]["initial_in"], obj_fifo_lookup[(row, col)]["out"], obj_fifos, zero_i32, accumulate_i32)
                        for _ in range_(sys.maxsize):
                            do_kernel(obj_fifo_lookup[(row, col)]["in"], obj_fifo_lookup[(row, col)]["out"], obj_fifos, zero_i32, accumulate_i32)
                            
                else:
                    @core(tiles[row][col], "accumulate.o",stack_size=0xFF0)
                    def core_body():
                        for _ in range_(sys.maxsize):
                            do_kernel(obj_fifo_lookup[(row, col)]["in"], obj_fifo_lookup[(row, col)]["out"], obj_fifos, zero_i32, accumulate_i32)

        # To/from AIE-array data movement
        @runtime_sequence(
            in_data_dtype,
            in_data_dtype,
            out_data_dtype,
        )
        def sequence(Input_one, Input_two, Output):
            in_tasks = []
            in_tasks.append(shim_dma_single_bd_task(
                obj_fifos[in_shim_to_mem_obj_fifo_id], 
                Input_one, 
                sizes=[1,1,1,inout_size],
            ))
            
            out_tasks = []
            out_tasks.append(shim_dma_single_bd_task(
                obj_fifos[out_mem_to_shim_obj_fifo_id],
                Output,
                sizes=[1,1,1,inout_size],
                issue_token=True
            ))

            dma_start_task(*in_tasks, *out_tasks)
            dma_await_task(*out_tasks)
            dma_free_task(*in_tasks)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="Microbenchmark",
        description="Emits MLIR code for microbenchmark of the given netlist topology.",
    )
    argparser.add_argument(
        "--dev", 
        type=str, 
        dest="dev",
        default="npu2",
    )
    argparser.add_argument(
        "--inout_size", 
        type=int, 
        dest="inout_size",
        default=1,
    )
    argparser.add_argument(
        "--num_rows", 
        type=int, 
        dest="num_rows",
        default=6
    )
    argparser.add_argument(
        "--num_cols", 
        type=int, 
        dest="num_cols",
        default=8
    )
    argparser.add_argument(
        "--dtype", 
        type=str, 
        dest="dtype_str",
        choices=["i32"], 
        default="i32"
    )
    argparser.add_argument(
        "--placement_seed", 
        type=int, 
        dest="placement_seed",
        default=0,
    )
    argparser.add_argument(
        "--trace_size", 
        type=int, 
        default=0
    )
    opts = argparser.parse_args()
    main(opts)