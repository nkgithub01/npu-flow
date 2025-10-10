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
    if dev == "npu2":
        dev_ty = AIEDevice.npu2
    else:
        raise AssertionError("Invalid device type: only NPU2 (Strix/Strix Halo/Krackan) is supported")

    @device(dev_ty)
    def device_body():
        # parse the netlist file and get the tiles and object_fifos and link the object_fifos
        in_data_dtype = np.ndarray[(inout_size,), np.dtype[dtype]]
        out_data_dtype = np.ndarray[(inout_size,), np.dtype[dtype]]
        intermediate_data_dtype = np.ndarray[(inout_size,), np.dtype[dtype]]

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
            accumulate(in_items, 1, out_items, zero_kernel, accumulate_kernel)
            for input_fifo_id in in_obj_fifo_ids:
                obj_fifos[input_fifo_id].release(ObjectFifoPort.Consume, 1)
            for output_fifo_id in out_obj_fifo_ids:
                obj_fifos[output_fifo_id].release(ObjectFifoPort.Produce, 1)

        # Tile declarations as tile[row][col]
        tiles = [
            [tile(col, row) for col in range(0, 8)] for row in range(0, 6)
        ]

        expand_rate = opts.expand_rate
        contract_rate = expand_rate
        fifo_depth = 2

        obj_fifo_lookup = dict()
        obj_fifos = []

        for row in range(2,6):
            for col in range(0,8):
                obj_fifo_lookup[(row,col)] = {"in":[], "out":[], "initial_in":[]}

        # object fifos between tiles all node connect to the top and right nodes
        for row in range(2,6):
            for col in range(0,8):
                curr_node = (row, col)
                top_node = (row+1, col) if row < 5 else (2, col)
                right_node = (row, col+1) if col < 7 else (row, 0)

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
                if row == 3 and col < 7:
                    obj_fifo_lookup[(right_node[0], right_node[1])]['initial_in'].append(to_right_obj_fifo_id)

        in_obj_fifo_id = len(obj_fifos)
        obj_fifos.append(object_fifo(
            f"obj_fifo_0_0_to_3_0",
            tiles[0][0],
            tiles[3][0],
            fifo_depth,
            in_data_dtype,
        ))
        out_obj_fifo_id = len(obj_fifos)
        obj_fifos.append(object_fifo(
            f"obj_fifo_2_7_to_0_7",
            tiles[2][7],
            tiles[0][7],
            fifo_depth,
            out_data_dtype,
        ))

        # Core function declarations
        for row in range(2,6):
            for col in range(0,8):
                # Input core that takes input from outside AIE-array then repeatedly sends to the next cores
                if (row == 3 and col == 0):
                    @core(tiles[row][col], "accumulate.o")
                    def core_body():
                        # Initial accumulation to avoid deadlock
                        do_kernel([in_obj_fifo_id], obj_fifo_lookup[(row, col)]["out"], obj_fifos, zero_i32, accumulate_i32)
                        for _ in range_(sys.maxsize):
                            do_kernel([in_obj_fifo_id] + obj_fifo_lookup[(row, col)]["in"], obj_fifo_lookup[(row, col)]["out"], obj_fifos, zero_i32, accumulate_i32)

                # Output core that takes input from the previous cores then repeatedly sends to outside AIE-array
                elif (row == 2 and col == 7):
                    @core(tiles[row][col], "accumulate.o")
                    def core_body():
                        for _ in range_(sys.maxsize):
                            do_kernel(obj_fifo_lookup[(row, col)]["in"], [out_obj_fifo_id] + obj_fifo_lookup[(row, col)]["out"], obj_fifos, zero_i32, accumulate_i32)
                            
                # Intermediate cores
                elif row == 3 or col == 0:
                    @core(tiles[row][col], "accumulate.o")
                    def core_body():
                        # Initial accumulation to avoid deadlock
                        do_kernel(obj_fifo_lookup[(row, col)]["initial_in"], obj_fifo_lookup[(row, col)]["out"], obj_fifos, zero_i32, accumulate_i32)
                        for _ in range_(sys.maxsize):
                            do_kernel(obj_fifo_lookup[(row, col)]["in"], obj_fifo_lookup[(row, col)]["out"], obj_fifos, zero_i32, accumulate_i32)
                            
                else:
                    @core(tiles[row][col], "accumulate.o")
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
            input_tile = TensorTiler2D.group_tiler(
                (inout_size,), # Shape of the input tensor
                (inout_size,),
                (1,),
                pattern_repeat=64,  # Repeat data
            )
            in_tasks.append(shim_dma_single_bd_task(
                obj_fifos[in_obj_fifo_id], 
                Input_one, 
                tap=input_tile[0],
            ))
            
            out_tasks = []
            output_tile = TensorTiler2D.group_tiler(
                (inout_size,), # Shape of the output tensor
                (inout_size,),
                (1,),
                pattern_repeat=64,  # Repeat data
            )
            out_tasks.append(shim_dma_single_bd_task(
                obj_fifos[out_obj_fifo_id],
                Output,
                tap=output_tile[0],
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
        "--placement", 
        type=str, 
        dest="placement",
        default="regular",
    )
    argparser.add_argument(
        "--placement_seed", 
        type=int, 
        dest="placement_seed",
        default=0,
    )
    argparser.add_argument(
        "--enable_feedback", 
        type=int, 
        dest="enable_feedback",
        choices=[0, 1], 
        default=0,
    )
    argparser.add_argument(
        "--expand_rate", 
        type=int, 
        dest="expand_rate",
        default=10**6,
    )
    argparser.add_argument(
        "--length", 
        type=int, 
        dest="length",
        default=16,
    )
    argparser.add_argument(
        "--dtype", 
        type=str, 
        dest="dtype_str",
        choices=["i32"], 
        default="i32"
    )
    argparser.add_argument(
        "--trace_size", 
        type=int, 
        default=0
    )
    opts = argparser.parse_args()
    main(opts)