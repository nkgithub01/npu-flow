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
    random.seed(opts.random_seed)

    with mlir_mod_ctx() as ctx:
        my_benchmark(opts)
        # Print the python-to-mlir conversion to stdout
        print(ctx.module)

def my_benchmark(opts):
    dev = opts.dev
    dtype_str = opts.dtype_str
    inout_size = opts.inout_size
    dtype = dtype_map[dtype_str]
    num_rows = opts.num_rows
    num_cols = opts.num_cols
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
        tiles = dict()
        for row in range(2, num_rows):
            for col in range(0, num_cols):
                tiles[(row, col)] = tile(col, row)

        total_num_comp_tiles = (num_rows - 2) * num_cols

        fifo_depth = 2
        obj_fifo_lookup = dict()
        obj_fifos = []

        for row in range(2,num_rows):
            for col in range(0,num_cols):
                obj_fifo_lookup[(row,col)] = {"in":[], "out":[], "initial_in":[]}

        # Each compute node is connected to 2 other nodes to form a binary tree structure
        in_shim_to_mem_obj_fifo_id = None
        in_mem_to_core_obj_fifo_id = None
        out_core_to_mem_obj_fifo_id = None
        out_mem_to_shim_obj_fifo_id = None
        feedback_fifo_id = None
        branching_factor = opts.branching_factor
        no_children_node_locs = []
        for node_idx in range(total_num_comp_tiles):
            curr_node_x = node_idx % num_cols
            curr_node_y = 2 + (node_idx // num_cols)
            if node_idx == 0:
                # define the MEM and SHIM tiles if not already defined
                if (0, curr_node_x) not in tiles:
                    tiles[(0, curr_node_x)] = tile(curr_node_x, 0)
                if (1, curr_node_x) not in tiles:
                    tiles[(1, curr_node_x)] = tile(curr_node_x, 1)
                # Output from COMP to MEM
                out_core_to_mem_obj_fifo_id = len(obj_fifos)
                obj_fifos.append(object_fifo(
                    f"obj_fifo_{curr_node_y}_{curr_node_x}_to_{1}_{curr_node_x}",
                    tiles[(curr_node_y, curr_node_x)],
                    tiles[(1, curr_node_x)],
                    fifo_depth,
                    intermediate_data_dtype,
                ))
                obj_fifo_lookup[(curr_node_y, curr_node_x)]['out'].append(out_core_to_mem_obj_fifo_id)
                # Output from MEM to SHIM
                out_mem_to_shim_obj_fifo_id = len(obj_fifos)
                obj_fifos.append(object_fifo(
                    f"obj_fifo_{1}_{curr_node_x}_to_{0}_{curr_node_x}",
                    tiles[(1, curr_node_x)],
                    tiles[(0, curr_node_x)],
                    fifo_depth,
                    intermediate_data_dtype,
                ))
                object_fifo_link(obj_fifos[out_core_to_mem_obj_fifo_id], obj_fifos[out_mem_to_shim_obj_fifo_id])

            # Connects from children nodes to current node
            has_no_children = True
            for child_idx in range(branching_factor):
                child_node_idx = node_idx * branching_factor + child_idx + 1
                child_x = child_node_idx % num_cols
                child_y = 2 + (child_node_idx // num_cols)
                if child_x < num_cols and child_y < num_rows and (child_x != (num_cols-1) or child_y != (num_rows-1)):
                    # Connects from child to the current node
                    obj_fifo_id = len(obj_fifos)
                    obj_fifos.append(object_fifo(
                        f"obj_fifo_{child_y}_{child_x}_to_{curr_node_y}_{curr_node_x}",
                        tiles[(child_y, child_x)],
                        tiles[(curr_node_y, curr_node_x)],
                        fifo_depth,
                        intermediate_data_dtype,
                    ))
                    obj_fifo_lookup[(child_y, child_x)]['out'].append(obj_fifo_id)
                    obj_fifo_lookup[(curr_node_y, curr_node_x)]['in'].append(obj_fifo_id)
                    obj_fifo_lookup[(curr_node_y, curr_node_x)]['initial_in'].append(obj_fifo_id)
                    has_no_children = False

            if has_no_children:
                if curr_node_x != (num_cols-1) or curr_node_y != (num_rows-1):
                    no_children_node_locs.append((curr_node_y, curr_node_x))

            if node_idx == total_num_comp_tiles-1:
                # define the MEM and SHIM tiles if not already defined
                if (0, curr_node_x) not in tiles:
                    tiles[(0, curr_node_x)] = tile(curr_node_x, 0)
                if (1, curr_node_x) not in tiles:
                    tiles[(1, curr_node_x)] = tile(curr_node_x, 1)
                # Input from SHIM to MEM
                in_shim_to_mem_obj_fifo_id = len(obj_fifos)
                obj_fifos.append(object_fifo(
                    f"obj_fifo_{0}_{curr_node_x}_to_{1}_{curr_node_x}",
                    tiles[(0, curr_node_x)],
                    tiles[(1, curr_node_x)],
                    fifo_depth,
                    intermediate_data_dtype,
                ))
                # Input from MEM to COMP
                in_mem_to_core_obj_fifo_id = len(obj_fifos)
                obj_fifos.append(object_fifo(
                    f"obj_fifo_{1}_{curr_node_x}_to_{curr_node_y}_{curr_node_x}",
                    tiles[(1, curr_node_x)],
                    tiles[(curr_node_y, curr_node_x)],
                    fifo_depth,
                    intermediate_data_dtype,
                ))
                object_fifo_link(obj_fifos[in_shim_to_mem_obj_fifo_id], obj_fifos[in_mem_to_core_obj_fifo_id])
                obj_fifo_lookup[(curr_node_y, curr_node_x)]['in'].append(in_mem_to_core_obj_fifo_id)
                obj_fifo_lookup[(curr_node_y, curr_node_x)]['initial_in'].append(in_mem_to_core_obj_fifo_id)

                # broadcast to all nodes that have no children
                if len(no_children_node_locs) != 0:
                    obj_fifo_id = len(obj_fifos)
                    obj_fifos.append(object_fifo(
                        f"obj_fifo_{curr_node_y}_{curr_node_x}_to_node_with_no_children",
                        tiles[(curr_node_y, curr_node_x)],
                        [tiles[(y, x)] for (y, x) in no_children_node_locs],
                        fifo_depth,
                        intermediate_data_dtype,
                    ))
                    obj_fifo_lookup[(curr_node_y, curr_node_x)]['out'].append(obj_fifo_id)
                    for (y, x) in no_children_node_locs:
                        obj_fifo_lookup[(y, x)]['in'].append(obj_fifo_id)
                        obj_fifo_lookup[(y, x)]['initial_in'].append(obj_fifo_id)
                
                # Feedback connection
                feedback_fifo_id = len(obj_fifos)
                obj_fifos.append(object_fifo(
                    f"obj_fifo_2_0_to_{curr_node_y}_{curr_node_x}_feedback",
                    tiles[(2, 0)],
                    tiles[(curr_node_y, curr_node_x)],
                    fifo_depth,
                    intermediate_data_dtype,
                ))
                obj_fifo_lookup[(curr_node_y, curr_node_x)]['in'].append(feedback_fifo_id)
                obj_fifo_lookup[(2, 0)]['out'].append(feedback_fifo_id)

        # Core function declarations
        for row in range(2,num_rows):
            for col in range(0,num_cols):
                @core(tiles[(row, col)], "accumulate.o")
                def core_body():
                    # Initial accumulation to avoid deadlock
                    do_kernel(obj_fifo_lookup[(row, col)]["initial_in"], obj_fifo_lookup[(row, col)]["out"], obj_fifos, zero_i32, accumulate_i32)
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
        "--branching_factor", 
        type=int, 
        dest="branching_factor",
        default=2
    )
    argparser.add_argument(
        "--dtype", 
        type=str, 
        dest="dtype_str",
        choices=["i32"], 
        default="i32"
    )
    argparser.add_argument(
        "--random_seed", 
        type=int, 
        dest="random_seed",
        default=0,
    )
    argparser.add_argument(
        "--trace_size", 
        type=int, 
        default=0
    )
    opts = argparser.parse_args()
    main(opts)