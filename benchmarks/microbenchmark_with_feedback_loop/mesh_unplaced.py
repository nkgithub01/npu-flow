import os
import sys
import json
import argparse
import random
import numpy as np

from aie.iron import LocalBuffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer, SAPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

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
    module = my_benchmark(opts)
    # Print the python-to-mlir conversion to stdout
    print(module)

def my_benchmark(opts):
    dev = opts.dev
    dtype_str = opts.dtype_str
    inout_size = opts.inout_size
    dtype = dtype_map[dtype_str]
    num_rows = opts.num_rows
    num_cols = opts.num_cols
    if dev == "npu2":
        dev_ty = NPU2()
    else:
        raise AssertionError("Invalid device type: only NPU2 (Strix/Strix Halo/Krackan) is supported")
    
    # parse the netlist file and get the tiles and object_fifos and link the object_fifos
    in_data_dtype = np.ndarray[(inout_size,), np.dtype[dtype]]
    out_data_dtype = np.ndarray[(inout_size,), np.dtype[dtype]]
    intermediate_data_dtype = np.ndarray[(1,), np.dtype[dtype]]

    # kernal function declarations
    zero_i32 = Kernel(
        "zero_int32_t_1", "accumulate.o", [intermediate_data_dtype]
    )
    accumulate_i32 = Kernel(
        "accumulate_int32_t_int32_t_1", "accumulate.o", [intermediate_data_dtype, np.int32, intermediate_data_dtype]
    )

    def accumulate(in_items, range, out_items, zero_kernel, accumulate_kernel):
        for out_item in out_items:
            zero_kernel(out_item)
            for in_item in in_items:
                accumulate_kernel(in_item, range, out_item)

    def do_kernel(zero_kernel=None, accumulate_kernel=None, num_inFIFO=2, num_outFIFO=2, *obj_FIFOs):
        in_items = []
        out_items = []
        for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
            out_items.append(outFIFO.acquire(1))
        for inFIFO in obj_FIFOs[:num_inFIFO]:
            in_items.append(inFIFO.acquire(1))
        accumulate(in_items, 0, out_items, zero_kernel, accumulate_kernel)
        for inFIFO in obj_FIFOs[:num_inFIFO]:
            inFIFO.release(1)
        for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
            outFIFO.release(1)

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
            obj_fifos.append(ObjectFifo(
                name=f"obj_fifo_{curr_node[0]}_{curr_node[1]}_to_{top_node[0]}_{top_node[1]}",
                default_depth=fifo_depth,
                obj_type=intermediate_data_dtype,
            ))
            obj_fifo_lookup[(curr_node[0], curr_node[1])]['out'].append(to_top_obj_fifo_id)
            obj_fifo_lookup[(top_node[0], top_node[1])]['in'].append(to_top_obj_fifo_id)
            # To avoid deadlock, we need to initialize some object fifos with data
            if col == 0 and row > 2:
                obj_fifo_lookup[(top_node[0], top_node[1])]['initial_in'].append(to_top_obj_fifo_id)

            # Create object fifos to right nodes
            to_right_obj_fifo_id = len(obj_fifos)
            obj_fifos.append(ObjectFifo(
                name=f"obj_fifo_{curr_node[0]}_{curr_node[1]}_to_{right_node[0]}_{right_node[1]}",
                default_depth=fifo_depth,
                obj_type=intermediate_data_dtype,
            ))
            obj_fifo_lookup[(curr_node[0], curr_node[1])]['out'].append(to_right_obj_fifo_id)
            obj_fifo_lookup[(right_node[0], right_node[1])]['in'].append(to_right_obj_fifo_id)
            # To avoid deadlock, we need to initialize some object fifos with data
            if row == 3 and col < num_cols-1:
                obj_fifo_lookup[(right_node[0], right_node[1])]['initial_in'].append(to_right_obj_fifo_id)

    # Input and output object fifos to/from AIE-array
    in_shim_to_mem_obj_fifo_id = len(obj_fifos)
    obj_fifos.append(ObjectFifo(
        name=f"obj_fifo_0_0_to_1_0",
        default_depth=fifo_depth,
        obj_type=intermediate_data_dtype,
    ))
    in_mem_to_core_obj_fifo_id = len(obj_fifos)
    obj_fifos.append(obj_fifos[in_shim_to_mem_obj_fifo_id].cons().forward(name=f"obj_fifo_1_0_to_3_0_forward"))
    obj_fifo_lookup[(3, 0)]['initial_in'].append(in_mem_to_core_obj_fifo_id)
    obj_fifo_lookup[(3, 0)]['in'].append(in_mem_to_core_obj_fifo_id)
    out_core_to_mem_obj_fifo_id = len(obj_fifos)
    obj_fifos.append(ObjectFifo(
        name=f"obj_fifo_2_{num_cols-1}_to_1_{num_cols-1}",
        default_depth=fifo_depth,
        obj_type=intermediate_data_dtype,
    ))
    out_mem_to_shim_obj_fifo_id = len(obj_fifos)
    obj_fifos.append(obj_fifos[out_core_to_mem_obj_fifo_id].cons().forward(name=f"obj_fifo_1_{num_cols-1}_to_0_{num_cols-1}_forward"))
    obj_fifo_lookup[(2, num_cols-1)]['out'].append(out_core_to_mem_obj_fifo_id)

    # Core function declaration
    workers = []
    def core_func_feedback_node(zero_kernel=None, accumulate_kernel=None, num_initial_inFIFO=2, num_inFIFO=2, num_outFIFO=2, *obj_FIFOs):
        out_fifo_start_idx = num_initial_inFIFO + num_inFIFO
        # Initial accumulation to avoid deadlock
        do_kernel(zero_kernel, accumulate_kernel, num_initial_inFIFO, num_outFIFO, *obj_FIFOs[0:num_initial_inFIFO], *obj_FIFOs[out_fifo_start_idx:out_fifo_start_idx+num_outFIFO])
        for _ in range_(sys.maxsize):
            do_kernel(zero_kernel, accumulate_kernel, num_initial_inFIFO + num_inFIFO, num_outFIFO, *obj_FIFOs[0:out_fifo_start_idx], *obj_FIFOs[out_fifo_start_idx:out_fifo_start_idx+num_outFIFO])

    def core_func_intermediate_node(zero_kernel=None, accumulate_kernel=None, num_inFIFO=2, num_outFIFO=2, *obj_FIFOs):
        for _ in range_(sys.maxsize):
            do_kernel(zero_kernel, accumulate_kernel, num_inFIFO, num_outFIFO, *obj_FIFOs[0:num_inFIFO], *obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO])

    for col in range(0,num_cols): 
        for row in range(2,num_rows):
            # Nodes that handles the feedback connections
            if row == 3 or col == 0:
                obj_fifo_not_initial_in = [idx for idx in obj_fifo_lookup[(row, col)]["in"] if idx not in obj_fifo_lookup[(row, col)]["initial_in"]]
                workers.append(
                    Worker(
                        core_func_feedback_node,
                        [
                        zero_i32, accumulate_i32, 
                        len(obj_fifo_lookup[(row, col)]["initial_in"]), len(obj_fifo_not_initial_in), len(obj_fifo_lookup[(row, col)]["out"]), 
                        *[obj_fifos[idx].cons() for idx in obj_fifo_lookup[(row, col)]["initial_in"]],
                        *[obj_fifos[idx].cons() for idx in obj_fifo_not_initial_in],
                        *[obj_fifos[idx].prod() for idx in obj_fifo_lookup[(row, col)]["out"]]
                        ],
                    )
                )
            # Intermediate nodes
            else:
                workers.append(
                    Worker(
                        core_func_intermediate_node,
                        [
                        zero_i32, accumulate_i32, 
                        len(obj_fifo_lookup[(row, col)]["in"]), len(obj_fifo_lookup[(row, col)]["out"]), 
                        *[obj_fifos[idx].cons() for idx in obj_fifo_lookup[(row, col)]["in"]],
                        *[obj_fifos[idx].prod() for idx in obj_fifo_lookup[(row, col)]["out"]]
                        ],
                    )
                )
    random.shuffle(workers)
    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(
        in_data_dtype,
        in_data_dtype,
        out_data_dtype,
    ) as (Input_one, Input_two, Output):
        rt.start(*workers)
        tap = TensorAccessPattern(
            tensor_dims=[1, 1, 1, inout_size], # unused dims are set to 1
            sizes=[1, 1, 1, inout_size],
            offset=0,
            strides=[1, 1, 1, 1] # strides for the tensor, at least 1
        )
        rt.fill(obj_fifos[in_shim_to_mem_obj_fifo_id].prod(), Input_one, tap=tap)
        
        tap = TensorAccessPattern(
            tensor_dims=[1, 1, 1, inout_size], # unused dims are set to 1
            sizes=[1, 1, 1, inout_size],
            offset=0,
            strides=[1, 1, 1, 1] # strides for the tensor, at least 1
        )
        rt.drain(obj_fifos[out_mem_to_shim_obj_fifo_id].cons(), Output, tap=tap, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    placer_function = PLACER_CONVERSION[opts.placer](opts.pnr_args)
    return Program(dev_ty, rt).resolve_program(placer_function)


PLACER_CONVERSION = {
    "sa_placer": lambda args: SAPlacer(args),
    "sequential_placer": lambda _: SequentialPlacer(),
}


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
    argparser.add_argument(
        "-p", 
        "--placer", 
        type=str,
        required=False,
        dest="placer",
        default="sequential_placer",
        choices=PLACER_CONVERSION.keys(),
        help="Placement strategy to use",
    )
    argparser.add_argument(
        "-pnr",
        "--pnr-args",
        type=str,
        required=False,
        dest="pnr_args",
        default="-n 1",
        help="PnR tool arguments (only used when placer is sa_placer)",
    )
    opts = argparser.parse_args()
    main(opts)