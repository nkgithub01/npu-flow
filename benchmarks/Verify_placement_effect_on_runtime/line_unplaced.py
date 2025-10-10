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
    random.seed(opts.placement_seed)
    module = my_benchmark(opts)
    # Print the python-to-mlir conversion to stdout
    print(module)

def my_benchmark(opts):
    dev = opts.dev
    dtype_str = opts.dtype_str
    inout_size = opts.inout_size
    dtype = dtype_map[dtype_str]
    if dev == "npu2":
        dev_ty = NPU2()
    else:
        raise AssertionError("Invalid device type: only NPU2 (Strix/Strix Halo/Krackan) is supported")

    if opts.enable_feedback and opts.length < 4:
        raise AssertionError("Invalid length: length should be at least 4 when feedback is enabled")
    elif not opts.enable_feedback and opts.length < 2:
        raise AssertionError("Invalid length: length should be at least 2 when feedback is disabled")
    elif opts.length > 32:
        raise AssertionError("Invalid length: length should be at most 32")

    # parse the netlist file and get the tiles and object_fifos and link the object_fifos
    in_data_dtype = np.ndarray[(inout_size,), np.dtype[dtype]]
    out_data_dtype = np.ndarray[(inout_size,), np.dtype[dtype]]
    intermediate_data_dtype = np.ndarray[(inout_size,), np.dtype[dtype]]

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

    def core_func_expand(expand_rate=2, zeroFunc=None, accumulateFunc=None, num_inFIFO=2, num_outFIFO=2, *obj_FIFOs):
        for _ in range_(sys.maxsize):
            in_items = []
            out_items = []
            for inFIFO in obj_FIFOs[:num_inFIFO]:
                in_items.append(inFIFO.acquire(1))
            for _ in range_(expand_rate):
                for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
                    out_items.append(outFIFO.acquire(1))
                accumulate(in_items, 1, out_items, zeroFunc, accumulateFunc)
                for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
                    outFIFO.release(1)
                out_items = []
            for inFIFO in obj_FIFOs[:num_inFIFO]:
                inFIFO.release(1)

    def core_func_feedback(feedback_FIFO_idx=0, zeroFunc=None, accumulateFunc=None, num_inFIFO=2, num_outFIFO=2, *obj_FIFOs):
        # initial run to fill the feedback FIFO
        in_items = []
        out_items = []
        for idx, inFIFO in enumerate(obj_FIFOs[:num_inFIFO]):
            if idx == feedback_FIFO_idx:
                continue
            else:
                in_items.append(inFIFO.acquire(1))
        for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
            out_items.append(outFIFO.acquire(1))
        accumulate(in_items, 1, out_items, zeroFunc, accumulateFunc)
        for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
            outFIFO.release(1)
        for idx, inFIFO in enumerate(obj_FIFOs[:num_inFIFO]):
            if idx == feedback_FIFO_idx:
                continue
            else:
                inFIFO.release(1)
        core_func_intermediate(zeroFunc, accumulateFunc, num_inFIFO, num_outFIFO, *obj_FIFOs)

    def core_func_intermediate(zeroFunc=None, accumulateFunc=None, num_inFIFO=2, num_outFIFO=2, *obj_FIFOs):
        for _ in range_(sys.maxsize):
            in_items = []
            out_items = []
            for inFIFO in obj_FIFOs[:num_inFIFO]:
                in_items.append(inFIFO.acquire(1))
            for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
                out_items.append(outFIFO.acquire(1))
            accumulate(in_items, 1, out_items, zeroFunc, accumulateFunc)
            for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
                outFIFO.release(1)
            for inFIFO in obj_FIFOs[:num_inFIFO]:
                inFIFO.release(1)

    def core_func_contract(contract_rate=2, zeroFunc=None, accumulateFunc=None, num_inFIFO=2, num_outFIFO=2, *obj_FIFOs):
        for _ in range_(sys.maxsize):
            in_items = []
            out_items = []
            for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
                out_items.append(outFIFO.acquire(1))
            for _ in range_(contract_rate):
                for inFIFO in obj_FIFOs[:num_inFIFO]:
                    in_items.append(inFIFO.acquire(1))
                accumulate(in_items, 1, out_items, zeroFunc, accumulateFunc)
                for inFIFO in obj_FIFOs[:num_inFIFO]:
                    inFIFO.release(1)
                in_items = []
            for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
                outFIFO.release(1)

    num_core_for_expanding = 1
    num_core_for_contracting = num_core_for_expanding
    compute_core_start_idx = 2
    compute_core_end_idx = -2
    expanding_till_node = compute_core_start_idx + num_core_for_expanding
    contracting_from_node = compute_core_end_idx + (-1) * num_core_for_contracting
    expand_rate = opts.expand_rate
    contract_rate = expand_rate
    fifo_depth = 2

    connection_order = list(range(opts.length+4)) # including shim and mem tiles
    obj_fifo_lookup = dict()
    obj_fifos = []

    for node_idx in connection_order:
        obj_fifo_lookup[node_idx] = {"in":[], "out":[]}

    # object fifos between tiles
    in_shim_to_mem_obj_fifo_id = 0
    in_mem_to_core_obj_fifo_id = 1
    out_core_to_mem_obj_fifo_id = -2
    out_mem_to_shim_obj_fifo_id = -1
    for cur_idx in range(len(connection_order)-1):
        next_idx = cur_idx + 1
        obj_fifo_id = len(obj_fifos)
        obj_fifos.append(ObjectFifo(
            name=f"obj_fifo_idx_{cur_idx}_to_idx_{next_idx}",
            default_depth=fifo_depth,
            obj_type=intermediate_data_dtype,
        ))
        if cur_idx == 0:
            in_shim_to_mem_obj_fifo_id = obj_fifo_id
        elif cur_idx == 1:
            in_mem_to_core_obj_fifo_id = obj_fifo_id
        elif cur_idx == len(connection_order)-2:
            out_mem_to_shim_obj_fifo_id = obj_fifo_id
        else:
            obj_fifo_lookup[cur_idx]['out'].append(obj_fifo_id)

        if next_idx == 1:
            in_shim_to_mem_obj_fifo_id = obj_fifo_id
        elif next_idx == len(connection_order)-2:
            out_core_to_mem_obj_fifo_id = obj_fifo_id
        elif next_idx == len(connection_order)-1:
            out_mem_to_shim_obj_fifo_id = obj_fifo_id
        else:
            obj_fifo_lookup[next_idx]['in'].append(obj_fifo_id)
    
    # Link input object fifos from shim to first core
    obj_fifos[in_mem_to_core_obj_fifo_id] = obj_fifos[in_shim_to_mem_obj_fifo_id].cons().forward(name=obj_fifos[in_mem_to_core_obj_fifo_id].name+"_forward")

    # Link output object fifos from last core to shim
    obj_fifos[out_mem_to_shim_obj_fifo_id] = obj_fifos[out_core_to_mem_obj_fifo_id].cons().forward(name=obj_fifos[out_mem_to_shim_obj_fifo_id].name+"_forward")

    # feedback object fifo from last core to the core at expanding_till_node to prevent pipelining
    feedback_obj_fifo_id = -1
    if opts.enable_feedback:
        feedback_obj_fifo_id = len(obj_fifos)
        obj_fifos.append(ObjectFifo(
            name=f"obj_fifo_idx_{connection_order[contracting_from_node-1]}_to_idx_{connection_order[expanding_till_node]}_feedback",
            default_depth=fifo_depth,
            obj_type=intermediate_data_dtype,
        ))
        obj_fifo_lookup[connection_order[contracting_from_node-1]]['out'].append(feedback_obj_fifo_id)

    # Core function declaration
    workers = []
    # Expanding phase, each core produces 'expand_rate' number of outputs for each input
    for node_idx in connection_order[compute_core_start_idx:expanding_till_node]:
        in_items = []
        out_items = []
        for input_fifo_id in obj_fifo_lookup[node_idx]["in"]:
            in_items.append(obj_fifos[input_fifo_id].cons())
        for output_fifo_id in obj_fifo_lookup[node_idx]["out"]:
            out_items.append(obj_fifos[output_fifo_id].prod())

        workers.append(
            Worker(
                core_func_expand,
                [expand_rate, zero_i32, accumulate_i32, len(in_items), len(out_items), *in_items, *out_items],
            )
        )

    # Feedback loop core, this makes sure the downstream cores are not pipelined, each data need to go through the whole chain before the next data can enter
    if expanding_till_node < len(connection_order) + contracting_from_node:
        in_items = []
        out_items = []
        if opts.enable_feedback:
            in_items.append(obj_fifos[feedback_obj_fifo_id].cons())
        for input_fifo_id in obj_fifo_lookup[expanding_till_node]["in"]:
            in_items.append(obj_fifos[input_fifo_id].cons())
        for output_fifo_id in obj_fifo_lookup[expanding_till_node]["out"]:
            out_items.append(obj_fifos[output_fifo_id].prod())

        if opts.enable_feedback:
            workers.append(
                Worker(
                    core_func_feedback,
                    [0, zero_i32, accumulate_i32, len(in_items), len(out_items), *in_items, *out_items],
                )
            )
        else:
            workers.append(
                Worker(
                    core_func_intermediate,
                    [zero_i32, accumulate_i32, len(in_items), len(out_items), *in_items, *out_items],
                )
            )

    # Intermediate phase, each core produces 1 output for each input
    for node_idx in connection_order[expanding_till_node+1:contracting_from_node]:
        in_items = []
        out_items = []
        for input_fifo_id in obj_fifo_lookup[node_idx]["in"]:
            in_items.append(obj_fifos[input_fifo_id].cons())
        for output_fifo_id in obj_fifo_lookup[node_idx]["out"]:
            out_items.append(obj_fifos[output_fifo_id].prod())

        workers.append(
            Worker(
                core_func_intermediate,
                [zero_i32, accumulate_i32, len(in_items), len(out_items), *in_items, *out_items],
            )
        )

    # Contracting phase, each core consumes 'contract_rate' number of inputs for each output 
    for node_idx in connection_order[contracting_from_node:compute_core_end_idx]:
        in_items = []
        out_items = []
        for input_fifo_id in obj_fifo_lookup[node_idx]["in"]:
            in_items.append(obj_fifos[input_fifo_id].cons())
        for output_fifo_id in obj_fifo_lookup[node_idx]["out"]:
            out_items.append(obj_fifos[output_fifo_id].prod())

        workers.append(
            Worker(
                core_func_contract,
                [contract_rate, zero_i32, accumulate_i32, len(in_items), len(out_items), *in_items, *out_items],
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