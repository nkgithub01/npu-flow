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
        my_benchmark(
            opts.dev,
            opts.dtype_str,
            opts.inout_size
        )

        # Print the python-to-mlir conversion to stdout
        print(ctx.module)

def my_benchmark(
    dev,
    dtype_str,
    inout_size
):
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
        accumulate_i32 = external_func(
            "accumulate_int32_t_int32_t_1", inputs=[intermediate_data_dtype, np.int32, intermediate_data_dtype]
    )
        
        # Tile declarations as tile[row][col]
        tiles = [
            [tile(col, row) for col in range(0, 8)] for row in range(0, 6)
        ]
        connection_order = []
        if opts.placement == "regular":
            connection_order = [dict(row=y, col=x) for x in range(4) for y in (range(2,6) if x%2==0 else range(5,1,-1))]
        elif opts.placement == "1-hop":
            connection_order = [dict(row=y, col=x) for x in range(0,8,2) for y in (range(2,6,2) if (x/2)%2==0 else range(4,1,-2))]
            connection_order += ([dict(row=y, col=x) for x in range(1,8,2) for y in (range(2,6,2) if (x/2)%2==1 else range(4,1,-2))])[::-1]
        elif opts.placement == "2-hop":
            connection_order = [
                dict(row=2, col=0), dict(row=5, col=0), dict(row=5, col=3), dict(row=5, col=6), dict(row=2, col=6), dict(row=2, col=3),
                dict(row=2, col=1), dict(row=5, col=1), dict(row=5, col=4), dict(row=5, col=7), dict(row=2, col=7), dict(row=2, col=4),
                dict(row=2, col=2), dict(row=5, col=2), dict(row=5, col=5), dict(row=2, col=5)
            ]
        elif opts.placement == "random":
            connection_order = [dict(row=y, col=x) for y in range(2,6) for x in range(8)]
            connection_order.remove(dict(row=2,col=0))
            connection_order.remove(dict(row=2,col=7))
            random.shuffle(connection_order)
            connection_order = [dict(row=2,col=0)] + connection_order[:14] + [dict(row=2,col=7)]
        elif "single_node" in opts.placement:
            temp = [dict(row=y, col=x) for x in range(8) for y in (range(2,6) if x%2==0 else range(5,1,-1))]
            distance = int(opts.placement.split("_")[-1])
            if distance < 4:
                src_node = temp[0]
                dst_node = dict(row=src_node["row"], col=src_node["col"]+distance)
                intermediates = [[[x,src_node["row"]] for x in range(distance+1)]]
            else:
                src_node = temp[0]
                dst_node = temp[distance]
                intermediates=[[[x,y] for x in range(8) for y in (range(2,6) if x%2==0 else range(5,1,-1))][:distance+1]]
            route = dict(src_col_x=src_node["col"], src_row_y=src_node["row"], dst_col_x=dst_node["col"], dst_row_y=dst_node["row"], intermediates=intermediates)
            connection_order = [
                src_node,
                dst_node,
            ]

            if 'neighbour' in opts.placement:
                route = dict()
            json.dump(route, open("./custom_route.json", "w"), indent=4)
        else:
            connection_order = [dict(row=y, col=x) for x in range(4) for y in (range(2,6) if x%2==0 else range(5,1,-1))]
        
        # add the shim and mem tiles in the same column at the beginning and the end
        connection_order = [dict(row=0,col=connection_order[0]["col"]), dict(row=1,col=connection_order[0]["col"])] + connection_order + [dict(row=1,col=connection_order[-1]["col"]), dict(row=0,col=connection_order[-1]["col"])]

        num_core_for_expanding = 1
        num_core_for_contracting = num_core_for_expanding
        compute_core_start_idx = 2
        compute_core_end_idx = -2
        expanding_till_node = compute_core_start_idx + num_core_for_expanding
        contracting_from_node = compute_core_end_idx + (-1) * num_core_for_contracting
        expand_rate = opts.expand_rate
        contract_rate = expand_rate
        fifo_depth = 2

        obj_fifo_lookup = dict()
        obj_fifos = []

        for node in connection_order:
            obj_fifo_lookup[(node['row'],node['col'])] = {"in":[], "out":[]}

        # object fifos between tiles
        in_obj_fifo_id = 0
        out_obj_fifo_id = -1
        for cur_idx in range(len(connection_order)-1):
            next_idx = cur_idx + 1
            obj_fifo_id = len(obj_fifos)
            obj_fifos.append(object_fifo(
                f"obj_fifo_{connection_order[cur_idx]['row']}_{connection_order[cur_idx]['col']}_to_{connection_order[next_idx]['row']}_{connection_order[next_idx]['col']}",
                tiles[connection_order[cur_idx]['row']][connection_order[cur_idx]['col']],
                tiles[connection_order[next_idx]['row']][connection_order[next_idx]['col']],
                fifo_depth,
                intermediate_data_dtype,
            ))
            if connection_order[cur_idx]['row'] == 0 and connection_order[next_idx]['row'] == 1:
                in_obj_fifo_id = obj_fifo_id
            elif connection_order[cur_idx]['row'] == 1 and connection_order[next_idx]['row'] == 0:
                out_obj_fifo_id = obj_fifo_id
            
            if connection_order[cur_idx]['row'] > 1:
                obj_fifo_lookup[(connection_order[cur_idx]['row'],connection_order[cur_idx]['col'])]['out'].append(obj_fifo_id)
            if connection_order[next_idx]['row'] > 1:
                obj_fifo_lookup[(connection_order[next_idx]['row'],connection_order[next_idx]['col'])]['in'].append(obj_fifo_id)
        
        # Link input object fifos from shim to first core
        object_fifo_link(obj_fifos[in_obj_fifo_id], obj_fifos[in_obj_fifo_id+1])

        # Link output object fifos from last core to shim
        object_fifo_link(obj_fifos[out_obj_fifo_id-1], obj_fifos[out_obj_fifo_id])
        
        # feedback object fifo from last core to the core at expanding_till_node to prevent pipelining
        feedback_obj_fifo_id = -1
        if opts.enable_feedback:
            feedback_obj_fifo_id = len(obj_fifos)
            obj_fifos.append(object_fifo(
                f"obj_fifo_{connection_order[contracting_from_node-1]['row']}_{connection_order[contracting_from_node-1]['col']}_to_{connection_order[expanding_till_node]['row']}_{connection_order[expanding_till_node]['col']}",
                tiles[connection_order[contracting_from_node-1]['row']][connection_order[contracting_from_node-1]['col']],
                tiles[connection_order[expanding_till_node]['row']][connection_order[expanding_till_node]['col']],
                fifo_depth,
                intermediate_data_dtype,
            ))
            obj_fifo_lookup[(connection_order[contracting_from_node-1]['row'],connection_order[contracting_from_node-1]['col'])]['out'].append(feedback_obj_fifo_id)

        # Core function declaration
        # Expanding phase, each core produces 'expand_rate' number of outputs for each input
        for node in connection_order[compute_core_start_idx:expanding_till_node]:
            row = node["row"]
            col = node["col"]
            @core(tiles[row][col], "accumulate.o")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    in_items = []
                    out_items = []
                    for input_fifo_id in obj_fifo_lookup[(row, col)]["in"]:
                        in_items.append(obj_fifos[input_fifo_id].acquire(ObjectFifoPort.Consume, 1))
                    for _ in range_(expand_rate):
                        for output_fifo_id in obj_fifo_lookup[(row, col)]["out"]:
                            out_items.append(obj_fifos[output_fifo_id].acquire(ObjectFifoPort.Produce, 1))
                        accumulate_i32(in_items[0], 1, out_items[0])
                        for output_fifo_id in obj_fifo_lookup[(row, col)]["out"]:
                            obj_fifos[output_fifo_id].release(ObjectFifoPort.Produce, 1)
                    for input_fifo_id in obj_fifo_lookup[(row, col)]["in"]:
                        obj_fifos[input_fifo_id].release(ObjectFifoPort.Consume, 1)

        # Feedback loop core, this makes sure the downstream cores are not pipelined, each data need to go through the whole chain before the next data can enter
        if expanding_till_node < len(connection_order) + contracting_from_node:
            row = connection_order[expanding_till_node]['row']
            col = connection_order[expanding_till_node]['col']
            @core(tiles[row][col], "accumulate.o")
            def core_body():
                in_items = []
                out_items = []
                for input_fifo_id in obj_fifo_lookup[(row, col)]["in"]:
                    in_items.append(obj_fifos[input_fifo_id].acquire(ObjectFifoPort.Consume, 1))
                for output_fifo_id in obj_fifo_lookup[(row, col)]["out"]:
                    out_items.append(obj_fifos[output_fifo_id].acquire(ObjectFifoPort.Produce, 1))
                accumulate_i32(in_items[0], 1, out_items[0])
                for output_fifo_id in obj_fifo_lookup[(row, col)]["out"]:
                    obj_fifos[output_fifo_id].release(ObjectFifoPort.Produce, 1)
                for input_fifo_id in obj_fifo_lookup[(row, col)]["in"]:
                    obj_fifos[input_fifo_id].release(ObjectFifoPort.Consume, 1)

                for _ in range_(0xFFFFFFFF):
                    if opts.enable_feedback:
                        tmp = obj_fifos[feedback_obj_fifo_id].acquire(ObjectFifoPort.Consume, 1)
                    for input_fifo_id in obj_fifo_lookup[(row, col)]["in"]:
                        in_items.append(obj_fifos[input_fifo_id].acquire(ObjectFifoPort.Consume, 1))
                    for output_fifo_id in obj_fifo_lookup[(row, col)]["out"]:
                        out_items.append(obj_fifos[output_fifo_id].acquire(ObjectFifoPort.Produce, 1))
                    accumulate_i32(in_items[0], 1, out_items[0])
                    for output_fifo_id in obj_fifo_lookup[(row, col)]["out"]:
                        obj_fifos[output_fifo_id].release(ObjectFifoPort.Produce, 1)
                    for input_fifo_id in obj_fifo_lookup[(row, col)]["in"]:
                        obj_fifos[input_fifo_id].release(ObjectFifoPort.Consume, 1)
                    if opts.enable_feedback:
                        obj_fifos[feedback_obj_fifo_id].release(ObjectFifoPort.Consume, 1)

        # Intermediate phase, each core produces 1 output for each input
        for node in connection_order[expanding_till_node+1:contracting_from_node]:
            row = node["row"]
            col = node["col"]
            @core(tiles[row][col], "accumulate.o")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    in_items = []
                    out_items = []
                    for input_fifo_id in obj_fifo_lookup[(row, col)]["in"]:
                        in_items.append(obj_fifos[input_fifo_id].acquire(ObjectFifoPort.Consume, 1))
                    for output_fifo_id in obj_fifo_lookup[(row, col)]["out"]:
                        out_items.append(obj_fifos[output_fifo_id].acquire(ObjectFifoPort.Produce, 1))
                    accumulate_i32(in_items[0], 1, out_items[0])
                    for output_fifo_id in obj_fifo_lookup[(row, col)]["out"]:
                        obj_fifos[output_fifo_id].release(ObjectFifoPort.Produce, 1)
                    for input_fifo_id in obj_fifo_lookup[(row, col)]["in"]:
                        obj_fifos[input_fifo_id].release(ObjectFifoPort.Consume, 1)

        # Contracting phase, each core consumes 'contract_rate' number of inputs for each output 
        for node in connection_order[contracting_from_node:compute_core_end_idx]:
            row = node["row"]
            col = node["col"]
            @core(tiles[row][col], "accumulate.o")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    in_items = []
                    out_items = []
                    for output_fifo_id in obj_fifo_lookup[(row, col)]["out"]:
                        out_items.append(obj_fifos[output_fifo_id].acquire(ObjectFifoPort.Produce, 1))
                    for _ in range_(contract_rate):
                        for input_fifo_id in obj_fifo_lookup[(row, col)]["in"]:
                            in_items.append(obj_fifos[input_fifo_id].acquire(ObjectFifoPort.Consume, 1))
                        accumulate_i32(in_items[0], 1, out_items[0])
                        for input_fifo_id in obj_fifo_lookup[(row, col)]["in"]:
                            obj_fifos[input_fifo_id].release(ObjectFifoPort.Consume, 1)
                    for output_fifo_id in obj_fifo_lookup[(row, col)]["out"]:
                        obj_fifos[output_fifo_id].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(
            in_data_dtype,
            in_data_dtype,
            out_data_dtype,
        )
        def sequence(Input_one, Input_two, Output):
            in_tasks = []
            in_tasks.append(shim_dma_single_bd_task(
                obj_fifos[in_obj_fifo_id], 
                Input_one, 
                offset=0, 
                sizes=[1, 1, 1, inout_size]
            ))
            
            out_tasks = []
            out_tasks.append(shim_dma_single_bd_task(
                obj_fifos[out_obj_fifo_id],
                Output,
                offset=0,
                sizes=[1, 1, 1, inout_size],
                issue_token=True
            ))

            dma_start_task(*in_tasks, *out_tasks)
            dma_await_task(*out_tasks)
            dma_free_task(*in_tasks)


# zeroing the values of the items
def zero_func(items):
    for data in items:
        for i in range_(data.shape[0]):
            data[i] = 0


# Accumulate the input and adding a constant for each element one by one (simple function to prevent optimization)
def add_func(in_items, out_items, increment=1):
    for data_out in out_items:
        for i in range_(data_out.shape[0]):
            data_out[i] = increment
            for data_in in in_items:
                data_out[i] += data_in[i]


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