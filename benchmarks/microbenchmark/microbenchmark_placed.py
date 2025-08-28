import json
import argparse
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
    with mlir_mod_ctx() as ctx:
        microbenchmark(
            opts.dev,
            opts.dtype_str,
            opts.input_netlist_file,
            opts.trace_size,
        )

        # Print the python-to-mlir conversion to stdout
        print(ctx.module)

def parse_netlist(dtype_str, netlist_file):
    netlist_info = dict(
        tiles={},
        core_tile_ids=[],
        mem_tile_ids=[],
        shim_tile_ids=[],
        obj_fifos={},
        fifo_links=dict(many2one={}, one2many={}),
        obj_fifos_data_shape={},
        core_tile_input_output_fifo_ids={},
        shim_tile_in_out_fifo_ids=dict(input=[], output=[]),
        in_data_shape=(),
        out_data_shape=(),
    )

    with open(netlist_file) as json_file:
        netlist = json.load(json_file)
        
        # Tile(s) declarations
        for node in netlist['nodes']:
            netlist_info["tiles"][node["id"]] = tile(node["col_x"], node["row_y"])
            if node["type"] == "COMP":
                netlist_info["core_tile_ids"].append(node["id"])
                netlist_info["core_tile_input_output_fifo_ids"][node["id"]] = dict(input=[], output=[])
            elif node["type"] == "MEM":
                netlist_info["mem_tile_ids"].append(node["id"])
            elif node["type"] == "SHIM":
                netlist_info["shim_tile_ids"].append(node["id"])

        # Object FIFO(s) declarations
        for net in netlist['nets']:
            dtype = dtype_map[dtype_str]
            data_size = np.dtype(dtype).itemsize
            data_shape = (net["byte_size_per_depth"] // data_size,)
            data_ty = np.ndarray[data_shape, np.dtype[dtype]]
            netlist_info["obj_fifos"][net["net_id"]] = object_fifo(
                f"obj_fifo_{net["net_id"]}",
                netlist_info["tiles"][net["src_id"]],
                [netlist_info["tiles"][idx] for idx in net["dst_id"]],
                net["depths"] if len(net["depths"]) > 1 else net["depths"][0],
                data_ty
            )
            netlist_info["obj_fifos_data_shape"][net["net_id"]] = data_shape

            # Track input and output object FIFOs for each core tile
            if net["src_id"] in netlist_info["core_tile_ids"]:
                netlist_info["core_tile_input_output_fifo_ids"][net["src_id"]]["output"].append(net["net_id"])
            for dst_tile_id in net["dst_id"]:
                if dst_tile_id in netlist_info["core_tile_ids"]:
                    netlist_info["core_tile_input_output_fifo_ids"][dst_tile_id]["input"].append(net["net_id"])

            # Track object FIFOs from or to shim tiles
            if net["src_id"] in netlist_info["shim_tile_ids"]:
                netlist_info["shim_tile_in_out_fifo_ids"]["input"].append(net["net_id"])
            if any(dst_tile_id in netlist_info["shim_tile_ids"] for dst_tile_id in net["dst_id"]):
                netlist_info["shim_tile_in_out_fifo_ids"]["output"].append(net["net_id"])

        # Init FIFO links data structure
        for net in netlist['nets']:
            netlist_info["fifo_links"]["many2one"][net["net_id"]] = []
            netlist_info["fifo_links"]["one2many"][net["net_id"]] = []

        for link in netlist['links']:
            if len(link["src_net_ids"]) > 1:
                netlist_info["fifo_links"]["many2one"][link["dst_net_ids"][0]].extend(link["src_net_ids"])
            else:
                netlist_info["fifo_links"]["one2many"][link["src_net_ids"][0]].extend(link["dst_net_ids"])

        # Link the Object FIFOs
        for link_src in netlist_info["fifo_links"]["one2many"]:
            if len(netlist_info["fifo_links"]["one2many"][link_src]) != 0:
                object_fifo_link(
                    netlist_info["obj_fifos"][link_src],
                    [netlist_info["obj_fifos"][dst] for dst in netlist_info["fifo_links"]["one2many"][link_src]]
                )
        for link_dst in netlist_info["fifo_links"]["many2one"]:
            if len(netlist_info["fifo_links"]["many2one"][link_dst]) != 0:
                object_fifo_link(
                    [netlist_info["obj_fifos"][src] for src in netlist_info["fifo_links"]["many2one"][link_dst]],
                    netlist_info["obj_fifos"][link_dst],
                    [0 for _ in netlist_info["fifo_links"]["many2one"][link_dst]],  # all srcs are one-to-one
                )

        in_data_shape = 0
        for input_fifo_id in netlist_info["shim_tile_in_out_fifo_ids"]["input"]:
            in_data_shape += netlist_info["obj_fifos_data_shape"][input_fifo_id][0]
        out_data_shape = 0
        for output_fifo_id in netlist_info["shim_tile_in_out_fifo_ids"]["output"]:
            out_data_shape += netlist_info["obj_fifos_data_shape"][output_fifo_id][0]
        netlist_info["in_data_shape"] = (in_data_shape,)
        netlist_info["out_data_shape"] = (out_data_shape,)

    return netlist_info


def microbenchmark(
    dev,
    dtype_str,
    netlist_file,
    trace_size
):
    dtype = dtype_map[dtype_str]
    if dev == "npu2":
        dev_ty = AIEDevice.npu2
    else:
        raise AssertionError("Invalid device type: only NPU2 (Strix/Strix Halo/Krackan) is supported")
    
    @device(dev_ty)
    def device_body():
        # parse the netlist file and get the tiles and object_fifos and link the object_fifos
        netlist_info = parse_netlist(dtype_str, netlist_file)

        # Core function declaration
        for tile_id in netlist_info["core_tile_ids"]:
            @core(netlist_info["tiles"][tile_id])
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    in_items = []
                    out_items = []
                    for input_fifo_id in netlist_info["core_tile_input_output_fifo_ids"][tile_id]["input"]:
                        in_items.append(netlist_info["obj_fifos"][input_fifo_id].acquire(ObjectFifoPort.Consume, 1))
                    for output_fifo_id in netlist_info["core_tile_input_output_fifo_ids"][tile_id]["output"]:
                        out_items.append(netlist_info["obj_fifos"][output_fifo_id].acquire(ObjectFifoPort.Produce, 1))
                    
                    zero_func(out_items)
                    add_func(in_items, out_items)

                    for output_fifo_id in netlist_info["core_tile_input_output_fifo_ids"][tile_id]["output"]:
                        netlist_info["obj_fifos"][output_fifo_id].release(ObjectFifoPort.Produce, 1)
                    for input_fifo_id in netlist_info["core_tile_input_output_fifo_ids"][tile_id]["input"]:
                        netlist_info["obj_fifos"][input_fifo_id].release(ObjectFifoPort.Consume, 1)
        
        # Set up a packet-switched flow from core/mem to shim for tracing information
        # Max can only trace 31 tiles
        tile_to_route_trace = netlist_info["tiles"][netlist_info["shim_tile_ids"][-1]]
        tiles_to_trace = [netlist_info["tiles"][tile_id] for tile_id in netlist_info["shim_tile_ids"]]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, tile_to_route_trace)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[netlist_info["in_data_shape"], np.dtype[dtype]],
            np.ndarray[netlist_info["in_data_shape"], np.dtype[dtype]],
            np.ndarray[netlist_info["out_data_shape"], np.dtype[dtype]],
        )
        def sequence(Input_one, Input_two, Output):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=tile_to_route_trace,
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
            offset = 0
            tensor_size = 0
            for fifo_id in netlist_info["shim_tile_in_out_fifo_ids"]["input"]:
                tensor_size = netlist_info["obj_fifos_data_shape"][fifo_id][0]
                in_tasks.append(shim_dma_single_bd_task(
                    netlist_info["obj_fifos"][fifo_id], 
                    Input_one, 
                    offset=offset, 
                    sizes=[1, 1, 1, tensor_size]
                ))
                offset += tensor_size
            
            out_tasks = []
            offset = 0
            tensor_size = 0
            for fifo_id in netlist_info["shim_tile_in_out_fifo_ids"]["output"]:
                tensor_size = netlist_info["obj_fifos_data_shape"][fifo_id][0]
                out_tasks.append(shim_dma_single_bd_task(
                    netlist_info["obj_fifos"][fifo_id],
                    Output,
                    offset=offset,
                    sizes=[1, 1, 1, tensor_size],
                    issue_token=True
                ))
                offset += tensor_size

            dma_start_task(*in_tasks, *out_tasks)
            dma_await_task(*out_tasks)
            dma_free_task(*in_tasks)

            trace_utils.gen_trace_done_aie2(tile_to_route_trace)


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
        "-nl",
        "--netlist", 
        type=str, 
        dest="input_netlist_file",
        default="AIE_data_flow_netlist.json",
    )
    argparser.add_argument(
        "--dtype", 
        type=str, 
        dest="dtype_str",
        choices=["i8", "i16", "i32"], 
        default="i32"
    )
    argparser.add_argument(
        "--trace_size", 
        type=int, 
        default=0
    )
    opts = argparser.parse_args()
    main(opts)