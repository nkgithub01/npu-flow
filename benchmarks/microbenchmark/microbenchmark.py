import json
import argparse
import numpy as np
import random

from aie.iron import LocalBuffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer, SAPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


dtype_map = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
}


def main(opts):
    module = microbenchmark(opts)

    # Print the python-to-mlir conversion to stdout
    print(module)

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
            netlist_info["obj_fifos_data_shape"][net["net_id"]] = data_shape
            netlist_info["obj_fifos"][net["net_id"]] = ObjectFifo(
                name=f"obj_fifo_{net["net_id"]}",
                default_depth=net["depths"] if len(net["depths"]) > 1 else net["depths"][0],
                obj_type=data_ty
            )

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
            if len(link["src_net_ids"]) > 1 and len(link["dst_net_ids"]) == 1:
                netlist_info["fifo_links"]["many2one"][link["dst_net_ids"][0]].extend(link["src_net_ids"])
            elif len(link["src_net_ids"]) == 1 and len(link["dst_net_ids"]) >= 1:
                netlist_info["fifo_links"]["one2many"][link["src_net_ids"][0]].extend(link["dst_net_ids"])
            else:
                exit("Error: Unsupported link configuration (many-to-many)")

        # Link the Object FIFOs
        for link in netlist['links']:
            if len(link['src_net_ids']) == 1 and len(link['dst_net_ids']) == 1:
                netlist_info["obj_fifos"][link["dst_net_ids"][0]] = netlist_info["obj_fifos"][link['src_net_ids'][0]].cons().forward(name=f"obj_fifo_{link['dst_net_ids'][0]}")
            elif len(link['src_net_ids']) == 1 and len(link['dst_net_ids']) > 1:
                obj_fifos = netlist_info["obj_fifos"][link['src_net_ids'][0]].cons().split(
                    offsets=[0]*len(link['dst_net_ids']),
                    names=[f"obj_fifo_{dst_id}" for dst_id in link['dst_net_ids']]
                )
                for idx, net_id in enumerate(link['dst_net_ids']):
                    netlist_info["obj_fifos"][net_id] = obj_fifos[idx]
            elif len(link['src_net_ids']) > 1 and len(link['dst_net_ids']) == 1:
                obj_fifos = netlist_info["obj_fifos"][link['dst_net_ids'][0]].prod().join(
                    offsets=[0]*len(link['src_net_ids']),
                    names=[f"obj_fifo_{src_id}" for src_id in link['src_net_ids']]
                )
                for idx, net_id in enumerate(link['src_net_ids']):
                    netlist_info["obj_fifos"][net_id] = obj_fifos[idx]
            else:
                exit("Error: Unsupported link configuration (many-to-many)")

        in_data_shape = 0
        for input_fifo_id in netlist_info["shim_tile_in_out_fifo_ids"]["input"]:
            in_data_shape += netlist_info["obj_fifos_data_shape"][input_fifo_id][0]
        out_data_shape = 0
        for output_fifo_id in netlist_info["shim_tile_in_out_fifo_ids"]["output"]:
            out_data_shape += netlist_info["obj_fifos_data_shape"][output_fifo_id][0]
        netlist_info["in_data_shape"] = (in_data_shape,)
        netlist_info["out_data_shape"] = (out_data_shape,)

    return netlist_info


def microbenchmark(opts):
    dev = opts.dev
    dtype_str = opts.dtype_str
    netlist_file = opts.input_netlist_file
    dtype = dtype_map[dtype_str]
    if dev == "npu2":
        dev_ty = NPU2()
    else:
        raise AssertionError("Invalid device type: only NPU2 (Strix/Strix Halo/Krackan) is supported")
    
    # parse the netlist file and get the tiles and object_fifos and link the object_fifos
    netlist_info = parse_netlist(dtype_str, netlist_file)
    datatype = np.ndarray[(1,), np.dtype[dtype]]

    zero_i32 = Kernel(
        "zero_int32_t_1", "accumulate.o", [datatype]
    )
    accumulate_i32 = Kernel(
        "accumulate_int32_t_int32_t_1", "accumulate.o", [datatype, np.int32, datatype]
    )

    # Wrap the zeroing and adding functions in a way that they can be used in the Worker
    def core_func(zeroFunc=None, accumulateFunc=None, num_inFIFO=2, num_outFIFO=2, *obj_FIFOs):
        in_items = []
        out_items = []
        for inFIFO in obj_FIFOs[:num_inFIFO]:
            in_items.append(inFIFO.acquire(1))
        for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
            out_items.append(outFIFO.acquire(1))

        for out_item in out_items:
            zeroFunc(out_item)
            for idx, in_item in enumerate(in_items):
                accumulateFunc(in_item, 1 if idx == 0 else 0, out_item)

        for inFIFO in obj_FIFOs[:num_inFIFO]:
            inFIFO.release(1)
        for outFIFO in obj_FIFOs[num_inFIFO:num_inFIFO+num_outFIFO]:
            outFIFO.release(1)

    # Core function declaration
    workers = []
    for tile_id in netlist_info["core_tile_ids"]:
        in_items = []
        out_items = []
        for input_fifo_id in netlist_info["core_tile_input_output_fifo_ids"][tile_id]["input"]:
            in_items.append(netlist_info["obj_fifos"][input_fifo_id].cons())
        for output_fifo_id in netlist_info["core_tile_input_output_fifo_ids"][tile_id]["output"]:
            out_items.append(netlist_info["obj_fifos"][output_fifo_id].prod())

        workers.append(
            Worker(
                core_func,
                [zero_i32, accumulate_i32, len(in_items), len(out_items), *in_items, *out_items],
            )
        )
    random.shuffle(workers)
    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(
        np.ndarray[netlist_info["in_data_shape"], np.dtype[dtype]],
        np.ndarray[netlist_info["in_data_shape"], np.dtype[dtype]],
        np.ndarray[netlist_info["out_data_shape"], np.dtype[dtype]]
    ) as (Input_one, Input_two, Output):
        rt.start(*workers)
        offset = 0
        tensor_size = 0
        whole_tensor_size = netlist_info["in_data_shape"][0]
        for fifo_id in netlist_info["shim_tile_in_out_fifo_ids"]["input"]:
            tensor_size = netlist_info["obj_fifos_data_shape"][fifo_id][0]
            tap = TensorAccessPattern(
                tensor_dims=[1, 1, 1, whole_tensor_size], # unused dims are set to 1
                sizes=[1, 1, 1, tensor_size],
                offset=offset,
                strides=[1, 1, 1, 1] # strides for the tensor, at least 1
            )
            offset += tensor_size
            rt.fill(netlist_info["obj_fifos"][fifo_id].prod(), Input_one, tap=tap)
        
        offset = 0
        tensor_size = 0
        whole_tensor_size = netlist_info["out_data_shape"][0]
        for fifo_id in netlist_info["shim_tile_in_out_fifo_ids"]["output"]:
            tensor_size = netlist_info["obj_fifos_data_shape"][fifo_id][0]
            tap = TensorAccessPattern(
                tensor_dims=[1, 1, 1, whole_tensor_size], # unused dims are set to 1
                sizes=[1, 1, 1, tensor_size],
                offset=offset,
                strides=[1, 1, 1, 1] # strides for the tensor, at least 1
            )
            offset += tensor_size
            rt.drain(netlist_info["obj_fifos"][fifo_id].cons(), Output, tap=tap, wait=True)

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
