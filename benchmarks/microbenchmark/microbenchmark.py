import json
import argparse
import numpy as np

from aie.iron import LocalBuffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


dtype_map = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
}


def main(opts):
    module = microbenchmark(
        opts.dev,
        opts.dtype_str,
        opts.input_netlist_file
    )

    # Print the python-to-mlir conversion to stdout
    print(module)

def parse_netlist(dtype_str, netlist_file):
    netlist_info = dict(
        tiles={},
        core_tile_ids=[],
        mem_tile_ids=[],
        shim_tile_ids=[],
        obj_fifos={},
        fifo_links={},
        obj_fifos_data_shape={},
        core_tile_producer_consumer_fifo_ids={},
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
                netlist_info["core_tile_producer_consumer_fifo_ids"][node["id"]] = dict(producer=[], consumer=[])
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
            netlist_info["obj_fifos"][net["net_id"]] = ObjectFifo(
                name=f"obj_fifo_{net["net_id"]}",
                default_depth=net["depths"] if len(net["depths"]) > 1 else net["depths"][0],
                obj_type=data_ty
            )
            netlist_info["obj_fifos_data_shape"][net["net_id"]] = data_shape

            # Track producer and consumer object FIFOs for each core tile
            if net["src_id"] in netlist_info["core_tile_ids"]:
                netlist_info["core_tile_producer_consumer_fifo_ids"][net["src_id"]]["producer"].append(net["net_id"])
            for dst_tile_id in net["dst_id"]:
                if dst_tile_id in netlist_info["core_tile_ids"]:
                    netlist_info["core_tile_producer_consumer_fifo_ids"][dst_tile_id]["consumer"].append(net["net_id"])

            # Track object FIFOs from or to shim tiles
            if net["src_id"] in netlist_info["shim_tile_ids"]:
                netlist_info["shim_tile_in_out_fifo_ids"]["input"].append(net["net_id"])
            if any(dst_tile_id in netlist_info["shim_tile_ids"] for dst_tile_id in net["dst_id"]):
                netlist_info["shim_tile_in_out_fifo_ids"]["output"].append(net["net_id"])

            netlist_info["fifo_links"][net["net_id"]] = []

        for net in netlist['nets']:
            if net["need_linking"]:
                netlist_info["fifo_links"][net["link_src_net_id"]].append(net["net_id"])
        
        # Link the Object FIFOs
        for link_src in netlist_info["fifo_links"]:
            for link_dst in netlist_info["fifo_links"][link_src]:
                netlist_info["obj_fifos"][link_dst] = netlist_info["obj_fifos"][link_src].cons().forward(name=f"obj_fifo_{link_dst}")

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
    netlist_file
):
    dtype = dtype_map[dtype_str]
    if dev == "npu2":
        dev_ty = NPU2()
    else:
        raise AssertionError("Invalid device type: only NPU2 (Strix/Strix Halo/Krackan) is supported")
    
    # parse the netlist file and get the tiles and object_fifos and link the object_fifos
    netlist_info = parse_netlist(dtype_str, netlist_file)
    
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

    # Wrap the zeroing and adding functions in a way that they can be used in the Worker
    def core_func(in_items, out_items):
        in_elements = []
        for item in in_items:
            in_elements.append(item.acquire(1))
        out_elements = []
        for item in out_items:
            out_elements.append(item.acquire(1))

        zero_func(out_elements)
        add_func(in_elements, out_elements)

        for item in in_items:
            item.release(1)
        for item in out_items:
            item.release(1)

    # Core function declaration
    workers = []
    for tile_id in netlist_info["core_tile_ids"]:
        in_items = []
        out_items = []
        for consumer_fifo_id in netlist_info["core_tile_producer_consumer_fifo_ids"][tile_id]["consumer"]:
            in_items.append(netlist_info["obj_fifos"][consumer_fifo_id].cons())
        for producer_fifo_id in netlist_info["core_tile_producer_consumer_fifo_ids"][tile_id]["producer"]:
            out_items.append(netlist_info["obj_fifos"][producer_fifo_id].prod())
        
        workers.append(
            Worker(
                core_func,
                [in_items, out_items],
            )
        )
      
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
                sizes=[1, 1, 1, whole_tensor_size],
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
                sizes=[1, 1, 1, whole_tensor_size],
                offset=offset,
                strides=[1, 1, 1, 1] # strides for the tensor, at least 1
            )
            offset += tensor_size
            rt.drain(netlist_info["obj_fifos"][fifo_id].cons(), Output, tap=tap, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev_ty, rt).resolve_program(SequentialPlacer())


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
    opts = argparser.parse_args()
    main(opts)