import json
import os
import sys
import time
import numpy as np

from aie.utils.xrt import setup_aie, write_out_trace, execute
import aie.utils.test as test_utils
from execute_aie import execute_aie_multi_with_timing

RELEASE_VERBOSITY_LEVEL = 1
DEBUG_VERBOSITY_LEVEL = 2
DETAILED_DEBUG_VERBOSITY_LEVEL = 3

dtype_map = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
}

def topologicalSortUtil(net_id, netlist_info, stack):
    # Mark the current net as visited
    netlist_info["netlist"][net_id]["visited"] = True

    # Recur for all downstream neighbor nets
    for id in netlist_info["netlist"][net_id]["downstream_neighbor_net_ids"]:
        if not netlist_info["netlist"][id]["visited"]:
            topologicalSortUtil(id, netlist_info, stack)

    # Push current vertex to stack which stores the result
    stack.append(net_id)

def parse_netlist(dtype_str, netlist_file):
    netlist_info = dict(
        tiles={},
        core_tile_ids=[],
        mem_tile_ids=[],
        shim_tile_ids=[],
        netlist={},
        obj_fifos_data_shape={},
        core_tile_input_output_fifo_ids={},
        shim_tile_in_out_fifo_ids=dict(input=[], output=[]),
        in_data_shape=(),
        out_data_shape=(),
        topological_order_of_net_ids=[],
    )

    with open(netlist_file) as json_file:
        netlist = json.load(json_file)
        
        # Tile(s) declarations
        for node in netlist['nodes']:
            netlist_info["tiles"][node["id"]] = None
            netlist_info["core_tile_input_output_fifo_ids"][node["id"]] = dict(input=[], output=[])
            if node["type"] == "COMP":
                netlist_info["core_tile_ids"].append(node["id"])
            elif node["type"] == "MEM":
                netlist_info["mem_tile_ids"].append(node["id"])
            elif node["type"] == "SHIM":
                netlist_info["shim_tile_ids"].append(node["id"])

        # Object FIFO(s) declarations
        for net in netlist['nets']:
            dtype = dtype_map[dtype_str]
            data_size = np.dtype(dtype).itemsize
            data_shape = (net["byte_size_per_depth"] // data_size,)
            netlist_info["obj_fifos_data_shape"][net["net_id"]] = data_shape
            netlist_info["netlist"][net["net_id"]] = dict(
                src_tile_id=net["src_id"],
                dst_tile_ids=net["dst_id"],
                carried_value = None,
                need_linking=False,
                carry_value_from_net_id=None,
                visited=False,
                downstream_neighbor_net_ids=[],
            )

            # Track input and output object FIFOs for each tile
            netlist_info["core_tile_input_output_fifo_ids"][net["src_id"]]["output"].append(net["net_id"])
            for dst_tile_id in net["dst_id"]:
                netlist_info["core_tile_input_output_fifo_ids"][dst_tile_id]["input"].append(net["net_id"])

            # Track object FIFOs from or to shim tiles
            if net["src_id"] in netlist_info["shim_tile_ids"]:
                netlist_info["shim_tile_in_out_fifo_ids"]["input"].append(net["net_id"])
            if any(dst_tile_id in netlist_info["shim_tile_ids"] for dst_tile_id in net["dst_id"]):
                netlist_info["shim_tile_in_out_fifo_ids"]["output"].append(net["net_id"])

        # tracking net linking information
        for link in netlist['links']:
            for dst_net_id in link["dst_net_ids"]:
                netlist_info["netlist"][dst_net_id]["need_linking"] = True
                netlist_info["netlist"][dst_net_id]["carry_value_from_net_id"] = link["src_net_ids"][-1]

        # tracking downstream neighbor net IDs
        for net in netlist['nets']:
            for dst_tile_id in net["dst_id"]:
                netlist_info["netlist"][net["net_id"]]["downstream_neighbor_net_ids"].extend(
                    netlist_info["core_tile_input_output_fifo_ids"][dst_tile_id]["output"]
                )

        # Topological order of net IDs
        for net_id in netlist_info["shim_tile_in_out_fifo_ids"]["input"]:
            topologicalSortUtil(net_id, netlist_info, netlist_info["topological_order_of_net_ids"])  
        netlist_info["topological_order_of_net_ids"].reverse()

        in_data_shape = 0
        for input_fifo_id in netlist_info["shim_tile_in_out_fifo_ids"]["input"]:
            in_data_shape += netlist_info["obj_fifos_data_shape"][input_fifo_id][0]
        out_data_shape = 0
        for output_fifo_id in netlist_info["shim_tile_in_out_fifo_ids"]["output"]:
            out_data_shape += netlist_info["obj_fifos_data_shape"][output_fifo_id][0]
        netlist_info["in_data_shape"] = (in_data_shape,)
        netlist_info["out_data_shape"] = (out_data_shape,)

    return netlist_info

def main(opts):
    # -----------------------------------------------------------------------------------
    # Program arguments parsing
    # -----------------------------------------------------------------------------------
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    verbosity = opts.verbosity
    do_verify = opts.verify
    num_iter = opts.iters
    num_warmup_iter = opts.warmup_iters
    trace_size = opts.trace_size
    enable_trace = False if not trace_size else True

    npu_time_total = 0

    output_folder = "output/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # -----------------------------------------------------------------------------------
    # Configure the design's buffer size
    # -----------------------------------------------------------------------------------
    netlist_info = parse_netlist(opts.dtype_str, opts.input_netlist_file)
    dtype = dtype_map[opts.dtype_str]

    # The test only supports integer.
    dtype_is_int = np.issubdtype(dtype, np.integer)
    assert dtype_is_int, "Input and output data type must be an integer type (i8, i16, i32)."
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max

    shape_in_one = netlist_info["in_data_shape"]
    shape_in_two = netlist_info["in_data_shape"]
    shape_out = netlist_info["out_data_shape"]

    # -----------------------------------------------------------------------------------
    # Generate the input and the reference output
    # -----------------------------------------------------------------------------------
    NPU_input_one = np.zeros(shape_in_one, dtype=dtype)
    NPU_input_two = np.zeros(shape_in_two, dtype=dtype)
    NPU_output_ref = np.zeros(shape_out, dtype=dtype)

    obj_fifo_shape = netlist_info["obj_fifos_data_shape"][netlist_info["shim_tile_in_out_fifo_ids"]["input"][0]]
    for tile_id in netlist_info["core_tile_ids"]:
        netlist_info["tiles"][tile_id] = np.ones(obj_fifo_shape, dtype=dtype)
    for tile_id in netlist_info["mem_tile_ids"]:
        netlist_info["tiles"][tile_id] = np.zeros(obj_fifo_shape, dtype=dtype)
    for tile_id in netlist_info["shim_tile_ids"]:
        netlist_info["tiles"][tile_id] = np.zeros(obj_fifo_shape, dtype=dtype)
    for net_id in netlist_info["shim_tile_in_out_fifo_ids"]["input"]:
        netlist_info["tiles"][netlist_info["netlist"][net_id]["src_tile_id"]] = np.zeros(obj_fifo_shape, dtype=dtype)
    
    for net_id in netlist_info["topological_order_of_net_ids"]:
        if netlist_info["netlist"][net_id]["need_linking"]:
            # If the net needs linking, carry the value from the linked net
            netlist_info["netlist"][net_id]["carried_value"] = netlist_info["netlist"][netlist_info["netlist"][net_id]["carry_value_from_net_id"]]["carried_value"]
        else:
            # If the net does not need linking, carry the value from the source tile
            netlist_info["netlist"][net_id]["carried_value"] = netlist_info["tiles"][netlist_info["netlist"][net_id]["src_tile_id"]]
        
        if verbosity >= DETAILED_DEBUG_VERBOSITY_LEVEL:
            print(f"Reference solution net ID: {net_id}, Carried value: {netlist_info['netlist'][net_id]['carried_value']}")
        
        for tile_id in netlist_info["netlist"][net_id]["dst_tile_ids"]:
            if tile_id in netlist_info["core_tile_ids"]:
                netlist_info["tiles"][tile_id] += netlist_info["netlist"][net_id]["carried_value"]
    
    for idx, net_id in enumerate(netlist_info["shim_tile_in_out_fifo_ids"]["output"]):
        NPU_output_ref[idx] = netlist_info["netlist"][netlist_info["shim_tile_in_out_fifo_ids"]["output"][idx]]["carried_value"][0]

    if verbosity >= DEBUG_VERBOSITY_LEVEL:
        print(f"NPU input one (Shape: {NPU_input_one.shape}):\n{NPU_input_one}")
        print(f"NPU input two (Shape: {NPU_input_two.shape}):\n{NPU_input_two}")
        print(f"NPU reference output (Shape: {NPU_output_ref.shape}):\n{NPU_output_ref}")
        np.savetxt(output_folder+"NPU_input_one.txt", NPU_input_one, fmt="%d")
        np.savetxt(output_folder+"NPU_input_two.txt", NPU_input_two, fmt="%d")
        np.savetxt(output_folder+"NPU_output_reference.txt", NPU_output_ref, fmt="%d")

    # -----------------------------------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # -----------------------------------------------------------------------------------
    app = setup_aie(
        xclbin_path,
        insts_path,
        shape_in_one,
        dtype,
        shape_in_two,
        dtype,
        shape_out,
        dtype,
        enable_trace=enable_trace,
        trace_size=trace_size,
        trace_after_output=False,
    )

    # -----------------------------------------------------------------------------------
    # Main run loop
    # -----------------------------------------------------------------------------------
    data_buffer = execute_aie_multi_with_timing(
        app,
        input_one=NPU_input_one,
        input_two=NPU_input_two,
        enable_trace=enable_trace,
        num_iters=num_iter,
        warmup_iters=num_warmup_iter,
        trace_file=opts.trace_file
    )

    # -----------------------------------------------------------------------------------
    # Compare the AIE output and the golden reference result
    # -----------------------------------------------------------------------------------
    NPU_output = np.array(data_buffer, dtype=dtype)
    if verbosity >= DEBUG_VERBOSITY_LEVEL:
        print(f"NPU output (Shape: {NPU_output.shape}):\n{NPU_output}")
        np.savetxt(output_folder+"NPU_output.txt", NPU_output, fmt="%d")

    relative_tolerance = 0
    absolute_tolerance = 0

    if do_verify:
        are_close = np.allclose(NPU_output, NPU_output_ref, rtol=relative_tolerance, atol=absolute_tolerance)

        if are_close:
            print("\nPASS!\n")
        else:
            print("\nFailed.")
            if verbosity >= DEBUG_VERBOSITY_LEVEL:
                if NPU_output.shape != NPU_output_ref.shape:
                    print(f"Output shape mismatch: AIE={NPU_output.shape}, Ref={NPU_output_ref.shape}")
                else:
                    with np.nditer(NPU_output, flags=['multi_index']) as it:
                        for x in it:
                            if not np.isclose(NPU_output[it.multi_index], NPU_output_ref[it.multi_index], rtol=relative_tolerance, atol=absolute_tolerance):
                                print(f"First mismatch at ({it.multi_index}): AIE={NPU_output[it.multi_index]}, Ref={NPU_output_ref[it.multi_index]}")
                print(f"\nRelative tolerance: {relative_tolerance}, Absolute tolerance: {absolute_tolerance}\n")
            exit(-1)
    else:
        print("\nVerification skipped, assuming PASS.\n")


if __name__ == "__main__":
    argparser = test_utils.create_default_argparser()
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
    opts = argparser.parse_args(sys.argv[1:])
    main(opts)