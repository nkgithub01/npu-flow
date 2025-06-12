import json
import os
import sys
import time
import numpy as np

from aie.utils.xrt import setup_aie, write_out_trace, execute
import aie.utils.test as test_utils

RELEASE_VERBOSITY_LEVEL = 1
DEBUG_VERBOSITY_LEVEL = 2

dtype_map = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
}


def parse_netlist(dtype_str, netlist_file):
    with open(netlist_file) as json_file:
        netlist = json.load(json_file)
        shim_tile_ids = []
        obj_fifos_data_shape = {}
        shim_tile_in_out_fifo_ids = dict(input=[], output=[])

        # Track SHIM tile ids
        for node in netlist['nodes']:
            if node["type"] == "SHIM":
                shim_tile_ids.append(node["tile_id"])

        # Track object FIFOs from or to shim tiles
        for net in netlist['nets']:
            dtype = dtype_map[dtype_str]
            data_size = np.dtype(dtype).itemsize
            data_shape = (net["byte_size_per_depth"] // data_size,)
            obj_fifos_data_shape[net["net_id"]] = data_shape

            if net["src_tile_id"] in shim_tile_ids:
                shim_tile_in_out_fifo_ids["input"].append(net["net_id"])
            if any(dst_tile_id in shim_tile_ids for dst_tile_id in net["dst_tile_ids"]):
                shim_tile_in_out_fifo_ids["output"].append(net["net_id"])

        in_data_shape = 0
        for input_fifo_id in shim_tile_in_out_fifo_ids["input"]:
            in_data_shape += obj_fifos_data_shape[input_fifo_id][0]
        out_data_shape = 0
        for output_fifo_id in shim_tile_in_out_fifo_ids["output"]:
            out_data_shape += obj_fifos_data_shape[output_fifo_id][0]

    return (in_data_shape,), (out_data_shape,)


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
    total_iters = num_iter + num_warmup_iter
    trace_size = opts.trace_size
    enable_trace = False if not trace_size else True

    npu_time_total = 0

    output_folder = "output/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # -----------------------------------------------------------------------------------
    # Configure the design's buffer size
    # -----------------------------------------------------------------------------------
    in_data_shape, out_data_shape = parse_netlist(opts.dtype_str, opts.input_netlist_file)
    dtype = dtype_map[opts.dtype_str]

    # The test only supports integer.
    dtype_is_int = np.issubdtype(dtype, np.integer)
    assert dtype_is_int, "Input and output data type must be an integer type (i8, i16, i32)."
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max

    shape_in_one = in_data_shape
    shape_in_two = in_data_shape
    shape_out = out_data_shape

    # -----------------------------------------------------------------------------------
    # Generate the input and the reference output
    # -----------------------------------------------------------------------------------
    NPU_input_one = np.zeros(shape_in_one, dtype=dtype)
    NPU_input_two = np.zeros(shape_in_two, dtype=dtype)
    NPU_output_ref = np.zeros(shape_out, dtype=dtype)

    if verbosity >= DEBUG_VERBOSITY_LEVEL:
        print(f"NPU input one: {NPU_input_one.shape}\n{NPU_input_one}")
        print(f"NPU input two: {NPU_input_two.shape}\n{NPU_input_two}")
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
    for i in range(total_iters):
        start = time.time_ns()
        if enable_trace:
            data_buffer, trace_buffer = execute(app=app, input_one=NPU_input_one, input_two=NPU_input_two, enable_trace=enable_trace, trace_after_output=False)
        else:
            data_buffer = execute(app=app, input_one=NPU_input_one, input_two=NPU_input_two, enable_trace=enable_trace, trace_after_output=False)
        stop = time.time_ns()

        if enable_trace and i == num_iter - 1:
            write_out_trace(trace_buffer.view(np.uint32), str(opts.trace_file))

        npu_time = stop - start
        # Warmup iterations are not counted in the average time
        if i >= num_warmup_iter:
            npu_time_total = npu_time_total + npu_time

    if verbosity >= RELEASE_VERBOSITY_LEVEL:  
        print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

    # -----------------------------------------------------------------------------------
    # Compare the AIE output and the golden reference result
    # -----------------------------------------------------------------------------------
    NPU_output = np.array(data_buffer, dtype=dtype)
    if verbosity >= DEBUG_VERBOSITY_LEVEL:
        print(f"NPU output: {NPU_output.shape}\n{NPU_output}")
        np.savetxt(output_folder+"NPU_output_two.txt", NPU_output, fmt="%d")

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