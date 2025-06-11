
import os
import sys
import time
import numpy as np

from aie.utils.xrt import setup_aie, write_out_trace, execute
import aie.utils.test as test_utils

dtype_map = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
}

RELEASE_VERBOSITY_LEVEL = 1
DEBUG_VERBOSITY_LEVEL = 2

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
    dtype_in = dtype_map[opts.dtype_in_str]
    dtype_out = dtype_map[opts.dtype_out_str]

    # The test only supports integer.
    dtype_is_int = np.issubdtype(dtype_in, np.integer) and np.issubdtype(dtype_out, np.integer)
    assert dtype_is_int, "Input and output data type must be an integer type (i8, i16, i32)."
    dtype_in_min = np.iinfo(dtype_in).min
    dtype_in_max = np.iinfo(dtype_in).max

    shape_in_A = None
    shape_in_B = None
    shape_out_C = None

    # -----------------------------------------------------------------------------------
    # Generate the input and the reference output
    # -----------------------------------------------------------------------------------
    NPU_input_one = None
    NPU_input_two = None
    NPU_output_ref = None

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
        shape_in_A,
        dtype_in,
        shape_in_B,
        dtype_in,
        shape_out_C,
        dtype_out,
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
    NPU_output = np.array(data_buffer, dtype=dtype_out)
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
    p = test_utils.create_default_argparser()
    p.add_argument(
        "-nl",
        "--netlist", 
        type=str, 
        dest="netlist",
        default="netlist.json",
    )
    p.add_argument(
        "--dtype_in", 
        type=str, 
        dest="dtype_in_str",
        choices=["i8", "i16"], 
        default="i16"
    )
    p.add_argument(
        "--dtype_out", 
        type=str, 
        dest="dtype_out_str",
        choices=["i8", "i16", "i32"], 
        default="i16"
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)