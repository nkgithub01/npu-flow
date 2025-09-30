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
    dtype = dtype_map[opts.dtype_str]

    # The test only supports integer.
    dtype_is_int = np.issubdtype(dtype, np.integer)
    assert dtype_is_int, "Input and output data type must be an integer type (i8, i16, i32)."
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max

    shape_in_one = (opts.inout_size,)
    shape_in_two = (opts.inout_size,)
    shape_out = (opts.inout_size,)

    # -----------------------------------------------------------------------------------
    # Generate the input and the reference output
    # -----------------------------------------------------------------------------------
    NPU_input_one = np.zeros(shape_in_one, dtype=dtype)
    NPU_input_two = np.zeros(shape_in_two, dtype=dtype)
    NPU_output_ref = np.zeros(shape_out, dtype=dtype)

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
            exit(0)
    else:
        print("\nVerification skipped, assuming PASS.\n")


if __name__ == "__main__":
    argparser = test_utils.create_default_argparser()
    argparser.add_argument(
        "--dtype", 
        type=str, 
        dest="dtype_str",
        choices=["i8", "i16", "i32"], 
        default="i32"
    )
    argparser.add_argument(
        "--inout_size", 
        type=int, 
        dest="inout_size",
        default=1
    )
    opts = argparser.parse_args(sys.argv[1:])
    main(opts)