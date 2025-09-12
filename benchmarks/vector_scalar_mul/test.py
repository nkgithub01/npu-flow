import numpy as np
import sys
import os
import time

import aie.utils.xrt as xrt_utils
import aie.utils.test as test_utils

from execute_aie import execute_aie_multi_with_timing

def main(opts):
    # -----------------------------------------------------------------------------------
    # Program arguments parsing
    # -----------------------------------------------------------------------------------
    in1_size = int(opts.in1_size)  # in bytes
    in2_size = int(opts.in2_size)  # in bytes
    out_size = int(opts.out_size)  # in bytes
    num_iters = opts.iters
    warmup_iters = opts.warmup_iters
    print(
        "\nNumber of iterations:",
        str(opts.iters),
        "(warmup iterations:",
        str(opts.warmup_iters),
        ")",
    )
    trace_file = opts.trace_file
    enable_trace = False if not opts.trace_size else True

    # -----------------------------------------------------------------------------------
    # Configure the design's buffer size
    # -----------------------------------------------------------------------------------

    in1_dtype = np.int16
    in2_dtype = np.int32
    out_dtype = in1_dtype

    in1_volume = in1_size // np.dtype(in1_dtype).itemsize
    in2_volume = in2_size // np.dtype(in2_dtype).itemsize
    out_volume = out_size // np.dtype(out_dtype).itemsize

    # check buffer sizes
    assert in2_size == 4
    assert out_size == in1_size

    scale_factor = 3

    # Initialize data
    in1_data = np.arange(1, in1_volume + 1, dtype=in1_dtype)
    in2_data = np.array([scale_factor], dtype=in2_dtype)
    out_data = np.zeros([out_volume], dtype=out_dtype)

    # -----------------------------------------------------------------------------------
    # Setup reference solution
    # -----------------------------------------------------------------------------------
    ref = np.arange(1, in1_volume + 1, dtype=out_dtype) * scale_factor

    # ------------------------------------------------------------
    # Set up AIE application
    # ------------------------------------------------------------
    app = xrt_utils.setup_aie(
        opts.xclbin,
        opts.instr,
        in1_volume,
        in1_dtype,
        in2_volume,
        in2_dtype,
        out_volume,
        out_dtype,
        enable_trace=enable_trace,
        trace_size=opts.trace_size,
        verbosity=opts.verbosity,
        trace_after_output=False,
    )

    # ------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------
    output_buf = execute_aie_multi_with_timing(
        app,
        input_one=in1_data,
        input_two=in2_data,
        enable_trace=enable_trace,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        trace_file=trace_file
    )
    aie_output = output_buf[:out_size].view(out_dtype)
    
    # ------------------------------------------------------------
    # Verify results
    # ------------------------------------------------------------
    if opts.verify:
        if opts.verbosity >= 1:
            print("Verifying results ...")
    e = np.equal(ref, aie_output)
    errors = np.size(e) - np.count_nonzero(e)

    if errors:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        sys.exit(0)
    else:
        print("\nPASS!\n")
        exit(0)

if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)