import numpy as np
import sys
import os
import time
import aie.utils.xrt as xrt_utils
import aie.utils.test as test_utils

def main(opts):
    # -----------------------------------------------------------------------------------
    # Program arguments parsing
    # -----------------------------------------------------------------------------------
    in1_size = int(opts.in1_size)  # in bytes
    in2_size = int(opts.in2_size)  # in bytes
    out_size = int(opts.out_size)  # in bytes
    num_iter = opts.iters + opts.warmup_iters
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    print(
        "\nNumber of iterations:",
        str(opts.iters),
        "(warmup iterations:",
        str(opts.warmup_iters),
        ")",
    )
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
    enable_trace = False if not opts.trace_size else True

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
    for i in range(num_iter):
        if opts.verbosity >= 1:
            print("Running Kernel.")
            
        start = time.time_ns()
        if enable_trace:
            output_buf, trace_buf = xrt_utils.execute(
                app, in1_data, in2_data, enable_trace, False
            )
        else:
            output_buf = xrt_utils.execute(app, in1_data, in2_data, enable_trace, False)
            trace_buf = None
        stop = time.time_ns()
        
        if i < opts.warmup_iters:
            continue

        aie_output = output_buf[:out_size].view(out_dtype)
        # Save trace for final iteration
        if enable_trace and i == num_iter - 1:
            xrt_utils.write_out_trace(trace_buf.view(np.uint32), str(opts.trace_file))
        # Calculate runtime for this iter
        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time
        npu_time_min = min(npu_time_min, npu_time)
        npu_time_max = max(npu_time_max, npu_time)

        # Compare results and verify they are correct
        if opts.verify:
            if opts.verbosity >= 1:
                print("Verifying results ...")
        e = np.equal(ref, aie_output)
        errors = np.size(e) - np.count_nonzero(e)

        if errors:
            print("\nError count: ", errors)
            print("\nFailed.\n")
            sys.exit(1)
    
    print("\nPASS!\n")
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / opts.iters) / 1000)))
    print("\nMin NPU time: {}us.".format(int((npu_time_min) / 1000)))
    print("\nMax NPU time: {}us.".format(int((npu_time_max) / 1000)))
            
    sys.exit(0)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)