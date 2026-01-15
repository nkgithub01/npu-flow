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
    num_iters = opts.iters
    warmup_iters = opts.warmup_iters
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
    
    in_dtype = np.int32
    out_dtype = in_dtype
    in_shape = (16,)
    out_shape = (16,)
    out_size = np.prod(out_shape) * np.dtype(out_dtype).itemsize

    # Initialize data
    in_data = np.ones(in_shape, dtype=in_dtype)
    out_data = np.zeros(out_shape, dtype=out_dtype)

    # -----------------------------------------------------------------------------------
    # Setup reference solution
    # -----------------------------------------------------------------------------------
    
    # ------------------------------------------------------------
    # Set up AIE application
    # ------------------------------------------------------------
    app = xrt_utils.AIE_Application(opts.xclbin, opts.instr, "MLIR_AIE")
    app.register_buffer(3, shape=in_shape, dtype=in_dtype)
    app.register_buffer(4, shape=in_shape, dtype=in_dtype)
    app.register_buffer(5, shape=out_shape, dtype=out_dtype)
    # ------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------
    exec_times = []
    app.buffers[3].write(in_data)
    for i in range(num_iters + warmup_iters):
        start = time.time_ns()
        app.run()
        stop = time.time_ns()
        if i >= warmup_iters:
            exec_times.append(stop - start)
    

    # Timing stats
    print(f"\nTotal NPU time: {sum(exec_times) // 1000} us")
    print(f"Avg NPU time: {sum(exec_times) // len(exec_times) // 1000} us")
    print(f"Min NPU time: {min(exec_times) // 1000} us")
    print(f"Max NPU time: {max(exec_times) // 1000} us")

    out_buf = app.buffers[5].read()

    out_data = out_buf[:out_size].view(out_dtype)
    # ------------------------------------------------------------
    # Verify results
    # ------------------------------------------------------------
    if np.allclose(out_data, 2):
        print("\nPASS!\n")
    else:
        print("\nFAIL!\n")
    exit(0)

if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)