import time
import numpy as np

import aie.utils.xrt as xrt_utils

def execute_aie_multi_with_timing(
    app: xrt_utils.AIE_Application,
    input_one=None,
    input_two=None,
    enable_trace=False,
    num_iters=1,
    warmup_iters=0,
    trace_file=None,
):
    """
    Run AIE app() with multi-iteration and warmup, prints timing stats. 
    Optionally writes trace to file.

    Args:
        app (AIE_Application): Initialized AIE app from xrt_utils.
        input_one (ndarray, optional): Input Data.
        input_two (ndarray, optional): Second input data.
        enable_trace (bool): Whether to enable tracing.
        num_iters (int): Number of iterations.
        warmup_iters (int): Number of warm-up runs (not timed).
        trace_file (str): Output file to write trace.

    Returns:
        output_buffer
    """
    # Write input to buffer
    if input_one is not None:
        app.buffers[3].write(input_one)
    if input_two is not None:
        app.buffers[4].write(input_two)

    exec_times = []

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


    output_index = 5 if input_two is not None else 4
    output_buffer = app.buffers[output_index].read()

    if enable_trace:
        trace_buffer = app.buffers[7].read()
        print("\nWriting trace to file...\n")
        xrt_utils.write_out_trace(trace_buffer.view(np.uint32), str(trace_file))

    return output_buffer
    