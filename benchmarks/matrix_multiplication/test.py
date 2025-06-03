
import sys
import time
import os
import numpy as np
from ml_dtypes import bfloat16
import cv2

from aie.utils.xrt import setup_aie, write_out_trace, execute
import aie.utils.test as test_utils


dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
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
    total_iters = num_iter + num_warmup_iter
    trace_size = opts.trace_size
    enable_trace = False if not trace_size else True

    b_col_maj = opts.b_col_maj
    M = opts.M
    K = opts.K
    N = opts.N
    
    npu_time_total = 0

    print(f"Input matrix A size: {M}x{K}")
    print(f"Input matrix B size: {K}x{N}")

    # -----------------------------------------------------------------------------------
    # Configure the design's buffer size
    # -----------------------------------------------------------------------------------
    dtype_in = dtype_map[opts.dtype_in_str]
    dtype_out = dtype_map[opts.dtype_out_str]


    shape_in_A = (M, K)
    shape_in_B = (K, N)
    shape_out_C = (M, N)

    # -----------------------------------------------------------------------------------
    # Generate the input matrices A and B and the reference output matrix C
    # -----------------------------------------------------------------------------------
    Mat_A = np.random.randint(0, 256, shape_in_A, dtype=dtype_in)
    Mat_B = np.random.randint(0, 256, shape_in_B, dtype=dtype_in)
    Mat_C_ref = Mat_A @ Mat_B
    


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
            data_buffer, trace_buffer = execute(app=app, input_one=Mat_A, input_two=Mat_B, enable_trace=enable_trace, trace_after_output=False)
        else:
            data_buffer = execute(app=app, input_one=Mat_A, input_two=Mat_B, enable_trace=enable_trace, trace_after_output=False)
        stop = time.time_ns()

        if enable_trace and i == num_iter - 1:
            write_out_trace(trace_buffer.view(np.uint32), str(opts.trace_file))

        npu_time = stop - start
        # Warmup iterations are not counted in the average time
        if i >= num_warmup_iter:
            npu_time_total = npu_time_total + npu_time
        
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

    # -----------------------------------------------------------------------------------
    # Compare the AIE output and the golden reference result
    # -----------------------------------------------------------------------------------
    Mat_C = data_buffer.view(dtype_out)

    if dtype_out == np.int8:
        relative_tolerance = 0
        absolute_tolerance = 0
    elif dtype_out == np.int16:
        relative_tolerance = 0
        absolute_tolerance = 0
    elif dtype_out == np.int32:
        relative_tolerance = 0
        absolute_tolerance = 0
    elif dtype_out == bfloat16:
        relative_tolerance = 0.05
        absolute_tolerance = 0.5
    elif dtype_out == np.float32:
        relative_tolerance = 0.05
        absolute_tolerance = 0.5
    else:
        relative_tolerance = 0
        absolute_tolerance = 0

    are_close = np.allclose(Mat_C, Mat_C_ref, rtol=relative_tolerance, atol=absolute_tolerance)

    if are_close:
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nFailed.")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "--b_col_maj",
        type=bool,
        dest="b_col_maj",
        default=False,
        help="Flag to indicate if the input matrix B is passed into the AIE array in column-major order",
    )
    p.add_argument(
        "-M",
        type=int,
        dest="M",
        default=512,
        help="Input matrix A height (number of rows)",
    )
    p.add_argument(
        "-K",
        type=int,
        dest="K",
        default=512,
        help="Input matrix A width (number of columns) and input matrix B height (number of rows)",
    )
    p.add_argument(
        "-N",
        type=int,
        dest="N",
        default=512,
        help="Input matrix B width (number of columns)",
    )
    p.add_argument(
        "-dtype_in",
        type=str,
        dest="N",
        default=512,
        help="Input matrix B width (number of columns)",
    )
    p.add_argument(
        "-dtype_in", 
        type=str, 
        dest="dtype_in_str",
        choices=["bf16", "i8", "i16"], 
        default="i16"
    )
    p.add_argument(
        "-dtype_out", 
        type=str, 
        dest="dtype_out_str",
        choices=["bf16", "i8", "i16", "f32", "i32"], 
        default="i16"
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)