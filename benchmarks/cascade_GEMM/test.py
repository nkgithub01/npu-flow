
import os
import sys
import time
import numpy as np

from aie.utils.xrt import setup_aie, write_out_trace, execute
import aie.utils.test as test_utils

from execute_aie import execute_aie_multi_with_timing

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
    total_iters = num_iter + num_warmup_iter
    trace_size = opts.trace_size
    enable_trace = False if not trace_size else True

    b_col_maj = opts.b_col_maj
    M = opts.M
    K = opts.K
    N = opts.N
    
    npu_time_total = 0

    output_folder = "output/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if verbosity > 0:
        print(f"Input matrix A size: {M}x{K}")
        print(f"Input matrix B size: {K}x{N}")

    # -----------------------------------------------------------------------------------
    # Configure the design's buffer size
    # -----------------------------------------------------------------------------------
    dtype_in = dtype_map[opts.dtype_in_str]
    dtype_out = dtype_map[opts.dtype_out_str]
    # The test only supports integer.
    dtype_is_int = np.issubdtype(dtype_in, np.integer)
    assert dtype_is_int, "Input data type must be an integer type (i8, i16)."
    dtype_in_min = np.iinfo(dtype_in).min
    dtype_in_max = np.iinfo(dtype_in).max

    shape_in_A = (M, K)
    shape_in_B = (K, N)
    shape_out_C = (M, N)

    # -----------------------------------------------------------------------------------
    # Generate the input matrices A and B and the reference output matrix C
    # -----------------------------------------------------------------------------------
    Mat_A = np.random.randint(dtype_in_min/2, dtype_in_max/2, shape_in_A, dtype=dtype_in)
    Mat_B = np.random.randint(dtype_in_min/2, dtype_in_max/2, shape_in_B, dtype=dtype_in)
    if b_col_maj:
        Mat_C_ref = Mat_A.astype(dtype_out) @ Mat_B.astype(dtype_out).T.reshape(K, N)
    else:
        Mat_C_ref = Mat_A.astype(dtype_out) @ Mat_B.astype(dtype_out)
    
    if verbosity > 1:
        print(f"Input matrix A: {Mat_A.shape}\n{Mat_A}")
        print(f"Input matrix B: {Mat_B.shape}\n{Mat_B}")
        print(f"Reference output matrix C: {Mat_C_ref.shape}\n{Mat_C_ref}")
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
    data_buffer = execute_aie_multi_with_timing(
        app,
        input_one=Mat_A,
        input_two=Mat_B,
        enable_trace=enable_trace,
        num_iters=num_iter,
        warmup_iters=num_warmup_iter,
        trace_file=opts.trace_file
    )
    # -----------------------------------------------------------------------------------
    # Compare the AIE output and the golden reference result
    # -----------------------------------------------------------------------------------
    Mat_C = np.array(data_buffer, dtype=dtype_out)
    if verbosity > 1:
        print(f"Output matrix C: {Mat_C.shape}\n{Mat_C}")
        np.savetxt(output_folder+"reference_matrix_C.txt", Mat_C_ref, fmt="%d")
        np.savetxt(output_folder+"output_matrix_C.txt", Mat_C, fmt="%d")

    relative_tolerance = 0
    absolute_tolerance = 0

    if do_verify:
        are_close = np.allclose(Mat_C, Mat_C_ref, rtol=relative_tolerance, atol=absolute_tolerance)

        if are_close:
            print("\nPASS!\n")
            exit(0)
        else:
            print("\nFailed.")
            for i in range(M):
                for j in range(N):
                    if not np.isclose(Mat_C[i, j], Mat_C_ref[i, j], rtol=relative_tolerance, atol=absolute_tolerance):
                        print(f"First mismatch at ({i}, {j}): AIE={Mat_C[i, j]}, Ref={Mat_C_ref[i, j]}")
                        exit(-1)
            print(f"\nRelative tolerance: {relative_tolerance}, Absolute tolerance: {absolute_tolerance}\n")
            exit(-1)
    else:
        print("\nVerification skipped, assuming PASS.\n")
        exit(0)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "--b_col_maj",
        type=int,
        choices=[0, 1],
        dest="b_col_maj",
        default=0,
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
        dest="dtype_in_str",
        choices=["i8", "i16"], 
        default="i16"
    )
    p.add_argument(
        "-dtype_out", 
        type=str, 
        dest="dtype_out_str",
        choices=["i8", "i16", "i32"], 
        default="i16"
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)