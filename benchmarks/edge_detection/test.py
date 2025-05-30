
import sys
import time
import os
import numpy as np
import cv2
from aie.utils.xrt import setup_aie, write_out_trace, execute
import aie.utils.test as test_utils


def edge_detect(in_image):
    in_gray = cv2.cvtColor(in_image, cv2.COLOR_RGBA2GRAY)
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)  # Laplacian

    image_filtered = cv2.filter2D(in_gray, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    _, image_thresh = cv2.threshold(image_filtered, 10, 255, cv2.THRESH_BINARY)
    image_thresh_bgr = cv2.cvtColor(image_thresh, cv2.COLOR_GRAY2RGBA)
    alpha = 1.0
    beta = 1.0
    gamma = 0.0
    out_image = cv2.addWeighted(image_thresh_bgr, alpha, in_image, beta, gamma)
    return out_image


def image_compare(img1, img2, verbose=True):
    diff = cv2.absdiff(img1, img2)
    l1_error = np.sum(diff) / diff.size
    num_diff = np.count_nonzero(diff)

    if verbose:
        print(f"Differences: {num_diff}, Avg L1 error: {l1_error}")
    return num_diff, l1_error
    

def main(opts):
    # -----------------------------------------------------------------------------------
    # Program arguments parsing
    # -----------------------------------------------------------------------------------
    test_image_width = int(opts.image_width)
    test_image_height = int(opts.image_height)
    num_compute_flow_column = int(opts.num_compute_flow_column)
    epsilon = 2.0
    
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    output_file = opts.outfile
    
    output_folder = "output/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_iter = opts.iters
    npu_time_total = 0
    trace_size = opts.trace_size
    enable_trace = False if not trace_size else True

    # -----------------------------------------------------------------------------------
    # Read the input image or generate random one if no input file argument provided
    # -----------------------------------------------------------------------------------
    if opts.image != '':
        in_image = cv2.imread(opts.image)
    else:
        in_image = np.random.randint(0, 256, (test_image_width, test_image_height, 4), dtype=np.uint8)
    in_image = cv2.resize(in_image, (test_image_width, test_image_height))
    in_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGBA)

    # -----------------------------------------------------------------------------------
    # Calculate OpenCV reference for edgeDetect
    # -----------------------------------------------------------------------------------
    golden_output_image = edge_detect(in_image)
    
    # -----------------------------------------------------------------------------------
    # Configure the design's buffer size
    # -----------------------------------------------------------------------------------
    dtype_in = np.dtype("uint8")
    dtype_out = np.dtype("uint8")

    shape_in = (num_compute_flow_column, *in_image.shape)
    shape_out = (num_compute_flow_column, *in_image.shape)

    image_buffer_in = np.array([in_image for _ in range(num_compute_flow_column)], dtype=dtype_in)

    # -----------------------------------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # -----------------------------------------------------------------------------------
    app = setup_aie(
        xclbin_path,
        insts_path,
        shape_in,
        dtype_in,
        shape_in,
        dtype_in,
        shape_out,
        dtype_out,
        enable_trace=enable_trace,
        trace_size=trace_size,
        trace_after_output=False,
    )

    # -----------------------------------------------------------------------------------
    # Main run loop
    # -----------------------------------------------------------------------------------
    for i in range(num_iter):
        start = time.time_ns()
        if enable_trace:
            data_buffer, trace_buffer = execute(app=app, input_one=image_buffer_in, enable_trace=enable_trace, trace_after_output=False)
        else:
            data_buffer = execute(app=app, input_one=image_buffer_in, enable_trace=enable_trace, trace_after_output=False)
        stop = time.time_ns()

        if enable_trace and i == num_iter - 1:
            write_out_trace(trace_buffer.view(np.uint32), str(opts.trace_file))

        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time
        
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

    # -----------------------------------------------------------------------------------
    # Save the AIE output image and Compare the AIE output and the golden reference
    # -----------------------------------------------------------------------------------
    golden_output_image = cv2.cvtColor(golden_output_image, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(output_folder + f"golden_{output_file}", golden_output_image)

    output_images = []
    output_L1_errors = []
    for idx in range(num_compute_flow_column):
        output_images.append(data_buffer[idx].squeeze())
        output_images[idx] = cv2.cvtColor(output_images[idx], cv2.COLOR_RGBA2BGR)
        cv2.imwrite(output_folder + f"Col_{idx}_{output_file}", output_images[idx])
        
        print(f"Column {idx}:")
        _, output_L1_error = image_compare(output_images[idx], golden_output_image, verbose=True)
        output_L1_errors.append(output_L1_error)

    if all(error < epsilon for error in output_L1_errors):
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nFailed.")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "-iwd",
        "--image_width",
        type=int,
        dest="image_width",
        default=1920,
        help="Width of image",
    )
    p.add_argument(
        "-iht",
        "--image_height",
        type=int,
        dest="image_height",
        default=1080,
        help="Height of image",
    )
    p.add_argument(
        "-nc", 
        "--num_compute_flow_column", 
        type=int,
        required=False,
        dest="num_compute_flow_column",
        default=4,
        help="Number of compute flow columns on the AIE array that will be used in parallel",
    )
    p.add_argument(
        "--image",
        type=str,
        dest="image",
        default='',
        help="Input image file path",
    )
    p.add_argument(
        "--outfile",
        type=str,
        dest="outfile",
        default='edgeDetectOut_test.jpg',
        help="Output image file path",
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)