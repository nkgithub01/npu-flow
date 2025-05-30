
import sys
import math
import time
import os
import numpy as np
import cv2
from aie.utils.xrt import setup_aie, extract_trace, write_out_trace, execute
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
    testImageWidth = int(opts.width)
    testImageHeight = int(opts.height)
    testImageSize = testImageWidth * testImageHeight
    epsilon = 2.0
    
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    output_file = opts.outfile
    
    output_folder = "output/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_iter = opts.iters
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    trace_size = opts.trace_size
    enable_trace = False if not trace_size else True

    # -----------------------------------------------------------------------------------
    # Read the input image or generate random one if no input file argument provided
    # -----------------------------------------------------------------------------------
    if opts.image != '':
        in_image = cv2.imread(opts.image)
    else:
        in_image = np.random.randint(0, 256, (testImageWidth, testImageHeight, 4), dtype=np.uint8)
    in_image = cv2.resize(in_image, (testImageWidth, testImageHeight))
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

    shape_in = in_image.shape
    shape_out = (3, *in_image.shape)

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
            data_buffer, trace_buffer = execute(app, in_image, in_image, enable_trace, False)
        else:
            data_buffer = execute(app, in_image, in_image, enable_trace, False)
        stop = time.time_ns()

        if enable_trace and i == num_iter - 1:
            write_out_trace(trace_buffer.view(np.uint32), str(opts.trace_file))

        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time
        
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

    # -----------------------------------------------------------------------------------
    # Save the AIE output image and Compare the AIE output and the golden reference
    # -----------------------------------------------------------------------------------
    output_image_1 = data_buffer[0].squeeze()
    output_image_2 = data_buffer[1].squeeze()
    output_image_3 = data_buffer[2].squeeze()
    golden_output_image = cv2.cvtColor(golden_output_image, cv2.COLOR_RGBA2BGR)
    output_image_1 = cv2.cvtColor(output_image_1, cv2.COLOR_RGBA2BGR)
    output_image_2 = cv2.cvtColor(output_image_2, cv2.COLOR_RGBA2BGR)
    output_image_3 = cv2.cvtColor(output_image_3, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(output_folder + f"golden_{output_file}", golden_output_image)
    cv2.imwrite(output_folder + f"Col_1_{output_file}", output_image_1)
    cv2.imwrite(output_folder + f"Col_2_{output_file}", output_image_2)
    cv2.imwrite(output_folder + f"Col_3_{output_file}", output_image_3)
    
    _, output_1_L1_error = image_compare(output_image_1, golden_output_image)
    _, output_2_L1_error = image_compare(output_image_2, golden_output_image)
    _, output_3_L1_error = image_compare(output_image_3, golden_output_image)

    if output_1_L1_error < epsilon and output_2_L1_error < epsilon and output_3_L1_error < epsilon:
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nFailed.")
        if not output_1_L1_error < epsilon:
            print("First Column Failed")
            image_compare(output_image_1, golden_output_image)
        if not output_2_L1_error < epsilon:
            print("Second Column Failed")
            image_compare(output_image_2, golden_output_image)
        if not output_3_L1_error < epsilon:
            print("Thrid Column Failed")
            image_compare(output_image_3, golden_output_image)
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "-wd",
        "--width",
        dest="width",
        default=32,
        help="Width of image",
    )
    p.add_argument(
        "-ht",
        "--height",
        dest="height",
        default=32,
        help="Height of image",
    )
    p.add_argument(
        "--image",
        dest="image",
        default='',
        help="Input image file path",
    )
    p.add_argument(
        "--outfile",
        dest="outfile",
        default='edgeDetectOut_test.jpg',
        help="Output image file path",
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)