import sys
import time
import os
import numpy as np

from math import cos, sin, sqrt, exp
import aie.utils.xrt as xrt_utils
import aie.utils.test as test_utils

from ml_dtypes import bfloat16

def main(opts):
    embed_dim = 288
    head_dim = 48
    QKV_COUNT = 3
    pos = 16
    seq_len = 256
    # -----------------------------------------------------------------------------------
    # Configure the design's buffer size
    # -----------------------------------------------------------------------------------
    in_dtype = bfloat16
    out_dtype = bfloat16
    
    out_0_dtype = bfloat16
    qkv_act_shape = (3, embed_dim)
    qkv_w_shape = (3, embed_dim, head_dim)
    kv_cache_shape = (2, pos+1, head_dim)
    score_shape = (head_dim)

    kv_cache_size = np.prod(kv_cache_shape) * np.dtype(in_dtype).itemsize
    score_size = np.prod(score_shape) * np.dtype(out_dtype).itemsize

    # -----------------------------------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # -----------------------------------------------------------------------------------
    app = xrt_utils.AIE_Application(opts.xclbin, opts.instr, "MLIR_AIE")
    app.register_buffer(3, shape=qkv_act_shape, dtype=in_dtype)
    app.register_buffer(4, shape=qkv_w_shape, dtype=in_dtype)
    app.register_buffer(5, shape=kv_cache_shape, dtype=in_dtype)
    app.register_buffer(6, shape=score_shape, dtype=in_dtype)

    # -----------------------------------------------------------------------------------
    # Setup input data
    # -----------------------------------------------------------------------------------
    # Activations: Q, K, V each has shape (288,)
    qkv_a_data = np.tile(np.linspace(-0.1, 0.1, embed_dim), (QKV_COUNT, 1)).astype(in_dtype)

    # Weights: Q, K, V each has shape (288, 48)
    qkv_w_data = np.zeros((QKV_COUNT, embed_dim, head_dim), dtype=np.float32)
    for i in range(QKV_COUNT):
        for j in range(embed_dim):
            qkv_w_data[i, j] = ((-1)**j) * 0.05
    qkv_w_data = qkv_w_data.astype(in_dtype)
    
    # Assume all previous KV entries are 0
    kv_cache_data = np.zeros(kv_cache_shape, dtype=in_dtype)

    # -----------------------------------------------------------------------------------
    # Main run loop
    # -----------------------------------------------------------------------------------
    app.buffers[3].write(qkv_a_data)
    app.buffers[4].write(qkv_w_data)
    app.buffers[5].write(kv_cache_data)
    start = time.time_ns()
    app.run()
    stop = time.time_ns()

    kv_cache_buf = app.buffers[5].read()
    final_kv_cache = kv_cache_buf[:kv_cache_size].view(in_dtype)
    
    score_buf = app.buffers[6].read()
    aie_score = score_buf[:score_size].view(out_dtype)    

    # -----------------------------------------------------------------------------------
    # Golden Reference
    # -----------------------------------------------------------------------------------
    gd_qkv_out = np.zeros(shape=(QKV_COUNT, head_dim), dtype=out_dtype)
    for qkv_iter in range(QKV_COUNT):
        gd_qkv_out[qkv_iter] = np.dot(
            qkv_a_data[qkv_iter].astype(out_dtype),
            qkv_w_data[qkv_iter].astype(out_dtype),
        ).astype(out_dtype)
    output_c = np.reshape(gd_qkv_out, (-1))
    
    for s in range(0, 48, 2):
        freq = 1.0 / pow(10000.0, float(s) / float(48))
        val = pos * freq

        fcr = cos(val)
        fci = sin(val)

        v0 = output_c[s]
        v1 = output_c[s + 1]

        output_c[s] = v0 * fcr - v1 * fci
        output_c[s + 1] = v0 * fci + v1 * fcr

        v0 = output_c[s + 48]
        v1 = output_c[s + 48 + 1]
        output_c[s + 48] = v0 * fcr - v1 * fci
        output_c[s + 48 + 1] = v0 * fci + v1 * fcr
    

    output_q = np.zeros(shape=(1, head_dim), dtype=out_dtype)
    output_kc = np.zeros(shape=(seq_len, head_dim), dtype=out_dtype)
    output_vc = np.zeros(shape=(seq_len, head_dim), dtype=out_dtype)
    output_xb = np.zeros(shape=(head_dim), dtype=out_dtype)
    softmax_output = np.zeros(shape=(seq_len), dtype=out_dtype)

    for i in range(0, head_dim):
        output_q[0][i] = output_c[i]
        output_kc[pos][i] = output_c[head_dim + i]
        output_vc[pos][i] = output_c[2 * head_dim + i]

    # Attn 1
    softmax_output.fill(-99.0)
    for t in range(0, pos + 1):
        score = out_dtype(0)
        for i in range(0, head_dim):
            score += output_q[0][i] * output_kc[t][i]
        score /= sqrt(head_dim)
        softmax_output[t] = score

    # Softmax
    max_val = softmax_output[0]
    for i in range(1, pos + 1):
        if softmax_output[i] > max_val:
            max_val = softmax_output[i]
    sum_val = 0.0
    for i in range(pos + 1):
        softmax_output[i] = exp(softmax_output[i] - max_val)
        sum_val += softmax_output[i]
    for i in range(pos + 1):
        softmax_output[i] = softmax_output[i] / sum_val

    # Attn 2
    for t in range(0, pos + 1):
        for i in range(0, head_dim):
            output_xb[i] += softmax_output[t] * output_vc[t][i]
    print("SCORE")
    print(aie_score)
    print("GOLDEN REFERENCE")
    print(output_xb)
    
    print(f"\nTotal NPU time: {(stop - start) // 1000} us")

    score_f32 = aie_score.astype(np.float32)
    golden_f32 = output_xb.astype(np.float32) 
    if np.allclose(score_f32, golden_f32, rtol=0.07, atol=1e-5):
        print("\nPASS!\n")
    else:
        print("\nFailed.")
    exit(0)

if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument("--embed-dim", type=int, required=True, help="Total embedding dimension.")
    p.add_argument("--head-dim", type=int, required=True, help="Dimension of a single attention head.")
    p.add_argument("--tile-dim", type=int, required=True, help="Embedding dimension processed per tile.")
    p.add_argument("--seq-len", type=int, required=True, help="Sequence length.")
    p.add_argument("--pos", type=int, required=True, help="Current position index.")

    opts = p.parse_args(sys.argv[1:])
    main(opts)
    