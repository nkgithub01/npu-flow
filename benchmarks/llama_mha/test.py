import sys
import time
import os
import numpy as np

from math import cos, sin, sqrt, exp
import aie.utils.xrt as xrt_utils
import aie.utils.test as test_utils

from ml_dtypes import bfloat16

def main(opts):
    embed_dim = opts.embed_dim
    head_dim = opts.head_dim
    tile_embed_dim = opts.tile_dim
    QKV_COUNT = 3
    seq_len = opts.seq_len
    # -----------------------------------------------------------------------------------
    # Configure the design's buffer size
    # -----------------------------------------------------------------------------------
    in_dtype = bfloat16
    out_dtype = bfloat16
    
    out_0_dtype = bfloat16
    qkv_act_shape = (seq_len, 3, embed_dim)
    qkv_w_shape = (seq_len, 3, embed_dim, head_dim)
    score_shape = (seq_len, head_dim)
    score_size = np.prod(score_shape) * np.dtype(out_dtype).itemsize

    # -----------------------------------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # -----------------------------------------------------------------------------------
    app = xrt_utils.AIE_Application(opts.xclbin, opts.instr, "MLIR_AIE")
    app.register_buffer(3, shape=qkv_act_shape, dtype=in_dtype)
    app.register_buffer(4, shape=qkv_w_shape, dtype=in_dtype)
    app.register_buffer(5, shape=score_shape, dtype=in_dtype)

    # -----------------------------------------------------------------------------------
    # Setup input data
    # -----------------------------------------------------------------------------------
    # Activations: Q, K, V each has shape (288,)
    qkv_a_data = np.tile(np.linspace(-0.1, 0.1, embed_dim), (QKV_COUNT, 1)).astype(in_dtype)
    qkv_a_data = np.tile(qkv_a_data, (seq_len, 1, 1))
    # Weights: Q, K, V each has shape (288, 48)
    qkv_w_data = np.zeros((QKV_COUNT, embed_dim, head_dim), dtype=np.float32)
    for i in range(QKV_COUNT):
        for j in range(embed_dim):
            qkv_w_data[i, j] = ((-1)**j) * 0.05
    qkv_w_data = qkv_w_data.astype(in_dtype)
    qkv_w_data = np.tile(qkv_w_data, (seq_len, 1, 1, 1))

    print("input shapes:", qkv_a_data.shape, qkv_w_data.shape)
    # -----------------------------------------------------------------------------------
    # Main run loop
    # -----------------------------------------------------------------------------------
    app.buffers[3].write(qkv_a_data)
    app.buffers[4].write(qkv_w_data)
    start = time.time_ns()
    app.run()
    stop = time.time_ns()
    
    score_buf = app.buffers[5].read()
    aie_score = score_buf[:score_size].view(out_dtype)   
    aie_score = np.reshape(aie_score, score_shape) 

    # -----------------------------------------------------------------------------------
    # Golden Reference
    # -----------------------------------------------------------------------------------
    output_kc = np.zeros((seq_len, head_dim), dtype=out_dtype)
    output_vc = np.zeros((seq_len, head_dim), dtype=out_dtype)
    output_qs = np.zeros((seq_len, head_dim), dtype=out_dtype)
    output_xbs = np.zeros((seq_len, head_dim), dtype=out_dtype)
    softmax_output = np.zeros(seq_len, dtype=out_dtype)

    for pos in range(seq_len):
        qkv_a_pos = qkv_a_data[pos]
        qkv_w_pos = qkv_w_data[pos]

        # Matmul Q, K, V
        gd_qkv_out = np.zeros((QKV_COUNT, head_dim), dtype=out_dtype)
        for qkv_iter in range(QKV_COUNT):
            gd_qkv_out[qkv_iter] = np.dot(
                qkv_a_pos[qkv_iter].astype(out_dtype),
                qkv_w_pos[qkv_iter].astype(out_dtype),
            ).astype(out_dtype)

        # Flatten and apply RoPE
        output_c = np.reshape(gd_qkv_out, (-1))
        for s in range(0, 48, 2):
            freq = 1.0 / pow(10000.0, float(s) / float(48))
            val = pos * freq
            fcr = cos(val)
            fci = sin(val)

            # Apply RoPE on Q
            v0, v1 = output_c[s], output_c[s + 1]
            output_c[s] = v0 * fcr - v1 * fci
            output_c[s + 1] = v0 * fci + v1 * fcr

            # Apply RoPE on K
            v0, v1 = output_c[s + 48], output_c[s + 49]
            output_c[s + 48] = v0 * fcr - v1 * fci
            output_c[s + 49] = v0 * fci + v1 * fcr

        # Store Q, K, V
        output_qs[pos] = output_c[0:head_dim]
        output_kc[pos] = output_c[head_dim : 2 * head_dim]
        output_vc[pos] = output_c[2 * head_dim : 3 * head_dim]

        # Attention 1 + Softmax + Attention 2
        output_xb = np.zeros(head_dim, dtype=out_dtype)
        softmax_output.fill(-99.0)

        for t in range(pos + 1):
            score = out_dtype(0)
            for i in range(head_dim):
                score += output_qs[pos][i] * output_kc[t][i]
            score /= sqrt(head_dim)
            softmax_output[t] = score

        max_val = np.max(softmax_output[:pos + 1])
        softmax_output[:pos + 1] = np.exp(softmax_output[:pos + 1] - max_val)
        softmax_output[:pos + 1] /= np.sum(softmax_output[:pos + 1])

        for t in range(pos + 1):
            output_xb += softmax_output[t] * output_vc[t]

        output_xbs[pos] = output_xb

    # === Validation Loop ===
    print("\n=== Result Check ===")
    score_f32 = aie_score.astype(np.float32)
    golden_f32 = output_xbs.astype(np.float32)

    if not np.allclose(score_f32[0], golden_f32[0], rtol=0.07, atol=1e-5):
        print(f"\nFailed")
        print("AIE SCORE     :", score_f32[0])
        print("GOLDEN OUTPUT :", golden_f32[0])
        exit(1)

    print("\nPASS!\n")
    print(f"Total NPU time: {(stop - start) // 1000} us")

if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument("--embed-dim", type=int, required=True, help="Total embedding dimension.")
    p.add_argument("--head-dim", type=int, required=True, help="Dimension of a single attention head.")
    p.add_argument("--tile-dim", type=int, required=True, help="Embedding dimension processed per tile.")
    p.add_argument("--seq-len", type=int, required=True, help="Sequence length.")

    opts = p.parse_args(sys.argv[1:])
    main(opts)
    