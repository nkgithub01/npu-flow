import numpy as np
import argparse

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects.ext import arith
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx
from ml_dtypes import bfloat16

def mha(embed_dim, head_dim, tile_embed_dim, seq_len, pos):
    @device(AIEDevice.npu2)
    def device_body():
        pos_p1 = pos + 1 # index to include pos

        # Types
        kv_full_ty = np.ndarray[(2, pos + 1, head_dim), np.dtype[bfloat16]]
        a_in_ty = np.ndarray[(3, embed_dim), np.dtype[bfloat16]]
        b_in_ty = np.ndarray[(3, embed_dim, head_dim), np.dtype[bfloat16]]
        a_tile_ty = np.ndarray[(tile_embed_dim,), np.dtype[bfloat16]]
        b_tile_ty = np.ndarray[(tile_embed_dim, head_dim), np.dtype[bfloat16]]
        c_out_ty = np.ndarray[(3, head_dim), np.dtype[bfloat16]]
        c_out_single_ty = np.ndarray[(1, head_dim), np.dtype[bfloat16]]
        attn_ty = np.ndarray[(seq_len,), np.dtype[bfloat16]]
        xb_ty = np.ndarray[(head_dim,), np.dtype[bfloat16]]
        head_div_2_ty = np.ndarray[(24,), np.dtype[bfloat16]]

        # Kernels
        linalg_fill_bf16 = external_func("linalg_fill_bf16", inputs=[np.int32, c_out_ty])
        vecmat_bf16_bf16 = external_func("vecmat_bf16_bf16", inputs=[np.int32, a_tile_ty, b_tile_ty, c_out_ty])
        cosf_poly = external_func("cosf_bf16_24_8",inputs=[head_div_2_ty, head_div_2_ty])
        sinf_poly = external_func("sinf_bf16_24_8",inputs=[head_div_2_ty, head_div_2_ty])
        freq_pos = external_func("freq_pos_bf16_24_8",inputs=[np.int32, head_div_2_ty])
        shuffle_apply_rope = external_func("shuffle_apply_rope_bf16_48", inputs=[np.int32, head_div_2_ty, head_div_2_ty, c_out_ty])
        fill_neg = external_func("fill_neg", inputs=[attn_ty])
        attn_qk = external_func("attn_1", inputs=[c_out_single_ty, c_out_single_ty, np.int32, attn_ty])
        softmax_k = external_func("softmax_bf16", inputs=[attn_ty, np.int32, attn_ty])
        attn_v = external_func("attn_2", inputs=[attn_ty, c_out_single_ty, np.int32, xb_ty])
        split_qkv = external_func("split_qkv", inputs=[c_out_ty, c_out_single_ty, c_out_single_ty, c_out_single_ty])
        fill_zero = external_func("fill_zero", inputs=[xb_ty])

        # Declare tiles
        qkv_shim, qkv_mem, qkv_core = tile(1, 0), tile(1, 1), tile(1, 2)
        attn_shim, attn_mem, attn_core = tile(0, 0), tile(0, 1), tile(0, 2)

        # Object FIFOs and links
        # Stage 1 Ins
        inOF_A = object_fifo("inOF_A", qkv_shim, qkv_mem, 1, a_in_ty)
        inOF_B = object_fifo("inOF_B", qkv_shim, qkv_mem, 1, b_in_ty)
        inOF_A_tiled = object_fifo("inOF_A_tiled", qkv_mem, qkv_core, 9, a_tile_ty)
        inOF_B_tiled = object_fifo("inOF_B_tiled", qkv_mem, qkv_core, 9, b_tile_ty)
        
        a_offsets = [r * embed_dim + c * tile_embed_dim for r in range(3) for c in range(embed_dim // tile_embed_dim)]
        b_offsets = [r * embed_dim * head_dim + c * tile_embed_dim * head_dim for r in range(3) for c in range(embed_dim // tile_embed_dim)]

        object_fifo_link(inOF_A, inOF_A_tiled, [], a_offsets)
        object_fifo_link(inOF_B, inOF_B_tiled, [], b_offsets)

        # Stage 1 Intermediates
        OF_freq = object_fifo(f"OF_freq", qkv_core, qkv_core, 1, head_div_2_ty)
        OF_sin = object_fifo(f"OF_sin", qkv_core, qkv_core, 1, head_div_2_ty)
        OF_cos = object_fifo(f"OF_cos", qkv_core, qkv_core, 1, head_div_2_ty)

        # Stage 1 Outs 
        OF_QKV = object_fifo("OF_QKV", qkv_core, qkv_core, 1, c_out_ty)
        OF_Q = object_fifo("OF_Q", qkv_core, attn_core, 1, c_out_single_ty)
        outOF_K_toM = object_fifo("outOF_K_toM", qkv_core, qkv_mem, 1, c_out_single_ty)
        outOF_K_toS = object_fifo("outOF_K_toS", qkv_mem, qkv_shim, 1, c_out_single_ty)
        outOF_V_toM = object_fifo("outOF_V_toM", qkv_core, qkv_mem, 1, c_out_single_ty)
        outOF_V_toS = object_fifo("outOF_V_toS", qkv_mem, qkv_shim, 1, c_out_single_ty)

        object_fifo_link(outOF_K_toM, outOF_K_toS)
        object_fifo_link(outOF_V_toM, outOF_V_toS)

        # Stage 2 Ins
        inOF_KV_toM = object_fifo("inOF_KV_toM", attn_shim, attn_mem, 2, c_out_single_ty)
        inOF_KV_single = object_fifo("inOF_KV_single", attn_mem, attn_core, 2, c_out_single_ty)
        object_fifo_link(inOF_KV_toM, inOF_KV_single)

        # Stage 2 Intermediates
        OF_attn1 = object_fifo("OF_attn1", attn_core, attn_core, 1, attn_ty)
        OF_sm = object_fifo("OF_sm", attn_core, attn_core, 1, attn_ty)

        # Stage 2 Outs
        outOF_score_toM = object_fifo("outOF_score_toM", attn_core, attn_mem, 1, xb_ty)
        outOF_score_toS = object_fifo("outOF_score_toS", attn_mem, attn_shim, 1, xb_ty)
        object_fifo_link(outOF_score_toM, outOF_score_toS)

        @core(qkv_core, "mha.cc.o")
        def compute():
            q_off, k_off, v_off = arith.constant(0), arith.constant(head_dim), arith.constant(2*head_dim) 
            qkv_out = OF_QKV.acquire(ObjectFifoPort.Produce, 1)
            linalg_fill_bf16(q_off, qkv_out)
            # Vec-mat multiply for Q, K, V
            for off in [q_off, k_off, v_off]:
                for _ in range_(embed_dim // tile_embed_dim):
                    a_tile = inOF_A_tiled.acquire(ObjectFifoPort.Consume, 1)
                    b_tile = inOF_B_tiled.acquire(ObjectFifoPort.Consume, 1)
                    vecmat_bf16_bf16(off, a_tile, b_tile, qkv_out)
                    inOF_A_tiled.release(ObjectFifoPort.Consume, 1)
                    inOF_B_tiled.release(ObjectFifoPort.Consume, 1)

            # Get frequency values
            freq_res = OF_freq.acquire(ObjectFifoPort.Produce, 1)
            freq_pos(arith.constant(pos), freq_res)
            OF_freq.release(ObjectFifoPort.Produce, 1)

            # Get cos/sin values
            freq_res = OF_freq.acquire(ObjectFifoPort.Consume, 1)
            sin_res = OF_sin.acquire(ObjectFifoPort.Produce, 1)
            cos_res = OF_cos.acquire(ObjectFifoPort.Produce, 1)
            sinf_poly(freq_res, sin_res) 
            cosf_poly(freq_res, cos_res)
            OF_freq.release(ObjectFifoPort.Consume, 1)
            OF_sin.release(ObjectFifoPort.Produce, 1)
            OF_cos.release(ObjectFifoPort.Produce, 1)

            # Apply RoPE
            sin_res = OF_sin.acquire(ObjectFifoPort.Consume, 1)
            cos_res = OF_cos.acquire(ObjectFifoPort.Consume, 1)
            shuffle_apply_rope(q_off, cos_res, sin_res, qkv_out)
            shuffle_apply_rope(k_off, cos_res, sin_res, qkv_out)
            OF_sin.release(ObjectFifoPort.Consume, 1)
            OF_cos.release(ObjectFifoPort.Consume, 1)
            OF_QKV.release(ObjectFifoPort.Produce, 1) 
            
            # Send Q to next core, KV back to host
            qkv_res = OF_QKV.acquire(ObjectFifoPort.Consume, 1)
            q_res = OF_Q.acquire(ObjectFifoPort.Produce, 1)
            v_out = outOF_V_toM.acquire(ObjectFifoPort.Produce, 1)
            k_out = outOF_K_toM.acquire(ObjectFifoPort.Produce, 1)
            split_qkv(qkv_res, q_res, k_out, v_out)
            OF_QKV.release(ObjectFifoPort.Consume, 1)
            outOF_K_toM.release(ObjectFifoPort.Produce, 1)
            outOF_V_toM.release(ObjectFifoPort.Produce, 1)
            OF_Q.release(ObjectFifoPort.Produce, 1)

        @core(attn_core, "attn.cc.o")
        def compute():
            # Attn 1st stage
            attn_1 = OF_attn1.acquire(ObjectFifoPort.Produce, 1)
            fill_neg(attn_1)
            q_res = OF_Q.acquire(ObjectFifoPort.Consume, 1)
            for i in range(pos_p1):
                k_data = inOF_KV_single.acquire(ObjectFifoPort.Consume, 1)
                attn_qk(q_res, k_data, i, attn_1)
                inOF_KV_single.release(ObjectFifoPort.Consume, 1)
            OF_Q.release(ObjectFifoPort.Consume, 1)
            OF_attn1.release(ObjectFifoPort.Produce, 1)

            # Softmax
            attn_1 = OF_attn1.acquire(ObjectFifoPort.Consume, 1)
            sm_res = OF_sm.acquire(ObjectFifoPort.Produce, 1)
            softmax_k(attn_1, arith.constant(pos_p1), sm_res)
            OF_sm.release(ObjectFifoPort.Produce, 1)
            OF_attn1.release(ObjectFifoPort.Consume, 1)

            # Attn 2nd stage
            sm_res = OF_sm.acquire(ObjectFifoPort.Consume, 1)
            score = outOF_score_toM.acquire(ObjectFifoPort.Produce, 1)
            fill_zero(score)
            for i in range(pos_p1):
                v_data = inOF_KV_single.acquire(ObjectFifoPort.Consume, 1)
                attn_v(sm_res, v_data, i, score)
                inOF_KV_single.release(ObjectFifoPort.Consume, 1)
            outOF_score_toM.release(ObjectFifoPort.Produce, 1)
                
        @runtime_sequence(a_in_ty, b_in_ty, kv_full_ty, xb_ty)
        def sequence(A, B, C, O):
            task_a = shim_dma_single_bd_task(inOF_A, A, sizes=[1, 1, 3, embed_dim], 
                                                        strides=[0, 0, embed_dim, 1],
                                                        issue_token=True)
            task_b = shim_dma_single_bd_task(inOF_B, B, sizes=[1, 3, embed_dim, head_dim], 
                                                        strides=[0, embed_dim*head_dim, head_dim, 1], 
                                                        issue_token=True)  
            task_ck = shim_dma_single_bd_task(outOF_K_toS, C, sizes=[1, 1, 1, head_dim], 
                                                              offset=pos*head_dim,
                                                              issue_token=True)
            task_cv = shim_dma_single_bd_task(outOF_V_toS, C, sizes=[1, 1, 1, head_dim], 
                                                              offset=pos_p1*head_dim + pos*head_dim,
                                                              issue_token=True)
            
            dma_start_task(task_a, task_b, task_ck)
            dma_await_task(task_ck)
            dma_start_task(task_cv)
            dma_await_task(task_cv)
            dma_free_task(task_a, task_b)  
              
            # Single DMA channel that passes K[0] to K[pos], then V[0] to V[pos]
            task_kv_in = shim_dma_single_bd_task(inOF_KV_toM, C, sizes=[pos_p1*2, 1, 1, head_dim], 
                                                                 strides=[head_dim, 0, 0, 1],
                                                                 issue_token=True)
            task_out = shim_dma_single_bd_task(outOF_score_toS, O, sizes=[1, 1, 1, head_dim], issue_token=True)
            dma_start_task(task_kv_in, task_out)
            dma_await_task(task_out)
            dma_free_task(task_kv_in)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embed-dim", type=int, required=True, help="Total embedding dimension.")
    p.add_argument("--head-dim", type=int, required=True, help="Dimension of a single attention head.")
    p.add_argument("--tile-dim", type=int, required=True, help="Embedding dimension processed per tile.")
    p.add_argument("--seq-len", type=int, required=True, help="Sequence length.")
    p.add_argument("--pos", type=int, required=True, help="Current position index.")

    opts = p.parse_args()
    with mlir_mod_ctx() as ctx:
        mha(opts.embed_dim, opts.head_dim, opts.tile_dim, opts.seq_len, opts.pos)
        res = ctx.module.operation.verify()
        if res == True:
            print(ctx.module)
        else:
            print(res)
