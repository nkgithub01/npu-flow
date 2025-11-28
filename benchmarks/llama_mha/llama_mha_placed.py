import numpy as np
import argparse

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects.ext import arith, memref
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx
from ml_dtypes import bfloat16
from aie.ir import MemRefType, IndexType, IntegerType, BF16Type
from aie.extras import types as T

def mha(embed_dim, head_dim, tile_embed_dim, seq_len):
    @device(AIEDevice.npu2)
    def device_body():

        # Host side types
        full_a_ty = np.ndarray[(seq_len, 3, embed_dim,), np.dtype[bfloat16]]
        full_w_ty = np.ndarray[(seq_len, 3, embed_dim, head_dim,), np.dtype[bfloat16]]
        f_score_ty = np.ndarray[(seq_len, head_dim,), np.dtype[bfloat16]]

        # Tile side types
        a_mem_ty = np.ndarray[(3, embed_dim), np.dtype[bfloat16]]
        w_mem_ty = np.ndarray[(3, embed_dim, head_dim), np.dtype[bfloat16]]
        a_in_ty = np.ndarray[(tile_embed_dim,), np.dtype[bfloat16]]
        w_in_ty = np.ndarray[(tile_embed_dim, head_dim), np.dtype[bfloat16]]
        qkv_ty = np.ndarray[(3, head_dim), np.dtype[bfloat16]]
        qkv_1_ty = np.ndarray[(head_dim,), np.dtype[bfloat16]]
        attn_ty = np.ndarray[(256,), np.dtype[bfloat16]]
        score_ty = np.ndarray[(head_dim,), np.dtype[bfloat16]]
        head_div_2_ty = np.ndarray[(head_dim // 2,), np.dtype[bfloat16]]
        cache_ty = np.ndarray[(256 * head_dim,), np.dtype[bfloat16]]

        mem_init = np.full((256 * head_dim,), bfloat16(0.0), dtype=bfloat16)
        
        # Kernels
        linalg_fill_bf16 = external_func("linalg_fill_bf16", inputs=[np.int32, qkv_ty])
        vecmat_bf16_bf16 = external_func("vecmat_bf16_bf16", inputs=[np.int32, a_in_ty, w_in_ty, qkv_ty])
        cosf_poly = external_func("cosf_bf16_24_8",inputs=[head_div_2_ty, head_div_2_ty])
        sinf_poly = external_func("sinf_bf16_24_8",inputs=[head_div_2_ty, head_div_2_ty])
        freq_pos = external_func("freq_pos_bf16_24_8",inputs=[np.int32, head_div_2_ty])
        shuffle_apply_rope = external_func("shuffle_apply_rope_bf16_48", inputs=[np.int32, head_div_2_ty, head_div_2_ty, qkv_ty])
        fill_neg = external_func("fill_neg", inputs=[attn_ty])
        attn_qk = external_func("attn_1", inputs=[qkv_1_ty, qkv_1_ty, np.int32, attn_ty])
        softmax_k = external_func("softmax_bf16", inputs=[attn_ty, np.int32, attn_ty])
        attn_v = external_func("attn_2", inputs=[attn_ty, qkv_1_ty, np.int32, score_ty])
        split_qkv = external_func("split_qkv", inputs=[np.int32, qkv_ty, qkv_1_ty, cache_ty, cache_ty])
        fill_zero = external_func("fill_zero", inputs=[score_ty])

        # Declare tiles
        Shim_0, Mem_0, CT_0, CT_1 = tile(1, 0), tile(1, 1), tile(1, 2), tile(1, 3)

        # Object FIFOs and links
        inOF_A_StoM = object_fifo("inOF_A_StoM", Shim_0, Mem_0, 6, a_mem_ty)
        inOF_W_StoM = object_fifo("inOF_W_StoM", Shim_0, Mem_0, 6, w_mem_ty)

        inOF_A_MtoC = object_fifo("inOF_A_MtoC", Mem_0, CT_0, 9, a_in_ty)
        inOF_W_MtoC = object_fifo("inOF_W_MtoC", Mem_0, CT_0, 9, w_in_ty)

        a_offsets = [r * embed_dim + c * tile_embed_dim for r in range(3) for c in range(embed_dim // tile_embed_dim)]
        w_offsets = [r * embed_dim * head_dim + c * tile_embed_dim * head_dim for r in range(3) for c in range(embed_dim // tile_embed_dim)]

        object_fifo_link(inOF_A_StoM, inOF_A_MtoC, [], a_offsets)
        object_fifo_link(inOF_W_StoM, inOF_W_MtoC, [], w_offsets)

        OF_freq = object_fifo(f"OF_freq", CT_0, CT_0, 1, head_div_2_ty)
        OF_sin = object_fifo(f"OF_sin", CT_0, CT_0, 1, head_div_2_ty)
        OF_cos = object_fifo(f"OF_cos", CT_0, CT_0, 1, head_div_2_ty)
        OF_QKV = object_fifo("OF_QKV", CT_0, CT_1, 1, qkv_ty)

        OF_Q = object_fifo("OF_Q", CT_1, CT_1, 1, qkv_1_ty)
        OF_attn1 = object_fifo("OF_attn1", CT_1, CT_1, 1, attn_ty)
        OF_sm = object_fifo("OF_sm", CT_1, CT_1, 1, attn_ty)

        outOF_score_CtoM = object_fifo("outOF_score_CtoM", CT_1, Mem_0, 1, score_ty)
        outOF_score_MtoS = object_fifo("outOF_score_MtoS", Mem_0, Shim_0, 1, score_ty)
        object_fifo_link(outOF_score_CtoM, outOF_score_MtoS)

        mem_k_buf = buffer(tile=CT_1, datatype=cache_ty, initial_value=mem_init, name="mem_k_buf")
        mem_v_buf = buffer(tile=CT_1, datatype=cache_ty, initial_value=mem_init, name="mem_v_buf")
        mem_k_cons_lock = lock(tile=CT_1, lock_id=0, init=1, sym_name="mem_k_cons_lock")
        mem_k_prod_lock = lock(tile=CT_1, lock_id=1, init=0, sym_name="mem_k_prod_lock")
        mem_v_cons_lock = lock(tile=CT_1, lock_id=2, init=1, sym_name="mem_v_cons_lock")
        mem_v_prod_lock = lock(tile=CT_1, lock_id=3, init=0, sym_name="mem_v_prod_lock")

        index_type = IndexType.get()
        mem_ty = MemRefType.get((48,), BF16Type.get())
        i32_type = IntegerType.get_signless(32)

        @core(CT_0, "mha.cc.o")
        def compute():
            q_off, k_off, v_off = arith.constant(0), arith.constant(head_dim), arith.constant(2*head_dim)
            for pos in range_(seq_len):
                pos_i32 = arith.IndexCastOp(i32_type, pos).result
                qkv_out = OF_QKV.acquire(ObjectFifoPort.Produce, 1)
                linalg_fill_bf16(q_off, qkv_out)
                # Vec-mat multiply for Q, K, V
                for offset in [q_off, k_off, v_off]:
                    for _ in range_(embed_dim // tile_embed_dim):
                        a_in = inOF_A_MtoC.acquire(ObjectFifoPort.Consume, 1)
                        w_in = inOF_W_MtoC.acquire(ObjectFifoPort.Consume, 1)
                        vecmat_bf16_bf16(offset, a_in, w_in, qkv_out)
                        inOF_A_MtoC.release(ObjectFifoPort.Consume, 1)
                        inOF_W_MtoC.release(ObjectFifoPort.Consume, 1)

                # Get frequency values
                freq_res = OF_freq.acquire(ObjectFifoPort.Produce, 1)
                freq_pos(pos_i32, freq_res)
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

        @core(CT_1, "mha.cc.o")
        def compute():
            for pos in range_(seq_len):
                pos_i32 = arith.IndexCastOp(i32_type, pos).result
                pos_p1 = arith.addi(pos, arith.constant(1, index_type))
                pos_p1_i32 = arith.IndexCastOp(i32_type, pos_p1).result
                # Split Q, K, V
                qkv_res = OF_QKV.acquire(ObjectFifoPort.Consume, 1)
                q_out = OF_Q.acquire(ObjectFifoPort.Produce, 1)
                use_lock(mem_k_cons_lock, LockAction.AcquireGreaterEqual, value=1)
                use_lock(mem_v_cons_lock, LockAction.AcquireGreaterEqual, value=1)
                split_qkv(pos_i32, qkv_res, q_out, mem_k_buf.buffer, mem_v_buf.buffer)
                use_lock(mem_k_prod_lock, LockAction.Release, value=1)
                use_lock(mem_v_prod_lock, LockAction.Release, value=1)
                OF_QKV.release(ObjectFifoPort.Consume, 1)
                OF_Q.release(ObjectFifoPort.Produce, 1)

                # Attn 1st Stage
                q_res = OF_Q.acquire(ObjectFifoPort.Consume, 1)
                attn_1 = OF_attn1.acquire(ObjectFifoPort.Produce, 1)
                fill_neg(attn_1)
                zero_idx = arith.constant(0, index_type)
                one_idx = arith.constant(1, index_type)
                head_dim_idx = arith.constant(head_dim, index_type)
                use_lock(mem_k_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                for i in range_(pos_p1):
                    i_32 = arith.IndexCastOp(i32_type, i).result
                    offset = arith.muli(i, head_dim_idx)
                    k_data = memref.subview(mem_k_buf.buffer, [offset], [48], [1])
                    k_data_ = memref.cast(mem_ty, k_data)
                    attn_qk(q_res, k_data_, i_32, attn_1)
                use_lock(mem_k_cons_lock, LockAction.Release, value=1)
                OF_Q.release(ObjectFifoPort.Consume, 1)
                OF_attn1.release(ObjectFifoPort.Produce, 1)

                # Softmax
                attn_1 = OF_attn1.acquire(ObjectFifoPort.Consume, 1)
                sm_out = OF_sm.acquire(ObjectFifoPort.Produce, 1)
                softmax_k(attn_1, pos_p1_i32, sm_out)
                OF_sm.release(ObjectFifoPort.Produce, 1)
                OF_attn1.release(ObjectFifoPort.Consume, 1)

                # Attn 2nd Stage
                sm_res = OF_sm.acquire(ObjectFifoPort.Consume, 1)
                score_out = outOF_score_CtoM.acquire(ObjectFifoPort.Produce, 1)
                fill_zero(score_out)
                use_lock(mem_v_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                for i in range_(pos_p1):
                    i_32 = arith.IndexCastOp(i32_type, i).result
                    offset = arith.muli(i, head_dim_idx)
                    v_data = memref.subview(mem_v_buf.buffer, [offset], [48], [1])
                    v_data_ = memref.cast(mem_ty, v_data)
                    attn_v(sm_res, v_data_, i_32, score_out)
                use_lock(mem_v_cons_lock, LockAction.Release, value=1)
                OF_sm.release(ObjectFifoPort.Consume, 1)
                outOF_score_CtoM.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(full_a_ty, full_w_ty, f_score_ty)
        def sequence(A, W, S):
            task_a = shim_dma_single_bd_task(inOF_A_StoM, A, sizes=[seq_len, 1, 3, embed_dim], 
                                                             strides=[3*embed_dim, 0, embed_dim, 1], issue_token=True)
            task_w = shim_dma_single_bd_task(inOF_W_StoM, W, sizes=[seq_len, 3, embed_dim, head_dim], 
                                                             strides=[3 * embed_dim * head_dim, embed_dim * head_dim, head_dim, 1], issue_token=True)
            task_s = shim_dma_single_bd_task(outOF_score_MtoS, S, sizes=[seq_len, 1, 1, head_dim], 
                                                                  strides=[head_dim, 0, 0, 1], issue_token=True)
            dma_start_task(task_a, task_w)
            dma_await_task(task_a, task_w)
            dma_start_task(task_s)
            dma_await_task(task_s)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embed-dim", type=int, required=True, help="Total embedding dimension.")
    p.add_argument("--head-dim", type=int, required=True, help="Dimension of a single attention head.")
    p.add_argument("--tile-dim", type=int, required=True, help="Embedding dimension processed per tile.")
    p.add_argument("--seq-len", type=int, required=True, help="Sequence length.")

    opts = p.parse_args()
    with mlir_mod_ctx() as ctx:
        mha(opts.embed_dim, opts.head_dim, opts.tile_dim, opts.seq_len)
        res = ctx.module.operation.verify()
        if res == True:
            print(ctx.module)
        else:
            print(res)