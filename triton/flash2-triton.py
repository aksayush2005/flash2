import triton
import triton.language as tl
import torch

##@triton.jit
##def fwd_pass_inner()

@triton.jit
def fwd_pass(
    Q,K,V, #BATCH_SIZE,NO. OF HEADS,SEQ_LEN,HEAD_DIM
    M,O,   #BATCH_SIZE,NO. OF HEADS,SEQ_LEN
    softmax_scaling,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS:tl.constexpr,
    SEQ_LEN:tl.consexpr,
    HEAD_DIM:tl.constexpr,
    BLOCK_SIZE_Q:tl.constexpr,
    BLOCK_SIZE_KV:tl.constexpr,
    mode:tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV<=HEAD_DIM) #For compiler performance
    blk_index_Q=tl.program_id(0)
    batch_head_index=tl.program_id(1)
    batch_index=batch_head_index//NUM_HEADS
    head_index=batch_head_index%NUM_HEADS
    qkv_offset=(batch_index.to(tl.int64)*stride_Q_batch+head_index.to(tl.int64)*stride_Q_head)
    
    Q_blk_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(blk_index_Q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0), #For compiler performance
    )

    K_blk_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim,stride_K_seq, ), 
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    V_blk_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )
    
    O_blk_ptr = tl.make_block_ptr(
        base=O + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_Q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    offs_q = blk_index_Q* BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    m=tl.zeros([BLOCK_SIZE_Q],dtype=tl.float32)-float("inf")
    l=tl.zeros([BLOCK_SIZE_Q],dtype=tl.float32)+1.0
    O_blk=tl.zeros([BLOCK_SIZE_Q,HEAD_DIM],dtype=tl.float32)

    Q_blk=tl.load(Q_blk_ptr)
    #mode 3 if causal,else 1
    if mode==1 or mode==3:
        O_blk,l,m = fwd_pass_inner(
            O_blk,
            l,
            m,
            Q_blk,
            K_blk_ptr,
            V_blk_ptr,
            blk_index_Q,
            softmax_scaling,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - mode,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
        if mode == 3:
            O_blk, l_i, m_i = fwd_pass_inner(
                O_blk,
                l,
                m,
                Q_blk,
                K_blk_ptr,
                V_blk_ptr,
                blk_index_Q,
                softmax_scaling,
                BLOCK_SIZE_Q,
                BLOCK_SIZE_KV,
                2,
                offs_q,
                offs_kv,
                SEQ_LEN,
             )
    m=m+tl.math.log(l)
    O_blk=O_blk/l[:,None]
    m_ptrs=M+batch_head_index*SEQ_LEN+offs_q
    tl.store(m_ptrs,m)
    tl.store(O_blk_ptr,O_blk.to(O.type.element_ty))