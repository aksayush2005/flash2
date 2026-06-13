import triton
import triton.language as tl
import torch

@triton.jit
def fwd_pass_inner(
    O_blk,
    l,
    m,
    Q_blk,
    K_blk_ptr,
    V_blk_ptr,
    blk_index_Q,
    softmax_scaling,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    mode: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,):

    if mode==1:
        low,high=0,blk_index_Q*BLOCK_SIZE_Q
    elif mode==2:
        low,high=blk_index_Q* BLOCK_SIZE_Q, (blk_index_Q + 1) * BLOCK_SIZE_Q
        low=tl.multiple_of(low,BLOCK_SIZE_Q) #for optimization
    else:
        low,high=0,SEQ_LEN
    
    K_blk_ptr = tl.advance(K_blk_ptr, (0, lo))
    V_blk_ptr = tl.advance(V_blk_ptr, (lo, 0)) 

    for kv_start in range(low,high,BLOCK_SIZE_KV):
        kv_start=tl.multiple_of(kv_start,BLOCK_SIZE_KV)
        K_blk=tl.load(K_blk_ptr)
        QK_blk=tl.dot(Q_blk,K_blk)
        
        if mode== 2:
            mask = offs_q[:, None] >= (kv_start + offs_kv[None, :])
            QK_blk = QK_blk * softmax_scaling+ tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m, tl.max(QK_blk, 1))
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m, tl.max(QK_block, 1) * softmax_scaling)
            QK_block = QK_block * softmax_scaling - m_ij[:, None]              

        
        P_blk = tl.math.exp(QK_blk)
        l_ij = tl.sum(P_blk,1)
        alpha = tl.math.exp(m - m_ij)
        l = l * alpha + l_ij
        V_blk = tl.load(V_blk_ptr)
        P_blk = P_blk.to(tl.float16)
        O_blk = O_blk * alpha[:, None]
        O_blk = tl.dot(P_blk, V_blk, O_blk) #O=O+PV

        m = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    return O_blk, l, m


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
        offsets=(blk_index_Q * BLOCK_SIZE_Q, 0),
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