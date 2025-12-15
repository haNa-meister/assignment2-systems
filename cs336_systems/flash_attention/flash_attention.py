import triton
import triton.language as tl
import torch
import math

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Outer loop
    # Each block handles [Q_TILE_SIZE, D] queries
    # Aim to compute [Q_TILE_SIZE, D] output tile per block on Softmax(Q_tile * K_tile^T) * scale * V_tile
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # O_i+1 = O_i * alpha + P_i * V_i
    # O_tile: [Q_TILE_SIZE, D]
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # L_i+1 = L_i * alpha + sum(P_i)
    # L_tile: [Q_TILE_SIZE]
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    o_i = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,1))

    # K_tile: [K_TILE_SIZE, D]
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # V_tile: [K_TILE_SIZE, D]
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Inner loop for K_tile: [K_TILE_SIZE, D] from 0 to N_KEYS
    # Inner loop to Q_tile: [Q_TILE_SIZE, D] over each K_tile and V_tile
    for k_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_i = tl.load(K_block_ptr, boundary_check=(0,1))
        V_i = tl.load(V_block_ptr, boundary_check=(0,1))

        # Compute S_tile: [Q_TILE_SIZE, K_TILE_SIZE]
        # S_tile = Q_tile * K_tile^T
        S_i = tl.dot(Q_i.to(tl.float32), tl.trans(K_i.to(tl.float32))) * scale

        # Compute max for numerical stability
        # m_i_cur: [Q_TILE_SIZE]
        m_i_cur = tl.maximum(tl.max(S_i, axis=1), m_i)

        # P_tile = exp(S_tile - m_i_cur)
        # Since m_i_cur reshapes to [Q_TILE_SIZE, 1], S_tile - m_i_cur broadcasts correctly by axis 1
        # P_i: [Q_TILE_SIZE, K_TILE_SIZE]
        P_i = tl.exp(S_i - m_i_cur[:, None])

        # Update L_i and O_i
        # alpha: [Q_TILE_SIZE]
        alpha = tl.exp(m_i - m_i_cur)
        l_i = l_i * alpha + tl.sum(P_i, axis=1)
        o_i = o_i * alpha[:, None] + tl.dot(P_i, V_i.to(tl.float32))
        m_i = m_i_cur

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    # O_final = O_i / L_i
    o_final = o_i * (1 / l_i[:, None])
    l_i = tl.log(l_i) + m_i

    tl.store(O_block_ptr, o_final, boundary_check=(0,1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))

@triton.jit
def flash_bwd_kernel(
    # Input Pointers
    Q_ptr, K_ptr, V_ptr,
    DO_ptr,        # Gradient of Output (Incoming Grad)
    L_ptr,         # LogSumExp (Saved from Forward Pass)
    D_ptr,         # The "Delta" term: sum(DO * O) (Pre-calculated)
    # Output Pointers (Gradients to write)
    DQ_ptr, DK_ptr, DV_ptr,
    # Strides
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq, # Delta strides
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    # Dimensions
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # -----------------------------------------------------------
    # 1. Grid Setup (Parallelize over Keys/Values aka Columns)
    # -----------------------------------------------------------
    # In Forward we used program_id(0) for Queries. 
    # In Backward, we use it for Keys to optimize dK/dV accumulation.
    k_tile_index = tl.program_id(0) 
    batch_index = tl.program_id(1)

    # -----------------------------------------------------------
    # 2. Setup "Fixed" Pointers (The Owners)
    # -----------------------------------------------------------
    # We own this chunk of K and V. We will compute dK and dV for them.
    # Shape: [K_TILE_SIZE, D]
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0)
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0)
    )

    # Pointers for writing gradients dK and dV
    DK_block_ptr = tl.make_block_ptr(
        DK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D), strides=(stride_dkk, stride_dkd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0)
    )
    
    DV_block_ptr = tl.make_block_ptr(
        DV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D), strides=(stride_dvk, stride_dvd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0)
    )

    # -----------------------------------------------------------
    # 3. Setup "Iterating" Pointers (The loop variables)
    # -----------------------------------------------------------
    # We will loop over chunks of Q. 
    # Start at Q_row=0. We will .advance() these later.
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(0, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
    )

    DO_block_ptr = tl.make_block_ptr(
        DO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D), strides=(stride_doq, stride_dod),
        offsets=(0, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
    )
    
    # Pointer for atomic updates to dQ
    DQ_block_ptr = tl.make_block_ptr(
        DQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D), strides=(stride_dqq, stride_dqd),
        offsets=(0, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0)
    )

    # Stats pointers (L and Delta)
    # Note: These are 1D vectors per row
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(0,), block_shape=(Q_TILE_SIZE,), order=(0,)
    )
    
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,), strides=(stride_dq,),
        offsets=(0,), block_shape=(Q_TILE_SIZE,), order=(0,)
    )

    # -----------------------------------------------------------
    # 4. Load Fixed Data & Initialize Accumulators
    # -----------------------------------------------------------
    # We load K and V once and keep them in SRAM (transposed for dot products)
    k_tile = tl.load(K_block_ptr) # [K_TILE, D]
    v_tile = tl.load(V_block_ptr) # [K_TILE, D]
    
    # We need Transposed versions for matmuls later
    k_tile_t = tl.trans(k_tile)   # [D, K_TILE]
    v_tile_t = tl.trans(v_tile)   # [D, K_TILE]

    # Accumulators for dK and dV
    dk_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dv_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    # -----------------------------------------------------------
    # 5. Main Loop: Iterate over Queries (Rows)
    # -----------------------------------------------------------
    # We scan the entire sequence of Queries to see who attended to "Us" (Our Keys)
    
    # Note: Looping Q_TILE_SIZE steps at a time
    for start_q in range(0, N_QUERIES, Q_TILE_SIZE):
        
        # --- A. Load Data for this Q chunk ---
        q_tile = tl.load(Q_block_ptr, boundary_check=(0,1))   # [Q_TILE, D]
        do_tile = tl.load(DO_block_ptr, boundary_check=(0,1)) # [Q_TILE, D]
        l_tile = tl.load(L_block_ptr, boundary_check=(0,))    # [Q_TILE]
        d_tile = tl.load(D_block_ptr, boundary_check=(0,))    # [Q_TILE]

        # --- B. Recompute Attention Probabilities (P) ---
        # S = Q * K^T
        # Shape: [Q_TILE, D] @ [D, K_TILE] -> [Q_TILE, K_TILE]
        s_tile = tl.dot(q_tile, k_tile_t) * scale
        
        # P = exp(S - L)
        # Note: L must be broadcasted [Q_TILE, 1]
        p_tile = tl.exp(s_tile - l_tile[:, None]) # [Q_TILE, K_TILE]
        
        # --- C. Compute Gradient dV ---
        # dV += P^T * dO
        # Shape: [K_TILE, Q_TILE] @ [Q_TILE, D] -> [K_TILE, D]
        # We accumulate this locally because we own dV
        dv_acc += tl.dot(tl.trans(p_tile.to(k_tile.dtype)), do_tile)

        # --- D. Compute Gradient dP (Gradient of Softmax input) ---
        # dP = dO * V^T
        # Shape: [Q_TILE, D] @ [D, K_TILE] -> [Q_TILE, K_TILE]
        dp_tile = tl.dot(do_tile, v_tile_t)

        # --- E. Compute Gradient dS (Scores) ---
        # dS = P * (dP - Delta)
        # Delta is the "cost" of softmax normalization
        # D_tile must be broadcasted: [Q_TILE, 1]
        ds_tile = p_tile * (dp_tile - d_tile[:, None]) # [Q_TILE, K_TILE]
        
        # Apply scale (from Attention formula S = QK^T / sqrt(d))
        ds_tile = ds_tile.to(q_tile.dtype) * scale

        # --- F. Compute Gradient dK ---
        # dK += dS^T * Q
        # Shape: [K_TILE, Q_TILE] @ [Q_TILE, D] -> [K_TILE, D]
        # We accumulate locally because we own dK
        dk_acc += tl.dot(tl.trans(ds_tile), q_tile)

        # --- G. Compute Gradient dQ (The tricky one) ---
        # dQ = dS * K
        # Shape: [Q_TILE, K_TILE] @ [K_TILE, D] -> [Q_TILE, D]
        dq_tile = tl.dot(ds_tile, k_tile)

        # CRITICAL: We do NOT own dQ. Other K-blocks are calculating dQ for 
        # these same queries right now. We must use Atomic Add.
        tl.atomic_add(DQ_block_ptr, dq_tile, boundary_check=(0,1))

        # --- H. Advance Pointers for next Q chunk ---
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        DO_block_ptr = DO_block_ptr.advance((Q_TILE_SIZE, 0))
        DQ_block_ptr = DQ_block_ptr.advance((Q_TILE_SIZE, 0)) # Advance atomic pointer too
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    # -----------------------------------------------------------
    # 6. Store Accumulated Gradients (dK, dV)
    # -----------------------------------------------------------
    # After the loop, we have the full gradients for our chunk of Keys/Values
    tl.store(DK_block_ptr, dk_acc, boundary_check=(0,1))
    tl.store(DV_block_ptr, dv_acc, boundary_check=(0,1))

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Inputs must be on CUDA"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Inputs must be contiguous"

        batch_size, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape

        # Allocate output tensors
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, N_QUERIES), device=Q.device, dtype=Q.dtype)

        if D <= 32:
            Q_TILE_SIZE, K_TILE_SIZE = 128, 128
        elif D <= 64:
            Q_TILE_SIZE, K_TILE_SIZE = 64, 64
        else:
            Q_TILE_SIZE, K_TILE_SIZE = 32, 32

        # Q_TILE = N_QUWERIES / Q_TILE_SIZE
        Q_TILE = math.ceil(N_QUERIES / Q_TILE_SIZE)

        # Launch forward kernel
        flash_fwd_kernel[(Q_TILE, batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            D ** (-0.5),
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        # Save for backward
        ctx.save_for_backward(Q, K, V, O, L)

        return O

    @staticmethod
    def backward(ctx, DO):
        assert DO.is_cuda, "Inputs must be on CUDA"
        assert DO.is_contiguous(), "Inputs must be contiguous"

        Q, K, V, O, L = ctx.saved_tensors
        
        batch_size, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape

        # Allocate output tensors
        DQ = torch.empty_like(Q)
        DK = torch.empty_like(K)
        DV = torch.empty_like(V)

        # Precompute Delta term
        D_D = torch.sum(DO * O, dim=-1)  # Shape: [N_QUERIES]

        if D <= 32:
            Q_TILE_SIZE, K_TILE_SIZE = 128, 128
        elif D <= 64:
            Q_TILE_SIZE, K_TILE_SIZE = 64, 64
        else:
            Q_TILE_SIZE, K_TILE_SIZE = 32, 32

        # K_TILE = N_KEYS / K_TILE_SIZE
        K_TILE = math.ceil(N_KEYS / K_TILE_SIZE)

        # Launch backward kernel
        flash_bwd_kernel[(K_TILE, batch_size)](
            Q, K, V,
            DO,
            L,
            D_D,
            DQ, DK, DV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            DO.stride(0), DO.stride(1), DO.stride(2),
            L.stride(0), L.stride(1),
            D_D.stride(0), D_D.stride(1),
            DQ.stride(0), DQ.stride(1), DQ.stride(2),
            DK.stride(0), DK.stride(1), DK.stride(2),
            DV.stride(0), DV.stride(1), DV.stride(2),
            N_QUERIES, N_KEYS,
            D ** (-0.5),
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        return DQ, DK, DV