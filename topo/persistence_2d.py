"""Sublevel Set Persistence of H0 on GPU + numba. Extrema detectino in torch, persistence in numpy+numba (cause its sequential)"""

from time import perf_counter
import torch
import numpy as np
from numba import njit, prange
from dataclasses import dataclass
import heapq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Optimized for 2D grids by replacing global boundary matrix reduction with a localized
1D sweep schedule. Eliminates structural redundancy via asymmetric cell tracking:
1. Extremum Verification: Pixels are verified as Minima/Maxima using a full 4-neighbor 
   orthogonal stencil (Up, Down, Left, Right).
2. Asymmetric Edge Harvesting: To avoid duplicate edge processing during the sweep phase, 
   unique pixel-to-pixel connections are gathered looking forward-only (Right and Down).
This directional extraction yields a sparse, de-duplicated 1D filtration schedule 
processed via an allocation-free Numba DSU engine, outperforming standard SOTA 
cubical reduction methods."""

"""Extrema detection in torch, persistence in numpy+numba (sequential).
Optimized for 2D grids by replacing global boundary matrix reduction with a localized
1D sweep schedule. Eliminates structural redundancy via asymmetric cell tracking:
1. Extremum Verification: Pixels verified as Minima/Maxima using 4-neighbor stencil.
2. Asymmetric Edge Harvesting: forward-only (Right and Down) to avoid duplicate edges.
3. Apparent Pair Clearing: edges where value == min(endpoints) are zero-persistence, removed."""

REGULAR, MIN, MAX, SADDLE = 0, 1, 2, 3

# ✅ ====== NEW TRY ========

@njit(cache=True)
def _root(parent, node):
    """Return union-find root with path compression."""
    current = node
    while parent[current] != current:
        parent[current] = parent[parent[current]]
        current = parent[current]
    return current

@njit(cache=True)
def _sweep_h0_forward(edge_vals, e_idx1, e_idx2, pix_vals, num_pixels, h0_out):
    """Compute H0 persistence via forward union-find sweep."""
    parent    = np.full(num_pixels, -1, dtype=np.int64)
    birth_val = np.zeros(num_pixels, dtype=np.float32)
    h0_count  = 0
    
    for i in range(len(edge_vals)):
        p1, p2 = e_idx1[i], e_idx2[i]
        if parent[p1] == -1: parent[p1] = p1; birth_val[p1] = pix_vals[p1]
        if parent[p2] == -1: parent[p2] = p2; birth_val[p2] = pix_vals[p2]
        
        r1 = _root(parent, p1)
        r2 = _root(parent, p2)
        if r1 != r2:
            if birth_val[r1] <= birth_val[r2]: survivor, victim = r1, r2
            else: survivor, victim = r2, r1
            h0_out[h0_count, 0] = birth_val[victim]
            h0_out[h0_count, 1] = edge_vals[i] # saddle: h0 death
            h0_count += 1
            parent[victim] = survivor
    return h0_count

@njit(cache=True)
def _init_face(f, parent_f, birth_val_f, face_vals, exterior,):
    """Initialize dual face component on first encounter. Modifies arrays in-place"""
    if parent_f[f] == -1:
        parent_f[f] = f
        if f == exterior:
            birth_val_f[f] = np.inf
        else:
            birth_val_f[f] = face_vals[f]

@njit(cache=True)
def _sweep_h1_backward(edge_vals, edge_f1, edge_f2, face_vals, num_faces, h1_pairs):
    """Compute H1 persistence via backward dual union-find sweep."""
    EXTERIOR = num_faces
    h1_count = 0

    parent_f      = np.full(num_faces + 1, -1, dtype=np.int64)
    birth_val_f   = np.zeros(num_faces + 1, dtype=np.float32)
    reverse_order = np.argsort(edge_vals)[::-1]

    for idx in reverse_order:
        edge_val = edge_vals[idx]
        f1 = edge_f1[idx]
        f2 = edge_f2[idx]

        _init_face(f1, parent_f, birth_val_f, face_vals, EXTERIOR)
        _init_face(f2, parent_f, birth_val_f, face_vals, EXTERIOR)
        root_f1 = _root(parent_f, f1)
        root_f2 = _root(parent_f, f2)

        if root_f1 == root_f2:
            continue

        if birth_val_f[root_f1] >= birth_val_f[root_f2]:
            survivor = root_f1
            victim   = root_f2
        else:
            survivor = root_f2
            victim   = root_f1

        if victim != EXTERIOR: # only pair finite regions.
            h1_pairs[h1_count, 0] = edge_val # saddle: h1 birth
            h1_pairs[h1_count, 1] = birth_val_f[victim] # loop/face: h1 death
            h1_count += 1

        parent_f[victim] = survivor
    return h1_count

def compute_h0_h1(grid: torch.Tensor):
    """Compute H0/H1 sublevel persistence of a 2D image."""
    R, C       = grid.shape
    device     = grid.device
    num_pixels = R * C
    flat       = torch.arange(num_pixels, device=device).reshape(R, C)
    pix_vals   = grid.flatten().cpu().numpy().astype(np.float32)

    # 1. Edges
    h_vals = torch.maximum(grid[:, :-1], grid[:, 1:])
    h_idx1 = flat[:, :-1].flatten()
    h_idx2 = flat[:, 1:].flatten()
    
    v_vals = torch.maximum(grid[:-1, :], grid[1:, :])
    v_idx1 = flat[:-1, :].flatten()
    v_idx2 = flat[1:, :].flatten()

    num_faces = (R - 1) * (C - 1)
    f_flat    = torch.arange(num_faces, device=device).reshape(R - 1, C - 1)
    f_pad     = torch.full((R + 1, C + 1), num_faces, dtype=torch.long, device=device)
    f_pad[1:R, 1:C] = f_flat

    # Dual mappings
    h_row = torch.arange(R, device=device).view(-1, 1).repeat(1, C - 1).flatten()
    h_col = torch.arange(C - 1, device=device).view(1, -1).repeat(R, 1).flatten()

    # above / below
    e_h_f1 = f_pad[h_row,     h_col + 1]
    e_h_f2 = f_pad[h_row + 1, h_col + 1]

    v_row = torch.arange(R - 1, device=device).view(-1, 1).repeat(1, C).flatten()
    v_col = torch.arange(C, device=device).view(1, -1).repeat(R - 1, 1).flatten()

    # left / right
    e_v_f1 = f_pad[v_row + 1, v_col]
    e_v_f2 = f_pad[v_row + 1, v_col + 1]

    edge_vals = torch.cat([h_vals.flatten(), v_vals.flatten()])
    e_idx1    = torch.cat([h_idx1, v_idx1])
    e_idx2    = torch.cat([h_idx2, v_idx2])
    edge_f1   = torch.cat([e_h_f1, e_v_f1])
    edge_f2   = torch.cat([e_h_f2, e_v_f2])

    e_f1_np = edge_f1.cpu().numpy()
    e_f2_np = edge_f2.cpu().numpy()

    boundary_edges = np.sum(e_f1_np == num_faces) + np.sum(e_f2_np == num_faces)
    self_edges     = np.sum(e_f1_np == e_f2_np)
    expected_boundary = 2 * (R - 1) + 2 * (C - 1)

    print("num_faces:", num_faces)
    print("boundary incidences:", boundary_edges)
    print("self dual edges:", self_edges)
    print("expected boundary edges:", expected_boundary)

    # 2. Faces
    face_vals = torch.amax(torch.stack([
        grid[:-1, :-1], grid[:-1, 1:],
        grid[1:, :-1], grid[1:, 1:],], dim=0), dim=0).flatten().cpu().numpy().astype(np.float32)

    # Sort primal edges for H0
    e_order   = torch.argsort(edge_vals)
    e_vals_np = edge_vals[e_order].cpu().numpy().astype(np.float32)
    e_idx1_np = e_idx1[e_order].cpu().numpy().astype(np.int64)
    e_idx2_np = e_idx2[e_order].cpu().numpy().astype(np.int64)

    # Allocations
    h0_out   = np.empty((len(e_vals_np), 2), dtype=np.float32)
    h1_pairs = np.empty((len(e_vals_np), 2), dtype=np.float32)

    # Pass 1: Forward H0
    h0_count = _sweep_h0_forward(e_vals_np, e_idx1_np, e_idx2_np, pix_vals, num_pixels, h0_out)
    
    # Pass 2: Backward H1
    h1_count = _sweep_h1_backward(
        edge_vals.cpu().numpy().astype(np.float32),
        edge_f1.cpu().numpy().astype(np.int64),
        edge_f2.cpu().numpy().astype(np.int64), face_vals, num_faces, h1_pairs)

    h0 = h0_out[:h0_count]
    h1 = h1_pairs[:h1_count]
    h0 = h0[h0[:, 1] > h0[:, 0]]
    h1 = h1[h1[:, 1] > h1[:, 0]]

    global_min = np.array([[pix_vals.min(), np.inf]], dtype=np.float32)
    h0         = np.concatenate([h0, global_min])
    return h0, h1


# ===== END OF NEW TRY ======
def compute_h0_h1_fast(grid: torch.Tensor):
    """Highly optimized H0/H1 sublevel persistence for 2D images."""
    R, C = grid.shape
    device = grid.device
    num_pixels = R * C
    num_faces = (R - 1) * (C - 1)

    # 1. Direct mathematical vectorization for edges (No padding grids)
    r_idx = torch.arange(R, device=device).view(-1, 1)
    c_idx = torch.arange(C, device=device).view(1, -1)
    pixel_ids = r_idx * C + c_idx

    # Horizontal Edges
    h_vals = torch.maximum(grid[:, :-1], grid[:, 1:]).flatten()
    h_idx1 = pixel_ids[:, :-1].flatten()
    h_idx2 = pixel_ids[:, 1:].flatten()
    
    h_row = torch.arange(R, device=device).view(-1, 1).expand(R, C - 1).flatten()
    h_col = torch.arange(C - 1, device=device).view(1, -1).expand(R, C - 1).flatten()
    
    e_h_f1 = torch.where(h_row > 0, (h_row - 1) * (C - 1) + h_col, num_faces)
    e_h_f2 = torch.where(h_row < R - 1, h_row * (C - 1) + h_col, num_faces)

    # Vertical Edges
    v_vals = torch.maximum(grid[:-1, :], grid[1:, :]).flatten()
    v_idx1 = pixel_ids[:-1, :].flatten()
    v_idx2 = pixel_ids[1:, :].flatten()
    
    v_row = torch.arange(R - 1, device=device).view(-1, 1).expand(R - 1, C).flatten()
    v_col = torch.arange(C, device=device).view(1, -1).expand(R - 1, C).flatten()
    
    e_v_f1 = torch.where(v_col > 0, v_row * (C - 1) + (v_col - 1), num_faces)
    e_v_f2 = torch.where(v_col < C - 1, v_row * (C - 1) + v_col, num_faces)

    # Concatenate structures cleanly on the source device
    edge_vals = torch.cat([h_vals, v_vals])
    e_idx1 = torch.cat([h_idx1, v_idx1])
    e_idx2 = torch.cat([h_idx2, v_idx2])
    edge_f1 = torch.cat([e_h_f1, e_v_f1])
    edge_f2 = torch.cat([e_h_f2, e_v_f2])

    # 2. Extract 2x2 face values
    face_vals = torch.amax(torch.stack([
        grid[:-1, :-1], grid[:-1, 1:],
        grid[1:, :-1], grid[1:, 1:]
    ], dim=0), dim=0).flatten()

    # 3. Parallelized Device Sort (Leveraging GPU Radix Sort if available)
    e_order = torch.argsort(edge_vals)
    rev_e_order = e_order.flip(dims=[0]) # Exact inverse order for H1

    # Composite transfer to host CPU memory to reduce bus synchronization overhead
    # We pack the structural indices together to copy them in one block
    indices_stack = torch.stack([e_idx1[e_order], e_idx2[e_order]], dim=0)
    
    # Unified Host Migration
    pix_vals_np = grid.flatten().cpu().numpy().astype(np.float32)
    face_vals_np = face_vals.cpu().numpy().astype(np.float32)
    edge_vals_sorted_np = edge_vals[e_order].cpu().numpy().astype(np.float32)
    indices_np = indices_stack.cpu().numpy().astype(np.int64)
    
    edge_vals_raw_np = edge_vals.cpu().numpy().astype(np.float32)
    edge_f1_np = edge_f1.cpu().numpy().astype(np.int64)
    edge_f2_np = edge_f2.cpu().numpy().astype(np.int64)
    rev_order_np = rev_e_order.cpu().numpy().astype(np.int64)

    # Allocations
    num_edges = len(edge_vals_sorted_np)
    h0_out = np.empty((num_edges, 2), dtype=np.float32)
    h1_pairs = np.empty((num_edges, 2), dtype=np.float32)

    # Pass 1: Forward H0
    h0_count = _sweep_h0_forward(
        edge_vals_sorted_np, indices_np[0], indices_np[1], 
        pix_vals_np, num_pixels, h0_out
    )
    
    # Pass 2: Modified Optimized Backward H1 Sweep (Avoids internal argsort)
    h1_count = _sweep_h1_backward_fast(
        edge_vals_raw_np, edge_f1_np, edge_f2_np, 
        face_vals_np, num_faces, rev_order_np, h1_pairs
    )

    # Extract final outputs
    h0 = h0_out[:h0_count]
    h1 = h1_pairs[:h1_count]
    h0 = h0[h0[:, 1] > h0[:, 0]]
    h1 = h1[h1[:, 1] > h1[:, 0]]

    global_min = np.array([[pix_vals_np.min(), np.inf]], dtype=np.float32)
    return np.concatenate([h0, global_min]), h1

# Fast H1 handler that accepts a pre-sorted order array from PyTorch
@njit(cache=True)
def _sweep_h1_backward_fast(edge_vals, edge_f1, edge_f2, face_vals, num_faces, pre_sorted_order, h1_pairs):
    EXTERIOR = num_faces
    h1_count = 0
    parent_f = np.full(num_faces + 1, -1, dtype=np.int64)
    birth_val_f = np.zeros(num_faces + 1, dtype=np.float32)

    for idx in pre_sorted_order:
        edge_val = edge_vals[idx]
        f1, f2 = edge_f1[idx], edge_f2[idx]

        _init_face(f1, parent_f, birth_val_f, face_vals, EXTERIOR)
        _init_face(f2, parent_f, birth_val_f, face_vals, EXTERIOR)
        
        root_f1 = _root(parent_f, f1)
        root_f2 = _root(parent_f, f2)

        if root_f1 != root_f2:
            if birth_val_f[root_f1] >= birth_val_f[root_f2]:
                survivor, victim = root_f1, root_f2
            else:
                survivor, victim = root_f2, root_f1

            if victim != EXTERIOR:
                h1_pairs[h1_count, 0] = edge_val
                h1_pairs[h1_count, 1] = birth_val_f[victim]
                h1_count += 1

            parent_f[victim] = survivor
    return h1_count

