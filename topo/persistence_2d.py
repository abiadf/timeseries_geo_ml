"""Sublevel Set Persistence of H0 on GPU + numba. Extrema detectino in torch, persistence in numpy+numba (cause its sequential)"""

from time import perf_counter
import torch
import numpy as np
from numba import njit, prange
from dataclasses import dataclass
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

# ✅ ====== BASE SCENARIO ========

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
def _sweep_h1_backward_surgical(edge_vals, edge_f1, edge_f2, face_vals, num_faces, pre_sorted_order, h1_pairs):
    EXTERIOR = num_faces
    h1_count = 0
    
    # Fast native array initialization
    parent_f = np.arange(num_faces + 1, dtype=np.int64)
    
    birth_val_f = np.empty(num_faces + 1, dtype=np.float32)
    birth_val_f[:num_faces] = face_vals
    birth_val_f[EXTERIOR] = np.inf

    for idx in pre_sorted_order:
        edge_val = edge_vals[idx]
        f1, f2 = edge_f1[idx], edge_f2[idx]

        root_f1 = f1
        while parent_f[root_f1] != root_f1:
            parent_f[root_f1] = parent_f[parent_f[root_f1]]
            root_f1 = parent_f[root_f1]
            
        root_f2 = f2
        while parent_f[root_f2] != root_f2:
            parent_f[root_f2] = parent_f[parent_f[root_f2]]
            root_f2 = parent_f[root_f2]

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

def compute_h0_h1_fast(grid: torch.Tensor):
    """Highly optimized H0/H1 sublevel persistence for 2D images."""
    # 1. Force typing on device upfront to allow zero-copy CPU transfers
    grid = grid.to(torch.float32)
    R, C = grid.shape
    device = grid.device
    num_pixels = R * C
    num_faces = (R - 1) * (C - 1)

    r_idx = torch.arange(R, device=device).view(-1, 1)
    c_idx = torch.arange(C, device=device).view(1, -1)
    pixel_ids = r_idx * C + c_idx

    # Horizontal Edges
    h_vals = torch.maximum(grid[:, :-1], grid[:, 1:]).reshape(-1)
    h_idx1 = pixel_ids[:, :-1].reshape(-1)
    h_idx2 = pixel_ids[:, 1:].reshape(-1)
    
    h_row = torch.arange(R, device=device).view(-1, 1).expand(R, C - 1).reshape(-1)
    h_col = torch.arange(C - 1, device=device).view(1, -1).expand(R, C - 1).reshape(-1)
    
    e_h_f1 = torch.where(h_row > 0, (h_row - 1) * (C - 1) + h_col, num_faces)
    e_h_f2 = torch.where(h_row < R - 1, h_row * (C - 1) + h_col, num_faces)

    # Vertical Edges
    v_vals = torch.maximum(grid[:-1, :], grid[1:, :]).reshape(-1)
    v_idx1 = pixel_ids[:-1, :].reshape(-1)
    v_idx2 = pixel_ids[1:, :].reshape(-1)
    
    v_row = torch.arange(R - 1, device=device).view(-1, 1).expand(R - 1, C).reshape(-1)
    v_col = torch.arange(C, device=device).view(1, -1).expand(R - 1, C).reshape(-1)
    
    e_v_f1 = torch.where(v_col > 0, v_row * (C - 1) + (v_col - 1), num_faces)
    e_v_f2 = torch.where(v_col < C - 1, v_row * (C - 1) + v_col, num_faces)

    # Concatenate structures on device
    edge_vals = torch.cat([h_vals, v_vals])
    e_idx1 = torch.cat([h_idx1, v_idx1]).to(torch.int64)
    e_idx2 = torch.cat([h_idx2, v_idx2]).to(torch.int64)
    edge_f1 = torch.cat([e_h_f1, e_v_f1]).to(torch.int64)
    edge_f2 = torch.cat([e_h_f2, e_v_f2]).to(torch.int64)

    face_vals = torch.amax(torch.stack([
        grid[:-1, :-1], grid[:-1, 1:],
        grid[1:, :-1], grid[1:, 1:]
    ], dim=0), dim=0).reshape(-1)

    # Parallelized Device Sort
    e_order = torch.argsort(edge_vals)
    rev_e_order = e_order.flip(dims=[0]).to(torch.int64)

    indices_stack = torch.stack([e_idx1[e_order], e_idx2[e_order]], dim=0)
    
    # 2. Unified Host Migration (Zero-Copy Transfer)
    pix_vals_np = grid.reshape(-1).cpu().numpy()
    face_vals_np = face_vals.cpu().numpy()
    edge_vals_sorted_np = edge_vals[e_order].cpu().numpy()
    indices_np = indices_stack.cpu().numpy()
    
    edge_vals_raw_np = edge_vals.cpu().numpy()
    edge_f1_np = edge_f1.cpu().numpy()
    edge_f2_np = edge_f2.cpu().numpy()
    rev_order_np = rev_e_order.cpu().numpy()

    # Allocations
    num_edges = len(edge_vals_sorted_np)
    h0_out = np.empty((num_edges, 2), dtype=np.float32)
    h1_pairs = np.empty((num_edges, 2), dtype=np.float32)

    # Pass 1: Forward H0
    h0_count = _sweep_h0_forward(
        edge_vals_sorted_np, indices_np[0], indices_np[1], 
        pix_vals_np, num_pixels, h0_out)
    
    # Pass 2: Surgical Backward H1 Sweep
    h1_count = _sweep_h1_backward_surgical(
        edge_vals_raw_np, edge_f1_np, edge_f2_np, 
        face_vals_np, num_faces, rev_order_np, h1_pairs)

    h0 = h0_out[:h0_count]
    h1 = h1_pairs[:h1_count]
    h0 = h0[h0[:, 1] > h0[:, 0]]
    h1 = h1[h1[:, 1] > h1[:, 0]]

    global_min = np.array([[pix_vals_np.min(), np.inf]], dtype=np.float32)
    return np.concatenate([h0, global_min]), h1

# 🚰 ===== STREAMING SCENARIO ======

@njit(cache=True)
def _flush_h0_numba(h0_edges, parent, birth_val, pix_vals, h0_pairs_out):
    h0_count = 0
    for i in range(len(h0_edges)):
        val, u, v = h0_edges[i, 0], int(h0_edges[i, 1]), int(h0_edges[i, 2])
        
        if parent[u] == -1:
            parent[u] = u
            birth_val[u] = pix_vals[u]
        if parent[v] == -1:
            parent[v] = v
            birth_val[v] = pix_vals[v]

        root_u = _root(parent, u)
        root_v = _root(parent, v)

        if root_u != root_v:
            if birth_val[root_u] <= birth_val[root_v]:
                survivor, victim = root_u, root_v
            else:
                survivor, victim = root_v, root_u

            h0_pairs_out[h0_count, 0] = birth_val[victim]
            h0_pairs_out[h0_count, 1] = val
            h0_count += 1
            parent[victim] = survivor
    return h0_count

@njit(cache=True)
def _flush_h1_numba(h1_edges, parent_f, birth_val_f, face_vals, exterior, h1_pairs_out):
    h1_count = 0
    for i in range(len(h1_edges)):
        neg_val, f1, f2 = h1_edges[i, 0], int(h1_edges[i, 1]), int(h1_edges[i, 2])
        val = -neg_val
        
        if parent_f[f1] == -1:
            parent_f[f1] = f1
            birth_val_f[f1] = np.inf if f1 == exterior else face_vals[f1]
        if parent_f[f2] == -1:
            parent_f[f2] = f2
            birth_val_f[f2] = np.inf if f2 == exterior else face_vals[f2]

        root_f1 = _root(parent_f, f1)
        root_f2 = _root(parent_f, f2)

        if root_f1 != root_f2:
            if birth_val_f[root_f1] >= birth_val_f[root_f2]:
                survivor, victim = root_f1, root_f2
            else:
                survivor, victim = root_f2, root_f1

            if victim != exterior:
                h1_pairs_out[h1_count, 0] = val
                h1_pairs_out[h1_count, 1] = birth_val_f[victim]
                h1_count += 1

            parent_f[victim] = survivor
    return h1_count

class StreamingPersistentBasinForest:
    def __init__(self, num_pixels, num_faces, pix_vals, face_vals):
        self.num_pixels = num_pixels
        self.num_faces = num_faces
        self.exterior = num_faces
        self.pix_vals = pix_vals
        self.face_vals = face_vals

        self.parent = np.full(num_pixels, -1, dtype=np.int64)
        self.birth_val = np.zeros(num_pixels, dtype=np.float32)

        self.parent_f = np.full(num_faces + 1, -1, dtype=np.int64)
        self.birth_val_f = np.zeros(num_faces + 1, dtype=np.float32)

        self.h0_edges = np.empty((0, 3), dtype=np.float32)
        self.h1_edges = np.empty((0, 3), dtype=np.float32)

        self.h0_pairs = np.empty((0, 2), dtype=np.float32)
        self.h1_pairs = np.empty((0, 2), dtype=np.float32)

    def flush_accumulated_edges(self):
        """Processes collected array states into global topological persistence components."""
        if len(self.h0_edges) > 0:
            h0_arr = self.h0_edges[self.h0_edges[:, 0].argsort()]
            self.h0_edges = np.empty((0, 3), dtype=np.float32)
            
            h0_out = np.empty((len(h0_arr), 2), dtype=np.float32)
            h0_count = _flush_h0_numba(h0_arr, self.parent, self.birth_val, self.pix_vals, h0_out)
            self.h0_pairs = h0_out[:h0_count]

        if len(self.h1_edges) > 0:
            h1_arr = self.h1_edges[self.h1_edges[:, 0].argsort()]
            self.h1_edges = np.empty((0, 3), dtype=np.float32)

            h1_out = np.empty((len(h1_arr), 2), dtype=np.float32)
            h1_count = _flush_h1_numba(h1_arr, self.parent_f, self.birth_val_f, self.face_vals, self.exterior, h1_out)
            self.h1_pairs = h1_out[:h1_count]

    def get_diagrams(self):
        h0, h1 = self.h0_pairs, self.h1_pairs
        if len(h0) > 0: h0 = h0[h0[:, 1] > h0[:, 0]]
        if len(h1) > 0: h1 = h1[h1[:, 1] > h1[:, 0]]
        if len(self.pix_vals) > 0:
            global_min = np.array([[self.pix_vals.min(), np.inf]], dtype=np.float32)
            h0 = np.concatenate([h0, global_min]) if len(h0) > 0 else global_min
        return h0, h1

@njit(parallel=True, cache=True)
def _prepare_streaming_data_numba(grid, split_row):
    R, C = grid.shape
    num_pixels = R * C
    num_faces = (R - 1) * (C - 1)
    pix_vals = grid.ravel()

    # 1. Compute face values
    face_vals = np.empty(num_faces, dtype=np.float32)
    for r in prange(R - 1):
        for c in range(C - 1):
            f_idx = r * (C - 1) + c
            v1 = grid[r, c]
            v2 = grid[r, c + 1]
            v3 = grid[r + 1, c]
            v4 = grid[r + 1, c + 1]
            face_vals[f_idx] = max(max(v1, v2), max(v3, v4))

    # 2. Count exact streaming segment allocations
    c1_h, c2_h, st_h = 0, 0, 0
    for r in range(R):
        if r < split_row:
            c1_h += C - 1
        elif r > split_row:
            c2_h += C - 1
        else:
            st_h += C - 1

    c1_v, c2_v, st_v = 0, 0, 0
    for r in range(R - 1):
        if r < split_row - 1:
            c1_v += C
        elif r >= split_row:
            c2_v += C
        else:
            st_v += C

    total_h0_h1 = (c1_h + c1_v) + (c2_h + c2_v) + (st_h + st_v)

    # 3. Pre-allocate continuous flat output evaluation matrices
    stream_h0 = np.empty((total_h0_h1, 3), dtype=np.float32)
    stream_h1 = np.empty((total_h0_h1, 3), dtype=np.float32)

    # Offsets for block packing: [Chunk 1 | Chunk 2 | Stitch]
    c1_ptr = 0
    c2_ptr = c1_h + c1_v
    st_ptr = c2_ptr + c2_h + c2_v

    # 4. Fill Horizontal Edges in parallel segments safely
    for r in prange(R):
        # Calculate destination pointer offsets based on the split row
        if r < split_row:
            h0_idx = c1_ptr + r * (C - 1)
        elif r > split_row:
            h0_idx = c2_ptr + (r - split_row - 1) * (C - 1)
        else:
            h0_idx = st_ptr

        for c in range(C - 1):
            u = r * C + c
            v = u + 1
            val = max(grid[r, c], grid[r, c + 1])

            f1 = (r - 1) * (C - 1) + c if r > 0 else num_faces
            f2 = r * (C - 1) + c if r < R - 1 else num_faces

            idx = h0_idx + c
            stream_h0[idx, 0] = val
            stream_h0[idx, 1] = float(u)
            stream_h0[idx, 2] = float(v)

            stream_h1[idx, 0] = -val
            stream_h1[idx, 1] = float(f1)
            stream_h1[idx, 2] = float(f2)

    # 5. Fill Vertical Edges
    # Account for horizontal offsets when packing the rest of the layout segments
    v_c1_start = c1_ptr + c1_h
    v_c2_start = c2_ptr + c2_h
    v_st_start = st_ptr + st_h

    for r in prange(R - 1):
        if r < split_row - 1:
            v0_idx = v_c1_start + r * C
        elif r >= split_row:
            v0_idx = v_c2_start + (r - split_row) * C
        else:
            v0_idx = v_st_start

        for c in range(C):
            u = r * C + c
            v = u + C
            val = max(grid[r, c], grid[r + 1, c])

            f1 = r * (C - 1) + (c - 1) if c > 0 else num_faces
            f2 = r * (C - 1) + c if c < C - 1 else num_faces

            idx = v0_idx + c
            stream_h0[idx, 0] = val
            stream_h0[idx, 1] = float(u)
            stream_h0[idx, 2] = float(v)

            stream_h1[idx, 0] = -val
            stream_h1[idx, 1] = float(f1)
            stream_h1[idx, 2] = float(f2)
    return num_pixels, num_faces, pix_vals, face_vals, stream_h0, stream_h1

def prepare_streaming_data_numba_wrapper(grid_tensor: torch.Tensor, split_row: int):
    """Python wrapper to clean entry/exit points for Numba engine processing."""
    grid = grid_tensor.detach().cpu().numpy().astype(np.float32)

    num_pixels, num_faces, pix_vals, face_vals, stream_h0, stream_h1 = (
        _prepare_streaming_data_numba(grid, split_row))

    return {
        "metadata": {
            "num_pixels": num_pixels,
            "num_faces": num_faces,
            "exterior": num_faces,},
        "pix_vals": pix_vals,
        "face_vals": face_vals,
        "stream_h0": stream_h0,
        "stream_h1": stream_h1,}

def run_streaming_persistence(data: dict):
    """Ingests pre-allocated array segments instantly without loop iterations.
    Pure streaming execution path."""
    meta = data["metadata"]
    
    stream_engine = StreamingPersistentBasinForest(
        num_pixels=meta["num_pixels"], num_faces=meta["num_faces"], 
        pix_vals=data["pix_vals"], face_vals=data["face_vals"])

    # Load data directly into staging variables
    stream_engine.h0_edges = data["stream_h0"]
    stream_engine.h1_edges = data["stream_h1"]

    # Compute persistence pairs and clear the edge arrays
    stream_engine.flush_accumulated_edges()
    return stream_engine.get_diagrams()

