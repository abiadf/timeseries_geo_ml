"""Sublevel Set Persistence of H0 on GPU + numba. Extrema detectino in torch, persistence in numpy+numba (cause its sequential)"""

from time import perf_counter
import torch
import numpy as np
from numba import njit, prange
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

GMIN, MIN, EDGE, MAX, GMAX = 1, 2, 3, 4, 5


def compute_2d_persistence(grid: torch.Tensor):
    R, C = grid.shape
    device = grid.device
    num_pixels = R * C

    gmin_val = grid.min()
    gmax_val = grid.max()

    # 1. Minima/Maxima detection (4-neighbor stencil)
    padded_min = torch.nn.functional.pad(grid, (1, 1, 1, 1), mode='constant', value=float('inf'))
    is_min = (grid <= padded_min[0:-2, 1:-1]) & \
             (grid <= padded_min[2:,   1:-1]) & \
             (grid <= padded_min[1:-1, 0:-2]) & \
             (grid <= padded_min[1:-1, 2:  ])

    padded_max = torch.nn.functional.pad(grid, (1, 1, 1, 1), mode='constant', value=float('-inf'))
    is_max = (grid >= padded_max[0:-2, 1:-1]) & \
             (grid >= padded_max[2:,   1:-1]) & \
             (grid >= padded_max[1:-1, 0:-2]) & \
             (grid >= padded_max[1:-1, 2:  ])

    type_grid = torch.zeros((R, C), dtype=torch.int32, device=device)
    type_grid[is_min] = MIN
    type_grid[is_max] = MAX
    # Global min/max override (they are always critical)
    type_grid[grid == gmin_val] = GMIN
    type_grid[grid == gmax_val] = GMAX

    flat_indices = torch.arange(num_pixels, device=device).reshape(R, C)
    min_mask = (type_grid == MIN) | (type_grid == GMIN)
    max_mask = (type_grid == MAX) | (type_grid == GMAX)

    min_flat_idx = flat_indices[min_mask]
    max_flat_idx = flat_indices[max_mask]

    # 2. Forward-only edge harvesting (right + down only, no duplicates)
    h_vals = torch.maximum(grid[:, :-1], grid[:, 1:])
    h_idx1 = flat_indices[:, :-1].flatten()
    h_idx2 = flat_indices[:, 1: ].flatten()

    v_vals = torch.maximum(grid[:-1, :], grid[1:, :])
    v_idx1 = flat_indices[:-1, :].flatten()
    v_idx2 = flat_indices[1:,  :].flatten()

    # 3. Apparent pair clearing:
    # Keep edge only if its value > min(endpoints), i.e. not a zero-persistence pair.
    h_keep = grid[:, :-1] != grid[:, 1:]
    v_keep = grid[:-1, :] != grid[1:, :]

    h_vals_f = h_vals[h_keep]
    h_idx1_f = h_idx1[h_keep.flatten()]
    h_idx2_f = h_idx2[h_keep.flatten()]

    v_vals_f = v_vals[v_keep]
    v_idx1_f = v_idx1[v_keep.flatten()]
    v_idx2_f = v_idx2[v_keep.flatten()]

    # 4. Build flat filtration arrays
    n_min   = len(min_flat_idx)
    n_max   = len(max_flat_idx)
    n_edges = len(h_vals_f) + len(v_vals_f)
    total   = n_min + n_max + n_edges

    vals_np  = np.empty(total, dtype=np.float32)
    types_np = np.empty(total, dtype=np.int32)
    idx1_np  = np.empty(total, dtype=np.int64)
    idx2_np  = np.empty(total, dtype=np.int64)

    # Minima
    vals_np [:n_min] = grid[min_mask].cpu().numpy()
    types_np[:n_min] = type_grid[min_mask].cpu().numpy()
    idx1_np [:n_min] = min_flat_idx.cpu().numpy()
    idx2_np [:n_min] = min_flat_idx.cpu().numpy()

    # Maxima
    s = n_min
    vals_np [s:s+n_max] = grid[max_mask].cpu().numpy()
    types_np[s:s+n_max] = type_grid[max_mask].cpu().numpy()
    idx1_np [s:s+n_max] = max_flat_idx.cpu().numpy()
    idx2_np [s:s+n_max] = max_flat_idx.cpu().numpy()

    # Edges
    s += n_max
    vals_np [s:] = torch.cat([h_vals_f, v_vals_f]).cpu().numpy()
    types_np[s:] = EDGE
    idx1_np [s:] = torch.cat([h_idx1_f, v_idx1_f]).cpu().numpy()
    idx2_np [s:] = torch.cat([h_idx2_f, v_idx2_f]).cpu().numpy()

    # 5. Sort by filtration value.
    # CRITICAL: use stable sort + type-priority tiebreak so that
    # pixels always come before edges at the same filtration value.
    # Without this, an edge fires before its endpoint pixels are active -> empty output.

    vals_for_sort = vals_np.copy()
    vals_for_sort[types_np == EDGE] += 1e-7
    sort_order = np.argsort(vals_for_sort, kind='stable')

    # Global cell identities (for recording birth/death indices)
    identities = np.arange(total, dtype=np.int64)

    max_pairs   = n_edges
    h0_pairs_out = np.empty((max_pairs, 2), dtype=np.int64)
    h1_pairs_out = np.empty((max_pairs, 2), dtype=np.int64)

    h0_count, h1_count = _run_2d_sweep_loop(
        vals_np [sort_order],
        types_np[sort_order],
        idx1_np [sort_order],
        idx2_np [sort_order],
        identities[sort_order],
        num_pixels, total,
        h0_pairs_out, h1_pairs_out,
        GMIN, MIN, EDGE, MAX, GMAX,)
    return h0_pairs_out[:h0_count], h1_pairs_out[:h1_count]


@njit(cache=True)
def _find_basin_root(parent, node):
    root = node
    while root != parent[root]:
        root = parent[root]
    # Path compression
    cursor = node
    while cursor != root:
        nxt = parent[cursor]
        parent[cursor] = root
        cursor = nxt
    return root


@njit(cache=True)
def _run_2d_sweep_loop(
    vals, types, idx1, idx2, identities,
    num_pixels, total_elements,
    h0_out, h1_out,
    GMIN, MIN, EDGE, MAX, GMAX,):
    parent          = np.arange(num_pixels, dtype=np.int64)
    active          = np.zeros(num_pixels, dtype=np.bool_)
    birth_id        = np.zeros(num_pixels, dtype=np.int64)  # identity of birth cell per component
    loop_registry   = np.zeros(num_pixels, dtype=np.int64)  # identity of H1-birth edge per component
    has_open_loop   = np.zeros(num_pixels, dtype=np.bool_)

    h0_count = 0
    h1_count = 0

    for i in range(len(vals)):
        t  = types[i]
        i1 = idx1[i]
        i2 = idx2[i]
        cid = identities[i]

        if t == MIN or t == GMIN:
            active[i1]   = True
            birth_id[i1] = cid

        elif t == EDGE:
            # Activate endpoints lazily in case of same-value ties
            # (handles edge sorted before its pixel at identical filtration value)
            if not active[i1]:
                active[i1]   = True
                birth_id[i1] = i1  # pixel index as fallback identity
            if not active[i2]:
                active[i2]   = True
                birth_id[i2] = i2

            r1 = _find_basin_root(parent, i1)
            r2 = _find_basin_root(parent, i2)

            if r1 != r2:
                # H0 death: younger component merges into older
                # Older = lower birth_id (processed earlier in filtration)
                if birth_id[r1] < birth_id[r2]:
                    victim, survivor = r2, r1
                else:
                    victim, survivor = r1, r2

                h0_out[h0_count, 0] = birth_id[victim]
                h0_out[h0_count, 1] = cid
                h0_count += 1

                # Merge: carry open loop from victim to survivor if survivor has none
                parent[victim] = survivor
                if has_open_loop[victim] and not has_open_loop[survivor]:
                    loop_registry[survivor] = loop_registry[victim]
                    has_open_loop[survivor] = True
                has_open_loop[victim] = False

            elif r1 == r2:
                # this branch must fire for H1 to work
                print("H1 birth candidate at filtration", vals[i])  # add this
                if not has_open_loop[r1]:
                    loop_registry[r1] = cid
                    has_open_loop[r1] = True

        elif t == MAX or t == GMAX:
            # find nearest active neighbor's root instead of the max pixel itself
            # max pixel sits at the top of a basin — its root is found via union-find
            # but we need it to be active first
            if not active[i1]:
                active[i1] = True
                birth_id[i1] = i1
            r = _find_basin_root(parent, i1)
            if has_open_loop[r]:
                h1_out[h1_count, 0] = loop_registry[r]
                h1_out[h1_count, 1] = cid
                h1_count += 1
                has_open_loop[r] = False
    return h0_count, h1_count

