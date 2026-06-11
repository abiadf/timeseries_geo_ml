
from __future__ import annotations
import numpy as np
from numba import njit
from dataclasses import dataclass

@njit(cache=True)
def _ceh_find(parent: np.ndarray, i: int) -> int:
    """Numba-compiled iterative path compression."""
    root = i
    while root != parent[root]:
        root = parent[root]
    cursor = i
    while cursor != root:
        nxt = parent[cursor]
        parent[cursor] = root
        cursor = nxt
    return root

@njit(cache=True)
def _run_ceh_sweep_loop(
    sorted_indices: np.ndarray,
    types: np.ndarray,
    heights: np.ndarray,
    active: np.ndarray,
    parent: np.ndarray,
    birth_idx: np.ndarray,
    max_capacity: int,
    pairs_out: np.ndarray
) -> int:
    """Numba-compiled inner sweep loop utilizing zero-allocation pre-allocated memory buffers."""
    p_count = 0
    
    for idx in sorted_indices:
        t = types[idx]
        
        # Minima create new components (handled implicitly as birth_idx starts as self)
        if t == 1 or t == 2:  # MIN or GMIN
            continue
            
        # Maxima trigger merges between adjacent components
        elif t == -1 or t == -2:  # MAX or GMAX
            left_idx = idx - 1
            right_idx = idx + 1
            
            left_root = _ceh_find(parent, left_idx) if (left_idx >= 0 and active[left_idx]) else -1
            right_root = _ceh_find(parent, right_idx) if (right_idx < max_capacity and active[right_idx]) else -1
            
            if left_root != -1 and right_root != -1 and left_root != right_root:
                left_birth = birth_idx[left_root]
                right_birth = birth_idx[right_root]
                
                h_left = heights[left_birth]
                h_right = heights[right_birth]
                
                if h_left > h_right:
                    pairs_out[p_count, 0] = left_birth
                    pairs_out[p_count, 1] = idx
                    p_count += 1
                    parent[left_root] = right_root
                else:
                    pairs_out[p_count, 0] = right_birth
                    pairs_out[p_count, 1] = idx
                    p_count += 1
                    parent[right_root] = left_root
                    
            elif left_root != -1:
                parent[idx] = left_root
            elif right_root != -1:
                parent[idx] = right_root
                
    return p_count

@dataclass
class DynamicMergeTreeCEH:
    """
    CEH-style dynamic merge tree accelerated via Numba and zero-allocation arrays
    to guarantee an identical execution field for benchmarking.
    """
    def __init__(self, max_capacity: int):
        self.max_capacity = max_capacity
        self.parent = np.arange(max_capacity, dtype=np.int64)
        self.birth_idx = np.arange(max_capacity, dtype=np.int64)
        self.heights = np.full(max_capacity, np.inf, dtype=np.float32)
        self.active = np.zeros(max_capacity, dtype=np.bool_)
        self.types = np.zeros(max_capacity, dtype=np.int64)

    def update_stream(self, new_indices: np.ndarray, new_heights: np.ndarray, new_types: np.ndarray) -> np.ndarray:
        # 1. Register data points into pre-allocated memory structures
        self.heights[new_indices] = new_heights
        self.active[new_indices] = True
        self.types[new_indices] = new_types

        # 2. Sort window context via optimized quicksort
        sort_order = np.argsort(new_heights)
        sorted_indices = new_indices[sort_order]

        # 3. Pre-allocate fixed pair buffer to block dynamic allocations
        max_possible_pairs = len(new_indices) // 2 + 1
        pairs_out = np.empty((max_possible_pairs, 2), dtype=np.int64)

        # 4. Fire compiled engine
        pair_count = _run_ceh_sweep_loop(
            sorted_indices, self.types, self.heights, self.active,
            self.parent, self.birth_idx, self.max_capacity, pairs_out
        )
        
        return pairs_out[:pair_count].copy()

