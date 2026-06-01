"""Sublevel Set Persistence of H0 on GPU + numba"""
from time import perf_counter

import torch
import numpy as np
from numba import njit, prange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN, MAX, GMIN, GMAX = 1, -1, 2, -2 # types of keypoints

# =========== 0) Base scenario ============

def find_extrema_in_timeseries(y_vals, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (indices, types) of minima and maxima, sorted by index.
    - Having many gmin/gmax values is correctly labeled
    - Interior gmin/gmax are upgraded from min/max via isin check
    - We assess whether boundary points are min/max based on their neighbor (if < neighbor they its a min, elts its a max);
    - boundary maxima are discarded (they mess up persistence)
    - If gmax only appears at boundaries, we ignore it and promote next highest interior max to GMAX
    NOTE: keeping only keypoints indirectly de-duplicates the dataset, solving the 'plateaus' problem"""

    # Vectorized comparison with neighbors
    is_min = (y_vals[1:-1] < y_vals[:-2]) & (y_vals[1:-1] < y_vals[2:])
    is_max = (y_vals[1:-1] > y_vals[:-2]) & (y_vals[1:-1] > y_vals[2:])
    
    # Adjust indices because we sliced off the boundaries
    min_idx  = torch.nonzero(is_min).flatten() + 1
    max_idx  = torch.nonzero(is_max).flatten() + 1

    gmin_val = y_vals.min()
    gmax_val = y_vals.max()
    gmin_idx = torch.nonzero(y_vals == gmin_val).flatten()
    gmax_idx = torch.nonzero(y_vals == gmax_val).flatten()

    # upgrade min/max to gmin/gmax before concat
    min_types = torch.where(torch.isin(min_idx, gmin_idx),
                            torch.tensor(GMIN, device=device),
                            torch.tensor(MIN,  device=device))
    max_types = torch.where(torch.isin(max_idx, gmax_idx),
                            torch.tensor(GMAX, device=device),
                            torch.tensor(MAX,  device=device))

    # boundaries: keep only if min/gmin, discard if max/gmax
    boundary_indices, boundary_types_list = [], []
    for idx, neighbor in [(0, 1), (len(y_vals) - 1, len(y_vals) - 2)]:
        val = y_vals[idx]
        if val == gmax_val:
            pass  # discard
        elif val < y_vals[neighbor]:
            boundary_indices.append(idx)
            boundary_types_list.append(GMIN if val == gmin_val else MIN)
        # else: boundary is a non-global max → discard

    # promote next interior max to GMAX if boundary was gmax and no interior gmax exists
    left_was_gmax  = y_vals[0]  == gmax_val
    right_was_gmax = y_vals[-1] == gmax_val
    boundary_stole_gmax  = (left_was_gmax or right_was_gmax) and not torch.isin(max_idx, gmax_idx).any()
    if boundary_stole_gmax:
        interior_max_val = y_vals[1:-1].max()
        promoted_idx     = torch.nonzero(y_vals[1:-1] == interior_max_val).flatten() + 1
        max_types        = torch.where(torch.isin(max_idx, promoted_idx),
                                        torch.tensor(GMAX, device=device), max_types)

    if boundary_indices:
        b_idx     = torch.tensor(boundary_indices,    device=device)
        b_types   = torch.tensor(boundary_types_list, device=device)
        all_idx   = torch.cat([min_idx, max_idx, b_idx])
        all_types = torch.cat([min_types, max_types, b_types])
    else:
        all_idx   = torch.cat([min_idx, max_idx])
        all_types = torch.cat([min_types, max_types])

    sort_key  = all_idx * 10 - all_types.abs()  # GMIN/GMAX before MIN/MAX at same index. *10 > max type abs val (2), so index dominates
    order     = torch.argsort(sort_key)
    all_idx   = all_idx[order]
    all_types = all_types[order]

    mask = torch.cat([torch.tensor([True], device=device), all_idx[1:] != all_idx[:-1]])
    keypoint_idx, keypoint_types = all_idx[mask], all_types[mask]
    return keypoint_idx, keypoint_types

@njit(fastmath=True)
def _find_basin_root_1d(basin_membership_ids: np.ndarray, rank: int):
    """Unified two-pass iterative path compression for both 1D and 2D arrays."""
    """Finds the ultimate root valley (deepest min) owning the basin at neighbor_rank.
    Flattens the tracking chain (path compression) for faster future lookups.
    Called 2x every time the sweep line hits a MAX (once for left neighbor, once for right)

    Params:
    - basin_membership_ids (np.ndarray): 1D array where index = keypoint rank, value = parent/root rank
    - neighbor_rank (int): rank of the keypoint immediately to the left or right of the current MAX
    * root = rank of deepest valley owning water

    Example:
    Keypoint indices: [5,  9, 13, 18, 23, 27, 33], and take the MAX at idx 18
    Ranks (positions):[0,  1,  2,  3,  4,  5,  6]
    basin_membership_ids = [0, 1, 0, 3, 2, 5, 6], this is after a few iterations
    Since max is at idx 18, we check its neighbors: idx 13 (rank 2) and idx 23 (rank 4)
    (Rank 4 points to 2; Rank 2 points to 0):

    when rank = 2, basin_membership_ids[2] = 0, cursor -> 0, basin_membership_ids[0] = 0, STOP
    when rank = 4, basin_membership_ids[4] = 2, cursor -> 2, basin_membership_ids[2] = 0,
    cursor -> 0, basin_membership_ids[0] = 0, STOP and update rank 4 to point directly to 0.
    updated basin_membership_ids = [0, 1, 0, 3, 0, 5, 6]
    For both neighbors, ultimate root = 0"""
    """Two-pass iterative path compression preventing C-stack overflow in 1D arrays."""
    cursor = rank
    root   = rank
    while root != basin_membership_ids[root]:
        root = basin_membership_ids[root]
    while cursor != root:
        next_node = basin_membership_ids[cursor]
        basin_membership_ids[cursor] = root
        cursor = next_node
    return root

@njit(fastmath=True)
def _find_basin_root_2d(basin_membership_ids: np.ndarray, row: int, rank: int):
    """Two-pass iterative path compression preventing C-stack overflow in 2D arrays."""
    cursor = rank
    root   = rank
    while root != basin_membership_ids[row, root]:
        root = basin_membership_ids[row, root]
    while cursor != root:
        next_node = basin_membership_ids[row, cursor]
        basin_membership_ids[row, cursor] = root
        cursor = next_node
    return root

@njit(fastmath=True)
def _run_1d_sweep_loop(sorted_ranks_idx: np.ndarray, keypoint_types: np.ndarray, keypoint_idx_array: np.ndarray, 
                              timeseries_values: np.ndarray, basin_membership_ids: np.ndarray,
                              submerged_keypoints: np.ndarray, num_keypoints: int, pairs_out: np.ndarray):
    """Zero-allocation inner sweep loop driven by an external pre-sorted schedule."""
    p_count = 0

    # Process elements using the external pre-sorted index track
    for active_sequence_rank in sorted_ranks_idx:
        if active_sequence_rank >= num_keypoints:
            continue
            
        keypoint_type     = keypoint_types[active_sequence_rank]
        active_series_idx = keypoint_idx_array[active_sequence_rank]

        if keypoint_type == MIN or keypoint_type == GMIN: 
            submerged_keypoints[active_sequence_rank] = True

        elif keypoint_type == MAX or keypoint_type == GMAX: 
            left_rank  = active_sequence_rank - 1
            right_rank = active_sequence_rank + 1

            left_root  = _find_basin_root_1d(basin_membership_ids, left_rank) if left_rank >= 0 else -1
            right_root = _find_basin_root_1d(basin_membership_ids, right_rank) if right_rank < num_keypoints else -1

            is_left_submerged  = (left_rank >= 0) and submerged_keypoints[left_root]
            is_right_submerged = (right_rank < num_keypoints) and submerged_keypoints[right_root]

            if is_left_submerged and is_right_submerged: 
                left_birth_height  = timeseries_values[keypoint_idx_array[left_root]]
                right_birth_height = timeseries_values[keypoint_idx_array[right_root]]

                if left_birth_height > right_birth_height: 
                    victim_birth_series_idx         = keypoint_idx_array[left_root]
                    basin_membership_ids[left_root] = right_root
                else: 
                    victim_birth_series_idx          = keypoint_idx_array[right_root]
                    basin_membership_ids[right_root] = left_root
                
                pairs_out[p_count, 0] = victim_birth_series_idx
                pairs_out[p_count, 1] = active_series_idx
                p_count += 1

            elif is_left_submerged: 
                basin_membership_ids[active_sequence_rank] = left_root
                submerged_keypoints[active_sequence_rank] = True
            elif is_right_submerged: 
                basin_membership_ids[active_sequence_rank] = right_root
                submerged_keypoints[active_sequence_rank] = True
    return p_count

def compute_1d_sublevel_persistence(timeseries_values: torch.Tensor, keypoint_series_idx: torch.Tensor, keypoint_types: torch.Tensor):
    """runs the persistence loop on a single timeseries"""
    num_keypoints       = len(keypoint_series_idx)
    keypoint_heights    = timeseries_values[keypoint_series_idx]
    sweep_line_schedule = torch.argsort(keypoint_heights) # ascending sort water level by y-height

    sweep_list_np   = sweep_line_schedule.cpu().numpy()
    types_np        = keypoint_types.cpu().numpy()
    kp_idx_np       = keypoint_series_idx.cpu().numpy()
    ts_values_np    = timeseries_values.cpu().numpy()

    # Preallocate temporary tracking arrays
    basin_membership_ids = np.arange(num_keypoints, dtype=np.int64) # initialized here
    submerged_keypoints  = np.zeros(num_keypoints, dtype=np.bool_)  # initialized here

    max_possible_pairs = num_keypoints // 2
    pairs_out = np.empty((max_possible_pairs, 2), dtype=np.int64)

    # # # birth_death_pairs_list = _run_fast_sweep_loop(sweep_list_np, types_np, kp_idx_np, ts_values_np, num_keypoints)
    # # birth_death_pairs_list = _run_1d_sweep_loop(sweep_list_np, types_np, kp_idx_np, ts_values_np, 
    # #                                                    basin_membership_ids, submerged_keypoints, num_keypoints)
    # # return birth_death_pairs_list

    # pair_count = _run_1d_sweep_loop(sweep_list_np, types_np, kp_idx_np, ts_values_np, 
    #                                         basin_membership_ids, submerged_keypoints, num_keypoints, pairs_out)
    # final_pairs = [tuple(pair) for pair in pairs_out[:pair_count]]
    # return final_pairs


    # Compute the sort order safely on the heap before hitting the Numba loop
    keypoint_heights = timeseries_values[keypoint_series_idx]
    sorted_ranks_idx = torch.argsort(keypoint_heights).cpu().numpy()

    # Pass 'sorted_ranks_idx' directly into the updated function
    pair_count = _run_1d_sweep_loop(sorted_ranks_idx, types_np, kp_idx_np, ts_values_np, 
                                            basin_membership_ids, submerged_keypoints, num_keypoints, pairs_out)
    final_pairs = [tuple(pair) for pair in pairs_out[:pair_count]]
    return final_pairs

# =========== 1) streaming case ============

class StreamingSublevelPersistence:
    """Handles additional updates to the timeseries. Maintains persistent, incrementally updated state in pre-allocated NumPy buffers"""
    def __init__(self, initial_yvals: torch.Tensor, initial_keypoint_idx: torch.Tensor, initial_keypoint_types: torch.Tensor, max_capacity: int = None):
        """initial_yvals = initial timeseries chunk"""
        self.device = initial_yvals.device

        n_yvals = len(initial_yvals)
        n_kps   = len(initial_keypoint_idx)
        
        # Ensure max_capacity accommodates BOTH time values and keypoint arrays safely
        required_minimum = max(n_yvals, n_kps)
        if max_capacity is None: # Add safety margin factor (ie 2x or 4x) to handle streaming chunks smoothly
            max_capacity = required_minimum * 2 
        elif max_capacity < required_minimum:
            raise ValueError(f"max_capacity ({max_capacity}) < required allocation bounds ({required_minimum})")
            
        self.max_capacity         = max_capacity
        self.history_values       = np.zeros(max_capacity, dtype=np.float32)
        self.keypoint_types       = np.zeros(max_capacity, dtype=np.int64)
        self.keypoint_idx_array   = np.zeros(max_capacity, dtype=np.int64)
        self.basin_membership_ids = np.arange(max_capacity, dtype=np.int64)
        self.submerged_keypoints  = np.zeros(max_capacity, dtype=np.bool_)
        
        self.current_yvals_len = n_yvals
        self.num_keypoints     = n_kps
        
        # Initialize the history values with the initial y-values
        self.history_values[:self.current_yvals_len] = initial_yvals.cpu().numpy()
        self.keypoint_types[:self.num_keypoints]     = initial_keypoint_types.cpu().numpy()
        self.keypoint_idx_array[:self.num_keypoints] = initial_keypoint_idx.cpu().numpy()

        # Instantiate local allocation context for the initial sweep run
        init_max_pairs        = self.num_keypoints // 2 # no more than half keypoints can be maxima
        init_pairs_scratchpad = np.empty((init_max_pairs, 2), dtype=np.int64)
        # initial_ranks = np.arange(self.num_keypoints, dtype=np.int64)        
        # _run_1d_sweep_loop(initial_ranks, self.keypoint_types, self.keypoint_idx_array,
        #                           self.history_values, self.basin_membership_ids, self.submerged_keypoints,
        #                           self.num_keypoints, init_pairs_scratchpad)

        initial_heights       = self.history_values[self.keypoint_idx_array[:self.num_keypoints]]
        sorted_initial_ranks  = np.argsort(initial_heights)
        _run_1d_sweep_loop(sorted_initial_ranks, self.keypoint_types, self.keypoint_idx_array,
                                self.history_values, self.basin_membership_ids, self.submerged_keypoints,
                                self.num_keypoints, init_pairs_scratchpad)

    def update_filtration(self, new_y_chunk: torch.Tensor, stream_keypoint_idx: torch.Tensor, stream_keypoint_types: torch.Tensor):
        """Incrementally updates the topological filtration state with new streaming data.
        Appends the incoming time-series chunk and resolves persistent features by 
        sweeping over unresolved historical keypoints combined with the new chunk. 
        Natively handles boundary crossings, new global extrema, and micro-oscillations 
        without re-computing history from scratch.
        Parameters:
            new_y_chunk (Tensor): Real-valued sequential observations.
            stream_keypoint_idx (Tensor): Local extrema indices relative to new_y_chunk.
            stream_keypoint_types (Tensor): Critical point identity flags (MIN/MAX/GMIN/GMAX).
        Returns:
            list[tuple[int, int]]: birth-death index pairs discovered in this stream window"""
        yvals_start_len = self.current_yvals_len
        yvals_end_len   = yvals_start_len + len(new_y_chunk)
        
        if yvals_end_len > self.max_capacity:
            raise MemoryError("Streaming data exceeds pre-allocated max_capacity.")
            
        self.history_values[yvals_start_len:yvals_end_len] = new_y_chunk.cpu().numpy()
        
        num_kp_start = self.num_keypoints
        num_kp_end   = num_kp_start + len(stream_keypoint_idx)
        
        self.keypoint_types[num_kp_start:num_kp_end]     = stream_keypoint_types.cpu().numpy()
        self.keypoint_idx_array[num_kp_start:num_kp_end] = stream_keypoint_idx.cpu().numpy() + yvals_start_len
        
        self.current_yvals_len = yvals_end_len
        self.num_keypoints     = num_kp_end
        
        # Combine unclosed (active) historical keypoints with the incoming chunk
        historical_active = np.where(~self.submerged_keypoints[:num_kp_start])[0]
        new_ranks         = np.arange(num_kp_start, num_kp_end, dtype=np.int64)
        ranks_to_process  = np.concatenate((historical_active, new_ranks))
        
        max_possible_pairs = len(ranks_to_process)
        pairs_out          = np.empty((max_possible_pairs, 2), dtype=np.int64)

        # Sort the entire active boundary domain by value height
        active_yvals        = self.history_values[self.keypoint_idx_array[ranks_to_process]]
        sorted_stream_ranks = ranks_to_process[np.argsort(active_yvals)]

        pair_count = _run_1d_sweep_loop(sorted_stream_ranks, self.keypoint_types, self.keypoint_idx_array,
                                               self.history_values, self.basin_membership_ids,
                                               self.submerged_keypoints, self.num_keypoints, pairs_out)
        return [tuple(pair) for pair in pairs_out[:pair_count]]

# =========== 2) BATCHING (no streaming) ============

@njit(fastmath=True, parallel=True) 
def _run_batch_sweep_loop(sweep_line_schedules, keypoint_types, keypoint_idx_arrays, timeseries_matrix,
                          actual_lengths, basin_membership_ids, submerged_keypoints, 
                          batch_pairs_storage, pair_counts):
    """Processes N independent series concurrently with zero internal allocations."""
    n_rows = timeseries_matrix.shape[0]

    for i in prange(n_rows):
        num_keypoints = actual_lengths[i]
        p_count       = 0
        schedule      = sweep_line_schedules[i]

        for active_sequence_rank in schedule:
            if active_sequence_rank >= num_keypoints:
                continue
                
            keypoint_type     = keypoint_types[i, active_sequence_rank]
            active_series_idx = keypoint_idx_arrays[i, active_sequence_rank]

            if keypoint_type == MIN or keypoint_type == GMIN:
                submerged_keypoints[i, active_sequence_rank] = True

            elif keypoint_type == MAX or keypoint_type == GMAX:
                left_rank  = active_sequence_rank - 1
                right_rank = active_sequence_rank + 1

                left_root  = _find_basin_root_2d(basin_membership_ids, i, left_rank) if left_rank >= 0 else -1
                right_root = _find_basin_root_2d(basin_membership_ids, i, right_rank) if right_rank < num_keypoints else -1

                is_left_submerged  = (left_rank >= 0) and submerged_keypoints[i, left_root]
                is_right_submerged = (right_rank < num_keypoints) and submerged_keypoints[i, right_root]

                if is_left_submerged and is_right_submerged:
                    left_birth_height  = timeseries_matrix[i, keypoint_idx_arrays[i, left_root]]
                    right_birth_height = timeseries_matrix[i, keypoint_idx_arrays[i, right_root]]

                    if left_birth_height > right_birth_height:
                        victim_birth_series_idx = keypoint_idx_arrays[i, left_root]
                        basin_membership_ids[i, left_root] = right_root
                    else:
                        victim_birth_series_idx = keypoint_idx_arrays[i, right_root]
                        basin_membership_ids[i, right_root] = left_root
                    
                    batch_pairs_storage[i, p_count, 0] = victim_birth_series_idx
                    batch_pairs_storage[i, p_count, 1] = active_series_idx
                    p_count += 1

                elif is_left_submerged:
                    basin_membership_ids[i, active_sequence_rank] = left_root
                    submerged_keypoints[i, active_sequence_rank] = True
                elif is_right_submerged:
                    basin_membership_ids[i, active_sequence_rank] = right_root
                    submerged_keypoints[i, active_sequence_rank] = True
        pair_counts[i] = p_count

def compute_batch_sublevel_persistence(timeseries_matrix: torch.Tensor, idx_list: list, types_list: list):
    """Pads tracking objects, fires parallel Numba loop, and formats output pairs in Python."""
    n_rows         = timeseries_matrix.shape[0]
    actual_lengths = np.array([len(t) for t in idx_list], dtype=np.int64)
    max_len        = int(actual_lengths.max())

    padded_idx   = np.zeros((n_rows, max_len), dtype=np.int64)
    padded_types = np.zeros((n_rows, max_len), dtype=np.int64)
    padded_sweep = np.zeros((n_rows, max_len), dtype=np.int64)

    # 1. Fully allocate tracking arrays out here on the heap
    basin_membership_ids = np.zeros((n_rows, max_len), dtype=np.int64)
    submerged_keypoints  = np.zeros((n_rows, max_len), dtype=np.bool_)
    batch_pairs_storage  = np.full((n_rows, max_len, 2), -1, dtype=np.int64)
    pair_counts          = np.zeros(n_rows, dtype=np.int64)

    for i in range(n_rows):
        length                   = actual_lengths[i]
        padded_idx[i, :length]   = idx_list[i].cpu().numpy()
        padded_types[i, :length] = types_list[i].cpu().numpy()
        basin_membership_ids[i, :length] = np.arange(length)
        
        row_heights  = timeseries_matrix[i][idx_list[i]]
        row_schedule = torch.argsort(row_heights).cpu().numpy()
        
        padded_sweep[i, :length] = row_schedule
        if length < max_len:
            padded_sweep[i, length:] = np.arange(length, max_len)

    ts_matrix_np = timeseries_matrix.cpu().numpy()

    # 2. Fire the zero-allocation loop
    _run_batch_sweep_loop(
        padded_sweep, padded_types, padded_idx, ts_matrix_np, actual_lengths,
        basin_membership_ids, submerged_keypoints, batch_pairs_storage, pair_counts)

    # 3. Parse output pairs efficiently back in standard Python space
    final_output = []
    for i in range(n_rows):
        count = pair_counts[i]
        final_output.append([
            (int(batch_pairs_storage[i, j, 0]), int(batch_pairs_storage[i, j, 1])) 
            for j in range(count)])
    return final_output

# =========== 3) streaming + BATCHING ===========

def compute_sublevel_persistence(timeseries: torch.Tensor, keypoint_idx, keypoint_types):
    """router that dynamically selects the fastest architecture"""
    if timeseries.ndim == 1 or (timeseries.ndim == 2 and timeseries.shape[0] == 1):
        # Squeeze down to 1D if necessary and run the hyper-fast sequential loop
        ts_1d = timeseries.squeeze(0) if timeseries.ndim == 2 else timeseries
        return compute_1d_sublevel_persistence(ts_1d, keypoint_idx, keypoint_types)
    else: # Route multi-signal tensors to the multi-core parallel engine
        return compute_batch_sublevel_persistence(timeseries, keypoint_idx, keypoint_types)

@njit(fastmath=True, parallel=True)
def _run_batch_streaming_sweep_loop(
    row_active_counts,          # 1D array: number of valid active keypoints to process per row
    padded_active_ranks,        # 2D array: pre-sorted active historical + new ranks to sweep
    keypoint_types, 
    keypoint_idx_matrix, 
    timeseries_matrix, 
    basin_membership_ids, 
    submerged_keypoints,
    batch_pairs_storage,        # Passed in to maintain zero-allocation
    pair_counts):                 # Passed in to maintain zero-allocation
    """Incremental parallel sweep engine processing N channels simultaneously with ZERO internal allocations."""
    n_rows = timeseries_matrix.shape[0]

    for i in prange(n_rows):
        num_to_process = row_active_counts[i]
        p_count = 0
        
        # Pull out the pre-sorted sequence schedule for this thread's row
        schedule = padded_active_ranks[i, :num_to_process]
        
        for active_sequence_rank in schedule:
            keypoint_type     = keypoint_types[i, active_sequence_rank]
            active_series_idx = keypoint_idx_matrix[i, active_sequence_rank]

            if keypoint_type == MIN or keypoint_type == GMIN:
                submerged_keypoints[i, active_sequence_rank] = True

            elif keypoint_type == MAX or keypoint_type == GMAX:
                left_rank  = active_sequence_rank - 1
                right_rank = active_sequence_rank + 1

                # Corrected tracking arrays bounds limit lookups
                left_root  = _find_basin_root_2d(basin_membership_ids, i, left_rank) if left_rank >= 0 else -1
                right_root = _find_basin_root_2d(basin_membership_ids, i, right_rank) if right_rank < keypoint_idx_matrix.shape[1] else -1

                is_left_submerged  = (left_rank >= 0) and submerged_keypoints[i, left_root]
                is_right_submerged = (right_rank < keypoint_idx_matrix.shape[1]) and submerged_keypoints[i, right_root]

                if is_left_submerged and is_right_submerged:
                    left_birth_height  = timeseries_matrix[i, keypoint_idx_matrix[i, left_root]]
                    right_birth_height = timeseries_matrix[i, keypoint_idx_matrix[i, right_root]]

                    if left_birth_height > right_birth_height:
                        victim_birth_series_idx = keypoint_idx_matrix[i, left_root]
                        basin_membership_ids[i, left_root] = right_root
                    else:
                        victim_birth_series_idx = keypoint_idx_matrix[i, right_root]
                        basin_membership_ids[i, right_root] = left_root
                    
                    batch_pairs_storage[i, p_count, 0] = victim_birth_series_idx
                    batch_pairs_storage[i, p_count, 1] = active_series_idx
                    p_count += 1

                elif is_left_submerged:
                    basin_membership_ids[i, active_sequence_rank] = left_root
                    submerged_keypoints[i, active_sequence_rank] = True
                elif is_right_submerged:
                    basin_membership_ids[i, active_sequence_rank] = right_root
                    submerged_keypoints[i, active_sequence_rank] = True
                    
        pair_counts[i] = p_count

class BatchStreamingSublevelPersistence:
    """Manages 2D contiguous state cache blocks for fast parallel multi-channel streaming."""
    def __init__(self, initial_matrix: torch.Tensor, idx_list: list, types_list: list, max_capacity: int = 10_000_000):
        self.n_rows = initial_matrix.shape[0]
        self.device = initial_matrix.device
        self.max_capacity = max_capacity
        
        # Pre-allocate large continuous memory tracking blocks
        self.history_values      = np.zeros((self.n_rows, max_capacity), dtype=np.float32)
        self.keypoint_types      = np.zeros((self.n_rows, max_capacity), dtype=np.int64)
        self.keypoint_idx_matrix = np.zeros((self.n_rows, max_capacity), dtype=np.int64)
        self.submerged_keypoints = np.zeros((self.n_rows, max_capacity), dtype=np.bool_)
        
        self.basin_membership_ids = np.zeros((self.n_rows, max_capacity), dtype=np.int64)
        for i in range(self.n_rows):
            self.basin_membership_ids[i] = np.arange(max_capacity)
            
        self.current_values_lens = np.full(self.n_rows, initial_matrix.shape[1], dtype=np.int64)
        self.num_keypoints       = np.array([len(t) for t in idx_list], dtype=np.int64)
        
        # Ingest base historical data matrices
        for i in range(self.n_rows):
            self.history_values[i, :self.current_values_lens[i]] = initial_matrix[i].cpu().numpy()
            self.keypoint_types[i, :self.num_keypoints[i]]       = types_list[i].cpu().numpy()
            self.keypoint_idx_matrix[i, :self.num_keypoints[i]]  = idx_list[i].cpu().numpy()

        # Gather historical baselines schedules
        row_active_counts   = self.num_keypoints.copy()
        max_active_elements = int(row_active_counts.max())
        padded_active_ranks = np.zeros((self.n_rows, max_active_elements), dtype=np.int64)
        
        for i in range(self.n_rows):
            ranks = np.arange(self.num_keypoints[i], dtype=np.int64)
            heights = self.history_values[i, self.keypoint_idx_matrix[i, ranks]]
            padded_active_ranks[i, :self.num_keypoints[i]] = ranks[np.argsort(heights)]

        # Allocations scratchpads for the initial baseline run
        batch_pairs_storage = np.full((self.n_rows, max_active_elements, 2), -1, dtype=np.int64)
        pair_counts         = np.zeros(self.n_rows, dtype=np.int64)

        _run_batch_streaming_sweep_loop(
            row_active_counts, padded_active_ranks, self.keypoint_types, self.keypoint_idx_matrix,
            self.history_values, self.basin_membership_ids, self.submerged_keypoints,
            batch_pairs_storage, pair_counts)

    def append_batch_stream(self, new_chunks_matrix: torch.Tensor, stream_idx_list: list, stream_types_list: list):
        """Pushes a multi-channel stream frame chunk down the parallel pipeline and resolves persistent structures."""
        # 1. Update positions, load metrics into shared 2D arrays
        active_rank_lists = []
        row_active_counts = np.zeros(self.n_rows, dtype=np.int64)
        
        for i in range(self.n_rows):
            v_start = self.current_values_lens[i]
            v_end   = v_start + new_chunks_matrix.shape[1]
            self.history_values[i, v_start:v_end] = new_chunks_matrix[i].cpu().numpy()
            
            kp_start = self.num_keypoints[i]
            kp_end   = kp_start + len(stream_idx_list[i])
            
            self.keypoint_types[i, kp_start:kp_end]      = stream_types_list[i].cpu().numpy()
            self.keypoint_idx_matrix[i, kp_start:kp_end] = stream_idx_list[i].cpu().numpy() + v_start
            
            self.current_values_lens[i] = v_end
            self.num_keypoints[i]       = kp_end
            
            # Combine unsubmerged active history boundaries with the new frame's ranks
            historical_active = np.where(~self.submerged_keypoints[i, :kp_start])[0]
            new_ranks         = np.arange(kp_start, kp_end, dtype=np.int64)
            ranks_to_process  = np.concatenate((historical_active, new_ranks))
            
            # Sort individual channel arrays by height
            heights = self.history_values[i, self.keypoint_idx_matrix[i, ranks_to_process]]
            sorted_ranks = ranks_to_process[np.argsort(heights)]
            
            active_rank_lists.append(sorted_ranks)
            row_active_counts[i] = len(sorted_ranks)

        # 2. Build tracking schedules on the heap before launching the parallel sweep loop
        max_active_elements = int(row_active_counts.max())
        padded_active_ranks = np.zeros((self.n_rows, max_active_elements), dtype=np.int64)
        for i in range(self.n_rows):
            padded_active_ranks[i, :row_active_counts[i]] = active_rank_lists[i]

        batch_pairs_storage = np.full((self.n_rows, max_active_elements, 2), -1, dtype=np.int64)
        pair_counts         = np.zeros(self.n_rows, dtype=np.int64)

        # 3. Fire parallel calculation loop
        _run_batch_streaming_sweep_loop(
            row_active_counts, padded_active_ranks, self.keypoint_types, self.keypoint_idx_matrix,
            self.history_values, self.basin_membership_ids, self.submerged_keypoints,
            batch_pairs_storage, pair_counts)

        # 4. Post-process structural pairs back out securely in Python
        final_output = []
        for i in range(self.n_rows):
            count = pair_counts[i]
            final_output.append([
                (int(batch_pairs_storage[i, j, 0]), int(batch_pairs_storage[i, j, 1])) 
                for j in range(count)])
        return final_output

