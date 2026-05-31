"""Persistence on GPU + numba"""
import torch
import numpy as np
from numba import njit, prange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN, MAX, GMIN, GMAX = 1, -1, 2, -2 # types of keypoints

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

# ========= 1) Base + streaming case ========

@njit(fastmath=True)
def _find_basin_root(basin_membership_ids: np.ndarray, rank: int):
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
    cursor = rank
    while cursor != basin_membership_ids[cursor]:
        basin_membership_ids[cursor] = basin_membership_ids[basin_membership_ids[cursor]]
        cursor = basin_membership_ids[cursor]
    return cursor

@njit(fastmath=True)
def _run_streaming_sweep_loop(ranks_to_process: np.ndarray, keypoint_types: np.ndarray, keypoint_idx_array: np.ndarray, 
                              timeseries_values: np.ndarray, basin_membership_ids: np.ndarray,
                              submerged_components: np.ndarray, num_keypoints: int):
    """sweep loop. Works for the normal case, and the incremental case"""
    heights           = timeseries_values[keypoint_idx_array[ranks_to_process]]
    sorted_ranks_idx  = np.argsort(heights)
    birth_death_pairs = []

    for idx in sorted_ranks_idx:
        active_sequence_rank = ranks_to_process[idx]
        keypoint_type        = keypoint_types[active_sequence_rank]
        active_series_idx    = keypoint_idx_array[active_sequence_rank]

        if keypoint_type == MIN or keypoint_type == GMIN: # we have a valley
            submerged_components[active_sequence_rank] = True

        elif keypoint_type == MAX or keypoint_type == GMAX: # we have a peak
            left_rank  = active_sequence_rank - 1
            right_rank = active_sequence_rank + 1

            left_root  = _find_basin_root(basin_membership_ids, left_rank) if left_rank >= 0 else -1
            right_root = _find_basin_root(basin_membership_ids, right_rank) if right_rank < num_keypoints else -1

            has_left   = (left_rank >= 0) and submerged_components[left_root]
            has_right  = (right_rank < num_keypoints) and submerged_components[right_root]

            if has_left and has_right: # lakes on both sides of peak => merge basins + record birth-death pair
                left_birth_height  = timeseries_values[keypoint_idx_array[left_root]]
                right_birth_height = timeseries_values[keypoint_idx_array[right_root]]

                if left_birth_height > right_birth_height: # lower basin is victim (right dies)
                    victim_birth_series_idx         = keypoint_idx_array[left_root]
                    basin_membership_ids[left_root] = right_root
                else: # lower basin (left here) is victim, dies
                    victim_birth_series_idx          = keypoint_idx_array[right_root]
                    basin_membership_ids[right_root] = left_root
                birth_death_pairs.append((int(victim_birth_series_idx), int(active_series_idx)))

            elif has_left: # lake only on left side => add to left basin
                basin_membership_ids[active_sequence_rank] = left_root
                submerged_components[active_sequence_rank] = True
            elif has_right: # lake only on right side => add to right basin
                basin_membership_ids[active_sequence_rank] = right_root
                submerged_components[active_sequence_rank] = True
    return birth_death_pairs

def compute_1d_sublevel_persistence(timeseries_values: torch.Tensor, keypoint_series_idx: torch.Tensor, keypoint_types: torch.Tensor):
    """runs the persistence loop on a single timeseries"""
    num_keypoints       = len(keypoint_series_idx)
    keypoint_heights    = timeseries_values[keypoint_series_idx]
    sweep_line_schedule = torch.argsort(keypoint_heights) # ascending sort water level by y-height

    sweep_list_np   = sweep_line_schedule.cpu().numpy()
    types_np        = keypoint_types.cpu().numpy()
    kp_idx_np       = keypoint_series_idx.cpu().numpy()
    ts_values_np    = timeseries_values.cpu().numpy()

    # birth_death_pairs_list = _run_fast_sweep_loop(sweep_list_np, types_np, kp_idx_np, ts_values_np, num_keypoints)

    # Preallocate temporary tracking arrays
    basin_membership_ids = np.arange(num_keypoints, dtype=np.int64) # initialized here
    submerged_components = np.zeros(num_keypoints, dtype=np.bool_)  # initialized here

    # FIX: Reuses the scenario 1 function to eliminate duplicate logic
    birth_death_pairs_list = _run_streaming_sweep_loop(sweep_list_np, types_np, kp_idx_np, ts_values_np, 
                                                       basin_membership_ids, submerged_components, num_keypoints)
    return birth_death_pairs_list

class OnlineSublevelPersistence:
    """Maintains a persistent, incrementally updated state in pre-allocated NumPy buffers """
    def __init__(self, initial_x: torch.Tensor, initial_keypoint_idx: torch.Tensor, initial_keypoint_types: torch.Tensor, max_capacity: int = None):
        self.device = initial_x.device
        
        n = len(initial_x)
        max_capacity = n if max_capacity is None else max_capacity
        if max_capacity < n:
            raise ValueError(f"max_capacity ({max_capacity}) < input size ({n})")
        self.max_capacity = max_capacity

        self.history_values       = np.zeros(max_capacity, dtype=np.float32)
        self.keypoint_types       = np.zeros(max_capacity, dtype=np.int64)
        self.keypoint_idx_array   = np.zeros(max_capacity, dtype=np.int64)
        self.basin_membership_ids = np.arange(max_capacity, dtype=np.int64)
        self.submerged_components = np.zeros(max_capacity, dtype=np.bool_)
        
        self.current_values_len = n
        self.num_keypoints      = len(initial_keypoint_idx)
        
        self.history_values[:self.current_values_len] = initial_x.cpu().numpy()
        self.keypoint_types[:self.num_keypoints]      = initial_keypoint_types.cpu().numpy()
        self.keypoint_idx_array[:self.num_keypoints]  = initial_keypoint_idx.cpu().numpy()
        
        initial_ranks = np.arange(self.num_keypoints, dtype=np.int64)
        _run_streaming_sweep_loop(initial_ranks, self.keypoint_types, self.keypoint_idx_array,
                                  self.history_values, self.basin_membership_ids, self.submerged_components,
                                  self.num_keypoints)

    def append_stream_data(self, new_x_chunk: torch.Tensor, stream_keypoint_idx: torch.Tensor, stream_keypoint_types: torch.Tensor):
            v_start = self.current_values_len
            v_end   = v_start + len(new_x_chunk)
            self.history_values[v_start:v_end] = new_x_chunk.cpu().numpy()
            
            kp_start = self.num_keypoints
            kp_end   = kp_start + len(stream_keypoint_idx)
            
            self.keypoint_types[kp_start:kp_end]     = stream_keypoint_types.cpu().numpy()
            self.keypoint_idx_array[kp_start:kp_end] = stream_keypoint_idx.cpu().numpy() + v_start
            
            self.current_values_len = v_end
            self.num_keypoints      = kp_end
            ranks_to_process        = np.arange(kp_start, kp_end, dtype=np.int64)
            
            new_pairs = _run_streaming_sweep_loop(ranks_to_process, self.keypoint_types, self.keypoint_idx_array,
                                                  self.history_values, self.basin_membership_ids,
                                                  self.submerged_components, self.num_keypoints)
            return new_pairs

# ===== 2) BATCHING (no streaming) ======

@njit(fastmath=True)
def _numba_find_root_batch(basin_membership_ids, row, rank):
    """Finds root with path compression inside a specific row of a 2D tracking array.
    This function also works for the online case"""
    cursor = rank
    while cursor != basin_membership_ids[row, cursor]:
        basin_membership_ids[row, cursor] = basin_membership_ids[row, basin_membership_ids[row, cursor]]
        cursor = basin_membership_ids[row, cursor]
    return cursor

@njit(fastmath=True, parallel=True) 
def _run_batch_sweep_loop(sweep_line_schedules, keypoint_types, keypoint_idx_arrays, timeseries_matrix,
                          actual_lengths, basin_membership_ids, submerged_components):  # <- Pre-allocated blocks passed here
    """Processes N independent time series concurrently across CPU cores with ZERO internal allocations"""
    n_rows              = timeseries_matrix.shape[0]
    max_keypoints       = sweep_line_schedules.shape[1]
    batch_pairs_storage = np.full((n_rows, max_keypoints, 2), -1, dtype=np.int64)
    pair_counts         = np.zeros(n_rows, dtype=np.int64)

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
                submerged_components[i, active_sequence_rank] = True

            elif keypoint_type == MAX or keypoint_type == GMAX:
                left_rank  = active_sequence_rank - 1
                right_rank = active_sequence_rank + 1

                left_root  = _numba_find_root_batch(basin_membership_ids, i, left_rank) if left_rank >= 0 else -1
                right_root = _numba_find_root_batch(basin_membership_ids, i, right_rank) if right_rank < num_keypoints else -1

                has_left  = (left_rank >= 0) and submerged_components[i, left_root]
                has_right = (right_rank < num_keypoints) and submerged_components[i, right_root]

                if has_left and has_right:
                    left_birth_height  = timeseries_matrix[i, keypoint_idx_arrays[i, left_root]]
                    right_birth_height = timeseries_matrix[i, keypoint_idx_arrays[i, right_root]]

                    if left_birth_height > right_birth_height:
                        victim_birth_series_idx = keypoint_idx_arrays[i, left_root]
                        basin_membership_ids[i, left_root] = right_root
                    else:
                        victim_birth_series_idx = keypoint_idx_arrays[i, right_root]
                        basin_membership_ids[i, right_root] = left_root
                    
                    batch_pairs_storage[i, p_count] = [victim_birth_series_idx, active_series_idx]
                    p_count += 1

                elif has_left:
                    basin_membership_ids[i, active_sequence_rank] = left_root
                    submerged_components[i, active_sequence_rank] = True
                elif has_right:
                    basin_membership_ids[i, active_sequence_rank] = right_root
                    submerged_components[i, active_sequence_rank] = True
        pair_counts[i] = p_count

    final_output = []
    for i in range(n_rows):
        count      = pair_counts[i]
        pairs_list = [(int(batch_pairs_storage[i, j, 0]), int(batch_pairs_storage[i, j, 1])) for j in range(count)]
        final_output.append(pairs_list)
    return final_output

def compute_batch_sublevel_persistence(timeseries_matrix: torch.Tensor, idx_list: list, types_list: list):
    """Pads different length keypoint tensors dynamically and triggers parallel execution."""
    n_rows         = timeseries_matrix.shape[0]
    actual_lengths = np.array([len(t) for t in idx_list], dtype=np.int64)
    max_len        = int(actual_lengths.max())

    padded_idx   = np.zeros((n_rows, max_len), dtype=np.int64)
    padded_types = np.zeros((n_rows, max_len), dtype=np.int64)
    padded_sweep = np.zeros((n_rows, max_len), dtype=np.int64)

    # 1. Allocate large shared 2D tracking blocks out here in Python space once
    basin_membership_ids = np.zeros((n_rows, max_len), dtype=np.int64)
    submerged_components = np.zeros((n_rows, max_len), dtype=np.bool_)

    for i in range(n_rows):
        length                   = actual_lengths[i]
        padded_idx[i, :length]   = idx_list[i].cpu().numpy()
        padded_types[i, :length] = types_list[i].cpu().numpy()
        
        # Initialize tracking arrays row identities
        basin_membership_ids[i, :length] = np.arange(length)
        
        row_heights  = timeseries_matrix[i][idx_list[i]]
        row_schedule = torch.argsort(row_heights).cpu().numpy()
        
        padded_sweep[i, :length] = row_schedule
        if length < max_len:
            padded_sweep[i, length:] = np.arange(length, max_len)

    ts_matrix_np = timeseries_matrix.cpu().numpy()

    # 2. Pass tracking blocks safely into Numba
    return _run_batch_sweep_loop(
        padded_sweep,
        padded_types,
        padded_idx,
        ts_matrix_np,
        actual_lengths,
        basin_membership_ids,
        submerged_components)

# ========= 3) streaming + BATCHING ========

def compute_sublevel_persistence(timeseries: torch.Tensor, keypoint_idx, keypoint_types):
    """router that dynamically selects the fastest architecture"""
    if timeseries.ndim == 1 or (timeseries.ndim == 2 and timeseries.shape[0] == 1):
        # Squeeze down to 1D if necessary and run the hyper-fast sequential loop
        ts_1d = timeseries.squeeze(0) if timeseries.ndim == 2 else timeseries
        return compute_1d_sublevel_persistence(ts_1d, keypoint_idx, keypoint_types)
    else: # Route multi-signal tensors to the multi-core parallel engine
        return compute_batch_sublevel_persistence(timeseries, keypoint_idx, keypoint_types)

@njit(fastmath=True, parallel=True)
def _run_batch_streaming_sweep_loop(start_ranks, end_ranks,          # 1D arrays tracking the new slice per row
    keypoint_types, keypoint_idx_matrix, timeseries_matrix, basin_membership_ids, submerged_components):
    """Incremental sweep engine executing streaming updates across N channels simultaneously."""
    n_rows         = timeseries_matrix.shape[0]
    max_kp_per_row = keypoint_idx_matrix.shape[1]
    
    # Pre-allocate output buffers safely for parallel execution threads
    batch_pairs_storage = np.full((n_rows, max_kp_per_row, 2), -1, dtype=np.int64)
    pair_counts         = np.zeros(n_rows, dtype=np.int64)

    for i in prange(n_rows):
        kp_start = start_ranks[i]
        kp_end   = end_ranks[i]
        
        if kp_start >= kp_end:
            continue  # No new keypoints found for this specific channel chunk
            
        # Extract and sort only the newly added keypoint span for this row
        ranks_to_process = np.arange(kp_start, kp_end, dtype=np.int64)
        heights          = timeseries_matrix[i, keypoint_idx_matrix[i, ranks_to_process]]
        sorted_ranks_idx = np.argsort(heights)
        
        p_count = 0
        for idx in sorted_ranks_idx:
            active_sequence_rank = ranks_to_process[idx]
            keypoint_type        = keypoint_types[i, active_sequence_rank]
            active_series_idx    = keypoint_idx_matrix[i, active_sequence_rank]

            if keypoint_type == MIN or keypoint_type == GMIN:
                submerged_components[i, active_sequence_rank] = True

            elif keypoint_type == MAX or keypoint_type == GMAX:
                left_rank  = active_sequence_rank - 1
                right_rank = active_sequence_rank + 1

                left_root  = _numba_find_root_batch(basin_membership_ids, i, left_rank) if left_rank >= 0 else -1
                right_root = _numba_find_root_batch(basin_membership_ids, i, right_rank) if right_rank < kp_end else -1

                has_left  = (left_rank >= 0) and submerged_components[i, left_root]
                has_right = (right_rank < kp_end) and submerged_components[i, right_root]

                if has_left and has_right:
                    left_birth_height  = timeseries_matrix[i, keypoint_idx_matrix[i, left_root]]
                    right_birth_height = timeseries_matrix[i, keypoint_idx_matrix[i, right_root]]

                    if left_birth_height > right_birth_height:
                        victim_birth_series_idx = keypoint_idx_matrix[i, left_root]
                        basin_membership_ids[i, left_root] = right_root
                    else:
                        victim_birth_series_idx = keypoint_idx_matrix[i, right_root]
                        basin_membership_ids[i, right_root] = left_root
                    
                    batch_pairs_storage[i, p_count] = [victim_birth_series_idx, active_series_idx]
                    p_count += 1

                elif has_left:
                    basin_membership_ids[i, active_sequence_rank] = left_root
                    submerged_components[i, active_sequence_rank] = True
                elif has_right:
                    basin_membership_ids[i, active_sequence_rank] = right_root
                    submerged_components[i, active_sequence_rank] = True
                    
        pair_counts[i] = p_count

    # Format raw array records into final Python results
    final_output = []
    for i in range(n_rows):
        count      = pair_counts[i]
        pairs_list = [(int(batch_pairs_storage[i, j, 0]), int(batch_pairs_storage[i, j, 1])) for j in range(count)]
        final_output.append(pairs_list)
    return final_output

class BatchOnlineSublevelPersistence:
    """Manages 2D contiguous state cache blocks for multi-channel streaming arrays."""
    def __init__(self, initial_matrix: torch.Tensor, idx_list: list, types_list: list, max_capacity: int = 10_000_000):
        self.n_rows = initial_matrix.shape[0]
        self.device = initial_matrix.device
        
        # Pre-allocate large continuous 2D matrices
        self.history_values      = np.zeros((self.n_rows, max_capacity), dtype=np.float32)
        self.keypoint_types      = np.zeros((self.n_rows, max_capacity), dtype=np.int64)
        self.keypoint_idx_matrix = np.zeros((self.n_rows, max_capacity), dtype=np.int64)
        
        # Tracking states
        self.basin_membership_ids = np.zeros((self.n_rows, max_capacity), dtype=np.int64)
        for i in range(self.n_rows):
            self.basin_membership_ids[i] = np.arange(max_capacity)
            
        self.submerged_components = np.zeros((self.n_rows, max_capacity), dtype=np.bool_)
        
        # Track individual filled boundary sizes per channel
        self.current_values_lens = np.full(self.n_rows, initial_matrix.shape[1], dtype=np.int64)
        self.num_keypoints       = np.array([len(t) for t in idx_list], dtype=np.int64)
        
        # Ingest the historical baseline components
        for i in range(self.n_rows):
            self.history_values[i, :self.current_values_lens[i]] = initial_matrix[i].cpu().numpy()
            self.keypoint_types[i, :self.num_keypoints[i]]       = types_list[i].cpu().numpy()
            self.keypoint_idx_matrix[i, :self.num_keypoints[i]]  = idx_list[i].cpu().numpy()

        # Seed initial system states
        start_ranks = np.zeros(self.n_rows, dtype=np.int64)
        _run_batch_streaming_sweep_loop(start_ranks, self.num_keypoints, self.keypoint_types, self.keypoint_idx_matrix,
                                        self.history_values, self.basin_membership_ids, self.submerged_components)

    def append_batch_stream(self, new_chunks_matrix: torch.Tensor, stream_idx_list: list, stream_types_list: list):
        """Pushes a new chunk matrix (N, Chunk_Size) straight down the parallel streaming pipeline."""
        start_ranks = self.num_keypoints.copy()
        end_ranks   = np.zeros(self.n_rows, dtype=np.int64)
        
        for i in range(self.n_rows):
            v_start = self.current_values_lens[i]
            v_end   = v_start + new_chunks_matrix.shape[1] # assumes standard uniform batch step shapes
            self.history_values[i, v_start:v_end] = new_chunks_matrix[i].cpu().numpy()
            
            kp_start = start_ranks[i]
            kp_end   = kp_start + len(stream_idx_list[i])
            
            self.keypoint_types[i, kp_start:kp_end]      = stream_types_list[i].cpu().numpy()
            self.keypoint_idx_matrix[i, kp_start:kp_end] = stream_idx_list[i].cpu().numpy() + v_start
            
            self.current_values_lens[i] = v_end
            self.num_keypoints[i]       = kp_end
            end_ranks[i]                = kp_end

        # Launch the underlying function across all threads
        return _run_batch_streaming_sweep_loop(start_ranks, end_ranks, self.keypoint_types,
                                               self.keypoint_idx_matrix, self.history_values,
                                               self.basin_membership_ids, self.submerged_components)

