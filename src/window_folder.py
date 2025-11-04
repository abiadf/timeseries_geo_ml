import __main__
import os
from typing import Dict, List, Literal, Tuple, Optional
import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.fft import fft, ifft, fftfreq
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(torch.cuda.memory_reserved(0) / 1e6, "MB reserved")
    print(torch.cuda.memory_allocated(0) / 1e6, "MB allocated")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WindowFolder:
    """Fold a 3D timeseries (samples x timesteps x features) into periodic windows inferred from its dominant
    temporal feature.
      1. Select feature most correlated with y (or highest variance if y=None), search entire dataset
      2. Optionally denoise using PSD mask or Butterworth filter
      3. Estimate dominant period from PSD peaks across multiple (not all) pages
      4. Fold and stack X (and y) into fixed-length windows for model input"""

    @staticmethod
    def _denoise_signal(x: np.ndarray, fs: float = 1.0, lowcut: float = 0.01, highcut: float = 0.2,
                        use_psd: bool = True, threshold_ratio: float = 0.1) -> np.ndarray:
        """Denoise a 1D signal using PSD mask or Butterworth bandpass."""
        if use_psd:
            f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
            Xf = fft(x)
            freqs = fftfreq(len(x), 1/fs)
            Pxx_interp = np.interp(np.abs(freqs), f, Pxx)
            mask = Pxx_interp >= threshold_ratio * np.max(Pxx_interp)
            return np.real(ifft(Xf * mask))
        else:
            b, a = butter(N=2, Wn=[lowcut, highcut], btype='band')
            return filtfilt(b, a, x)

    @staticmethod
    def X_select_dominant_feature(X: np.ndarray, y: Optional[np.ndarray] = None) -> int:
        """Pick feature index most correlated with y (or highest variance if y=None)."""
        _, _, n_features = X.shape
        if y is not None:
            if y.ndim == 1:
                y = y[:, None]
            corrs = np.zeros(n_features)
            for f in range(n_features):
                feature_mean = X[:, :, f].mean(axis=1)
                corrs[f]     = np.max([np.corrcoef(feature_mean, y[:, t])[0, 1] for t in range(y.shape[1])])
            return int(np.argmax(corrs))
        return int(np.argmax(X.var(axis=(0, 1))))

    @staticmethod
    def _select_dominant_feature(X: np.ndarray, y: Optional[np.ndarray] = None) -> int:
        """Pick feature index most correlated with y (or highest variance if y=None).
        X: (pages, timesteps, n_features)
        y: (pages, timesteps, n_targets) or (pages, n_targets)
        """
        _, T, n_features = X.shape
        if y is None:
            return int(np.argmax(X.var(axis=(0, 1))))

        corrs = np.zeros(n_features)
        for f in range(n_features):
            feature_series = X[:, :, f]  # shape (pages, timesteps)
            feature_corrs = []
            if y.ndim == 3:  # (pages, timesteps, targets)
                for t in range(y.shape[2]):
                    target_series = y[:, :, t]
                    # flatten page+time dimension
                    f_flat = feature_series.ravel()
                    t_flat = target_series.ravel()
                    if f_flat.size == t_flat.size:
                        feature_corrs.append(np.corrcoef(f_flat, t_flat)[0, 1])
            else:  # 2D y: (pages, targets)
                for t in range(y.shape[1]):
                    f_flat = feature_series.mean(axis=1)  # average over timesteps
                    t_flat = y[:, t]
                    if f_flat.size == t_flat.size:
                        feature_corrs.append(np.corrcoef(f_flat, t_flat)[0, 1])
            corrs[f] = np.nanmax(feature_corrs) if feature_corrs else 0.0

        return int(np.argmax(corrs))

    @staticmethod
    def _estimate_period_of_feature(X_feature: np.ndarray, fs: float = 1.0, peak_strength: float = 2.0,
                                    fallback_window: int = 50, max_pages: int = 10) -> int:
        """Estimate dominant period (timesteps) from a feature array (samples, timesteps), round to nearest 2^n"""
        n_pages      = min(max_pages, X_feature.shape[0])
        peak_periods = []
        for i in range(n_pages):
            f, Pxx     = welch(X_feature[i], fs=fs, nperseg=X_feature.shape[1] // 2)
            peak_ratio = np.max(Pxx) / np.mean(Pxx)
            if peak_ratio > peak_strength:
                f_peak = f[np.argmax(Pxx)]
                period = max(1, int(round(1 / f_peak)))
                peak_periods.append(period)
        if peak_periods:
            window_len = int(np.median(peak_periods))
            print(f"[INFO] Estimated window from {n_pages} pages, median period: {window_len}")
            window_len = int(2 ** np.ceil(np.log2(window_len)))  # take the 2^n above it, batches better
            print(f"[INFO] Setting the window_len to the succeeding 2^n, {window_len}")
            return window_len
        print(f"[INFO] No clear peak found in {n_pages} pages, using fallback window: {fallback_window}")
        return fallback_window

    @staticmethod #remove
    def X_fold_and_stack(X: np.ndarray, window_size: int, y: Optional[np.ndarray] = None,
                        max_windows_per_page: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fold X (and y) into windows of given size; discard leftovers."""
        samples, timesteps, n_features = X.shape
        all_X, all_y = [], []

        for i in range(samples):
            n_windows = timesteps // window_size
            if max_windows_per_page is not None:
                n_windows = min(n_windows, max_windows_per_page)
            if n_windows == 0:
                continue

            folded_X  = X[i, :n_windows * window_size, :].reshape(n_windows, window_size, n_features)
            all_X.append(folded_X)
            if y is not None:
                # repeat y[i] for each window of this sample
                all_y.append(np.tile(y[i], (n_windows, 1)))
        X_out = np.vstack(all_X)
        y_out = np.vstack(all_y) if y is not None else None
        return X_out, y_out

    @staticmethod
    def _fold_and_stack_fixed_pages(X: np.ndarray, window_size: int, y: Optional[np.ndarray] = None,
                                    num_pages_to_use: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fold only the first `num_pages_to_use` pages into fixed-length windows.
        Pads shorter pages with zeros, discards remainder that doesn't fit full windows.
        Returns stacked windows for X and last-timestep-per-window for y."""
        if num_pages_to_use is None:
            num_pages_to_use = X.shape[0]
        samples = min(num_pages_to_use, X.shape[0])

        # pad pages to max length
        max_len  = max(X[i].shape[0] for i in range(samples))
        X_padded = np.stack([np.pad(X[i], ((0, max_len - X[i].shape[0]), (0,0))) for i in range(samples)])
        if y is not None:
            if y.ndim == 2:  # already last-timestep style
                y_padded = y[:samples]
            else:
                y_padded = np.stack([np.pad(y[i], ((0, max_len - y[i].shape[0]), (0,0))) for i in range(samples)])

        all_X, all_y = [], []
        for i in range(samples):
            n_windows = X_padded.shape[1] // window_size
            if n_windows == 0:
                continue

            folded_X = X_padded[i, :n_windows*window_size, :].reshape(n_windows, window_size, X.shape[2])
            all_X.append(folded_X)

            if y is not None:
                # last timestep of each window
                if y.ndim == 3:  # full sequence
                    y_per_window = y_padded[i, window_size-1::window_size, :]
                else:  # already single row per page
                    y_per_window = np.tile(y_padded[i], (n_windows, 1))
                all_y.append(y_per_window)

        X_out = np.vstack(all_X)
        y_out = np.vstack(all_y) if y is not None else None
        return X_out, y_out

    @staticmethod #remove
    def X_auto_fold_timeseries(X: np.ndarray, y: Optional[np.ndarray] = None, denoise: bool = True, max_pages: int = 10,
                             peak_strength: float = 2.0, fs: Optional[float] = None, fallback_window: int = 50,
                             max_windows_per_page: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray], int, int]:
        """Auto-fold X (and y) into stacked windows based on dominant periodicity."""
        fs      = fs or 1.0
        dom_idx = WindowFolder._select_dominant_feature(X, y)
        X_dom   = X[:, :, dom_idx].copy()
        if denoise:
            for i in range(X_dom.shape[0]):
                X_dom[i] = WindowFolder._denoise_signal(X_dom[i], fs=fs)
        window_size = WindowFolder._estimate_period_of_feature(X_dom[:max_pages], fs=fs, peak_strength=peak_strength,
                                                               fallback_window=fallback_window, max_pages=max_pages)
        X_folded, y_folded = WindowFolder._fold_and_stack(X, window_size, y, max_windows_per_page=max_windows_per_page)
        print(f"[INFO] Selected dominant feature index: {dom_idx}, window size: {window_size}")
        return X_folded, y_folded, window_size, dom_idx

    @staticmethod
    def X_auto_fold_timeseries(X: np.ndarray, y: Optional[np.ndarray] = None, denoise: bool = True,
                             max_pages: int = 10, peak_strength: float = 2.0, fs: Optional[float] = None,
                             fallback_window: int = 50, num_pages_to_use: int = 10) -> Tuple[np.ndarray, Optional[np.ndarray], int, int]:
        """Fold only the first `num_pages_to_use` pages into fixed-length windows."""
        print(f"Input to windowfolder: X={X.shape}, y={y.shape if y is not None else None}")
        fs = fs or 1.0
        # 1. Dominant feature
        dom_idx = WindowFolder._select_dominant_feature(X, y)
        X_dom   = X[:num_pages_to_use, :, dom_idx].copy()

        # 2. Denoise if requested
        if denoise:
            for i in range(X_dom.shape[0]):
                X_dom[i] = WindowFolder._denoise_signal(X_dom[i], fs=fs)

        # 3. Estimate window size
        window_size = WindowFolder._estimate_period_of_feature(X_dom, fs=fs,
                                                            peak_strength=peak_strength,
                                                            fallback_window=fallback_window,
                                                            max_pages=max_pages)
        # 4. Fold X and y
        X_folded, y_folded = WindowFolder._fold_and_stack_fixed_pages(X, window_size, y,
                                                                    num_pages_to_use=num_pages_to_use)
        print(f"[INFO] Selected dominant feature index: {dom_idx}, window size: {window_size}")
        return X_folded, y_folded, window_size, dom_idx

    @staticmethod
    def XX_auto_fold_timeseries(X: np.ndarray, y: Optional[np.ndarray] = None, denoise: bool = True,
                            max_pages: int = 10, peak_strength: float = 2.0, fs: Optional[float] = None,
                            fallback_window: int = 50, num_pages_to_use: int = 10) -> Tuple[np.ndarray, Optional[np.ndarray], int, int]:
        """Fold only the first `num_pages_to_use` pages into fixed-length windows."""

        print(f"Shape input to window folder: X={X.shape}, y={y.shape if y is not None else None}")
        fs = fs or 1.0

        # --- 1. Dominant feature selection ---
        if y is not None and y.ndim == 3:
            # summarize y per page (e.g., last timestep)
            y_summary = y[:, -1, :]  # shape (pages, n_targets)
        else:
            y_summary = y

        dom_idx = WindowFolder._select_dominant_feature(X[:num_pages_to_use], y_summary[:num_pages_to_use])
        X_dom   = X[:num_pages_to_use, :, dom_idx].copy()

        # --- 2. Denoise if requested ---
        if denoise:
            for i in range(X_dom.shape[0]):
                X_dom[i] = WindowFolder._denoise_signal(X_dom[i], fs=fs)

        # --- 3. Estimate window size ---
        window_size = WindowFolder._estimate_period_of_feature(X_dom, fs=fs,
                                                            peak_strength=peak_strength,
                                                            fallback_window=fallback_window,
                                                            max_pages=max_pages)

        # --- 4. Fold X and y into windows ---
        X_folded, y_folded = WindowFolder._fold_and_stack_fixed_pages(X, window_size, y,
                                                                    num_pages_to_use=num_pages_to_use)

        print(f"[INFO] Selected dominant feature index: {dom_idx}, window size: {window_size}")
        return X_folded, y_folded, window_size, dom_idx

    @staticmethod
    def auto_fold_timeseries(X: np.ndarray, y: Optional[np.ndarray] = None, denoise: bool = True,
                            max_pages: int = 10, peak_strength: float = 2.0, fs: Optional[float] = None,
                            fallback_window: int = 50, num_pages_to_use: int = 10, window_size=None) -> Tuple[np.ndarray, Optional[np.ndarray], int, int]:
        """Fold only the first `num_pages_to_use` pages into fixed-length windows, using full y for correlation."""
        print(f"Shape input to window folder: X={X.shape}, y={y.shape if y is not None else None}")
        fs = fs or 1.0

        # --- 1. Prepare y summary for correlation ---
        if y is not None:
            if y.ndim == 3:  # (pages, timesteps, targets)
                y_for_corr = y[:num_pages_to_use]  # select pages
            else:  # (pages, targets)
                y_for_corr = y[:num_pages_to_use]
        else:
            y_for_corr = None

        # --- 2. Select dominant feature ---
        dom_idx = WindowFolder._select_dominant_feature(X[:num_pages_to_use], y_for_corr)
        X_dom   = X[:num_pages_to_use, :, dom_idx].copy()

        # --- 3. Denoise if requested ---
        if denoise:
            for i in range(X_dom.shape[0]):
                X_dom[i] = WindowFolder._denoise_signal(X_dom[i], fs=fs)

        # --- 4. Estimate window size ---
        if window_size is None:
            window_size = WindowFolder._estimate_period_of_feature(X_dom, fs=fs,
                                                                peak_strength=peak_strength,
                                                                fallback_window=fallback_window,
                                                                max_pages=max_pages)
        else:
            max_possible = X_dom.shape[1]
            if window_size > max_possible:
                print(f"⚠️ Provided window_size {window_size} exceeds sequence length {max_possible}, using {max_possible}")
                window_size = max_possible

        X_folded, y_folded = WindowFolder._fold_and_stack_fixed_pages(X, window_size, y,
                                                                    num_pages_to_use=num_pages_to_use)
        print(f"[INFO] Selected dominant feature index: {dom_idx}, window size: {window_size}")
        return X_folded, y_folded, window_size, dom_idx

