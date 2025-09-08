"""Downloads 3D datasets"""

from __future__ import annotations
import ast
import json
import os
import wfdb

from pathlib import Path
from typing import Dict, List, Literal, Tuple, Optional

import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ECGLoader:
    """Load + process the 'PTB-XL' ECG dataset (https://physionet.org/content/ptb-xl/1.0.3/), including SCP code parsing and label encoding
    Attributes:
        X: 3D array of shape (num_records, num_samples, num_leads), ECG signals.
        y: 2D array of labels, either multi-hot (multi-class) or code confidences (0-100).
        fs: Sampling frequency of the signals (Hz).
        lead_names: List of ECG channel names
    Parameters:
        base_dir: Path to PTB-XL dataset.
        sampling: 'hr' for high-resolution (5000 samples/10s), 'lr' for low-resolution (1000 samples/10s).
        target: 'multi' for multi-hot labels, 'single' for majority-superclass integer labels.
        segment_duration_sec: Desired segment length in seconds; signals are cropped or zero-padded.
        max_records: Optional limit on number of records to load.
        continuous_target: If True, returns code confidence values (0-100) instead of binary labels."""

    def __init__(self, base_dir: str | Path):
        self.base = Path(base_dir)

    @staticmethod
    def parse_scp_codes(s: str) -> dict[str, float]:
        """Parse SCP code string into a dictionary of code → confidence (0-100)."""
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return ast.literal_eval(s)

    @staticmethod
    def extract_superclasses(scp_codes_str: str, scp_super) -> List[str]:
        """Return list of diagnostic conditions 'superclasses' present in SCP string."""
        d            = ECGLoader.parse_scp_codes(scp_codes_str)
        present_scps = [k for k, v in d.items() if float(v) > 0]
        supers       = [scp_super[s] for s in present_scps if s in scp_super]
        return sorted(set(supers))

    @staticmethod
    def compute_majority_superclass(scp_codes_str: str, scp_super) -> str | None:
        """Return the diagnostic conditions 'superclass' with the highest summed confidence."""
        d = ECGLoader.parse_scp_codes(scp_codes_str)
        agg: Dict[str, float] = {}
        for k, w in d.items():
            if k in scp_super:
                agg[scp_super[k]] = agg.get(scp_super[k], 0.0) + float(w)
        return max(agg.items(), key=lambda kv: kv[1])[0] if agg else None

    @staticmethod
    def encode_scp_vector(scp_str: str, all_codes: list[str]) -> np.ndarray:
        """Convert SCP string into a vector of code confidences for all_codes."""
        d = ECGLoader.parse_scp_codes(scp_str)
        return np.array([d.get(code, 0.0) for code in all_codes], dtype=np.float32)

    def load_dataset(self, sampling: Literal["hr", "lr"] = "lr",
                     target: Literal["diagnostic_superclass_multi", "diagnostic_superclass_single"] = "diagnostic_superclass_multi",
                     segment_duration_sec: float | None = 10.0, max_records: int | None = None,
                     continuous_target: bool=False) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
        """Load PTB-XL ECGs.
        - sampling: "hr" = high-res signal (500hz), "lr" = low-res signal (100Hz)
        - target: label (y) format. "multi" = vector of values, "single" = 1 value
        - segment_duration_sec: desired duration of each ECG segment in s (up to the full record length, typically 10s);
        signals are truncated if longer or zero-padded if shorter to produce a uniform number of samples per segment
        - max_records: max # of records to load (otherwise it becomes too big)
        - continuous_target: if True, we get y labels as raw confidences (0-100) instead of binary values
        Returns X (signals), y (labels), sampling_rate, leads."""
        meta      = pd.read_csv(self.base / "ptbxl_database.csv")
        scp       = pd.read_csv(self.base / "scp_statements.csv", index_col=0)
        scp_diag  = scp[scp["diagnostic"] == 1].index.tolist()
        fname_col = "filename_hr" if sampling == "hr" else "filename_lr"
        meta      = meta[[fname_col, "scp_codes"]].copy()

        if max_records is not None:
            meta = meta.iloc[:max_records]

        scp_super    = scp.loc[scp_diag, "diagnostic_class"].to_dict()
        classes      = sorted(set(scp_super.values()))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        rec_supers: List[List[str]] = meta["scp_codes"].map(lambda s: ECGLoader.extract_superclasses(s, scp_super)).tolist()

        if continuous_target:
            all_codes = sorted({code for scp_str in meta["scp_codes"] for code in ECGLoader.parse_scp_codes(scp_str)})
            y         = np.stack([ECGLoader.encode_scp_vector(s, all_codes) for s in meta["scp_codes"]], axis=0)
        elif target == "diagnostic_superclass_multi":
            y = np.zeros((len(rec_supers), len(classes)), dtype=np.float32)
            for i, supers in enumerate(rec_supers):
                for s in supers:
                    y[i, class_to_idx[s]] = 1.0
        else:
            majors = meta["scp_codes"].map(lambda s: ECGLoader.compute_majority_superclass(s, scp_super)).tolist()
            y      = np.array([class_to_idx[m] if m is not None else -1 for m in majors], dtype=np.int64)

        sample_path   = self.base / meta.iloc[0][fname_col]
        sig0, fields0 = wfdb.rdsamp(str(sample_path))
        sampling_rate = int(fields0["fs"])
        leads         = fields0["sig_name"]
        target_len    = int(sampling_rate * segment_duration_sec) if segment_duration_sec is not None else None

        X_list: List[np.ndarray] = []
        for p in meta[fname_col].tolist():
            sig, _ = wfdb.rdsamp(str(self.base / p))
            sig    = sig.astype(np.float32)
            if segment_duration_sec is not None:
                T = sig.shape[0]
                if T >= target_len:
                    sig = sig[:target_len, :]
                else:
                    sig = np.pad(sig, ((0, target_len - T), (0, 0)), mode="constant")
            X_list.append(sig)
        X = np.stack(X_list, axis=0)
        return X, y, sampling_rate, leads

