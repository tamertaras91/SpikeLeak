#!/usr/bin/env python
# coding: utf-8

# # SNN Gradient Leakage — Mini-Batch Experimental Harness
# 
# This notebook is the cleaned experimental version of the working N-MNIST mini-batch attack.
# 
# It keeps the existing two-stage methodology unchanged:
# 
# 1. **Stage 1 — algebraic candidate recovery**
#    - uses only the observed FC weight/bias gradients,
#    - bias augmentation,
#    - row-space recovery,
#    - coordinate compression,
#    - guided MILP candidate enumeration,
#    - **no temporal constraints**.
# 
# 2. **Stage 2 — batch separation and temporal ordering**
#    - uses the known SNN dynamics,
#    - uses recovered downstream spike-pattern multisets,
#    - recovers `B` sequences of length `T`.
# 
# The notebook supports the requested experiment grid
# 
# \[
# T\in\{4,8,12,16\},\qquad
# B\in\{1,4,6,8,12,16,24,32\},
# \]
# 
# with \(L=BT\).
# 
# For each setting it records:
# - `rank(S)`, `rank(S_tilde)`, `rank(G)`, and `rank(D)` per layer, where
#   \(D=[\nabla W\mid\nabla b]\);
# - the row-space condition `rank(D) == rank(S_tilde)`;
# - spike density and number of distinct spike patterns;
# - Stage-1 candidate-pool size and true-pattern coverage;
# - reconstruction success up to a global batch permutation;
# - bit accuracy, Hamming distance, precision, recall, and F1;
# - Stage-1, Stage-2, and total attack time;
# - Stage-2 visited nodes and backtracks;
# - status and explicit failure reason.
# 
# The full experiment grid is enabled by default through `RUN_GRID = True`.
# 
# 
# **Final grid-run version.** The full $T\times B$ sweep is enabled, results are saved incrementally, and Stage-1, Stage-2, attack-only, and full-setting wall times are recorded for every configuration.
# 

# In[ ]:


# ============================================================
# Imports and numerical defaults
# ============================================================

import gc
import sys
import io
import json
import random
import time
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
if str(_REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))
from src.local_datasets import LocalNMNIST

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    import snntorch as snn
    from snntorch import surrogate
except Exception as e:
    raise ImportError("Please install snntorch: pip install snntorch") from e

import tonic
import tonic.transforms as transforms

from scipy.optimize import (
    milp,
    LinearConstraint,
    Bounds,
    linear_sum_assignment,
)

torch.set_default_dtype(torch.float64)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", DEVICE)
print("default dtype:", torch.get_default_dtype())


# In[ ]:


# ============================================================
# Experiment configuration
# ============================================================

# Requested experiment grid
T_VALUES = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
B_VALUES = [1]

# Convenient single-setting debug run
SINGLE_T = 8
SINGLE_B = 1

# Reproducibility
SEED = 0

# Default: same seed as the main experiments.
# For an optional 3-seed run, change to: SEEDS = [0, 1, 2]
SEEDS = [SEED]

# N-MNIST input/model dimensions
IN_DIM = 34 * 34 * 2
NUM_CLASSES = 10

H1 = 512
H2 = 256
H3 = 128

BETA = 0.90
U_THR = 1.0


# FAST-RUNNER OPTIMIZATIONS
# -------------------------
# - Rank/statistics reported only for x0 and s1.
# - Stage 1 reconstructs only x0 and s1.
# - Stage 2 uses s1 only; each x0 candidate test stops after fc1 + LIF1.
# - One forward/backward pass is shared by rank diagnostics and Stage 1.
# - exact_temporal_batch_match is not computed or reported.

# Attack target and Stage-2 cover layers
TARGET_LAYER = "x0"
COVER_LAYERS = ("s1",)

# Rank diagnostics are collected only for the target x0 and Stage-2 cover s1
RANK_LAYERS = ("x0", "s1")
RANK_TOL = None

# Stage 1
MAX_POOL = 1000
EPS_EQ = 1e-8
EPS_PROJ = 1e-7

ZERO_COL_TOL = 1e-12
ONE_ATOL = 1e-13
ONE_RTOL = 1e-10
DUPLICATE_DECIMALS = 12

USE_ZERO_ELIMINATION = True
USE_ONE_ELIMINATION = True
USE_DUPLICATE_ELIMINATION = True

UNAVERAGE_GRADIENTS_BY_BATCH = True

MILP_TIME_LIMIT = 300.0
SELECTION_MODE = "residual"
ENUMERATION_MODE = "guided"
GUIDED_RANDOM_WEIGHT = 1e-10
GUIDED_SEED = 0

# Stage 2
STAGE2_MAX_NODES = 2_000_000
MAX_SEQUENCE_SOLUTIONS = 5

# Files
DATA_DIR = _REPO_IMPORT_ROOT / "data"
RESULTS_DIR = Path("./results_nmnist_T_sweep_B1")
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"
LAYER_CSV = RESULTS_DIR / "layer_diagnostics.csv"

# Qualitative figures saved for every successful grid setting.
SAVE_GRID_FIGURES = False
FIGURE_MAX_COLUMNS = 8
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_CSV = RESULTS_DIR / "t_sweep_report.csv"
AGGREGATE_CSV = RESULTS_DIR / "t_sweep_aggregate.csv"

# Print live Stage-1 / Stage-2 progress during the grid.
# Set False on long unattended runs if the console output is too large.
GRID_VERBOSE = True

# Notebook execution guards.
# Final grid-run configuration.
# RUN_GRID=True means executing the grid cell launches all requested settings.
# RESUME_GRID=True skips settings already present in experiment_summary.csv.
RUN_SINGLE = False
RUN_GRID = True
RESUME_GRID = True

assert MAX_POOL >= max(T_VALUES) * max(B_VALUES), (
    "MAX_POOL must be at least max(T_VALUES)*max(B_VALUES) "
    "if Stage 2 is expected to have access to at least BT unique pool entries."
)

print("T_VALUES:", T_VALUES)
print("B_VALUES:", B_VALUES)
print("largest L=B*T:", max(T_VALUES) * max(B_VALUES))


# In[ ]:


# ============================================================
# Reproducibility and timing helpers
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()



def format_duration(seconds):
    """Human-readable wall-clock duration for progress messages."""
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    seconds = float(seconds)
    hours, rem = divmod(seconds, 3600.0)
    minutes, secs = divmod(rem, 60.0)
    if hours >= 1:
        return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"
    return f"{int(minutes):02d}:{secs:05.2f}"


def deterministic_index_pool(dataset_len, max_batch_size, seed=0):
    """One fixed pool so larger B settings contain the smaller B settings."""
    if max_batch_size > dataset_len:
        raise ValueError("max_batch_size exceeds dataset length.")
    rng = random.Random(seed)
    return rng.sample(range(dataset_len), max_batch_size)


set_seed(SEED)


# ## 1. Model
# 
# The model is recreated from the same seed for each \((T,B)\) setting so that experiments do not share membrane state, gradients, or parameter updates.
# 

# In[ ]:


# ============================================================
# Float64-safe fully connected SNN
# ============================================================

class FCSNN(nn.Module):
    def __init__(
        self,
        in_dim=IN_DIM,
        h1=H1,
        h2=H2,
        h3=H3,
        out=NUM_CLASSES,
        beta=BETA,
        threshold=U_THR,
    ):
        super().__init__()
        sg = surrogate.fast_sigmoid()

        self.fc1  = nn.Linear(in_dim, h1, bias=True)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=sg)

        self.fc2  = nn.Linear(h1, h2, bias=True)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=sg)

        self.fc3  = nn.Linear(h2, h3, bias=True)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=sg)

        self.fc4  = nn.Linear(h3, out, bias=True)
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=sg)

    def _init_mems(self, dtype, device, batch_size=None):
        # snn.Leaky.init_leaky may return scalar-like tensors. The first forward call will broadcast.
        return (
            self.lif1.init_leaky().to(device=device, dtype=dtype),
            self.lif2.init_leaky().to(device=device, dtype=dtype),
            self.lif3.init_leaky().to(device=device, dtype=dtype),
            self.lif4.init_leaky().to(device=device, dtype=dtype),
        )

    def forward(self, x, return_rec=True):
        """
        x shape: [T, B, IN_DIM]

        Returns if return_rec=True:
            out_sum: [B, NUM_CLASSES]
            rec: dict with x0, s1, s2, s3, s4 each [T,B,width]
        """
        dtype = x.dtype
        dev = x.device
        T_local = x.shape[0]

        mem1, mem2, mem3, mem4 = self._init_mems(dtype=dtype, device=dev, batch_size=x.shape[1])

        spk_seq = []
        rec = {"x0": [], "s1": [], "s2": [], "s3": [], "s4": []}

        for t in range(T_local):
            x_t = x[t].to(device=dev, dtype=dtype)

            cur1 = self.fc1(x_t)
            s1, mem1 = self.lif1(cur1, mem1)
            s1 = s1.to(dtype=dtype)
            mem1 = mem1.to(dtype=dtype)

            cur2 = self.fc2(s1)
            s2, mem2 = self.lif2(cur2, mem2)
            s2 = s2.to(dtype=dtype)
            mem2 = mem2.to(dtype=dtype)

            cur3 = self.fc3(s2)
            s3, mem3 = self.lif3(cur3, mem3)
            s3 = s3.to(dtype=dtype)
            mem3 = mem3.to(dtype=dtype)

            cur4 = self.fc4(s3)
            s4, mem4 = self.lif4(cur4, mem4)
            s4 = s4.to(dtype=dtype)
            mem4 = mem4.to(dtype=dtype)

            spk_seq.append(s4)

            if return_rec:
                rec["x0"].append(x_t.detach())
                rec["s1"].append(s1.detach())
                rec["s2"].append(s2.detach())
                rec["s3"].append(s3.detach())
                rec["s4"].append(s4.detach())

        spk_seq = torch.stack(spk_seq, dim=0)  # [T,B,C]

        if not return_rec:
            return spk_seq

        rec = {k: torch.stack(v, dim=0) for k, v in rec.items()}
        out_sum = spk_seq.sum(dim=0)  # [B,C]
        return out_sum, rec


def init_snn_weights(net, w_gain=4.0, b_val=0.1):
    """Useful for avoiding silent hidden layers in untrained experiments."""
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=w_gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, b_val)
    return net


def build_model(seed=SEED, device=DEVICE, w_gain=4.0, b_val=0.1):
    """
    Construct a fresh float64 model.

    Recreating the model per setting avoids gradient/state contamination
    between experiments and makes runs reproducible.
    """
    set_seed(seed)

    net = FCSNN(
        in_dim=IN_DIM,
        h1=H1,
        h2=H2,
        h3=H3,
        out=NUM_CLASSES,
        beta=BETA,
        threshold=U_THR,
    ).to(device).double()

    return init_snn_weights(net, w_gain=w_gain, b_val=b_val)


# ## 2. Dataset
# 
# N-MNIST is framed separately for each temporal horizon \(T\).  
# The same deterministic pool of dataset indices is reused across the grid, so comparisons across \(T\) and \(B\) are controlled.
# 

# In[ ]:


# ============================================================
# Dataset construction and mini-batch loading
# ============================================================

def build_nmnist_dataset(T, save_to=DATA_DIR, train=True):
    """Local-only N-MNIST loader; no download is attempted."""
    split = "Train" if train else "Test"
    root = Path(save_to) / "NMNIST" / split
    return LocalNMNIST(root=root, T=int(T), denoise_filter_time=10000)

def _fit_time_length(frames, T_target):
    frames = np.asarray(frames)

    if frames.shape[0] == T_target:
        return frames

    if frames.shape[0] > T_target:
        return frames[:T_target]

    pad_shape = (T_target - frames.shape[0],) + frames.shape[1:]
    pad = np.zeros(pad_shape, dtype=frames.dtype)

    return np.concatenate([frames, pad], axis=0)


def _label_to_int(label):
    if torch.is_tensor(label):
        return int(label.detach().cpu().item())
    return int(label)


def _normalize_nmnist_frames(frames):
    """
    Normalize a transformed N-MNIST sample to [T,2,34,34].
    """
    frames = np.asarray(frames)

    if frames.ndim == 2 and frames.shape[1] == IN_DIM:
        return frames.reshape(frames.shape[0], 2, 34, 34)

    if frames.ndim != 4:
        raise ValueError(
            f"Expected [T,2,34,34], [T,34,34,2], or [T,{IN_DIM}], "
            f"got {frames.shape}"
        )

    if frames.shape[1:] == (2, 34, 34):
        return frames

    if frames.shape[1:3] == (34, 34) and frames.shape[-1] == 2:
        return np.moveaxis(frames, -1, 1)

    raise ValueError(f"Could not normalize N-MNIST frames with shape {frames.shape}")


def load_batch_from_dataset(
    dataset,
    T,
    B,
    indices,
    device=DEVICE,
    binarize=True,
    dtype=torch.float64,
):
    """
    Returns
    -------
    x : [T,B,2312]
    y : [B]
    raw_frames : list of B arrays [T,2,34,34]
    """
    indices = list(indices)

    if len(indices) != B:
        raise ValueError(f"Expected {B} indices, got {len(indices)}.")

    xs = []
    ys = []
    raw_frames = []

    for idx in indices:
        frames, label = dataset[idx][:2]

        frames = _normalize_nmnist_frames(frames)
        frames = _fit_time_length(frames, T)

        if frames.shape != (T, 2, 34, 34):
            raise ValueError(
                f"Expected [{T},2,34,34], got {frames.shape} for index {idx}."
            )

        if binarize:
            frames = (frames > 0).astype(np.float64)
        else:
            frames = frames.astype(np.float64)

        raw_frames.append(frames)
        xs.append(frames.reshape(T, -1))
        ys.append(_label_to_int(label))

    # Time-major layout used throughout the attack.
    x_np = np.stack(xs, axis=1)  # [T,B,M]
    y_np = np.asarray(ys, dtype=np.int64)

    x = torch.tensor(x_np, dtype=dtype, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    assert x.shape == (T, B, IN_DIM)
    assert y.shape == (B,)

    return x, y, raw_frames


# ## 3. Batch-time and candidate utilities
# 
# All attack tensors use a **time-major** convention:
# 
# ```text
# [T, B, M]
# ```
# 
# and matrices use t-major column order:
# 
# ```text
# (t=0,b=0..B-1), (t=1,b=0..B-1), ...
# ```
# 

# In[ ]:


# ============================================================
# General utilities for batch-time matrices
# ============================================================

def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_candidate_tensor(X, device=None):
    """Ensure candidates have shape [K,1,M]."""
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float64, device=device)
    if not torch.is_tensor(X):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(X)}")
    if device is not None:
        X = X.to(device)
    if X.ndim == 2:
        X = X.unsqueeze(1)
    if X.ndim != 3:
        raise ValueError(f"Expected [K,M] or [K,1,M], got {tuple(X.shape)}")
    return X.to(dtype=torch.float64)


def _time_batch_to_columns(x):
    """
    Convert [T,B,M] to matrix S [M, T*B] using t-major order:
        column index = t*B + b.
    """
    x_np = _to_numpy(x)
    if x_np.ndim != 3:
        raise ValueError(f"Expected [T,B,M], got {x_np.shape}")
    T_local, B_local, M = x_np.shape
    return np.transpose(x_np, (2, 0, 1)).reshape(M, T_local * B_local)


def make_S_tilde_from_rec(spike_rec):
    """Convert recorded [T,B,M] spikes to augmented [M+1,T*B]."""
    S = _time_batch_to_columns(spike_rec)
    S = (S > 0).astype(int)
    bias = np.ones((1, S.shape[1]), dtype=int)
    return np.vstack([S, bias])


def candidate_tuple(z):
    return tuple(np.asarray(z).astype(int).tolist())


def candidate_counter_from_S_tilde(S_tilde):
    return Counter(candidate_tuple(S_tilde[:, i]) for i in range(S_tilde.shape[1]))


def candidates_to_spike_tensor(candidates, width, device):
    """Convert augmented candidates into [K,1,width]."""
    if len(candidates) == 0:
        return torch.empty((0, 1, width), dtype=torch.float64, device=device)
    X = np.stack([np.asarray(z[:-1]).astype(float) for z in candidates], axis=0)
    return torch.tensor(X, dtype=torch.float64, device=device).unsqueeze(1)


def _spike_key(x):
    """Hash one vector [M], [1,M], or [B,M] only if B=1."""
    x = x.detach().cpu() if torch.is_tensor(x) else torch.tensor(x)
    if x.ndim == 2:
        if x.shape[0] != 1:
            raise ValueError("_spike_key expects a single vector, not a batch.")
        x = x[0]
    return tuple((x > 0).to(torch.int64).tolist())


def _make_counter_from_candidate_tensor(X):
    """X shape [K,1,M]."""
    X = _as_candidate_tensor(X)
    return Counter(_spike_key(X[k]) for k in range(X.shape[0]))


def _make_counter_from_time_batch(X):
    """X shape [T,B,M]. Counts all batch-time vectors."""
    X = X.detach().cpu() if torch.is_tensor(X) else torch.tensor(X)
    c = Counter()
    for t in range(X.shape[0]):
        for b in range(X.shape[1]):
            c[_spike_key(X[t, b])] += 1
    return c


def _counter_has(counter, keys):
    need = Counter(keys)
    for k, v in need.items():
        if counter[k] < v:
            return False
    return True


def _counter_has_one(counter, key):
    return counter[key] > 0


def _counter_decrement_one(counter, key):
    counter[key] -= 1


def _counter_increment_one(counter, key):
    counter[key] += 1


def _counter_decrement(counter, keys):
    for k in keys:
        counter[k] -= 1


def _counter_increment(counter, keys):
    for k in keys:
        counter[k] += 1


def matrix_rank_np(A, tol=None):
    A = np.asarray(A)
    if tol is None:
        return np.linalg.matrix_rank(A)
    return np.linalg.matrix_rank(A, tol=tol)


# ## 4. Forward/backward and rank diagnostics
# 
# For each selected FC layer we construct
# 
# \[
# D=[\nabla W\mid\nabla b]
#   =G\widetilde S^\top.
# \]
# 
# The experiment records both the stronger sufficient condition `rank(G)=L` and the directly relevant row-space equality condition
# 
# \[
# \operatorname{rank}(D)=\operatorname{rank}(\widetilde S).
# \]
# 
# Singular values immediately around the numerical rank cutoff are also saved to diagnose numerical-rank ambiguity.
# 

# In[ ]:


# ============================================================
# Forward/backward and layer-wise rank diagnostics
# ============================================================

def _init_mems(net, dtype, device):
    return (
        net.lif1.init_leaky().to(device=device, dtype=dtype),
        net.lif2.init_leaky().to(device=device, dtype=dtype),
        net.lif3.init_leaky().to(device=device, dtype=dtype),
        net.lif4.init_leaky().to(device=device, dtype=dtype),
    )


def forward_with_current_recording(net, x_seq):
    """
    One forward pass for both attack gradients and rank diagnostics.

    The network still propagates through all layers to compute the loss, but
    only x0/s1 spike tensors and cur1/cur2 temporal currents are retained.
    This avoids storing/retaining unused s2/s3/s4 diagnostic tensors.
    """
    dtype = x_seq.dtype
    dev = x_seq.device

    mem1, mem2, mem3, mem4 = _init_mems(net, dtype, dev)

    spk_out_sum = None

    rec_rank = {
        "x0": [], "s1": [],
        "cur1": [], "cur2": [],
    }

    for t in range(x_seq.shape[0]):
        x_t = x_seq[t].to(device=dev, dtype=dtype)

        cur1 = net.fc1(x_t)
        cur1.retain_grad()
        s1, mem1 = net.lif1(cur1, mem1)
        s1 = s1.to(dtype=dtype)
        mem1 = mem1.to(dtype=dtype)

        cur2 = net.fc2(s1)
        cur2.retain_grad()
        s2, mem2 = net.lif2(cur2, mem2)
        s2 = s2.to(dtype=dtype)
        mem2 = mem2.to(dtype=dtype)

        # Deeper layers are required for the loss, but are not retained for
        # diagnostics because this fast runner only reports x0 and s1.
        cur3 = net.fc3(s2)
        s3, mem3 = net.lif3(cur3, mem3)
        s3 = s3.to(dtype=dtype)
        mem3 = mem3.to(dtype=dtype)

        cur4 = net.fc4(s3)
        s4, mem4 = net.lif4(cur4, mem4)
        s4 = s4.to(dtype=dtype)
        mem4 = mem4.to(dtype=dtype)

        spk_out_sum = s4 if spk_out_sum is None else spk_out_sum + s4

        rec_rank["x0"].append(x_t.detach())
        rec_rank["s1"].append(s1.detach())
        rec_rank["cur1"].append(cur1)
        rec_rank["cur2"].append(cur2)

    rec_rank["x0"] = torch.stack(rec_rank["x0"], dim=0)
    rec_rank["s1"] = torch.stack(rec_rank["s1"], dim=0)

    return spk_out_sum, rec_rank


def _G_from_cur_list(cur_list):
    """
    Build G [N,BT] in t-major order:
        columns = (t=0,b=0..B-1), then (t=1,b=0..B-1), ...
    """
    grads = []

    for cur in cur_list:
        if cur.grad is None:
            raise RuntimeError(
                "cur.grad is None. Did you call retain_grad() and backward()?"
            )
        grads.append(cur.grad.detach().cpu())  # [B,N]

    G_tbn = torch.stack(grads, dim=0)  # [T,B,N]
    T_local, B_local, N = G_tbn.shape

    return (
        G_tbn
        .permute(2, 0, 1)
        .reshape(N, T_local * B_local)
        .numpy()
    )


def _rank_and_spectrum(A, tol=None):
    """
    Numerical rank plus the singular values immediately around the cutoff.
    """
    A = np.asarray(A, dtype=np.float64)

    svals = np.linalg.svd(A, compute_uv=False)

    if svals.size == 0:
        return {
            "rank": 0,
            "tol": 0.0 if tol is None else float(tol),
            "smallest_kept": np.nan,
            "next_singular": np.nan,
            "spectral_gap_ratio": np.nan,
            "largest_singular": np.nan,
        }

    if tol is None:
        used_tol = max(A.shape) * np.finfo(np.float64).eps * svals[0]
    else:
        used_tol = float(tol)

    rank = int(np.sum(svals > used_tol))

    smallest_kept = float(svals[rank - 1]) if rank > 0 else np.nan
    next_singular = float(svals[rank]) if rank < len(svals) else np.nan

    if (
        rank > 0
        and rank < len(svals)
        and np.isfinite(next_singular)
        and next_singular > 0
    ):
        spectral_gap_ratio = smallest_kept / next_singular
    else:
        spectral_gap_ratio = np.inf if rank > 0 else np.nan

    return {
        "rank": rank,
        "tol": used_tol,
        "smallest_kept": smallest_kept,
        "next_singular": next_singular,
        "spectral_gap_ratio": float(spectral_gap_ratio),
        "largest_singular": float(svals[0]),
    }


def layer_rank_diagnostics(
    net,
    x,
    y,
    criterion,
    layers=RANK_LAYERS,
    rank_tol=RANK_TOL,
    print_results=False,
):
    """
    Compute layer-wise diagnostics.

    Notation used in the experiment tables:
        D := [grad_W | grad_b]

    The row-space condition tested experimentally is:
        rank(D) == rank(S_tilde)
    """
    selected_layers = list(layers)

    specs = {
        "x0": {
            "layer_name": "fc1",
            "presynaptic": "x0",
            "cur": "cur1",
            "layer": net.fc1,
        },
        "s1": {
            "layer_name": "fc2",
            "presynaptic": "s1",
            "cur": "cur2",
            "layer": net.fc2,
        },
    }

    net.zero_grad(set_to_none=True)

    out, rec_rank = forward_with_current_recording(net, x)
    loss = criterion(out, y)
    loss.backward()

    T_local, B_local = x.shape[:2]
    L = T_local * B_local

    results = {}

    for name in selected_layers:
        if name not in specs:
            raise ValueError(f"Unknown layer {name}.")

        spec = specs[name]
        layer = spec["layer"]

        G = _G_from_cur_list(rec_rank[spec["cur"]])  # [N,L]

        S = _time_batch_to_columns(
            rec_rank[spec["presynaptic"]]
        )  # [M,L]

        S_bin = (S > 0).astype(np.float64)

        S_tilde = np.vstack([
            S_bin,
            np.ones((1, L), dtype=np.float64),
        ])

        grad_W = layer.weight.grad.detach().cpu().numpy().astype(np.float64)
        grad_b = layer.bias.grad.detach().cpu().numpy().astype(np.float64)

        # D is the observed bias-augmented gradient matrix.
        D = np.concatenate(
            [grad_W, grad_b[:, None]],
            axis=1,
        )

        rank_G = matrix_rank_np(G, rank_tol)
        rank_S = matrix_rank_np(S_bin, rank_tol)
        rank_S_tilde = matrix_rank_np(S_tilde, rank_tol)

        D_info = _rank_and_spectrum(D, tol=rank_tol)
        rank_D = D_info["rank"]

        # Verify the analytical factorization numerically.
        grad_W_hat = G @ S_bin.T
        grad_b_hat = G @ np.ones((L,), dtype=G.dtype)

        rel_W_err = (
            np.linalg.norm(grad_W - grad_W_hat)
            /
            (np.linalg.norm(grad_W) + 1e-12)
        )

        rel_b_err = (
            np.linalg.norm(grad_b - grad_b_hat)
            /
            (np.linalg.norm(grad_b) + 1e-12)
        )

        distinct_patterns = int(
            np.unique(
                S_bin.T.astype(np.uint8),
                axis=0,
            ).shape[0]
        )

        results[name] = {
            "layer_name": spec["layer_name"],
            "presynaptic": spec["presynaptic"],
            "L": L,

            "G_shape": tuple(G.shape),
            "S_shape": tuple(S_bin.shape),
            "S_tilde_shape": tuple(S_tilde.shape),
            "D_shape": tuple(D.shape),

            "rank_G": int(rank_G),
            "rank_S": int(rank_S),
            "rank_S_tilde": int(rank_S_tilde),
            "rank_D": int(rank_D),

            "rank_condition": bool(rank_D == rank_S_tilde),
            "rank_G_full_L": bool(rank_G == L),
            "rank_S_tilde_full_L": bool(rank_S_tilde == L),

            "spike_density": float(S_bin.mean()),
            "num_distinct_spike_patterns": distinct_patterns,

            "factorization_rel_W_error": float(rel_W_err),
            "factorization_rel_b_error": float(rel_b_err),

            "D_rank_tolerance": float(D_info["tol"]),
            "D_largest_singular": D_info["largest_singular"],
            "D_smallest_kept_singular": D_info["smallest_kept"],
            "D_next_singular": D_info["next_singular"],
            "D_spectral_gap_ratio": D_info["spectral_gap_ratio"],
        }

    if print_results:
        print(
            f"T={T_local}, B={B_local}, L={L}, "
            f"loss={float(loss.detach().cpu()):.6g}"
        )

        for name, r in results.items():
            print(
                f"{name:>2s} | "
                f"rank(S)={r['rank_S']:4d} | "
                f"rank(S~)={r['rank_S_tilde']:4d} | "
                f"rank(G)={r['rank_G']:4d} | "
                f"rank(D)={r['rank_D']:4d} | "
                f"condition={r['rank_condition']} | "
                f"density={r['spike_density']:.4f} | "
                f"distinct={r['num_distinct_spike_patterns']}"
            )

    # Return only the spike tensors needed by Stage 1/Stage 2.
    rec_spikes = {"x0": rec_rank["x0"], "s1": rec_rank["s1"]}
    return results, rec_spikes, loss


# ## 5. Stage 1 — coordinate compression and row-space recovery
# 
# The zero-column, one-column, and duplicate-column rules reduce the binary dimension before MILP enumeration.
# 

# In[ ]:


# ============================================================
# Augmented gradient, coordinate compression, SVD, projection
# ============================================================

def _close_vector(a, b, atol=1e-12, rtol=1e-10):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.linalg.norm(a - b)
    scale = max(np.linalg.norm(a), np.linalg.norm(b), 1.0)
    return diff <= atol + rtol * scale


def compress_input_coordinates_from_gradient(
    gW,
    gb,
    zero_tol=1e-12,
    one_atol=1e-12,
    one_rtol=1e-10,
    duplicate_decimals=12,
    verify_duplicates=True,
):
    """Detect all-zero, all-one, and duplicate input-neuron columns."""
    gW = np.asarray(gW)
    gb = np.asarray(gb)
    full_width = gW.shape[1]

    zero_idx, one_idx, variable_idx = [], [], []

    for j in range(full_width):
        col = gW[:, j]
        if np.linalg.norm(col) <= zero_tol:
            zero_idx.append(j)
        elif _close_vector(col, gb, atol=one_atol, rtol=one_rtol):
            one_idx.append(j)
        else:
            variable_idx.append(j)

    zero_idx = np.asarray(zero_idx, dtype=int)
    one_idx = np.asarray(one_idx, dtype=int)
    variable_idx = np.asarray(variable_idx, dtype=int)

    buckets = {}
    for j in variable_idx:
        key = np.round(gW[:, j], decimals=duplicate_decimals).tobytes()
        buckets.setdefault(key, []).append(int(j))

    groups = []
    for bucket in buckets.values():
        if not verify_duplicates or len(bucket) == 1:
            groups.append(np.asarray(bucket, dtype=int))
        else:
            used = set()
            for j in bucket:
                if j in used:
                    continue
                group = [j]
                used.add(j)
                for k in bucket:
                    if k in used:
                        continue
                    if _close_vector(gW[:, j], gW[:, k], atol=one_atol, rtol=one_rtol):
                        group.append(k)
                        used.add(k)
                groups.append(np.asarray(group, dtype=int))

    rep_idx = np.asarray([g[0] for g in groups], dtype=int) if groups else np.array([], dtype=int)

    return {
        "full_width": full_width,
        "zero_idx": zero_idx,
        "one_idx": one_idx,
        "variable_idx": variable_idx,
        "groups": groups,
        "rep_idx": rep_idx,
        "num_zero": len(zero_idx),
        "num_one": len(one_idx),
        "num_variable_original": len(variable_idx),
        "num_representatives": len(rep_idx),
        "num_removed_by_duplicates": len(variable_idx) - len(rep_idx),
        "duplicate_decimals": duplicate_decimals,
    }


def build_compressed_augmented_gradient(
    fc_layer,
    zero_tol=1e-12,
    one_atol=1e-12,
    one_rtol=1e-10,
    duplicate_decimals=12,
    use_zero_elimination=True,
    use_one_elimination=True,
    use_duplicate_elimination=True,
    unaverage_gradients_by_batch=False,
    batch_size=1,
):
    """Build M=[grad_W_compressed | grad_b] and return compression map."""
    gW = fc_layer.weight.grad.detach().cpu().numpy().astype(np.float64)
    gb = fc_layer.bias.grad.detach().cpu().numpy().astype(np.float64)

    # CrossEntropyLoss(reduction='mean') scales gradients by 1/B.
    # Multiplying by B does not change row space, but can be useful for interpretation.
    if unaverage_gradients_by_batch and batch_size is not None:
        gW = gW * float(batch_size)
        gb = gb * float(batch_size)

    full_width = gW.shape[1]

    if not (use_zero_elimination or use_one_elimination or use_duplicate_elimination):
        groups = [np.asarray([j], dtype=int) for j in range(full_width)]
        compression = {
            "full_width": full_width,
            "zero_idx": np.array([], dtype=int),
            "one_idx": np.array([], dtype=int),
            "variable_idx": np.arange(full_width, dtype=int),
            "groups": groups,
            "rep_idx": np.arange(full_width, dtype=int),
            "num_zero": 0,
            "num_one": 0,
            "num_variable_original": full_width,
            "num_representatives": full_width,
            "num_removed_by_duplicates": 0,
        }
    else:
        base = compress_input_coordinates_from_gradient(
            gW, gb,
            zero_tol=zero_tol,
            one_atol=one_atol,
            one_rtol=one_rtol,
            duplicate_decimals=duplicate_decimals,
            verify_duplicates=True,
        )

        groups = []
        if use_duplicate_elimination:
            groups.extend(base["groups"])
        else:
            for j in base["variable_idx"]:
                groups.append(np.asarray([j], dtype=int))

        zero_idx = base["zero_idx"] if use_zero_elimination else np.array([], dtype=int)
        one_idx = base["one_idx"] if use_one_elimination else np.array([], dtype=int)

        if not use_zero_elimination:
            for j in base["zero_idx"]:
                groups.append(np.asarray([j], dtype=int))
        if not use_one_elimination:
            for j in base["one_idx"]:
                groups.append(np.asarray([j], dtype=int))

        groups = sorted(groups, key=lambda g: int(g[0]))
        rep_idx = np.asarray([g[0] for g in groups], dtype=int) if groups else np.array([], dtype=int)

        compression = dict(base)
        compression["zero_idx"] = zero_idx
        compression["one_idx"] = one_idx
        compression["groups"] = groups
        compression["rep_idx"] = rep_idx
        compression["num_zero"] = len(zero_idx)
        compression["num_one"] = len(one_idx)
        compression["num_representatives"] = len(rep_idx)

    gW_comp = gW[:, compression["rep_idx"]]
    M_comp = np.concatenate([gW_comp, gb[:, None]], axis=1)
    return M_comp, compression, gW, gb


def expand_compressed_candidate(z_compressed, compression):
    full_width = compression["full_width"]
    z_full = np.zeros(full_width + 1, dtype=int)
    z_full[compression["zero_idx"]] = 0
    z_full[compression["one_idx"]] = 1
    for local_pos, group in enumerate(compression["groups"]):
        z_full[group] = int(z_compressed[local_pos])
    z_full[-1] = int(z_compressed[-1])
    return z_full


def low_rank_factor_from_matrix(M_tilde, target_rank, rank_mode="target", rank_tol=1e-10):
    M_tilde = np.asarray(M_tilde, dtype=np.float64)
    U, svals, Vt = np.linalg.svd(M_tilde, full_matrices=False)
    if rank_mode in ["target", "T"]:
        r = min(target_rank, len(svals))
    elif rank_mode == "numerical":
        r = int(np.sum(svals > rank_tol)); r = max(r, 1)
    else:
        raise ValueError("rank_mode must be 'target'/'T' or 'numerical'.")
    L = U[:, :r] @ np.diag(svals[:r])
    R = Vt[:r, :]
    return L, R, svals


def projection_residual(z, R):
    """Distance from z to rowspace(R). R rows from SVD are orthonormal."""
    z = np.asarray(z).astype(float)
    proj = R.T @ (R @ z)
    return np.linalg.norm(z - proj)


def candidate_metrics(z, h, R):
    zf = np.asarray(z).astype(float)
    v = R.T @ h
    return {
        "proj": projection_residual(zf, R),
        "eq_l2": float(np.linalg.norm(v - zf)),
        "eq_max": float(np.max(np.abs(v - zf))),
        "bin_res": float(np.sum(np.minimum(v**2, (v - 1.0)**2))),
        "bias_err": float(abs(v[-1] - 1.0)),
    }


# ### Guided MILP candidate enumeration
# 
# The guided mode minimizes coordinate slacks subject to binary candidate variables and the augmented bias coordinate.
# 

# In[ ]:


# ============================================================
# MILP candidate enumeration: ranked feasibility and guided slack-minimizing modes
# ============================================================

def solve_one_binary_candidate_ranked(R, previous_candidates=None, eps_eq=1e-8, time_limit=300.0):
    if previous_candidates is None:
        previous_candidates = []
    r, d = R.shape
    num_vars = r + d
    c = np.zeros(num_vars)

    bounds = Bounds(
        np.concatenate([-np.inf * np.ones(r), np.zeros(d)]),
        np.concatenate([ np.inf * np.ones(r), np.ones(d)]),
    )
    integrality = np.concatenate([np.zeros(r), np.ones(d)])
    constraints = []

    A = np.zeros((d, num_vars))
    A[:, :r] = R.T
    A[:, r:] = -np.eye(d)
    constraints.append(LinearConstraint(A, -eps_eq * np.ones(d), eps_eq * np.ones(d)))

    A_bias = np.zeros((1, num_vars)); A_bias[0, r + d - 1] = 1.0
    constraints.append(LinearConstraint(A_bias, np.array([1.0]), np.array([1.0])))

    for z_prev in previous_candidates:
        z_prev = np.asarray(z_prev).astype(int)
        coeff = np.zeros(num_vars)
        num_ones = int(z_prev.sum())
        for j in range(d):
            coeff[r + j] = 1.0 if z_prev[j] == 0 else -1.0
        constraints.append(LinearConstraint(coeff.reshape(1, -1), np.array([1.0 - num_ones]), np.array([np.inf])))

    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints,
                  options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if not result.success or result.x is None:
        return None, None, result
    h = result.x[:r]
    z = np.rint(result.x[r:]).astype(int)
    return h, z, result


def solve_one_binary_candidate_guided(
    R,
    previous_candidates=None,
    eps_eq_bound=1e-8,
    time_limit=300.0,
    random_weight=0.0,
    rng=None,
):
    """Minimize sum of coordinate slacks u_j for |R.T h - z|."""
    if previous_candidates is None:
        previous_candidates = []
    if rng is None:
        rng = np.random.default_rng(0)

    r, d = R.shape
    # variables: [h, z, u]
    num_vars = r + d + d
    c = np.zeros(num_vars)
    c[r + d:] = 1.0
    if random_weight > 0:
        rc = rng.normal(size=d); rc[-1] = 0.0
        c[r:r+d] = random_weight * rc

    lb = np.concatenate([-np.inf * np.ones(r), np.zeros(d), np.zeros(d)])
    ub = np.concatenate([ np.inf * np.ones(r), np.ones(d), eps_eq_bound * np.ones(d)])
    bounds = Bounds(lb, ub)
    integrality = np.concatenate([np.zeros(r), np.ones(d), np.zeros(d)])
    constraints = []

    # R.T h - z - u <= 0
    A1 = np.zeros((d, num_vars))
    A1[:, :r] = R.T
    A1[:, r:r+d] = -np.eye(d)
    A1[:, r+d:] = -np.eye(d)
    constraints.append(LinearConstraint(A1, -np.inf * np.ones(d), np.zeros(d)))

    # R.T h - z + u >= 0
    A2 = np.zeros((d, num_vars))
    A2[:, :r] = R.T
    A2[:, r:r+d] = -np.eye(d)
    A2[:, r+d:] = np.eye(d)
    constraints.append(LinearConstraint(A2, np.zeros(d), np.inf * np.ones(d)))

    # bias z[-1] = 1
    A_bias = np.zeros((1, num_vars)); A_bias[0, r + d - 1] = 1.0
    constraints.append(LinearConstraint(A_bias, np.array([1.0]), np.array([1.0])))

    # no-good cuts on z
    for z_prev in previous_candidates:
        z_prev = np.asarray(z_prev).astype(int)
        coeff = np.zeros(num_vars)
        num_ones = int(z_prev.sum())
        for j in range(d):
            coeff[r + j] = 1.0 if z_prev[j] == 0 else -1.0
        constraints.append(LinearConstraint(coeff.reshape(1, -1), np.array([1.0 - num_ones]), np.array([np.inf])))

    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints,
                  options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if not result.success or result.x is None:
        return None, None, None, result
    h = result.x[:r]
    z = np.rint(result.x[r:r+d]).astype(int)
    u = result.x[r+d:]
    return h, z, u, result


def sort_candidate_pool(pool):
    return sorted(pool, key=lambda item: (
        item["metrics"].get("proj", np.inf),
        item["metrics"].get("slack_l1", np.inf),
        item["metrics"].get("eq_max", np.inf),
        item["metrics"].get("eq_l2", np.inf),
        item["metrics"].get("bin_res", np.inf),
        item["metrics"].get("bias_err", np.inf),
    ))


def enumerate_binary_candidates(
    R,
    top_k,
    max_pool=1000,
    eps_eq=1e-8,
    eps_proj=np.inf,
    time_limit=300.0,
    enumeration_mode="guided",
    guided_random_weight=1e-10,
    guided_seed=0,
    verbose=True,
):
    """Enumerate candidate augmented spike vectors using ranked or guided MILP."""
    pool = []
    rejected_by_proj = []
    excluded = []
    rng = np.random.default_rng(guided_seed)
    start = time.perf_counter()
    certified_complete = False
    stop_reason = None

    for k in range(max_pool):
        t0 = time.perf_counter()
        if enumeration_mode == "guided":
            h, z, u, result = solve_one_binary_candidate_guided(
                R=R,
                previous_candidates=excluded,
                eps_eq_bound=eps_eq,
                time_limit=time_limit,
                random_weight=guided_random_weight,
                rng=rng,
            )
        elif enumeration_mode == "ranked":
            h, z, result = solve_one_binary_candidate_ranked(
                R=R,
                previous_candidates=excluded,
                eps_eq=eps_eq,
                time_limit=time_limit,
            )
            u = None
        else:
            raise ValueError("enumeration_mode must be 'guided' or 'ranked'.")

        solve_time = time.perf_counter() - t0
        elapsed = time.perf_counter() - start

        if z is None:
            stop_reason = result.message
            if getattr(result, "status", None) == 2:
                certified_complete = True
                stop_reason = "infeasible: no more candidates"
            if verbose:
                print("\nStopping enumeration.")
                print("mode:", enumeration_mode)
                print("status:", getattr(result, "status", None))
                print("message:", result.message)
            break

        excluded.append(z)
        metrics = candidate_metrics(z, h, R)
        if u is not None:
            metrics["slack_l1"] = float(np.sum(u))
            metrics["slack_max"] = float(np.max(u))
        else:
            metrics["slack_l1"] = metrics["eq_l2"]
            metrics["slack_max"] = metrics["eq_max"]

        item = {"z": z, "h": h, "u": u, "metrics": metrics, "solve_time": solve_time, "elapsed": elapsed}

        if metrics["proj"] <= eps_proj:
            pool.append(item)
            if verbose:
                print(
                    f"{enumeration_mode} stored {len(pool):04d} | "
                    f"proj={metrics['proj']:.3e} | slack_l1={metrics['slack_l1']:.3e} | "
                    f"eq_max={metrics['eq_max']:.3e} | solve={solve_time:.3f}s | elapsed={elapsed:.3f}s"
                )
        else:
            rejected_by_proj.append(item)

    pool_sorted = sort_candidate_pool(pool)
    top_items = pool_sorted[:top_k]
    return {
        "candidates": [item["z"] for item in top_items],
        "hs": [item["h"] for item in top_items],
        "top_items": top_items,
        "pool": pool_sorted,
        "rejected_by_proj": rejected_by_proj,
        "excluded_candidates": excluded,
        "certified_complete": certified_complete,
        "stop_reason": stop_reason,
        "time": time.perf_counter() - start,
        "enumeration_mode": enumeration_mode,
    }


# ### Candidate selection and Stage-1 reconstruction
# 

# In[ ]:


# ============================================================
# Candidate selection and reconstruction
# ============================================================

def score_candidate_set_reconstruction(M_tilde, candidates):
    """Score how well selected candidates reconstruct M_tilde = G Z.T."""
    if len(candidates) == 0:
        return {"rel_recon": np.inf, "bias_rel": np.inf, "rank_Z": 0}
    Z = np.stack([np.asarray(z).astype(float) for z in candidates], axis=1)  # [M+1,K]
    G_hat = M_tilde @ Z @ np.linalg.pinv(Z.T @ Z)
    M_hat = G_hat @ Z.T
    rel = np.linalg.norm(M_tilde - M_hat, ord="fro") / (np.linalg.norm(M_tilde, ord="fro") + 1e-12)
    bias_true = M_tilde[:, -1]
    bias_hat = M_hat[:, -1]
    bias_rel = np.linalg.norm(bias_true - bias_hat) / (np.linalg.norm(bias_true) + 1e-12)
    return {"rel_recon": rel, "bias_rel": bias_rel, "rank_Z": np.linalg.matrix_rank(Z), "G_hat": G_hat, "M_hat": M_hat}


def select_candidates_by_global_reconstruction(M_tilde, pool, top_k, candidate_limit=100, rank_penalty=10.0, verbose=True):
    """Greedy set-level candidate selection using augmented-gradient reconstruction."""
    candidate_pool = pool[:candidate_limit] if candidate_limit is not None else pool
    selected = []
    remaining = list(range(len(candidate_pool)))
    history = []

    for step in range(min(top_k, len(candidate_pool))):
        best_idx, best_score, best_diag = None, np.inf, None
        for idx in remaining:
            trial = selected + [idx]
            cand = [candidate_pool[i]["z"] for i in trial]
            diag = score_candidate_set_reconstruction(M_tilde, cand)
            rank_deficit = max(0, len(cand) - diag["rank_Z"])
            score = diag["rel_recon"] + diag["bias_rel"] + rank_penalty * rank_deficit
            if score < best_score:
                best_idx, best_score, best_diag = idx, score, diag
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        history.append({"step": step+1, "idx": best_idx, "score": best_score, **{k:v for k,v in best_diag.items() if k not in ['G_hat','M_hat']}})
        if verbose:
            print(f"global step {step+1:02d} | idx={best_idx} | rel={best_diag['rel_recon']:.3e} | bias={best_diag['bias_rel']:.3e} | rank_Z={best_diag['rank_Z']}")

    selected_items = [candidate_pool[i] for i in selected]
    final_diag = score_candidate_set_reconstruction(M_tilde, [item["z"] for item in selected_items])
    return selected_items, {"history": history, "final_diag": final_diag, "selected_indices": selected}


def verify_candidates(pool, recovered_candidates, S_tilde_gt, name="layer"):
    true_counter = candidate_counter_from_S_tilde(S_tilde_gt)
    pool_counter = Counter(candidate_tuple(item["z"]) for item in pool)
    rec_counter = Counter(candidate_tuple(z) for z in recovered_candidates)

    true_subset_pool = all(pool_counter[k] >= v for k, v in true_counter.items())
    rec_subset_true = all(true_counter[k] >= v for k, v in rec_counter.items())
    true_in_rec = sum(min(v, rec_counter[k]) for k, v in true_counter.items())
    missing = {k: v - pool_counter[k] for k, v in true_counter.items() if pool_counter[k] < v}
    false = {k: v - true_counter[k] for k, v in rec_counter.items() if rec_counter[k] < v}

    print("\n" + "-" * 70)
    print(f"Verification for {name}")
    print("-" * 70)
    print("true total candidates BT:", sum(true_counter.values()))
    print("true unique vectors:", len(true_counter))
    print("pool unique candidates:", len(pool_counter))
    print("recovered/top total:", sum(rec_counter.values()))
    print("recovered/top unique:", len(rec_counter))
    print("Question 1: Are true vectors subset of the whole candidate pool?")
    print("Answer:", true_subset_pool)
    print("Question 2: Are recovered/top candidates subset of true vectors?")
    print("Answer:", rec_subset_true)
    print("true vectors in recovered/top multiset:", true_in_rec, "/", sum(true_counter.values()))
    print("missing true multiplicity from pool:", sum(missing.values()))
    print("false recovered multiplicity:", sum(false.values()))

    return {
        "true_subset_of_pool": true_subset_pool,
        "recovered_subset_of_true": rec_subset_true,
        "num_true_total": sum(true_counter.values()),
        "num_true_unique": len(true_counter),
        "num_pool_unique": len(pool_counter),
        "num_recovered_total": sum(rec_counter.values()),
        "num_recovered_unique": len(rec_counter),
        "num_true_in_recovered": true_in_rec,
        "missing_true": missing,
        "false_recovered": false,
    }


def recover_input_spikes_from_layer(
    fc_layer,
    name,
    width,
    target_rank,
    device,
    S_tilde_gt=None,
    top_k=None,
    max_pool=1000,
    eps_eq=1e-8,
    eps_proj=np.inf,
    zero_col_tol=1e-12,
    one_atol=1e-12,
    one_rtol=1e-10,
    duplicate_decimals=12,
    use_zero_elimination=True,
    use_one_elimination=True,
    use_duplicate_elimination=True,
    unaverage_gradients_by_batch=False,
    batch_size=1,
    time_limit=300.0,
    rank_mode="target",
    selection_mode="residual",
    global_candidate_limit=100,
    rank_penalty=10.0,
    enumeration_mode="guided",
    guided_random_weight=1e-10,
    guided_seed=0,
    verbose=True,
):
    if top_k is None:
        top_k = target_rank

    print("\n" + "=" * 80)
    print(f"Recovering {name}")
    print("=" * 80)

    M_comp, compression, gW, gb = build_compressed_augmented_gradient(
        fc_layer,
        zero_tol=zero_col_tol,
        one_atol=one_atol,
        one_rtol=one_rtol,
        duplicate_decimals=duplicate_decimals,
        use_zero_elimination=use_zero_elimination,
        use_one_elimination=use_one_elimination,
        use_duplicate_elimination=use_duplicate_elimination,
        unaverage_gradients_by_batch=unaverage_gradients_by_batch,
        batch_size=batch_size,
    )

    print("Original grad_W shape:", gW.shape)
    print("Compressed M shape:", M_comp.shape)
    print("compression zero/one/reps:", compression["num_zero"], compression["num_one"], compression["num_representatives"])

    L, R, svals = low_rank_factor_from_matrix(M_comp, target_rank=target_rank, rank_mode=rank_mode)
    print("rank(M_comp):", np.linalg.matrix_rank(M_comp))
    print("R shape:", R.shape)
    print("target_rank:", target_rank)
    if len(svals) >= target_rank:
        print("smallest kept singular value:", svals[target_rank-1])
    if len(svals) > target_rank:
        print("next singular value:", svals[target_rank])

    enum = enumerate_binary_candidates(
        R=R,
        top_k=top_k,
        max_pool=max_pool,
        eps_eq=eps_eq,
        eps_proj=eps_proj,
        time_limit=time_limit,
        enumeration_mode=enumeration_mode,
        guided_random_weight=guided_random_weight,
        guided_seed=guided_seed,
        verbose=verbose,
    )

    if selection_mode == "residual":
        selected_items = enum["top_items"]
        global_selection = None
    elif selection_mode == "global":
        selected_items, global_selection = select_candidates_by_global_reconstruction(
            M_tilde=M_comp,
            pool=enum["pool"],
            top_k=top_k,
            candidate_limit=global_candidate_limit,
            rank_penalty=rank_penalty,
            verbose=verbose,
        )
    else:
        raise ValueError("selection_mode must be 'residual' or 'global'.")

    comp_candidates = [item["z"] for item in selected_items]
    full_candidates = [expand_compressed_candidate(z, compression) for z in comp_candidates]

    full_pool = []
    for item in enum["pool"]:
        item_full = dict(item)
        item_full["z_compressed"] = item["z"]
        item_full["z"] = expand_compressed_candidate(item["z"], compression)
        item_full["compression"] = compression
        full_pool.append(item_full)

    verification = None
    if S_tilde_gt is not None:
        verification = verify_candidates(full_pool, full_candidates, S_tilde_gt, name=name)

    spikes = candidates_to_spike_tensor(full_candidates, width, device)
    return {
        "name": name,
        "spikes": spikes,
        "candidates_raw": full_candidates,
        "candidates_compressed": comp_candidates,
        "pool": full_pool,
        "pool_compressed": enum["pool"],
        "top_items": selected_items,
        "compression": compression,
        "M_tilde": M_comp,
        "L": L,
        "R": R,
        "singular_values": svals,
        "verification": verification,
        "pool_verification": verification,
        "global_selection": global_selection,
        "enumeration": enum,
        "enumeration_mode": enumeration_mode,
        "selection_mode": selection_mode,
        "target_rank": target_rank,
        "time": enum["time"],
    }


def recover_selected_layers(
    net,
    rec,
    T,
    B,
    device,
    layers="all",
    top_k=None,
    max_pool=1000,
    eps_eq=1e-8,
    eps_proj=np.inf,
    zero_col_tol=1e-12,
    one_atol=1e-12,
    one_rtol=1e-10,
    duplicate_decimals=12,
    use_zero_elimination=True,
    use_one_elimination=True,
    use_duplicate_elimination=True,
    unaverage_gradients_by_batch=False,
    time_limit=300.0,
    rank_mode="target",
    selection_mode="residual",
    global_candidate_limit=100,
    rank_penalty=10.0,
    enumeration_mode="guided",
    guided_random_weight=1e-10,
    guided_seed=0,
    verbose=True,
):
    target_rank = T * B
    layer_specs = {
        "x0": (net.fc1, net.fc1.in_features, rec["x0"]),
        "s1": (net.fc2, net.fc2.in_features, rec["s1"]),
        "s2": (net.fc3, net.fc3.in_features, rec["s2"]),
        "s3": (net.fc4, net.fc4.in_features, rec["s3"]),
    }
    selected = ["x0", "s1", "s2", "s3"] if layers == "all" else ([layers] if isinstance(layers, str) else list(layers))
    recovered = {}
    for name in selected:
        if name not in layer_specs:
            raise ValueError(f"Unknown layer {name}; choose from {list(layer_specs)}")
        layer, width, rec_spikes = layer_specs[name]
        S_tilde_gt = make_S_tilde_from_rec(rec_spikes)
        recovered[name] = recover_input_spikes_from_layer(
            fc_layer=layer,
            name=name,
            width=width,
            target_rank=target_rank,
            device=device,
            S_tilde_gt=S_tilde_gt,
            top_k=top_k if top_k is not None else target_rank,
            max_pool=max_pool,
            eps_eq=eps_eq,
            eps_proj=eps_proj,
            zero_col_tol=zero_col_tol,
            one_atol=one_atol,
            one_rtol=one_rtol,
            duplicate_decimals=duplicate_decimals,
            use_zero_elimination=use_zero_elimination,
            use_one_elimination=use_one_elimination,
            use_duplicate_elimination=use_duplicate_elimination,
            unaverage_gradients_by_batch=unaverage_gradients_by_batch,
            batch_size=B,
            time_limit=time_limit,
            rank_mode=rank_mode,
            selection_mode=selection_mode,
            global_candidate_limit=global_candidate_limit,
            rank_penalty=rank_penalty,
            enumeration_mode=enumeration_mode,
            guided_random_weight=guided_random_weight,
            guided_seed=guided_seed,
            verbose=verbose,
        )
    return recovered


# ## 6. Stage 2 — sequence separation and temporal ordering
# 
# This notebook keeps the scalable **sequential sequence recovery** version and removes the older factorial `permutations(available, B)` implementation.
# 
# A complete sequence is recovered, its target candidates and matched downstream patterns are consumed, and the search proceeds to the next batch member with backtracking when required.
# 

# In[ ]:


def extract_candidates_from_recovered_layer(recovered_layer, source="spikes", device=None, sort_pool=True):
    """Extract candidates as tensor [K,1,M]."""
    if torch.is_tensor(recovered_layer):
        X = _as_candidate_tensor(recovered_layer, device=device)
        return X, list(range(X.shape[0])), [{"source_index": i} for i in range(X.shape[0])]

    if source == "spikes":
        X = _as_candidate_tensor(recovered_layer["spikes"], device=device)
        return X, list(range(X.shape[0])), [{"source_index": i} for i in range(X.shape[0])]

    if source == "pool":
        pool = list(recovered_layer["pool"])
        if sort_pool:
            pool = sort_candidate_pool(pool)
        vecs, ids, items = [], [], []
        for i, item in enumerate(pool):
            z = np.asarray(item["z"]).astype(float)
            vecs.append(z[:-1]); ids.append(item.get("source_index", i)); items.append(item)
        X = torch.tensor(np.stack(vecs), dtype=torch.float64, device=device).unsqueeze(1)
        return X, ids, items

    if source == "raw":
        raw = recovered_layer["candidates_raw"]
        vecs = [np.asarray(z[:-1]).astype(float) for z in raw]
        X = torch.tensor(np.stack(vecs), dtype=torch.float64, device=device).unsqueeze(1)
        return X, list(range(X.shape[0])), [{"source_index": i} for i in range(X.shape[0])]

    raise ValueError("source must be 'spikes', 'pool', or 'raw'.")


def _init_mems_dict(net, dtype, device, target_layer="x0", cover_layers=("s1",)):
    """Initialize only the membrane states that Stage 2 can actually use."""
    order = ["x0", "s1", "s2", "s3", "s4"]
    target_pos = order.index(target_layer)
    deepest_pos = max(order.index(layer) for layer in cover_layers)

    mems = {}
    if target_pos < 1 <= deepest_pos:
        mems["m1"] = net.lif1.init_leaky().to(device=device, dtype=dtype)
    if target_pos < 2 <= deepest_pos:
        mems["m2"] = net.lif2.init_leaky().to(device=device, dtype=dtype)
    if target_pos < 3 <= deepest_pos:
        mems["m3"] = net.lif3.init_leaky().to(device=device, dtype=dtype)
    if target_pos < 4 <= deepest_pos:
        mems["m4"] = net.lif4.init_leaky().to(device=device, dtype=dtype)
    return mems


def _clone_mems_dict(mems):
    return {
        k: v.clone() if hasattr(v, "clone") else v
        for k, v in mems.items()
    }


def propagate_from_target_one_step(net, target_layer, x_t, mems, cover_layers=("s1",)):
    """
    Propagate one candidate only as far as the deepest requested cover layer.

    With the default experimental configuration
        TARGET_LAYER = "x0"
        COVER_LAYERS = ("s1",)
    each candidate test performs only fc1 + lif1 and returns immediately.
    """
    dtype = x_t.dtype
    device = x_t.device
    produced = {}

    order = ["x0", "s1", "s2", "s3", "s4"]
    deepest_cover = max(order.index(layer) for layer in cover_layers)

    if target_layer == "x0":
        cur1 = net.fc1(x_t)
        s1, mems["m1"] = net.lif1(cur1, mems["m1"])
        s1 = s1.to(dtype=dtype)
        mems["m1"] = mems["m1"].to(dtype=dtype)
        produced["s1"] = s1
        if deepest_cover == order.index("s1"):
            return produced, mems

        cur2 = net.fc2(s1)
        s2, mems["m2"] = net.lif2(cur2, mems["m2"])
        s2 = s2.to(dtype=dtype)
        mems["m2"] = mems["m2"].to(dtype=dtype)
        produced["s2"] = s2
        if deepest_cover == order.index("s2"):
            return produced, mems

        cur3 = net.fc3(s2)
        s3, mems["m3"] = net.lif3(cur3, mems["m3"])
        s3 = s3.to(dtype=dtype)
        mems["m3"] = mems["m3"].to(dtype=dtype)
        produced["s3"] = s3
        if deepest_cover == order.index("s3"):
            return produced, mems

        cur4 = net.fc4(s3)
        s4, mems["m4"] = net.lif4(cur4, mems["m4"])
        s4 = s4.to(dtype=dtype)
        mems["m4"] = mems["m4"].to(dtype=dtype)
        produced["s4"] = s4
        return produced, mems

    elif target_layer == "s1":
        s1 = x_t.to(device=device, dtype=dtype)
        cur2 = net.fc2(s1)
        s2, mems["m2"] = net.lif2(cur2, mems["m2"])
        s2 = s2.to(dtype=dtype)
        mems["m2"] = mems["m2"].to(dtype=dtype)
        produced["s2"] = s2
        if deepest_cover == order.index("s2"):
            return produced, mems

        cur3 = net.fc3(s2)
        s3, mems["m3"] = net.lif3(cur3, mems["m3"])
        s3 = s3.to(dtype=dtype)
        mems["m3"] = mems["m3"].to(dtype=dtype)
        produced["s3"] = s3
        if deepest_cover == order.index("s3"):
            return produced, mems

        cur4 = net.fc4(s3)
        s4, mems["m4"] = net.lif4(cur4, mems["m4"])
        s4 = s4.to(dtype=dtype)
        mems["m4"] = mems["m4"].to(dtype=dtype)
        produced["s4"] = s4
        return produced, mems

    elif target_layer == "s2":
        s2 = x_t.to(device=device, dtype=dtype)
        cur3 = net.fc3(s2)
        s3, mems["m3"] = net.lif3(cur3, mems["m3"])
        s3 = s3.to(dtype=dtype)
        mems["m3"] = mems["m3"].to(dtype=dtype)
        produced["s3"] = s3
        if deepest_cover == order.index("s3"):
            return produced, mems

        cur4 = net.fc4(s3)
        s4, mems["m4"] = net.lif4(cur4, mems["m4"])
        s4 = s4.to(dtype=dtype)
        mems["m4"] = mems["m4"].to(dtype=dtype)
        produced["s4"] = s4
        return produced, mems

    elif target_layer == "s3":
        s3 = x_t.to(device=device, dtype=dtype)
        cur4 = net.fc4(s3)
        s4, mems["m4"] = net.lif4(cur4, mems["m4"])
        s4 = s4.to(dtype=dtype)
        mems["m4"] = mems["m4"].to(dtype=dtype)
        produced["s4"] = s4
        return produced, mems

    raise ValueError(f"Unknown target_layer: {target_layer}")


def _valid_cover_layers(target_layer, cover_layers):
    order = ["x0", "s1", "s2", "s3", "s4"]
    target_pos = order.index(target_layer)

    for layer in cover_layers:
        if layer not in order:
            raise ValueError(f"Unknown cover layer: {layer}")
        if order.index(layer) <= target_pos:
            raise ValueError(
                f"Cover layer {layer} must be downstream of target layer {target_layer}."
            )


def select_and_order_batch_candidates_sequential(
    net,
    recovered,
    T,
    B,
    target_layer="x0",
    cover_layers=("s1",),
    target_source="pool",
    cover_source="spikes",
    max_nodes=2_000_000,
    max_candidate_pool=None,
    max_sequence_solutions=10,
    verbose=True,
):
    """
    Faster mini-batch Stage 2.

    Instead of selecting B candidates at each timestep using permutations,
    this method recovers one complete sequence at a time:

        sequence 1: choose T candidates and order them
        remove used target candidates and matched cover spikes

        sequence 2: choose T candidates and order them
        remove used target candidates and matched cover spikes

        ...

        sequence B

    This avoids permutations(available, B), which is very expensive for B=8.
    """

    device = next(net.parameters()).device
    dtype = next(net.parameters()).dtype

    cover_layers = tuple(cover_layers)
    _valid_cover_layers(target_layer, cover_layers)

    # --------------------------------------------------------
    # Extract target candidate pool
    # --------------------------------------------------------
    target_X, target_ids, target_items = extract_candidates_from_recovered_layer(
        recovered[target_layer],
        source=target_source,
        device=device,
        sort_pool=True,
    )

    # [K,1,M] -> [K,M]
    target_X = target_X[:, 0, :].to(device=device, dtype=dtype)

    if max_candidate_pool is not None:
        target_X = target_X[:max_candidate_pool]
        target_ids = target_ids[:max_candidate_pool]
        target_items = target_items[:max_candidate_pool]

    K = target_X.shape[0]

    if K < T * B:
        raise ValueError(f"Target pool has K={K}, but need B*T={B*T} candidates.")

    # --------------------------------------------------------
    # Build cover counters from recovered downstream layers
    # --------------------------------------------------------
    cover_counters = {}

    for layer in cover_layers:
        X_layer, _, _ = extract_candidates_from_recovered_layer(
            recovered[layer],
            source=cover_source,
            device=device,
            sort_pool=True,
        )

        cover_counters[layer] = _make_counter_from_candidate_tensor(X_layer)

    used_target = [False] * K

    batch_sequences = []          # list of length B, each item list of T target indices
    batch_cover_keys = []         # list of length B, each item list of per-time dicts

    stats = {
        "outer_nodes": 0,
        "sequence_nodes": 0,
        "candidate_tests": 0,
        "accepted_steps": 0,
        "backtracks": 0,
        "max_depth": 0,
    }

    start = time.perf_counter()

    # --------------------------------------------------------
    # Inner search: find possible one-example sequences
    # --------------------------------------------------------
    def find_one_sequence_solutions(current_used, current_cover_counters):
        """
        Find up to max_sequence_solutions complete sequences of length T
        for one example, under the current remaining target/cover pools.
        """

        solutions = []

        local_cover = {
            layer: current_cover_counters[layer].copy()
            for layer in cover_layers
        }

        seq_indices = []
        seq_cover_keys = []
        seq_used = set()

        def seq_backtrack(t, mems):
            if len(solutions) >= max_sequence_solutions:
                return True

            stats["sequence_nodes"] += 1
            stats["max_depth"] = max(stats["max_depth"], t)

            if stats["sequence_nodes"] > max_nodes:
                return True

            if t == T:
                solutions.append(
                    {
                        "indices": list(seq_indices),
                        "cover_keys": list(seq_cover_keys),
                    }
                )
                return len(solutions) >= max_sequence_solutions

            for i in range(K):
                if current_used[i] or i in seq_used:
                    continue

                stats["candidate_tests"] += 1

                x_t = target_X[i].unsqueeze(0)  # [1,M]

                produced, new_mems = propagate_from_target_one_step(
                    net=net,
                    target_layer=target_layer,
                    x_t=x_t,
                    mems=_clone_mems_dict(mems),
                    cover_layers=cover_layers,
                )

                keys = {}
                ok = True

                for layer in cover_layers:
                    key = _spike_key(produced[layer][0])
                    keys[layer] = key

                    if not _counter_has_one(local_cover[layer], key):
                        ok = False
                        break

                if not ok:
                    continue

                # accept candidate at this timestep
                seq_indices.append(i)
                seq_cover_keys.append(keys)
                seq_used.add(i)

                for layer, key in keys.items():
                    _counter_decrement_one(local_cover[layer], key)

                stats["accepted_steps"] += 1

                stop = seq_backtrack(t + 1, new_mems)

                # undo
                for layer, key in keys.items():
                    _counter_increment_one(local_cover[layer], key)

                seq_used.remove(i)
                seq_cover_keys.pop()
                seq_indices.pop()

                if stop and len(solutions) >= max_sequence_solutions:
                    return True

            stats["backtracks"] += 1
            return False

        mems0 = _init_mems_dict(
            net,
            dtype=dtype,
            device=device,
            target_layer=target_layer,
            cover_layers=cover_layers,
        )
        seq_backtrack(0, mems0)

        return solutions

    # --------------------------------------------------------
    # Outer search: recover B sequences one by one
    # --------------------------------------------------------
    def outer_backtrack(example_id):
        stats["outer_nodes"] += 1

        if example_id == B:
            return True

        if stats["sequence_nodes"] > max_nodes:
            return False

        sequence_solutions = find_one_sequence_solutions(
            current_used=used_target,
            current_cover_counters=cover_counters,
        )

        if verbose:
            print(
                f"example {example_id}: found {len(sequence_solutions)} candidate sequences"
            )

        for sol in sequence_solutions:
            indices = sol["indices"]
            cover_keys = sol["cover_keys"]

            # commit this sequence
            for i in indices:
                used_target[i] = True

            for time_keys in cover_keys:
                for layer, key in time_keys.items():
                    _counter_decrement_one(cover_counters[layer], key)

            batch_sequences.append(indices)
            batch_cover_keys.append(cover_keys)

            if outer_backtrack(example_id + 1):
                return True

            # undo this sequence
            batch_sequences.pop()
            batch_cover_keys.pop()

            for time_keys in cover_keys:
                for layer, key in time_keys.items():
                    _counter_increment_one(cover_counters[layer], key)

            for i in indices:
                used_target[i] = False

        return False

    success = outer_backtrack(0)
    elapsed = time.perf_counter() - start

    # --------------------------------------------------------
    # Build ordered tensor
    # --------------------------------------------------------
    ordered = None
    selected_source_ids = None
    selected_items = None

    if success:
        # batch_sequences[b] is a list of T candidate indices
        ordered = torch.stack(
            [
                torch.stack([target_X[i] for i in seq], dim=0)
                for seq in batch_sequences
            ],
            dim=1,
        )  # [T,B,M]

        selected_source_ids = [
            [target_ids[i] for i in seq]
            for seq in batch_sequences
        ]

        selected_items = [
            [target_items[i] for i in seq]
            for seq in batch_sequences
        ]

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------
    if verbose:
        print("\nSequential batch selection and ordering")
        print("---------------------------------------")
        print("target_layer:", target_layer)
        print("cover_layers:", cover_layers)
        print("target pool size:", K)
        print("T:", T, "B:", B, "need:", T * B)
        print("success:", success)
        print("stats:", stats)
        print(f"time: {elapsed:.4f}s")

        if success:
            print("\nSelected source ids by recovered sequence:")
            print(selected_source_ids)


    return {
        "success": success,
        "ordered": ordered,
        "selected_indices_by_sequence": batch_sequences if success else None,
        "selected_source_ids_by_sequence": selected_source_ids,
        "selected_items_by_sequence": selected_items,
        "target_layer": target_layer,
        "cover_layers": cover_layers,
        "target_pool_size": K,
        "stats": stats,
        "time": elapsed,
        "method": "sequential_batch_ordering",
    }


# ## 7. Metrics
# 
# Final reconstruction is evaluated **up to one global batch permutation** using a Hungarian assignment over full temporal sequences.  
# The assignment is used only for evaluation; it is not used by Stage 1 or Stage 2.
# 

# In[ ]:


# ============================================================
# Evaluation metrics
# ============================================================

def _binary_metrics(true_arr, pred_arr):
    true_arr = np.asarray(true_arr).astype(bool)
    pred_arr = np.asarray(pred_arr).astype(bool)

    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"Metric shape mismatch: {true_arr.shape} vs {pred_arr.shape}"
        )

    tp = int(np.sum(true_arr & pred_arr))
    fp = int(np.sum((~true_arr) & pred_arr))
    fn = int(np.sum(true_arr & (~pred_arr)))

    hamming = int(np.sum(true_arr != pred_arr))
    bit_accuracy = float(np.mean(true_arr == pred_arr))

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0 if tp + fn == 0 else 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 1.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "bit_accuracy": bit_accuracy,
        "hamming_distance": hamming,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exact_match": bool(hamming == 0),
    }


def permutation_invariant_sequence_metrics(true_seq, reconstructed_seq):
    """
    Compare [T,B,M] sequences up to one global permutation of B.

    Hungarian assignment replaces factorial permutation enumeration,
    so evaluation remains practical up to B=32.
    """
    true_seq = _to_numpy(true_seq)
    reconstructed_seq = _to_numpy(reconstructed_seq)

    if true_seq.ndim != 3 or reconstructed_seq.ndim != 3:
        raise ValueError("Expected true/reconstructed arrays with shape [T,B,M].")

    if true_seq.shape != reconstructed_seq.shape:
        raise ValueError(
            f"Shape mismatch: true={true_seq.shape}, "
            f"reconstructed={reconstructed_seq.shape}"
        )

    T_local, B_local, _ = true_seq.shape

    true_bin = true_seq > 0.5
    rec_bin = reconstructed_seq > 0.5

    cost = np.zeros((B_local, B_local), dtype=np.int64)

    for i in range(B_local):
        for j in range(B_local):
            cost[i, j] = np.sum(
                true_bin[:, i, :] != rec_bin[:, j, :]
            )

    true_ids, rec_ids = linear_sum_assignment(cost)

    # linear_sum_assignment returns each true index once.
    aligned = np.empty_like(rec_bin)

    for i, j in zip(true_ids, rec_ids):
        aligned[:, i, :] = rec_bin[:, j, :]

    global_metrics = _binary_metrics(true_bin, aligned)

    matched_hamming = np.array(
        [cost[i, j] for i, j in zip(true_ids, rec_ids)],
        dtype=np.int64,
    )

    global_metrics.update({
        "B": B_local,
        "T": T_local,
        "best_batch_assignment": rec_ids.tolist(),
        "exact_sequences": int(np.sum(matched_hamming == 0)),
        "success_up_to_permutation": bool(np.all(matched_hamming == 0)),
        "per_sequence_hamming": matched_hamming.tolist(),
    })

    return global_metrics, aligned


def candidate_pool_pattern_coverage(recovered_layer, true_spike_rec):
    """
    Coverage of distinct true spike patterns by the Stage-1 candidate pool.

    This deliberately measures distinct-pattern coverage rather than
    temporal multiplicity, because the MILP pool contains unique binary vectors.
    """
    S_tilde = make_S_tilde_from_rec(true_spike_rec)

    true_patterns = {
        candidate_tuple(S_tilde[:, i])
        for i in range(S_tilde.shape[1])
    }

    pool_patterns = {
        candidate_tuple(item["z"])
        for item in recovered_layer["pool"]
    }

    top_patterns = {
        candidate_tuple(z)
        for z in recovered_layer["candidates_raw"]
    }

    found_pool = true_patterns & pool_patterns
    found_top = true_patterns & top_patterns

    denom = max(len(true_patterns), 1)

    return {
        "num_true_distinct_patterns": len(true_patterns),
        "candidate_pool_size": len(recovered_layer["pool"]),
        "candidate_pool_unique": len(pool_patterns),

        "true_patterns_in_pool": len(found_pool),
        "true_pattern_coverage": len(found_pool) / denom,
        "all_true_patterns_in_pool": found_pool == true_patterns,

        "true_patterns_in_top": len(found_top),
        "top_pattern_coverage": len(found_top) / denom,
        "all_true_patterns_in_top": found_top == true_patterns,
    }


# ## 8. Incremental result saving
# 
# Two CSV files are maintained:
# 
# - `experiment_summary.csv`: one row per \((T,B,\text{seed})\);
# - `layer_diagnostics.csv`: one row per layer and setting.
# 
# Rows are upserted by experiment identifier, so a rerun replaces the previous version instead of creating duplicates.
# 

# In[ ]:


# ============================================================
# Result persistence
# ============================================================

def _upsert_rows_csv(rows, path, key_columns):
    """
    Incrementally save rows while replacing existing rows with the same key.
    """
    if not rows:
        return pd.DataFrame()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)

    if path.exists() and path.stat().st_size > 0:
        try:
            old_df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            old_df = pd.DataFrame()
    else:
        old_df = pd.DataFrame()

    if len(old_df) > 0:
        incoming_keys = {
            tuple(row[col] for col in key_columns)
            for row in rows
        }

        keep_mask = []

        for _, old_row in old_df.iterrows():
            old_key = tuple(old_row[col] for col in key_columns)
            keep_mask.append(old_key not in incoming_keys)

        old_df = old_df[np.asarray(keep_mask, dtype=bool)]

        out_df = pd.concat(
            [old_df, new_df],
            ignore_index=True,
            sort=False,
        )
    else:
        out_df = new_df

    out_df.to_csv(path, index=False)

    return out_df


def save_experiment_rows(summary_row, layer_rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _upsert_rows_csv(
        [summary_row],
        SUMMARY_CSV,
        key_columns=["experiment_id"],
    )

    if layer_rows:
        _upsert_rows_csv(
            layer_rows,
            LAYER_CSV,
            key_columns=["experiment_id", "layer"],
        )


# ## 9. One-setting experiment runner
# 
# `run_one_experiment(T, B, ...)` is the single entry point for one experimental setting.
# 
# A failed rank condition, Stage-1 failure, Stage-2 node limit, or unsuccessful final reconstruction is recorded explicitly rather than crashing the full grid.
# 

# In[ ]:


# ============================================================
# One complete experiment: one (T,B) setting
# ============================================================

def _layer_specs_for_stage1(net, rec):
    # Fast runner: Stage 1 reconstructs only x0 and the s1 cover layer.
    return {
        "x0": (net.fc1, net.fc1.in_features, rec["x0"]),
        "s1": (net.fc2, net.fc2.in_features, rec["s1"]),
    }


def run_one_experiment(
    T,
    B,
    dataset=None,
    sample_indices=None,
    seed=SEED,
    save_results=True,
    verbose=False,
):
    """
    Run one complete setting.

    Saved outputs:
      - experiment_summary.csv : one row per (T,B,seed)
      - layer_diagnostics.csv  : one row per layer per experiment

    Fast x0/s1 runner. Stage 1 is attempted only for x0 and s1 when
    rank(D) == rank(S_tilde). Rank statistics are reported only for those
    two representations. Stage 2 uses s1 only and stops propagation after
    fc1+LIF1 for each x0 candidate test.
    """
    experiment_id = f"nmnist_T{T}_B{B}_seed{seed}"

    # Synchronize before starting the full-setting wall-clock timer so
    # outstanding CUDA work from a previous setting is not charged here.
    sync_device()
    setting_wall_start = time.perf_counter()

    summary = {
        "experiment_id": experiment_id,
        "dataset": "N-MNIST",
        "seed": int(seed),
        "T": int(T),
        "B": int(B),
        "L": int(T * B),

        "status": "started",
        "failure_stage": "",
        "failure_reason": "",

        # Timing (seconds).
        # attack_time_s = Stage 1 + Stage 2 only.
        # setting_time_s = complete run_one_experiment wall time, including
        # data preparation, forward/backward, rank diagnostics and evaluation.
        "stage1_time_s": np.nan,
        "stage2_time_s": np.nan,
        "attack_time_s": np.nan,
        "setting_time_s": np.nan,

        # Legacy aliases retained for compatibility with earlier result files.
        "stage1_time": np.nan,
        "stage2_time": np.nan,
        "total_time": np.nan,
        "experiment_wall_time": np.nan,

        "candidate_pool_size": np.nan,
        "true_pattern_coverage": np.nan,

        # Explicit Stage-1 target-pattern recovery reporting.
        # Stage 1 enumerates UNIQUE binary candidates, so these quantities
        # refer to distinct true temporal patterns.
        "true_candidate_recovery_rate": np.nan,
        "true_candidate_recovery_rate_pct": np.nan,
        "true_patterns_recovered": np.nan,
        "num_true_distinct_patterns": np.nan,

        # Stage 2 depends on the recovered s1 cover layer as well.
        "s1_true_pattern_coverage": np.nan,
        "s1_true_candidate_recovery_rate_pct": np.nan,

        # Target-input (x0) diagnostics reported in the T sweep.
        "spike_density": np.nan,
        "rank_S_tilde": np.nan,
        "rank_D": np.nan,
        "rank_condition": False,
        "num_distinct_spike_patterns": np.nan,

        "reconstruction_success": False,
        "success_up_to_permutation": False,

        "bit_accuracy": np.nan,
        "hamming_distance": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "exact_sequences": np.nan,

        "visited_nodes": np.nan,
        "backtracks": np.nan,
    }

    layer_rows = []

    try:
        set_seed(seed)

        if dataset is None:
            dataset = build_nmnist_dataset(T=T)

        if sample_indices is None:
            pool = deterministic_index_pool(
                dataset_len=len(dataset),
                max_batch_size=B,
                seed=seed,
            )
            sample_indices = pool[:B]
        else:
            sample_indices = list(sample_indices)

        summary["sample_indices"] = json.dumps(sample_indices)

        # ----------------------------------------------------
        # Data + fresh model
        # ----------------------------------------------------
        x, y, raw_frames = load_batch_from_dataset(
            dataset=dataset,
            T=T,
            B=B,
            indices=sample_indices,
            device=DEVICE,
            binarize=True,
            dtype=torch.float64,
        )

        net = build_model(seed=seed, device=DEVICE)
        criterion = nn.CrossEntropyLoss(reduction="mean")

        # ----------------------------------------------------
        # Rank diagnostics
        # ----------------------------------------------------
        # One forward/backward pass serves both rank diagnostics and the attack.
        # Parameter gradients remain available on net.fc1/net.fc2 for Stage 1.
        rank_results, rec, diagnostic_loss = layer_rank_diagnostics(
            net=net,
            x=x,
            y=y,
            criterion=criterion,
            layers=RANK_LAYERS,
            rank_tol=RANK_TOL,
            print_results=verbose,
        )

        summary["loss"] = float(diagnostic_loss.detach().cpu())

        # Start layer rows with rank/pattern statistics.
        for layer in RANK_LAYERS:
            r = rank_results[layer]

            layer_rows.append({
                "experiment_id": experiment_id,
                "dataset": "N-MNIST",
                "seed": int(seed),
                "T": int(T),
                "B": int(B),
                "L": int(T * B),
                "layer": layer,
                **r,

                "stage1_attempted": False,
                "stage1_success": False,
                "stage1_time_s": np.nan,
                "stage1_time": np.nan,

                "candidate_pool_size": np.nan,
                "candidate_pool_unique": np.nan,
                "true_patterns_in_pool": np.nan,
                "true_pattern_coverage": np.nan,
                "all_true_patterns_in_pool": np.nan,
                "true_patterns_in_top": np.nan,
                "top_pattern_coverage": np.nan,
            })

        layer_row_by_name = {
            row["layer"]: row
            for row in layer_rows
        }

        required_stage1_layers = [
            TARGET_LAYER,
            *COVER_LAYERS,
        ]

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------
        recovered = {}

        sync_device()
        stage1_start = time.perf_counter()

        layer_specs = _layer_specs_for_stage1(net, rec)

        for layer_name in required_stage1_layers:
            row = layer_row_by_name[layer_name]

            if not row["rank_condition"]:
                if verbose:
                    print(
                        f"Skipping Stage 1 for {layer_name}: "
                        "rank(D) != rank(S_tilde)."
                    )
                continue

            fc_layer, width, true_spikes = layer_specs[layer_name]

            row["stage1_attempted"] = True

            # The observable row-space dimension is rank(D).
            recovery_rank = int(row["rank_D"])

            # Suppress the verbose legacy Stage-1 print stream for grid runs.
            stream = None if verbose else io.StringIO()

            try:
                if verbose:
                    recovered_layer = recover_input_spikes_from_layer(
                        fc_layer=fc_layer,
                        name=layer_name,
                        width=width,
                        target_rank=recovery_rank,
                        device=DEVICE,
                        S_tilde_gt=make_S_tilde_from_rec(true_spikes),
                        top_k=T * B,
                        max_pool=MAX_POOL,
                        eps_eq=EPS_EQ,
                        eps_proj=EPS_PROJ,
                        zero_col_tol=ZERO_COL_TOL,
                        one_atol=ONE_ATOL,
                        one_rtol=ONE_RTOL,
                        duplicate_decimals=DUPLICATE_DECIMALS,
                        use_zero_elimination=USE_ZERO_ELIMINATION,
                        use_one_elimination=USE_ONE_ELIMINATION,
                        use_duplicate_elimination=USE_DUPLICATE_ELIMINATION,
                        unaverage_gradients_by_batch=UNAVERAGE_GRADIENTS_BY_BATCH,
                        batch_size=B,
                        time_limit=MILP_TIME_LIMIT,
                        rank_mode="target",
                        selection_mode=SELECTION_MODE,
                        enumeration_mode=ENUMERATION_MODE,
                        guided_random_weight=GUIDED_RANDOM_WEIGHT,
                        guided_seed=GUIDED_SEED,
                        verbose=True,
                    )
                else:
                    with redirect_stdout(stream):
                        recovered_layer = recover_input_spikes_from_layer(
                            fc_layer=fc_layer,
                            name=layer_name,
                            width=width,
                            target_rank=recovery_rank,
                            device=DEVICE,
                            S_tilde_gt=make_S_tilde_from_rec(true_spikes),
                            top_k=T * B,
                            max_pool=MAX_POOL,
                            eps_eq=EPS_EQ,
                            eps_proj=EPS_PROJ,
                            zero_col_tol=ZERO_COL_TOL,
                            one_atol=ONE_ATOL,
                            one_rtol=ONE_RTOL,
                            duplicate_decimals=DUPLICATE_DECIMALS,
                            use_zero_elimination=USE_ZERO_ELIMINATION,
                            use_one_elimination=USE_ONE_ELIMINATION,
                            use_duplicate_elimination=USE_DUPLICATE_ELIMINATION,
                            unaverage_gradients_by_batch=UNAVERAGE_GRADIENTS_BY_BATCH,
                            batch_size=B,
                            time_limit=MILP_TIME_LIMIT,
                            rank_mode="target",
                            selection_mode=SELECTION_MODE,
                            enumeration_mode=ENUMERATION_MODE,
                            guided_random_weight=GUIDED_RANDOM_WEIGHT,
                            guided_seed=GUIDED_SEED,
                            verbose=False,
                        )

                recovered[layer_name] = recovered_layer

                coverage = candidate_pool_pattern_coverage(
                    recovered_layer=recovered_layer,
                    true_spike_rec=true_spikes,
                )

                row.update(coverage)
                row["stage1_success"] = bool(
                    coverage["all_true_patterns_in_pool"]
                )
                row["stage1_time_s"] = float(recovered_layer["time"])
                row["stage1_time"] = row["stage1_time_s"]

            except Exception as layer_exc:
                row["stage1_success"] = False
                row["stage1_error"] = repr(layer_exc)

                if verbose:
                    print(
                        f"Stage 1 exception for {layer_name}:",
                        repr(layer_exc),
                    )

        sync_device()
        stage1_elapsed = time.perf_counter() - stage1_start

        summary["stage1_time_s"] = float(stage1_elapsed)
        summary["stage1_time"] = summary["stage1_time_s"]

        target_row = layer_row_by_name[TARGET_LAYER]

        summary["candidate_pool_size"] = target_row["candidate_pool_size"]
        summary["true_pattern_coverage"] = target_row["true_pattern_coverage"]

        # Explicit x0 recovery rate.
        target_cov = target_row.get("true_pattern_coverage", np.nan)
        summary["true_candidate_recovery_rate"] = target_cov
        summary["true_candidate_recovery_rate_pct"] = (
            100.0 * float(target_cov)
            if np.isfinite(target_cov)
            else np.nan
        )
        summary["true_patterns_recovered"] = target_row.get(
            "true_patterns_in_pool", np.nan
        )
        summary["num_true_distinct_patterns"] = target_row.get(
            "num_true_distinct_patterns", np.nan
        )

        # Also report s1 recovery because Stage 2 relies on it.
        s1_row = layer_row_by_name.get("s1", {})
        s1_cov = s1_row.get("true_pattern_coverage", np.nan)
        summary["s1_true_pattern_coverage"] = s1_cov
        summary["s1_true_candidate_recovery_rate_pct"] = (
            100.0 * float(s1_cov)
            if np.isfinite(s1_cov)
            else np.nan
        )

        summary["spike_density"] = target_row["spike_density"]
        summary["rank_S_tilde"] = target_row["rank_S_tilde"]
        summary["rank_D"] = target_row["rank_D"]
        summary["rank_condition"] = bool(target_row["rank_condition"])
        summary["num_distinct_spike_patterns"] = target_row[
            "num_distinct_spike_patterns"
        ]

        # ----------------------------------------------------
        # Decide whether Stage 2 can run
        # ----------------------------------------------------
        missing_required = [
            name
            for name in required_stage1_layers
            if name not in recovered
        ]

        if missing_required:
            rank_failed = [
                name
                for name in missing_required
                if not layer_row_by_name[name]["rank_condition"]
            ]

            summary["status"] = "failed"
            summary["failure_stage"] = "stage1"

            if rank_failed:
                summary["failure_reason"] = (
                    "rank_condition_failed:"
                    + ",".join(rank_failed)
                )
            else:
                summary["failure_reason"] = (
                    "stage1_recovery_failed:"
                    + ",".join(missing_required)
                )

            summary["stage2_time_s"] = 0.0
            summary["stage2_time"] = 0.0
            summary["attack_time_s"] = float(stage1_elapsed)
            summary["total_time"] = summary["attack_time_s"]

            sync_device()
            summary["setting_time_s"] = (
                time.perf_counter() - setting_wall_start
            )
            summary["experiment_wall_time"] = summary["setting_time_s"]

            if save_results:
                save_experiment_rows(summary, layer_rows)

            return {
                "summary": summary,
                "layers": layer_rows,
                "x": x.detach().cpu(),
                "y": y.detach().cpu(),
                "true_rec": {
                    k: v.detach().cpu()
                    for k, v in rec.items()
                    if torch.is_tensor(v)
                },
                "recovered": recovered,
                "stage2": None,
            }

        # ----------------------------------------------------
        # Stage-1 completeness / Stage-2 precondition check
        # ----------------------------------------------------
        # A setting with incomplete TRUE DISTINCT-pattern recovery is a
        # failed experimental outcome, not a reason to terminate the sweep.
        incomplete_layers = []

        for name in required_stage1_layers:
            row = layer_row_by_name[name]
            cov = row.get("true_pattern_coverage", np.nan)

            if (not np.isfinite(cov)) or (float(cov) < 1.0 - 1e-12):
                incomplete_layers.append(
                    f"{name}={float(cov):.6f}"
                    if np.isfinite(cov)
                    else f"{name}=nan"
                )

        if incomplete_layers:
            summary["status"] = "failed"
            summary["failure_stage"] = "stage1"
            summary["failure_reason"] = (
                "incomplete_true_pattern_coverage:"
                + ",".join(incomplete_layers)
            )

            summary["stage2_time_s"] = 0.0
            summary["stage2_time"] = 0.0
            summary["attack_time_s"] = float(stage1_elapsed)
            summary["total_time"] = summary["attack_time_s"]
            summary["reconstruction_success"] = False
            summary["success_up_to_permutation"] = False
            summary["visited_nodes"] = 0
            summary["backtracks"] = 0

            sync_device()
            summary["setting_time_s"] = (
                time.perf_counter() - setting_wall_start
            )
            summary["experiment_wall_time"] = summary["setting_time_s"]

            if save_results:
                save_experiment_rows(summary, layer_rows)

            if verbose:
                print(
                    "[SKIP Stage 2] incomplete Stage-1 recovery "
                    f"| x0={summary['true_candidate_recovery_rate_pct']:.2f}% "
                    f"| s1={summary['s1_true_candidate_recovery_rate_pct']:.2f}% "
                    f"| reason={summary['failure_reason']}",
                    flush=True,
                )

            return {
                "summary": summary,
                "layers": layer_rows,
                "x": x.detach().cpu(),
                "y": y.detach().cpu(),
                "true_rec": {
                    k: v.detach().cpu()
                    for k, v in rec.items()
                    if torch.is_tensor(v)
                },
                "recovered": recovered,
                "stage2": None,
            }

        # The current sequential Stage-2 implementation requires at least
        # B*T entries in the UNIQUE x0 candidate pool. Check this explicitly
        # so it never raises and stops the sweep.
        try:
            target_X_precheck, _, _ = extract_candidates_from_recovered_layer(
                recovered[TARGET_LAYER],
                source="pool",
                device=DEVICE,
                sort_pool=True,
            )
            target_pool_for_stage2 = int(target_X_precheck.shape[0])
        except Exception:
            target_pool_for_stage2 = 0

        required_target_count = int(B * T)

        if target_pool_for_stage2 < required_target_count:
            summary["status"] = "failed"
            summary["failure_stage"] = "stage2_precondition"
            summary["failure_reason"] = (
                f"target_pool_smaller_than_BT:"
                f"{target_pool_for_stage2}/{required_target_count}"
            )

            summary["stage2_time_s"] = 0.0
            summary["stage2_time"] = 0.0
            summary["attack_time_s"] = float(stage1_elapsed)
            summary["total_time"] = summary["attack_time_s"]
            summary["reconstruction_success"] = False
            summary["success_up_to_permutation"] = False
            summary["visited_nodes"] = 0
            summary["backtracks"] = 0

            sync_device()
            summary["setting_time_s"] = (
                time.perf_counter() - setting_wall_start
            )
            summary["experiment_wall_time"] = summary["setting_time_s"]

            if save_results:
                save_experiment_rows(summary, layer_rows)

            if verbose:
                print(
                    "[SKIP Stage 2] "
                    f"candidate pool={target_pool_for_stage2}, "
                    f"required B*T={required_target_count}. "
                    "Continuing sweep.",
                    flush=True,
                )

            return {
                "summary": summary,
                "layers": layer_rows,
                "x": x.detach().cpu(),
                "y": y.detach().cpu(),
                "true_rec": {
                    k: v.detach().cpu()
                    for k, v in rec.items()
                    if torch.is_tensor(v)
                },
                "recovered": recovered,
                "stage2": None,
            }

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------
        sync_device()
        stage2_start = time.perf_counter()

        stage2_result = select_and_order_batch_candidates_sequential(
            net=net,
            recovered=recovered,
            T=T,
            B=B,
            target_layer=TARGET_LAYER,
            cover_layers=COVER_LAYERS,
            target_source="pool",
            cover_source="spikes",
            max_nodes=STAGE2_MAX_NODES,
            max_candidate_pool=None,
            max_sequence_solutions=MAX_SEQUENCE_SOLUTIONS,
            verbose=verbose,
        )

        sync_device()
        stage2_elapsed = time.perf_counter() - stage2_start

        summary["stage2_time_s"] = float(stage2_elapsed)
        summary["stage2_time"] = summary["stage2_time_s"]
        summary["attack_time_s"] = float(stage1_elapsed + stage2_elapsed)
        summary["total_time"] = summary["attack_time_s"]

        stats = stage2_result.get("stats", {})

        summary["visited_nodes"] = int(
            stats.get("outer_nodes", 0)
            +
            stats.get("sequence_nodes", 0)
        )

        summary["backtracks"] = int(
            stats.get("backtracks", 0)
        )

        summary["reconstruction_success"] = bool(
            stage2_result["success"]
        )

        if not stage2_result["success"]:
            summary["status"] = "failed"
            summary["failure_stage"] = "stage2"

            if stats.get("sequence_nodes", 0) > STAGE2_MAX_NODES:
                summary["failure_reason"] = "node_limit"
            else:
                summary["failure_reason"] = "search_exhausted"

        else:
            metrics, aligned = permutation_invariant_sequence_metrics(
                true_seq=rec[TARGET_LAYER],
                reconstructed_seq=stage2_result["ordered"],
            )

            summary.update({
                "success_up_to_permutation": metrics["success_up_to_permutation"],
                "reconstruction_success": metrics["success_up_to_permutation"],
                "bit_accuracy": metrics["bit_accuracy"],
                "hamming_distance": metrics["hamming_distance"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "exact_sequences": metrics["exact_sequences"],
                "best_batch_assignment": json.dumps(
                    metrics["best_batch_assignment"]
                ),
            })

            if metrics["success_up_to_permutation"]:
                summary["status"] = "success"
                summary["failure_stage"] = ""
                summary["failure_reason"] = ""
            else:
                summary["status"] = "partial"
                summary["failure_stage"] = "evaluation"
                summary["failure_reason"] = (
                    "stage2_returned_sequence_but_not_exact_up_to_permutation"
                )

        sync_device()
        summary["setting_time_s"] = (
            time.perf_counter() - setting_wall_start
        )
        summary["experiment_wall_time"] = summary["setting_time_s"]

        if save_results:
            save_experiment_rows(summary, layer_rows)

        return {
            "summary": summary,
            "layers": layer_rows,

            "x": x.detach().cpu(),
            "y": y.detach().cpu(),

            "true_rec": {
                k: v.detach().cpu()
                for k, v in rec.items()
            },

            "recovered": recovered,
            "stage2": stage2_result,
        }

    except Exception as exc:
        summary["status"] = "failed"
        summary["failure_stage"] = (
            summary["failure_stage"] or "experiment"
        )
        summary["failure_reason"] = repr(exc)
        sync_device()
        summary["setting_time_s"] = (
            time.perf_counter() - setting_wall_start
        )
        summary["experiment_wall_time"] = summary["setting_time_s"]

        # Preserve any timing completed before the exception.
        s1 = summary.get("stage1_time_s", np.nan)
        s2 = summary.get("stage2_time_s", np.nan)
        finite_parts = [v for v in (s1, s2) if np.isfinite(v)]
        if finite_parts:
            summary["attack_time_s"] = float(sum(finite_parts))
            summary["total_time"] = summary["attack_time_s"]

        if save_results:
            save_experiment_rows(summary, layer_rows)

        if verbose:
            print(
                f"[FAILED setting] T={T} B={B} "
                f"| stage={summary['failure_stage']} "
                f"| reason={summary['failure_reason']}",
                flush=True,
            )

        # Failed settings are experimental outcomes; continue the sweep.
        return {
            "summary": summary,
            "layers": layer_rows,
            "x": None,
            "y": None,
            "true_rec": None,
            "recovered": None,
            "stage2": None,
        }


# ## 10. Run one setting
# 
# Set `RUN_SINGLE = True` in the configuration cell when you want an interactive/debug run.
# 
# For example:
# 
# ```python
# SINGLE_T = 8
# SINGLE_B = 8
# RUN_SINGLE = True
# ```
# 

# In[ ]:


single_result = None

if RUN_SINGLE:
    dataset_single = build_nmnist_dataset(T=SINGLE_T)

    single_indices = deterministic_index_pool(
        dataset_len=len(dataset_single),
        max_batch_size=SINGLE_B,
        seed=SEED,
    )

    single_result = run_one_experiment(
        T=SINGLE_T,
        B=SINGLE_B,
        dataset=dataset_single,
        sample_indices=single_indices,
        seed=SEED,
        save_results=True,
        verbose=True,
    )

    print("\nSingle-setting summary")
    print("----------------------")

    for key, value in single_result["summary"].items():
        print(f"{key}: {value}")
else:
    print("RUN_SINGLE=False — no single-setting attack launched.")



# ============================================================
# Qualitative figure helpers used by the grid
# ============================================================

def make_event_rgb_from_flat_sequence(seq_flat, T):
    """
    seq_flat [T,2312] -> temporally aggregated RGB composite [34,34,3].

    ON  -> red
    OFF -> blue
    empty -> white
    """
    arr = _to_numpy(seq_flat).reshape(T, 2, 34, 34)

    on_map = arr[:, 1].sum(axis=0)
    off_map = arr[:, 0].sum(axis=0)

    on_norm = on_map / max(on_map.max(), 1)
    off_norm = off_map / max(off_map.max(), 1)

    rgb = np.ones((34, 34, 3), dtype=np.float32)
    rgb[..., 0] = 1.0 - off_norm
    rgb[..., 1] = 1.0 - np.maximum(on_norm, off_norm)
    rgb[..., 2] = 1.0 - on_norm

    return np.clip(rgb, 0.0, 1.0)


def plot_original_vs_reconstructed_batch(
    details,
    save_path=None,
    show=False,
    max_columns=8,
):
    """
    Save/display all original and reconstructed examples for one setting.

    The examples are wrapped into blocks of at most `max_columns` examples:

        Original      ex 1 ... ex 8
        Reconstructed ex 1 ... ex 8

        Original      ex 9 ... ex 16
        Reconstructed ex 9 ... ex 16
        ...

    Thus B=16, 24, and 32 remain readable instead of being compressed into
    one extremely wide 2 x B figure.

    Both tensors use the attack's time-major layout [T,B,M].  The figure
    preserves the Stage-2 output order; it does not use ground truth to
    reorder the reconstructed batch.
    """
    if details is None or details.get("stage2") is None:
        raise ValueError("No Stage-2 result is available.")

    stage2 = details["stage2"]
    if not stage2.get("success", False):
        raise ValueError("Stage 2 did not succeed.")

    true_x = details["true_rec"][TARGET_LAYER]
    recon_x = stage2["ordered"].detach().cpu()

    if torch.is_tensor(true_x):
        true_x = true_x.detach().cpu()

    if true_x.ndim != 3 or recon_x.ndim != 3:
        raise ValueError(
            f"Expected [T,B,M] tensors, got true={tuple(true_x.shape)}, "
            f"reconstructed={tuple(recon_x.shape)}"
        )

    T_local, B_local, M_local = true_x.shape
    if tuple(recon_x.shape) != (T_local, B_local, M_local):
        raise ValueError(
            f"Shape mismatch: true={tuple(true_x.shape)}, "
            f"reconstructed={tuple(recon_x.shape)}"
        )

    ncols = min(int(max_columns), B_local)
    nblocks = int(np.ceil(B_local / ncols))
    nrows = 2 * nblocks

    # Keep each event image large enough to remain readable in the saved PNG.
    fig_width = 2.05 * ncols
    fig_height = 4.15 * nblocks

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    # Hide unused slots in the final partial block (e.g. B=12).
    for ax in axes.flat:
        ax.axis("off")

    for b in range(B_local):
        block = b // ncols
        col = b % ncols
        original_row = 2 * block
        reconstructed_row = original_row + 1

        axes[original_row, col].imshow(
            make_event_rgb_from_flat_sequence(
                true_x[:, b, :],
                T=T_local,
            ),
            interpolation="nearest",
        )
        axes[original_row, col].axis("off")

        axes[reconstructed_row, col].imshow(
            make_event_rgb_from_flat_sequence(
                recon_x[:, b, :],
                T=T_local,
            ),
            interpolation="nearest",
        )
        axes[reconstructed_row, col].axis("off")

    # Label every block, not only the first one.
    for block in range(nblocks):
        original_row = 2 * block
        reconstructed_row = original_row + 1

        axes[original_row, 0].set_ylabel(
            "Original",
            fontsize=12,
            fontweight="bold",
            labelpad=10,
        )
        axes[reconstructed_row, 0].set_ylabel(
            "Reconstructed",
            fontsize=12,
            fontweight="bold",
            labelpad=10,
        )

    fig.suptitle(
        f"Original vs Reconstructed (B={B_local}, T={T_local})",
        fontsize=15,
        y=0.995,
    )

    # Compact spacing, with enough left margin for row labels.
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.955,
        bottom=0.02,
        wspace=0.04,
        hspace=0.12,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


# ## 11. Full \(T\times B\) grid
# 
# The grid runner:
# - reuses the same fixed sample-index pool across settings;
# - saves each setting immediately;
# - can skip settings already present in `experiment_summary.csv`;
# - releases large candidate pools/tensors before the next run.
# 
# Full grid execution is controlled by RUN_GRID in the configuration section.
# 

# In[ ]:


# ============================================================
# Grid runner with per-setting and whole-grid timing
# ============================================================

REPORT_COLUMNS = [
    "seed",
    "T",
    "L",
    "status",
    "reconstruction_success",
    "bit_accuracy",
    "f1",
    "attack_time_s",
    "stage1_time_s",
    "stage2_time_s",
    "visited_nodes",
    "backtracks",
    "spike_density",
    "rank_S_tilde",
    "rank_D",
    "rank_condition",
    "true_pattern_coverage",
    "true_candidate_recovery_rate",
    "true_candidate_recovery_rate_pct",
    "true_patterns_recovered",
    "num_true_distinct_patterns",
    "s1_true_pattern_coverage",
    "s1_true_candidate_recovery_rate_pct",
    "candidate_pool_size",
    "num_distinct_spike_patterns",
    "failure_reason",
]


def _write_t_sweep_report():
    """Create the compact report requested for the T-sweep experiment."""
    if not SUMMARY_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(SUMMARY_CSV)
    if df.empty:
        return df

    for col in REPORT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    report = (
        df[REPORT_COLUMNS]
        .sort_values(["seed", "T"])
        .reset_index(drop=True)
    )
    report.to_csv(REPORT_CSV, index=False)

    # If several seeds are requested, also save a compact per-T aggregate.
    if report["seed"].nunique() > 1:
        numeric_cols = [
            "reconstruction_success",
            "bit_accuracy",
            "f1",
            "attack_time_s",
            "stage1_time_s",
            "stage2_time_s",
            "visited_nodes",
            "backtracks",
            "spike_density",
            "rank_S_tilde",
            "rank_D",
            "rank_condition",
            "true_pattern_coverage",
            "true_candidate_recovery_rate",
            "true_candidate_recovery_rate_pct",
            "true_patterns_recovered",
            "num_true_distinct_patterns",
            "s1_true_pattern_coverage",
            "s1_true_candidate_recovery_rate_pct",
            "candidate_pool_size",
            "num_distinct_spike_patterns",
        ]

        agg = report.groupby("T", as_index=False)[numeric_cols].agg(
            ["mean", "std"]
        )
        agg.columns = [
            "T" if c[0] == "T" else f"{c[0]}_{c[1]}"
            for c in agg.columns.to_flat_index()
        ]
        agg.to_csv(AGGREGATE_CSV, index=False)

    return report


def existing_experiment_ids():
    if not SUMMARY_CSV.exists():
        return set()

    try:
        df = pd.read_csv(SUMMARY_CSV)
    except pd.errors.EmptyDataError:
        return set()

    if "experiment_id" not in df.columns:
        return set()

    return set(df["experiment_id"].astype(str))


def run_T_sweep(
    T_values=T_VALUES,
    seeds=SEEDS,
    resume=True,
    verbose=False,
):
    """
    Run the temporal-horizon experiment with B=1.

    For each seed, the same N-MNIST sample index is reused across every T,
    so changing T changes only the temporal binning, not the underlying event
    recording.  The SNN architecture and attack settings are exactly those of
    the fast x0/s1 main-experiment runner.
    """
    B = 1
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sync_device()
    sweep_start = time.perf_counter()

    completed = existing_experiment_ids() if resume else set()
    summaries = []

    total_requested = len(T_values) * len(seeds)
    pending = 0
    for seed in seeds:
        for T_local in T_values:
            exp_id = f"nmnist_T{T_local}_B1_seed{seed}"
            if not (resume and exp_id in completed):
                pending += 1

    print("=" * 72, flush=True)
    print("N-MNIST TEMPORAL-HORIZON SWEEP", flush=True)
    print("B = 1", flush=True)
    print("T_VALUES =", list(T_values), flush=True)
    print("SEEDS =", list(seeds), flush=True)
    print("Requested settings =", total_requested, flush=True)
    print("Pending settings =", pending, flush=True)
    print("=" * 72, flush=True)

    completed_now = 0

    for seed in seeds:
        # Fix one underlying event recording per seed and reuse it for all T.
        reference_dataset = build_nmnist_dataset(T=T_values[0])
        sample_index = deterministic_index_pool(
            dataset_len=len(reference_dataset),
            max_batch_size=1,
            seed=seed,
        )[0]

        print(
            f"\nSeed {seed}: fixed sample index = {sample_index}",
            flush=True,
        )

        for T_local in T_values:
            experiment_id = f"nmnist_T{T_local}_B1_seed{seed}"

            if resume and experiment_id in completed:
                print("skip:", experiment_id, flush=True)
                continue

            dataset_T = build_nmnist_dataset(T=T_local)

            print(
                f"\n{'=' * 72}\n"
                f"RUN {experiment_id} | L={T_local}\n"
                f"{'=' * 72}",
                flush=True,
            )

            try:
                details = run_one_experiment(
                    T=T_local,
                    B=B,
                    dataset=dataset_T,
                    sample_indices=[sample_index],
                    seed=seed,
                    save_results=True,
                    verbose=verbose,
                )
                row = details["summary"]

            except Exception as unexpected_exc:
                # Last-resort guard: one unexpected T must never stop the sweep.
                row = {
                    "experiment_id": experiment_id,
                    "dataset": "N-MNIST",
                    "seed": int(seed),
                    "T": int(T_local),
                    "B": 1,
                    "L": int(T_local),
                    "status": "failed",
                    "failure_stage": "experiment",
                    "failure_reason": repr(unexpected_exc),
                    "reconstruction_success": False,
                    "success_up_to_permutation": False,
                    "bit_accuracy": np.nan,
                    "f1": np.nan,
                    "attack_time_s": np.nan,
                    "stage1_time_s": np.nan,
                    "stage2_time_s": np.nan,
                    "visited_nodes": 0,
                    "backtracks": 0,
                    "spike_density": np.nan,
                    "rank_S_tilde": np.nan,
                    "rank_D": np.nan,
                    "rank_condition": False,
                    "true_pattern_coverage": np.nan,
                    "true_candidate_recovery_rate": np.nan,
                    "true_candidate_recovery_rate_pct": np.nan,
                    "true_patterns_recovered": np.nan,
                    "num_true_distinct_patterns": np.nan,
                    "s1_true_pattern_coverage": np.nan,
                    "s1_true_candidate_recovery_rate_pct": np.nan,
                    "candidate_pool_size": np.nan,
                    "num_distinct_spike_patterns": np.nan,
                }
                save_experiment_rows(row, [])
                details = None

                print(
                    f"[UNEXPECTED FAILURE] {experiment_id}: "
                    f"{unexpected_exc!r}. Continuing.",
                    flush=True,
                )

            summaries.append(row)
            completed_now += 1

            # Refresh the compact report immediately so partial runs are useful.
            report = _write_t_sweep_report()

            print(
                f"status={row['status']}"
                f" | reconstruction_success={row['reconstruction_success']}"
                f" | rank_condition={row['rank_condition']}"
                f" | pool={row['candidate_pool_size']}"
                f" | x0_coverage={row.get('true_candidate_recovery_rate_pct', np.nan):.2f}%"
                f" | s1_coverage={row.get('s1_true_candidate_recovery_rate_pct', np.nan):.2f}%\n"
                f"Stage1={format_duration(row['stage1_time_s'])}"
                f" | Stage2={format_duration(row['stage2_time_s'])}"
                f" | attack={format_duration(row['attack_time_s'])}",
                flush=True,
            )

            if details is not None:
                del details
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    sync_device()
    total_s = time.perf_counter() - sweep_start

    report = _write_t_sweep_report()

    print("\n" + "=" * 72, flush=True)
    print("T SWEEP COMPLETE", flush=True)
    print("Settings run now:", completed_now, flush=True)
    print("Sweep wall time:", format_duration(total_s), flush=True)
    print("Compact report:", REPORT_CSV.resolve(), flush=True)
    print("Full summary:", SUMMARY_CSV.resolve(), flush=True)
    print("Layer diagnostics:", LAYER_CSV.resolve(), flush=True)
    if len(SEEDS) > 1:
        print("Per-T aggregate:", AGGREGATE_CSV.resolve(), flush=True)
    print("=" * 72, flush=True)

    return report


# ============================================================
# Launch the temporal-horizon experiment
# ============================================================

t_sweep_results = None

if RUN_GRID:
    t_sweep_results = run_T_sweep(
        T_values=T_VALUES,
        seeds=SEEDS,
        resume=RESUME_GRID,
        verbose=GRID_VERBOSE,
    )

    if t_sweep_results is not None and len(t_sweep_results):
        print(
            "\nT-sweep report:\n"
            + t_sweep_results.to_string(index=False)
        )
else:
    print("RUN_GRID=False — T sweep not launched.")


# ============================================================
# Output locations
# ============================================================
print("\nCompact T-sweep report:", REPORT_CSV.resolve())
print("Full experiment summary:", SUMMARY_CSV.resolve())
print("Layer diagnostics:", LAYER_CSV.resolve())
if len(SEEDS) > 1:
    print("Per-T aggregate:", AGGREGATE_CSV.resolve())
