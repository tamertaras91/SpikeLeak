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
# Model initialization is selected with `INIT_MODE = "active"` or `"default"`.
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

# ============================================================
# Layer-count ablation
# ============================================================
# Here "num_layers" means the TOTAL number of fully connected
# spiking layers, including the final classification layer.
#
#   2 layers: IN -> 512 -> 10
#   3 layers: IN -> 512 -> 256 -> 10
#   4 layers: IN -> 512 -> 256 -> 128 -> 10
#   6 layers: IN -> 512 -> 256 -> 128 -> 128 -> 64 -> 10
#   9 layers: IN -> 512 -> 256 -> 128 -> 128 -> 64 -> 64 -> 32 -> 32 -> 10
#
# The attack itself remains unchanged:
#   Stage 1: recover x0 from fc1 gradients and s1 from fc2 gradients
#   Stage 2: order/separate x0 using only recovered s1.
NUM_LAYERS_VALUES = [2, 3, 4, 6, 9]

# Attack is intentionally fixed across all depths:
TARGET_LAYER = "x0"
COVER_LAYERS = ("s1",)
RANK_LAYERS = ("x0", "s1")

# Controlled setting for the depth ablation.
# IMPORTANT: for a 2-layer network, fc2 is the 10-class output layer,
# hence rank(G_fc2) <= 10. To make all depths comparable, keep B*T <= 10.
LAYER_SWEEP_T = 8
LAYER_SWEEP_B = 1

# Aliases retained for generic helpers.
T_VALUES = [LAYER_SWEEP_T]
B_VALUES = [LAYER_SWEEP_B]

# Convenient single-setting debug run
SINGLE_T = LAYER_SWEEP_T
SINGLE_B = LAYER_SWEEP_B
SINGLE_NUM_LAYERS = 4

# Reproducibility
SEED = 0
SEEDS = [SEED]  # change to [0, 1, 2] for a three-seed sweep

# N-MNIST input/model dimensions
IN_DIM = 34 * 34 * 2
NUM_CLASSES = 10

H1 = 512
H2 = 256
H3 = 128

# Hidden widths used for each total FC-layer count.
# The original 4-layer architecture is retained as the reference.
HIDDEN_DIMS_BY_NUM_LAYERS = {
    2: [512],
    3: [512, 256],
    4: [512, 256, 128],
    6: [1024,512, 256, 128, 100, 100],
    9: [2096,1024,728,512,256,128,128,100],
}

BETA = 0.90
U_THR = 1.0

# Model initialization mode
# -------------------------
# "active": Xavier-uniform weights with gain 4.0 and constant bias 0.1.
#           This is the activity-preserving untrained initialization used
#           in the original experiments.
# "default": retain PyTorch's native nn.Linear initialization.
INIT_MODE = "active"          # choose: "active" or "default"
ACTIVE_W_GAIN = 4.0
ACTIVE_B_VAL = 0.1

if INIT_MODE not in {"active", "default"}:
    raise ValueError(
        f"INIT_MODE must be 'active' or 'default', got {INIT_MODE!r}"
    )


# DEPTH-ABLATION ATTACK CONFIGURATION
# -----------------------------------
# For a network with K total FC/LIF layers:
#   Stage 1 recovers x0 from fc1 and every hidden spike representation
#   s1,...,s_{K-1} from fc2,...,fcK.
#   Stage 2 orders x0 using ALL recovered downstream hidden spike sets.
#
# This is deliberate: the experiment asks whether additional downstream
# layers provide more dynamical constraints to Stage 2, or instead make
# candidate recovery/search harder.
TARGET_LAYER = "x0"
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

MILP_TIME_LIMIT = 300000000.0
SELECTION_MODE = "residual"
ENUMERATION_MODE = "guided"
GUIDED_RANDOM_WEIGHT = 1e-10
GUIDED_SEED = 0

# Stage 2
STAGE2_MAX_NODES = 2_000_000
MAX_SEQUENCE_SOLUTIONS = 5

# Files
DATA_DIR = _REPO_IMPORT_ROOT / "data"
RESULTS_DIR = Path("./results_depth_sweep")
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"
LAYER_CSV = RESULTS_DIR / "layer_diagnostics.csv"

# Qualitative figures saved for every successful grid setting.
SAVE_GRID_FIGURES = False
FIGURE_MAX_COLUMNS = 8
FIGURES_DIR = RESULTS_DIR / "figures"

# Print live Stage-1 / Stage-2 progress during the grid.
# Set False on long unattended runs if the console output is too large.
GRID_VERBOSE = True

# Notebook execution guards.
# Final grid-run configuration.
# RUN_GRID=True means executing the grid cell launches all requested settings.
# RESUME_GRID=True skips settings already present in experiment_summary.csv.
RUN_SINGLE = False
RUN_GRID = False
RUN_LAYER_SWEEP = True
RESUME_GRID = True

assert MAX_POOL >= LAYER_SWEEP_T * LAYER_SWEEP_B, (
    "MAX_POOL must be at least LAYER_SWEEP_T*LAYER_SWEEP_B."
)

print("NUM_LAYERS_VALUES:", NUM_LAYERS_VALUES)
print("Layer ablation setting: T=", LAYER_SWEEP_T,
      "| B=", LAYER_SWEEP_B,
      "| L=", LAYER_SWEEP_T * LAYER_SWEEP_B)
print("INIT_MODE:", INIT_MODE)
if INIT_MODE == "active":
    print("active initialization: Xavier-uniform gain=", ACTIVE_W_GAIN,
          "| constant bias=", ACTIVE_B_VAL)
else:
    print("default initialization: PyTorch nn.Linear defaults")


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
    """Fully connected SNN with configurable total FC/LIF depth."""

    def __init__(
        self,
        in_dim=IN_DIM,
        hidden_dims=None,
        out=NUM_CLASSES,
        beta=BETA,
        threshold=U_THR,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS_BY_NUM_LAYERS[4]

        self.hidden_dims = list(hidden_dims)
        dims = [in_dim] + self.hidden_dims + [out]
        self.num_layers = len(dims) - 1

        sg = surrogate.fast_sigmoid()
        self.fcs = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=True)
            for i in range(self.num_layers)
        ])
        self.lifs = nn.ModuleList([
            snn.Leaky(beta=beta, threshold=threshold, spike_grad=sg)
            for _ in range(self.num_layers)
        ])

    @property
    def fc1(self): return self.fcs[0]
    @property
    def fc2(self): return self.fcs[1]
    @property
    def lif1(self): return self.lifs[0]
    @property
    def lif2(self): return self.lifs[1]

    def init_mems(self, dtype, device):
        return [
            lif.init_leaky().to(device=device, dtype=dtype)
            for lif in self.lifs
        ]

    def forward(self, x, return_rec=True):
        dtype = x.dtype
        dev = x.device
        mems = self.init_mems(dtype=dtype, device=dev)

        out_sum = None
        rec = {"x0": []}
        for i in range(1, self.num_layers):
            rec[f"s{i}"] = []

        for t in range(x.shape[0]):
            z = x[t].to(device=dev, dtype=dtype)
            if return_rec:
                rec["x0"].append(z.detach())

            for layer_idx, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                cur = fc(z)
                z, mems[layer_idx] = lif(cur, mems[layer_idx])
                z = z.to(dtype=dtype)
                mems[layer_idx] = mems[layer_idx].to(dtype=dtype)

                # s1,...,s_{K-1} are presynaptic representations for
                # downstream FC layers and can therefore be recovered.
                if return_rec and layer_idx < self.num_layers - 1:
                    rec[f"s{layer_idx + 1}"].append(z.detach())

            out_sum = z if out_sum is None else out_sum + z

        if not return_rec:
            return out_sum

        rec = {k: torch.stack(v, dim=0) for k, v in rec.items()}
        return out_sum, rec


def init_snn_weights_active(net, w_gain=ACTIVE_W_GAIN, b_val=ACTIVE_B_VAL):
    """
    Activity-preserving initialization used in the original untrained runs.

    Linear weights: Xavier uniform with gain ``w_gain``.
    Linear biases : constant ``b_val``.
    """
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=w_gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, b_val)
    return net


def build_model(
    seed=SEED,
    device=DEVICE,
    init_mode=INIT_MODE,
    num_layers=4,
    w_gain=ACTIVE_W_GAIN,
    b_val=ACTIVE_B_VAL,
):
    """Construct a fresh float64 SNN with configurable depth."""
    if init_mode not in {"active", "default"}:
        raise ValueError(
            f"Unknown init_mode={init_mode!r}. Choose 'active' or 'default'."
        )
    if num_layers not in HIDDEN_DIMS_BY_NUM_LAYERS:
        raise ValueError(
            f"Unsupported num_layers={num_layers}. "
            f"Available: {sorted(HIDDEN_DIMS_BY_NUM_LAYERS)}"
        )

    set_seed(seed)
    hidden_dims = HIDDEN_DIMS_BY_NUM_LAYERS[num_layers]

    net = FCSNN(
        in_dim=IN_DIM,
        hidden_dims=hidden_dims,
        out=NUM_CLASSES,
        beta=BETA,
        threshold=U_THR,
    ).to(device).double()

    if init_mode == "active":
        init_snn_weights_active(net, w_gain=w_gain, b_val=b_val)

    return net


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
    """
    Build N-MNIST with exactly T time bins.

    Tonic downloads/extracts the dataset automatically if it is not already
    cached under the repository ``data`` folder.
    """
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=int(T),
        ),
    ])
    return tonic.datasets.NMNIST(
        save_to=str(save_to),
        train=train,
        transform=frame_transform,
    )

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
    """Return True iff one copy of ``key`` is still available."""
    return counter[key] > 0


def _counter_decrement_one(counter, key):
    """Consume one occurrence of ``key`` from a multiset counter."""
    counter[key] -= 1


def _counter_increment_one(counter, key):
    """Restore one occurrence of ``key`` to a multiset counter."""
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
    return net.init_mems(dtype=dtype, device=device)


def forward_with_current_recording(net, x_seq):
    """One forward/backward-compatible pass recording all recoverable layers."""
    dtype = x_seq.dtype
    dev = x_seq.device
    mems = _init_mems(net, dtype, dev)

    spk_out_sum = None
    rec_rank = {"x0": []}
    for i in range(1, net.num_layers):
        rec_rank[f"s{i}"] = []
    for i in range(1, net.num_layers + 1):
        rec_rank[f"cur{i}"] = []

    for t in range(x_seq.shape[0]):
        z = x_seq[t].to(device=dev, dtype=dtype)
        rec_rank["x0"].append(z.detach())

        for layer_idx, (fc, lif) in enumerate(zip(net.fcs, net.lifs)):
            cur = fc(z)
            cur.retain_grad()
            rec_rank[f"cur{layer_idx + 1}"].append(cur)

            z, mems[layer_idx] = lif(cur, mems[layer_idx])
            z = z.to(dtype=dtype)
            mems[layer_idx] = mems[layer_idx].to(dtype=dtype)

            if layer_idx < net.num_layers - 1:
                rec_rank[f"s{layer_idx + 1}"].append(z.detach())

        spk_out_sum = z if spk_out_sum is None else spk_out_sum + z

    rec_rank["x0"] = torch.stack(rec_rank["x0"], dim=0)
    for i in range(1, net.num_layers):
        rec_rank[f"s{i}"] = torch.stack(rec_rank[f"s{i}"], dim=0)

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
    layers=None,
    rank_tol=RANK_TOL,
    print_results=False,
):
    """Compute rank/pattern diagnostics for x0 and all hidden spike layers."""
    if layers is None:
        layers = ["x0"] + [f"s{i}" for i in range(1, net.num_layers)]
    selected_layers = list(layers)

    specs = {
        "x0": {
            "layer_name": "fc1",
            "presynaptic": "x0",
            "cur": "cur1",
            "layer": net.fcs[0],
        }
    }
    for i in range(1, net.num_layers):
        specs[f"s{i}"] = {
            "layer_name": f"fc{i + 1}",
            "presynaptic": f"s{i}",
            "cur": f"cur{i + 1}",
            "layer": net.fcs[i],
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
            raise ValueError(f"Unknown layer {name}; available={list(specs)}")
        spec = specs[name]
        layer = spec["layer"]

        G = _G_from_cur_list(rec_rank[spec["cur"]])
        S = _time_batch_to_columns(rec_rank[spec["presynaptic"]])
        S_bin = (S > 0).astype(np.float64)
        S_tilde = np.vstack([S_bin, np.ones((1, L), dtype=np.float64)])

        grad_W = layer.weight.grad.detach().cpu().numpy().astype(np.float64)
        grad_b = layer.bias.grad.detach().cpu().numpy().astype(np.float64)
        D = np.concatenate([grad_W, grad_b[:, None]], axis=1)

        rank_G = matrix_rank_np(G, rank_tol)
        rank_S = matrix_rank_np(S_bin, rank_tol)
        rank_S_tilde = matrix_rank_np(S_tilde, rank_tol)
        D_info = _rank_and_spectrum(D, tol=rank_tol)
        rank_D = D_info["rank"]

        grad_W_hat = G @ S_bin.T
        grad_b_hat = G @ np.ones((L,), dtype=G.dtype)
        rel_W_err = np.linalg.norm(grad_W - grad_W_hat) / (np.linalg.norm(grad_W) + 1e-12)
        rel_b_err = np.linalg.norm(grad_b - grad_b_hat) / (np.linalg.norm(grad_b) + 1e-12)

        # ----------------------------------------------------
        # G diagnostics: magnitude / sparsity / potential vanishing
        # ----------------------------------------------------
        G_abs = np.abs(G)
        G_max_abs = float(np.max(G_abs)) if G_abs.size > 0 else 0.0
        G_mean_abs = float(np.mean(G_abs)) if G_abs.size > 0 else 0.0
        G_median_abs = float(np.median(G_abs)) if G_abs.size > 0 else 0.0
        G_rms = float(np.sqrt(np.mean(G ** 2))) if G.size > 0 else 0.0
        G_fro_norm = float(np.linalg.norm(G)) if G.size > 0 else 0.0
        G_l1_norm = float(np.sum(G_abs)) if G_abs.size > 0 else 0.0
        G_abs_mean_over_max = (
            float(G_mean_abs / (G_max_abs + 1e-30))
            if G_abs.size > 0 else np.nan
        )

        # Fraction of entries effectively zero under several thresholds.
        G_sparsity_le_1e_12 = float(np.mean(G_abs <= 1e-12)) if G_abs.size > 0 else np.nan
        G_sparsity_le_1e_10 = float(np.mean(G_abs <= 1e-10)) if G_abs.size > 0 else np.nan
        G_sparsity_le_1e_08 = float(np.mean(G_abs <= 1e-8)) if G_abs.size > 0 else np.nan
        G_sparsity_le_1e_06 = float(np.mean(G_abs <= 1e-6)) if G_abs.size > 0 else np.nan

        G_density_gt_1e_12 = 1.0 - G_sparsity_le_1e_12 if np.isfinite(G_sparsity_le_1e_12) else np.nan
        G_density_gt_1e_10 = 1.0 - G_sparsity_le_1e_10 if np.isfinite(G_sparsity_le_1e_10) else np.nan
        G_density_gt_1e_08 = 1.0 - G_sparsity_le_1e_08 if np.isfinite(G_sparsity_le_1e_08) else np.nan
        G_density_gt_1e_06 = 1.0 - G_sparsity_le_1e_06 if np.isfinite(G_sparsity_le_1e_06) else np.nan

        # Column norms: each column corresponds to one (time,batch) factor g_{b,t}.
        G_col_l2 = np.linalg.norm(G, axis=0) if G.ndim == 2 else np.asarray([], dtype=np.float64)
        G_col_norm_min = float(np.min(G_col_l2)) if G_col_l2.size > 0 else 0.0
        G_col_norm_mean = float(np.mean(G_col_l2)) if G_col_l2.size > 0 else 0.0
        G_col_norm_median = float(np.median(G_col_l2)) if G_col_l2.size > 0 else 0.0
        G_col_norm_max = float(np.max(G_col_l2)) if G_col_l2.size > 0 else 0.0
        G_col_norm_zero_frac_1e_12 = float(np.mean(G_col_l2 <= 1e-12)) if G_col_l2.size > 0 else np.nan
        G_col_norm_zero_frac_1e_10 = float(np.mean(G_col_l2 <= 1e-10)) if G_col_l2.size > 0 else np.nan
        G_col_norm_zero_frac_1e_08 = float(np.mean(G_col_l2 <= 1e-8)) if G_col_l2.size > 0 else np.nan
        G_col_norm_mean_over_max = (
            float(G_col_norm_mean / (G_col_norm_max + 1e-30))
            if G_col_l2.size > 0 else np.nan
        )

        distinct_patterns = int(np.unique(S_bin.T.astype(np.uint8), axis=0).shape[0])

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

            # G statistics for diagnosing vanishing / sparse temporal factors.
            "G_mean_abs": G_mean_abs,
            "G_median_abs": G_median_abs,
            "G_rms": G_rms,
            "G_max_abs": G_max_abs,
            "G_fro_norm": G_fro_norm,
            "G_l1_norm": G_l1_norm,
            "G_abs_mean_over_max": G_abs_mean_over_max,

            "G_sparsity_le_1e_12": G_sparsity_le_1e_12,
            "G_sparsity_le_1e_10": G_sparsity_le_1e_10,
            "G_sparsity_le_1e_08": G_sparsity_le_1e_08,
            "G_sparsity_le_1e_06": G_sparsity_le_1e_06,

            "G_density_gt_1e_12": G_density_gt_1e_12,
            "G_density_gt_1e_10": G_density_gt_1e_10,
            "G_density_gt_1e_08": G_density_gt_1e_08,
            "G_density_gt_1e_06": G_density_gt_1e_06,

            "G_col_norm_min": G_col_norm_min,
            "G_col_norm_mean": G_col_norm_mean,
            "G_col_norm_median": G_col_norm_median,
            "G_col_norm_max": G_col_norm_max,
            "G_col_norm_zero_frac_1e_12": G_col_norm_zero_frac_1e_12,
            "G_col_norm_zero_frac_1e_10": G_col_norm_zero_frac_1e_10,
            "G_col_norm_zero_frac_1e_08": G_col_norm_zero_frac_1e_08,
            "G_col_norm_mean_over_max": G_col_norm_mean_over_max,

            "factorization_rel_W_error": float(rel_W_err),
            "factorization_rel_b_error": float(rel_b_err),
            "D_rank_tolerance": float(D_info["tol"]),
            "D_largest_singular": D_info["largest_singular"],
            "D_smallest_kept_singular": D_info["smallest_kept"],
            "D_next_singular": D_info["next_singular"],
            "D_spectral_gap_ratio": D_info["spectral_gap_ratio"],
        }

    if print_results:
        print(f"T={T_local}, B={B_local}, L={L}, loss={float(loss.detach().cpu()):.6g}")
        for name, r in results.items():
            print(
                f"{name:>3s} | rank(S~)={r['rank_S_tilde']:3d} | "
                f"rank(G)={r['rank_G']:3d} | rank(D)={r['rank_D']:3d} | "
                f"condition={r['rank_condition']} | density={r['spike_density']:.4f} | "
                f"Gmean={r['G_mean_abs']:.3e} | Gdens(>1e-8)={r['G_density_gt_1e_08']:.4f} | "
                f"GcolMean={r['G_col_norm_mean']:.3e}"
            )

    return results, rec_rank, loss


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
        "x0": (net.fcs[0], net.fcs[0].in_features, rec["x0"]),
    }
    for i in range(1, net.num_layers):
        layer_specs[f"s{i}"] = (
            net.fcs[i],
            net.fcs[i].in_features,
            rec[f"s{i}"],
        )
    selected = list(layer_specs) if layers == "all" else ([layers] if isinstance(layers, str) else list(layers))
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
    """Initialize only the membrane state needed for x0 -> s1 Stage 2."""
    if target_layer != "x0":
        raise ValueError("Depth ablation uses target_layer='x0'.")
    deepest_idx = max(int(layer[1:]) for layer in cover_layers)
    return {
        f"m{i}": net.lifs[i - 1].init_leaky().to(device=device, dtype=dtype)
        for i in range(1, deepest_idx + 1)
    }


def _clone_mems_dict(mems):
    return {k: v.clone() if hasattr(v, "clone") else v for k, v in mems.items()}


def propagate_from_target_one_step(net, target_layer, x_t, mems, cover_layers=("s1",)):
    """
    Optimized Stage 2 for the depth sweep.

    Propagate ONLY:
        x0 -> fc1 -> LIF1 -> s1

    Deeper hidden states are deliberately not computed or used, even when
    the tested network contains 3, 4, 6, or 9 layers.
    """
    if target_layer != "x0":
        raise ValueError("Depth ablation uses target_layer='x0'.")
    if tuple(cover_layers) != ("s1",):
        raise ValueError("This optimized depth sweep uses cover_layers=('s1',) only.")

    dtype = x_t.dtype
    produced = {}

    cur1 = net.fcs[0](x_t)
    s1, mems["m1"] = net.lifs[0](cur1, mems["m1"])
    s1 = s1.to(dtype=dtype)
    mems["m1"] = mems["m1"].to(dtype=dtype)

    produced["s1"] = s1
    return produced, mems


def _valid_cover_layers(target_layer, cover_layers):
    if target_layer != "x0":
        raise ValueError("Depth ablation uses target_layer='x0'.")
    if tuple(cover_layers) != ("s1",):
        raise ValueError("Optimized depth sweep supports cover_layers=('s1',) only.")


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
    """
    Optimized depth-ablation attack specification.

    Only two representations are reconstructed:
      x0 <- fc1 gradients
      s1 <- fc2 gradients

    No Stage-1 recovery is attempted for s2, s3, ... .
    """
    if net.num_layers < 2:
        raise ValueError("At least two FC layers are required to recover s1 from fc2.")

    return {
        "x0": (net.fcs[0], net.fcs[0].in_features, rec["x0"]),
        "s1": (net.fcs[1], net.fcs[1].in_features, rec["s1"]),
    }


def run_one_experiment(
    T,
    B,
    num_layers=4,
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
    experiment_id = f"nmnist_{INIT_MODE}_layers{num_layers}_T{T}_B{B}_seed{seed}"

    # Synchronize before starting the full-setting wall-clock timer so
    # outstanding CUDA work from a previous setting is not charged here.
    sync_device()
    setting_wall_start = time.perf_counter()

    summary = {
        "experiment_id": experiment_id,
        "dataset": "N-MNIST",
        "init_mode": INIT_MODE,
        "active_w_gain": float(ACTIVE_W_GAIN) if INIT_MODE == "active" else np.nan,
        "active_b_val": float(ACTIVE_B_VAL) if INIT_MODE == "active" else np.nan,
        "seed": int(seed),
        "num_layers": int(num_layers),
        "hidden_dims": json.dumps(HIDDEN_DIMS_BY_NUM_LAYERS[num_layers]),
        "layer_widths": json.dumps([IN_DIM] + HIDDEN_DIMS_BY_NUM_LAYERS[num_layers] + [NUM_CLASSES]),
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

        # Figure reporting: whether the saved qualitative figure came from
        # Stage 2 (ordered) or from Stage 1 only (unordered top candidates).
        "figure_mode": "",
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

        net = build_model(seed=seed, device=DEVICE, init_mode=INIT_MODE, num_layers=num_layers)
        criterion = nn.CrossEntropyLoss(reduction="mean")

        # ----------------------------------------------------
        # Rank diagnostics
        # ----------------------------------------------------
        # One forward/backward pass serves both rank diagnostics and the attack.
        # Parameter gradients remain available on net.fc1/net.fc2 for Stage 1.
        # ----------------------------------------------------
        # Optimized depth sweep: keep the ATTACK fixed across depths.
        # Only x0 and s1 are reconstructed/reported.
        # Stage 2 uses only s1 to order x0.
        # ----------------------------------------------------
        rank_layers = ["x0", "s1"]
        cover_layers = ("s1",)

        rank_results, rec, diagnostic_loss = layer_rank_diagnostics(
            net=net,
            x=x,
            y=y,
            criterion=criterion,
            layers=rank_layers,
            rank_tol=RANK_TOL,
            print_results=verbose,
        )

        summary["loss"] = float(diagnostic_loss.detach().cpu())

        # Start layer rows with rank/pattern statistics.
        for layer in rank_layers:
            r = rank_results[layer]

            layer_rows.append({
                "experiment_id": experiment_id,
                "dataset": "N-MNIST",
                "init_mode": INIT_MODE,
                "active_w_gain": float(ACTIVE_W_GAIN) if INIT_MODE == "active" else np.nan,
                "active_b_val": float(ACTIVE_B_VAL) if INIT_MODE == "active" else np.nan,
                "seed": int(seed),
                "num_layers": int(num_layers),
                "hidden_dims": json.dumps(HIDDEN_DIMS_BY_NUM_LAYERS[num_layers]),
                "layer_widths": json.dumps([IN_DIM] + HIDDEN_DIMS_BY_NUM_LAYERS[num_layers] + [NUM_CLASSES]),
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

        required_stage1_layers = ["x0", "s1"]

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

        summary["candidate_pool_size_per_layer"] = json.dumps({
            name: (
                None if not np.isfinite(layer_row_by_name[name].get("candidate_pool_size", np.nan))
                else int(layer_row_by_name[name]["candidate_pool_size"])
            )
            for name in rank_layers
        })
        summary["rank_condition_per_layer"] = json.dumps({
            name: bool(layer_row_by_name[name]["rank_condition"])
            for name in rank_layers
        })
        summary["true_pattern_coverage_per_layer"] = json.dumps({
            name: (
                None if not np.isfinite(layer_row_by_name[name].get("true_pattern_coverage", np.nan))
                else float(layer_row_by_name[name]["true_pattern_coverage"])
            )
            for name in rank_layers
        })

        # Compact G-diagnostic summaries for the depth study.
        summary["G_mean_abs_per_layer"] = json.dumps({
            name: float(layer_row_by_name[name].get("G_mean_abs", np.nan))
            for name in rank_layers
        })
        summary["G_density_gt_1e_08_per_layer"] = json.dumps({
            name: float(layer_row_by_name[name].get("G_density_gt_1e_08", np.nan))
            for name in rank_layers
        })
        summary["G_sparsity_le_1e_08_per_layer"] = json.dumps({
            name: float(layer_row_by_name[name].get("G_sparsity_le_1e_08", np.nan))
            for name in rank_layers
        })
        summary["G_col_norm_mean_per_layer"] = json.dumps({
            name: float(layer_row_by_name[name].get("G_col_norm_mean", np.nan))
            for name in rank_layers
        })
        summary["G_col_norm_min_per_layer"] = json.dumps({
            name: float(layer_row_by_name[name].get("G_col_norm_min", np.nan))
            for name in rank_layers
        })

        target_row = layer_row_by_name[TARGET_LAYER]

        summary["candidate_pool_size"] = target_row["candidate_pool_size"]
        summary["true_pattern_coverage"] = target_row["true_pattern_coverage"]

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

            summary["figure_mode"] = (
                "stage1" if ("x0" in recovered and "spikes" in recovered["x0"]) else ""
            )

            if save_results:
                save_experiment_rows(summary, layer_rows)

            return {
                "summary": summary,
                "layers": layer_rows,
                "x": x.detach().cpu(),
                "y": y.detach().cpu(),
                "true_rec": {
                    k: rec[k].detach().cpu()
                    for k in rank_layers
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
            cover_layers=cover_layers,
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

        summary["figure_mode"] = (
            "stage2"
            if stage2_result is not None and stage2_result.get("success", False)
            else ("stage1" if ("x0" in recovered and "spikes" in recovered["x0"]) else "")
        )

        return {
            "summary": summary,
            "layers": layer_rows,

            "x": x.detach().cpu(),
            "y": y.detach().cpu(),

            "true_rec": {
                k: rec[k].detach().cpu()
                for k in rank_layers
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
            raise

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
        num_layers=SINGLE_NUM_LAYERS,
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
    Save/display original and best-available reconstructed examples for one setting.

    Priority:
      1) Stage 2 ordered reconstruction if available.
      2) Stage 1 x0 top candidates as an unordered surrogate reconstruction.

    For B=1 the aggregate event image is still meaningful even when Stage 1 has
    not temporally ordered the recovered candidates.
    """
    if details is None:
        raise ValueError("No experiment details are available.")

    true_x = details["true_rec"][TARGET_LAYER]
    if torch.is_tensor(true_x):
        true_x = true_x.detach().cpu()

    stage2 = details.get("stage2")
    recovered = details.get("recovered", {})

    recon_x = None
    recon_label = None

    if stage2 is not None and stage2.get("success", False):
        recon_x = stage2["ordered"].detach().cpu()
        recon_label = "Reconstructed (Stage 2)"
    elif (
        isinstance(recovered, dict)
        and TARGET_LAYER in recovered
        and "spikes" in recovered[TARGET_LAYER]
        and recovered[TARGET_LAYER]["spikes"] is not None
    ):
        recon_x = recovered[TARGET_LAYER]["spikes"].detach().cpu()
        recon_label = "Recovered candidates (Stage 1)"
    else:
        raise ValueError(
            "No Stage-2 reconstruction or Stage-1 x0 candidate tensor is available."
        )

    if true_x.ndim != 3 or recon_x.ndim != 3:
        raise ValueError(
            f"Expected [T,B,M] tensors, got true={tuple(true_x.shape)}, "
            f"reconstructed={tuple(recon_x.shape)}"
        )

    _, B_true, M_true = true_x.shape
    _, B_recon, M_recon = recon_x.shape

    if B_true != B_recon or M_true != M_recon:
        raise ValueError(
            f"Batch/feature mismatch: true={tuple(true_x.shape)}, "
            f"reconstructed={tuple(recon_x.shape)}"
        )

    B_local = B_true
    ncols = min(int(max_columns), B_local)
    nblocks = int(np.ceil(B_local / ncols))
    nrows = 2 * nblocks

    fig_width = 2.05 * ncols
    fig_height = 4.15 * nblocks

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for b in range(B_local):
        block = b // ncols
        col = b % ncols
        original_row = 2 * block
        reconstructed_row = original_row + 1

        axes[original_row, col].imshow(
            make_event_rgb_from_flat_sequence(true_x[:, b, :], T=true_x.shape[0]),
            interpolation="nearest",
        )
        axes[original_row, col].axis("off")

        axes[reconstructed_row, col].imshow(
            make_event_rgb_from_flat_sequence(recon_x[:, b, :], T=recon_x.shape[0]),
            interpolation="nearest",
        )
        axes[reconstructed_row, col].axis("off")

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
            recon_label,
            fontsize=12,
            fontweight="bold",
            labelpad=10,
        )

    fig.suptitle(
        f"Original vs {recon_label} (B={B_local}, T_true={true_x.shape[0]}, T_rec={recon_x.shape[0]})",
        fontsize=15,
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.095,
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

    return save_path, recon_label


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


def run_layer_count_sweep(
    num_layers_values=NUM_LAYERS_VALUES,
    T=LAYER_SWEEP_T,
    B=LAYER_SWEEP_B,
    seed=SEED,
    resume=True,
    verbose=True,
):
    """Run the same attack for several total FC-layer counts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_GRID_FIGURES:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    L = int(T * B)
    if 2 in num_layers_values and L > NUM_CLASSES:
        print(
            "WARNING: for num_layers=2, fc2 is the 10-class classifier, "
            f"so rank(G_s1) <= {NUM_CLASSES} < L={L}. "
            "s1 recovery is therefore expected to fail the full-rank condition.",
            flush=True,
        )

    dataset = build_nmnist_dataset(T=T)
    sample_indices = deterministic_index_pool(
        dataset_len=len(dataset), max_batch_size=B, seed=seed
    )[:B]

    completed = existing_experiment_ids() if resume else set()
    sync_device()
    sweep_start = time.perf_counter()
    summaries = []

    print("\n" + "=" * 72)
    print("LAYER-COUNT ABLATION")
    print("num_layers:", list(num_layers_values))
    print(f"T={T} | B={B} | L={L}")
    print("INIT_MODE:", INIT_MODE)
    print("sample_indices:", sample_indices)
    print("=" * 72)

    for num_layers in num_layers_values:
        experiment_id = (
            f"nmnist_{INIT_MODE}_layers{num_layers}_T{T}_B{B}_seed{seed}"
        )
        if resume and experiment_id in completed:
            print("skip:", experiment_id)
            continue

        print(
            f"\n{'=' * 72}\n"
            f"RUN {experiment_id}\n"
            f"hidden_dims={HIDDEN_DIMS_BY_NUM_LAYERS[num_layers]}\n"
            f"{'=' * 72}",
            flush=True,
        )

        details = run_one_experiment(
            T=T,
            B=B,
            num_layers=num_layers,
            dataset=dataset,
            sample_indices=sample_indices,
            seed=seed,
            save_results=True,
            verbose=verbose,
        )
        row = details["summary"]
        summaries.append(row)

        if SAVE_GRID_FIGURES:
            figure_mode = str(row.get("figure_mode", "")).strip().lower()

            if figure_mode:
                if figure_mode == "stage2":
                    suffix = "original_vs_stage2_reconstructed"
                elif figure_mode == "stage1":
                    suffix = "original_vs_stage1_candidates"
                else:
                    suffix = "original_vs_reconstruction"

                figure_path = FIGURES_DIR / f"{experiment_id}_{suffix}.png"

                try:
                    _, recon_label = plot_original_vs_reconstructed_batch(
                        details,
                        save_path=figure_path,
                        show=False,
                        max_columns=FIGURE_MAX_COLUMNS,
                    )
                    print(
                        f"figure saved ({recon_label}): {figure_path.resolve()}",
                        flush=True,
                    )
                except Exception as figure_error:
                    print(
                        f"WARNING: figure saving failed for {experiment_id}: {figure_error}",
                        flush=True,
                    )

        print(
            f"status={row['status']} | reason={row['failure_reason']} "
            f"| figure_mode={row.get('figure_mode', '')}\n"
            f"G_mean_abs={row.get('G_mean_abs_per_layer', '')} "
            f"| G_density(>1e-8)={row.get('G_density_gt_1e_08_per_layer', '')}\n"
            f"Stage 1={format_duration(row.get('stage1_time_s', np.nan))}"
            f" | Stage 2={format_duration(row.get('stage2_time_s', np.nan))}"
            f" | attack={format_duration(row.get('attack_time_s', np.nan))}"
            f" | setting={format_duration(row.get('setting_time_s', np.nan))}",
            flush=True,
        )

        del details
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sync_device()
    sweep_time = time.perf_counter() - sweep_start
    print("\n" + "=" * 72)
    print("LAYER-COUNT ABLATION COMPLETE")
    print("Total wall time:", format_duration(sweep_time))
    print("Summary CSV:", SUMMARY_CSV.resolve())
    print("Layer CSV:", LAYER_CSV.resolve())
    print("=" * 72)
    return pd.DataFrame(summaries)


depth_sweep_frames = []

if RUN_LAYER_SWEEP:
    for sweep_seed in SEEDS:
        df_seed = run_layer_count_sweep(
            num_layers_values=NUM_LAYERS_VALUES,
            T=LAYER_SWEEP_T,
            B=LAYER_SWEEP_B,
            seed=sweep_seed,
            resume=RESUME_GRID,
            verbose=GRID_VERBOSE,
        )
        if df_seed is not None and not df_seed.empty:
            depth_sweep_frames.append(df_seed)

    # Build compact report from the persisted summary so resume mode is included.
    if SUMMARY_CSV.exists():
        full_summary = pd.read_csv(SUMMARY_CSV)
        wanted_ids = {
            f"nmnist_{INIT_MODE}_layers{d}_T{LAYER_SWEEP_T}_B{LAYER_SWEEP_B}_seed{s}"
            for s in SEEDS for d in NUM_LAYERS_VALUES
        }
        report = full_summary[full_summary["experiment_id"].astype(str).isin(wanted_ids)].copy()

        report_columns = [
            "seed",
            "num_layers",
            "layer_widths",
            "T",
            "B",
            "status",
            "reconstruction_success",
            "bit_accuracy",
            "f1",
            "attack_time_s",
            "visited_nodes",
            "backtracks",
            "candidate_pool_size_per_layer",
            "rank_condition_per_layer",
            "true_pattern_coverage_per_layer",
            "G_mean_abs_per_layer",
            "G_density_gt_1e_08_per_layer",
            "G_sparsity_le_1e_08_per_layer",
            "G_col_norm_mean_per_layer",
            "G_col_norm_min_per_layer",
            "figure_mode",
            "failure_reason",
        ]
        report_columns = [c for c in report_columns if c in report.columns]
        report = report[report_columns].sort_values(["seed", "num_layers"])
        report_path = RESULTS_DIR / "depth_sweep_report.csv"
        report.to_csv(report_path, index=False)
        print("\nDepth-sweep report:")
        print(report.to_string(index=False))
        print("Saved:", report_path.resolve())
else:
    print("RUN_LAYER_SWEEP=False — no depth sweep launched.")
