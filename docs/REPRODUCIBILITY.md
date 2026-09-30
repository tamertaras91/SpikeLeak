# Reproducibility guide

## 1. Environment

Install:

```bash
pip install -r requirements.txt
```



## 2. Static check

```bash
python scripts/check_repository.py
```

This compiles the Python files but does not download data or run experiments.

## 3. One reviewer-selected setting

Example:

```bash
python scripts/run_setting.py \
    --dataset nmnist \
    --T 8 \
    --B 8 \
    --seed 0
```

The launcher sets only:

```text
SNN_T_VALUES
SNN_B_VALUES
SNN_SEED
SNN_VERBOSE
SNN_RESUME
SNN_REPO_ROOT
```

The attack implementation remains inside the dataset-specific runner.

## 4. Output locations

Main runs write to:

```text
results/reproduced/<dataset>/
```

including experiment summaries, layer diagnostics, and qualitative figures
when produced by the runner.

## 5. Failure semantics

Failed settings are experimental outcomes and should be retained.

Important reasons include:

```text
rank_condition_failed
incomplete_true_pattern_coverage
insufficient_stage2_candidate_pool
node_limit
search_exhausted
```

These should not be silently converted into missing rows.

## 6. Reference results

Place the exact CSVs and images used in the paper in:

```text
results/reference/
figures/reference/
```

They are intentionally left empty in this initial artifact because no final
paper-result exports were supplied with the repository-building request.




## B=1 event-sequence figures for all main datasets

For a successful `B=1` run on any main dataset runner, the aggregate figures are
kept:

```text
<experiment_id>_original.png
<experiment_id>_reconstructed.png
```

and the repository additionally writes time-resolved event-sequence figures:

```text
<experiment_id>_original_events.png
<experiment_id>_reconstructed_events.png
```

This now applies to:

```text
nmnist
dvsgesture
ncaltech101
cifar10dvs
```


## Diagnosing long-running settings

All main runners now provide live stage checkpoints. The last terminal line
therefore identifies whether a long run is currently in:

- dataset loading;
- batch loading/binarization;
- model allocation;
- forward/backward and rank diagnostics;
- a specific Stage-I MILP solve;
- Stage-II temporal search.

The launcher is unbuffered, so these messages should appear immediately.

For DVS128 Gesture and CIFAR10-DVS, batch size is not the main determinant of
model-allocation cost because their first FC layer is `32768 -> 7384`.
Run `scripts/preflight.py` to inspect the approximate float64 parameter
footprint before executing the attack.




## Dataset acquisition

The artifact intentionally keeps the Git repository small.

`N-MNIST`, `DVS128 Gesture`, and `N-Caltech101` are acquired through Tonic
when first used and cached under `data/`. Their cache directories are ignored
by Git.

`CIFAR10-DVS` is not downloaded automatically; the experiment reads the
repository-local `.aedat` files from `data/CIFAR10/`.
