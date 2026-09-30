# Reproducibility guide

## 1. Environment

Install:

```bash
pip install -r requirements.txt
```

The supplied source snapshots do not encode the exact historical
`snnTorch` and `Tonic` package versions. Before public release, the authors
should replace the unpinned entries in `requirements.txt` with the exact
versions used for the final reported runs.

The analytical experiments use `float64`.

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





## N-MNIST B=1 qualitative figures

Run:

```bash
python scripts/run_setting.py --dataset nmnist --T 8 --B 1 --seed 0
```

On a successful reconstruction, the runner keeps the aggregate figures:

```text
nmnist_T8_B1_seed0_original.png
nmnist_T8_B1_seed0_reconstructed.png
```

and additionally writes the time-resolved event-sequence figures:

```text
nmnist_T8_B1_seed0_original_events.png
nmnist_T8_B1_seed0_reconstructed_events.png
```





## Local-data-only execution


```text
data/NMNIST/Train
data/DVSGesture/ibmGestureTrain
data/NCALTECH101/Caltech101
data/CIFAR10
```


