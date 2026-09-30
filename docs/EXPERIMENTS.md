# Experiment inventory

## Main dataset runners

```text
experiments/main/nmnist.py
experiments/main/dvsgesture.py
experiments/main/ncaltech101.py
experiments/main/cifar10dvs.py
```

Use `scripts/run_setting.py` to choose `T`, `B`, and `seed` without editing
the files manually.

## Ablations

```text
experiments/ablations/nmnist_temporal_horizon_sweep.py
experiments/ablations/nmnist_depth_sweep.py
```

The depth runner includes diagnostics for the magnitude and sparsity of the
temporal gradient factor `G`, intended to help diagnose whether deeper SNNs
suffer gradient attenuation.

## Training checkpoint experiment

```text
experiments/training/nmnist_checkpoint_sweep.py
```

## Optimization-based comparison

```text
experiments/baselines/nmnist_dlg_style_B1_T8.py
```

This is the DLG-style gradient-matching baseline with the true label known
to the baseline.

## Qualitative dataset-only utilities

The scripts under `scripts/figures/` only read the corresponding datasets and
create aggregate event-image grids. They do not execute the attack.
