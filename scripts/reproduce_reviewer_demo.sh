#!/usr/bin/env bash
set -euo pipefail



python scripts/run_setting.py --dataset nmnist     --T 8 --B 8 --seed 0
python scripts/run_setting.py --dataset dvsgesture --T 8 --B 8 --seed 0
python scripts/run_setting.py --dataset ncaltech101 --T 8 --B 8 --seed 0
python scripts/run_setting.py --dataset cifar10dvs --T 8 --B 8 --seed 0
