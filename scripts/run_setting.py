#!/usr/bin/env python
"""Run one reviewer-selected experiment with live terminal progress."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

DATASETS = {
    "nmnist": "nmnist.py",
    "dvsgesture": "dvsgesture.py",
    "ncaltech101": "ncaltech101.py",
    "cifar10dvs": "cifar10dvs.py",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true",
                           help="Detailed output (already the default).")
    verbosity.add_argument("--quiet", action="store_true",
                           help="Hide detailed solver/search lines; stage checkpoints remain.")

    parser.add_argument("--milp-time-limit", type=float, default=None)
    parser.add_argument("--max-pool", type=int, default=None)
    parser.add_argument("--stage2-max-nodes", type=int, default=None)
    parser.add_argument("--max-sequence-solutions", type=int, default=None)

    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    script = repo / "experiments" / "main" / DATASETS[args.dataset]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SNN_REPO_ROOT"] = str(repo)
    env["SNN_T_VALUES"] = str(args.T)
    env["SNN_B_VALUES"] = str(args.B)
    env["SNN_SEED"] = str(args.seed)
    env["SNN_VERBOSE"] = "0" if args.quiet else "1"
    env["SNN_RESUME"] = "1" if args.resume else "0"

    if args.milp_time_limit is not None:
        env["SNN_MILP_TIME_LIMIT"] = str(args.milp_time_limit)
    if args.max_pool is not None:
        env["SNN_MAX_POOL"] = str(args.max_pool)
    if args.stage2_max_nodes is not None:
        env["SNN_STAGE2_MAX_NODES"] = str(args.stage2_max_nodes)
    if args.max_sequence_solutions is not None:
        env["SNN_MAX_SEQUENCE_SOLUTIONS"] = str(args.max_sequence_solutions)

    print("=" * 78, flush=True)
    print("SNN REVIEWER RUN", flush=True)
    print(f"dataset : {args.dataset}", flush=True)
    print(f"T       : {args.T}", flush=True)
    print(f"B       : {args.B}", flush=True)
    print(f"seed    : {args.seed}", flush=True)
    print(f"runner  : {script}", flush=True)
    print(f"verbose : {not args.quiet}", flush=True)
    print("=" * 78, flush=True)

    subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=repo,
        env=env,
        check=True,
    )

if __name__ == "__main__":
    main()
