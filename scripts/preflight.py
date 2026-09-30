#!/usr/bin/env python
"""Resource/configuration preflight without executing the SNN attack."""

import argparse
from pathlib import Path

LOCAL_ONLY_DATA = {
    "cifar10dvs": ("CIFAR10", ".aedat"),
}

DOWNLOADABLE_DATA = {
    "nmnist": "NMNIST",
    "dvsgesture": "DVSGesture",
    "ncaltech101": "NCALTECH101",
}

ARCH = {
    "nmnist": [2312, 512, 256, 128, 10],
    "dvsgesture": [32768, 7384, 4096, 1024, 11],
    "ncaltech101": [8192, 1024, 512, 256, 101],
    "cifar10dvs": [32768, 7384, 4096, 1024, 10],
}

def parameter_count(dims):
    return sum(
        dims[i] * dims[i + 1] + dims[i + 1]
        for i in range(len(dims) - 1)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(ARCH), required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dims = ARCH[args.dataset]
    n = parameter_count(dims)
    params_gib = n * 8 / (1024 ** 3)
    params_grads_gib = 2 * params_gib

    print("=" * 78)
    print("REVIEWER PREFLIGHT")
    print(f"dataset: {args.dataset}")
    print("architecture:", " -> ".join(map(str, dims)))
    print(f"parameters: {n:,}")
    print(f"float64 parameters: ~{params_gib:.2f} GiB")
    print(f"float64 parameters + gradients: ~{params_grads_gib:.2f} GiB")
    print("Additional memory is required for activations, temporal recordings,")
    print("SVD/MILP arrays, solver state, and framework overhead.")

    if args.dataset in LOCAL_ONLY_DATA:
        rel_folder, extension = LOCAL_ONLY_DATA[args.dataset]
        data_folder = repo / "data" / Path(rel_folder)
        files = list(data_folder.glob(f"**/*{extension}")) if data_folder.exists() else []
        print(f"dataset mode: repository-local")
        print(f"local data folder: {data_folder}")
        print(f"matching {extension} files: {len(files)}")
        if not data_folder.exists():
            print("ERROR: expected local CIFAR10-DVS folder does not exist.")
        elif not files:
            print(f"ERROR: no {extension} files were found under the expected folder.")
        else:
            print("local data check: OK")
    else:
        cache_name = DOWNLOADABLE_DATA[args.dataset]
        cache_folder = repo / "data" / cache_name
        print("dataset mode: Tonic auto-download/cache")
        print(f"cache root: {cache_folder}")
        if cache_folder.exists():
            print("cache status: present; Tonic will reuse available files")
        else:
            print("cache status: not present; first experiment run will download it")

    if args.dataset in {"dvsgesture", "cifar10dvs"}:
        print()
        print("NOTE: the large 32768 -> 7384 first layer is allocated even for B=1.")
        print("Therefore a small batch does not make model allocation inexpensive.")

    print("=" * 78)

if __name__ == "__main__":
    main()
