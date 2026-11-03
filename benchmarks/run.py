"""Run on a GPU node from the repo root: python3 benchmarks/run.py --m 1 64 1024 --n 256 4096 [--print]"""

import argparse
import csv
import os
from itertools import product
from pathlib import Path
import re
import subprocess
import pandas as pd

import numpy as np
from data import generate_data, save_data


VARIANTS = ["rank_search_k.fut"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", required=True)
    parser.add_argument("--n", type=int, nargs="+", required=True)
    parser.add_argument("--distribution", nargs="+", choices=["regular", "irregular"],
                        default=["regular", "irregular"])
    parser.add_argument("--k-mode", nargs="+", choices=["min", "max", "median", "random"],
                        default=["median", "random"])
    parser.add_argument("--value-mode", nargs="+", choices=["full", "segment", "segment10", "small"],
                        default=["full", "small"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--print", action="store_true",
                        help="only print timings and validation to stdout; write no CSV files")
    args = parser.parse_args()
    if min(args.m + args.n) < 1:
        parser.error("m and n must be positive")

    root = Path(__file__).resolve().parents[1]
    datasets = root / "benchmarks/datasets"
    build = datasets / "_build"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(["make", "-C", str(root / "cub-baseline"), "test-batch"], check=True)
    
    executables = {}

    for variant in VARIANTS:
        executable = build / Path(variant).stem
        subprocess.run(["futhark", "cuda", "-o", str(executable), str(root / variant)], check=True)
        executables[variant] = executable

    output = root / args.output
    if not args.print:
        output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "m", "n", "total_elements", "distribution", "k_mode", "value_mode", "seed",
          "cub_us", "futhark_us", "speedup"]
    # Append so later experiment runs keep earlier measurements.
    with (open(os.devnull, "w") if args.print else output.open("a", newline="")) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if not args.print and csv_file.tell() == 0:
            writer.writeheader()
        for m, n, distribution, k_mode, value_mode, seed in product(
                args.m, args.n, args.distribution, args.k_mode, args.value_mode, args.seeds):
            name = f"m{m}-n{n}-{distribution}-{k_mode}-{value_mode}-s{seed}"
            case = datasets / name
            values, offsets, ks = generate_data(
                m, n, distribution == "irregular", k_mode, value_mode, seed)
            save_data(values, offsets, ks, case)

            cub = subprocess.run([str(root / "cub-baseline/test-batch"), str(case)],
                                 check=True, stdout=subprocess.PIPE, text=True)
            match = re.search(r"Sorting \+ selection time: ([0-9.]+) us, VALID: 1\b", cub.stdout)
            if match is None:
                raise RuntimeError(f"CUB failed for {name}:\n{cub.stdout}")
            cub_us = float(match[1])

            expected = [int(np.sort(values[start:end])[int(k) - 1])
                    for start, end, k in zip(offsets[:-1], offsets[1:], ks)]

            # Match CUB's first visible GPU, warm-up, and 400 timed calls.
            for variant, executable in executables.items():
                timings = case / f"{Path(variant).stem}-times.txt"
                with (case / "input.in").open() as input_file:
                    fut = subprocess.run([str(executable), "-d", "#0", "-r", "400", "-t", str(timings)],
                                        stdin=input_file, check=True, stdout=subprocess.PIPE,
                                        text=True)
                actual = [int(x) for x in re.findall(r"(\d+)u32", fut.stdout)]
                if actual != expected:
                    raise RuntimeError(f"Incorrect Futhark result for {name}")
                samples = np.loadtxt(timings, ndmin=1)
                if samples.size != 400 or not np.all(np.isfinite(samples)) or np.any(samples < 0):
                    raise RuntimeError(f"Invalid Futhark timings for {name}")
                futhark_us = float(samples.mean())
                if futhark_us <= 0:
                    raise RuntimeError(f"Futhark timings too small to measure for {name}")

                speedup = cub_us / futhark_us
                result = dict(zip(fields, [variant, m, n, len(values), distribution, k_mode,
                           value_mode, seed, cub_us, futhark_us, speedup]))
                writer.writerow(result)

                csv_file.flush()
                # Reaching this line means CUB and Futhark both matched the NumPy reference.
                print(f"{name} | {variant}: CUB={cub_us:.2f} us, Futhark={futhark_us:.2f} us, "
                        f"speedup={speedup:.3f}x, VALIDATED", flush=True)
    if args.print:
        return
    print(f"Results: {output}")
    create_summary(output)

def create_summary(output):
    df = pd.read_csv(output)
    group_cols = ["variant", "m", "n", "distribution", "k_mode", "value_mode"]

    summary = df.groupby(group_cols).agg(
        seeds=("seed", "nunique"),
        cub_us_mean=("cub_us", "mean"),
        futhark_us_mean=("futhark_us", "mean"),
        futhark_us_std=("futhark_us", "std"),
        speedup_mean=("speedup", "mean"),
        speedup_std=("speedup", "std")
    ).reset_index()

    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()
