"""Run on a GPU node: python3 benchmarks/run.py --m 1 64 1024 --n 256 4096"""

import argparse
import csv
from itertools import product
from pathlib import Path
import re
import subprocess

import numpy as np
from data import generate_data, save_data


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.csv"))
    args = parser.parse_args()
    if min(args.m + args.n) < 1:
        parser.error("m and n must be positive")

    root = Path(__file__).resolve().parents[1]
    datasets = root / "benchmarks/datasets"
    build = datasets / "_build"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(["make", "-C", str(root / "cub-baseline"), "test-batch"], check=True)
    executable = build / "rank_search"
    subprocess.run(["futhark", "cuda", "-o", str(executable), str(root / "rank_search.fut")],
                   check=True)

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["m", "n", "total_elements", "distribution", "k_mode", "value_mode", "seed",
              "cub_us", "futhark_us", "speedup"]
    # Append so later experiment runs keep earlier measurements.
    with output.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if csv_file.tell() == 0:
            writer.writeheader()
        for m, n, distribution, k_mode, value_mode in product(
                args.m, args.n, args.distribution, args.k_mode, args.value_mode):
            name = f"m{m}-n{n}-{distribution}-{k_mode}-{value_mode}-s{args.seed}"
            case = datasets / name
            values, offsets, ks = generate_data(
                m, n, distribution == "irregular", k_mode, value_mode, args.seed)
            save_data(values, offsets, ks, case)

            cub = subprocess.run([str(root / "cub-baseline/test-batch"), str(case)],
                                 check=True, stdout=subprocess.PIPE, text=True)
            match = re.search(r"Sorting \+ selection time: ([0-9.]+) us, VALID: 1\b", cub.stdout)
            if match is None:
                raise RuntimeError(f"CUB failed for {name}:\n{cub.stdout}")
            cub_us = float(match[1])

            # Match CUB's first visible GPU, warm-up, and 400 timed calls.
            timings = case / "futhark-times.txt"
            with (case / "input.in").open() as input_file:
                fut = subprocess.run([str(executable), "-d", "#0", "-r", "400", "-t", str(timings)],
                                     stdin=input_file, check=True, stdout=subprocess.PIPE,
                                     text=True)
            actual = [int(x) for x in re.findall(r"(\d+)u32", fut.stdout)]
            expected = [int(np.sort(values[start:end])[int(k) - 1])
                        for start, end, k in zip(offsets[:-1], offsets[1:], ks)]
            if actual != expected:
                raise RuntimeError(f"Incorrect Futhark result for {name}")
            samples = np.loadtxt(timings, ndmin=1)
            if samples.size != 400 or not np.all(np.isfinite(samples)) or np.any(samples < 0):
                raise RuntimeError(f"Invalid Futhark timings for {name}")
            futhark_us = float(samples.mean())
            if futhark_us <= 0:
                raise RuntimeError(f"Futhark timings too small to measure for {name}")

            speedup = cub_us / futhark_us
            writer.writerow(dict(zip(fields, [m, n, len(values), distribution, k_mode,
                                             value_mode, args.seed, cub_us, futhark_us, speedup])))
            csv_file.flush()
            print(f"{name}: CUB={cub_us:.2f} us, Futhark={futhark_us:.2f} us, "
                  f"speedup={speedup:.3f}x", flush=True)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
