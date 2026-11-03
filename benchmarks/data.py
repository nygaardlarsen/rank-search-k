
import argparse
from pathlib import Path
import numpy as np
import json
import re

def generate_data(m, n, irregular=False, k_mode="random", value_mode="full", seed=42):
    rng = np.random.default_rng(seed)

    if m < 1 or n < 1:
        raise ValueError("m and n must be positive")

    if m > np.iinfo(np.int32).max or n > np.iinfo(np.int32).max:
        raise ValueError("Input exceeds int32 limits")

    if irregular and n * 19 // 10 > np.iinfo(np.int32).max:
        raise ValueError("Irregular segment size exceeds int32 limits")
        
    if irregular:
        # sizes from 0.1 to 1.9 times og segment size
        # still in expectation 1, so comparable with regular sizes
        sizes = rng.integers(max(1, n // 10), n * 19 // 10 + 1, size=m)
    else:
        # same sizes
        sizes = np.full(m, n, dtype=np.int64)

    offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    if offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError("Too many elements for int32 offsets")

    
    N = int(offsets[-1])
    values = np.empty(N, dtype=np.uint32)

    if value_mode not in ("full", "segment", "segment10", "small"):
        raise ValueError("Unknown value_mode")

    for i, size in enumerate(sizes):
        # all possible 32 bit integers
        # few duplicates in expectation, unless very large size
        if value_mode == "full":
            value_max = 2**32
        # max = number of values in segment
        elif value_mode == "segment":
            value_max = int(size)
        # max =  number of values in segment / 10
        # should produce duplicates, which changes how the algorithms might behave
        elif value_mode == "segment10":
            value_max = max(1, int(size) // 10)
        # max = 100
        # tons of duplicates
        else:
            value_max = 100

        values[offsets[i]:offsets[i + 1]] = rng.integers(0, value_max, size=int(size), dtype=np.uint32)

    # different ways to select k
    # to investigate hoow the algorithms behave
    if k_mode == "min":
        ks = np.ones(m, dtype=np.int32)
    elif k_mode == "max":
        ks = sizes.astype(np.int32)
    elif k_mode == "median":
        ks = ((sizes + 1) // 2).astype(np.int32)
    elif k_mode == "random":
        ks = rng.integers(1, sizes + 1).astype(np.int32)
    else:
        raise ValueError("Unknown k_mode")


    assert len(values) == offsets[-1]
    assert len(ks) == m
    assert np.all(ks >= 1)
    assert np.all(ks <= sizes)
    assert np.all(np.diff(offsets) == sizes)

    return values, offsets.astype(np.int32), ks


def write_futhark_array(f, array, futhark_type):
    # Futhark binary data format: 'b', version 2, rank 1, 4-char type, int64 length, data.
    f.write(b"b" + bytes([2, 1]) + f"{futhark_type:>4}".encode())
    f.write(np.int64(len(array)).astype("<i8").tobytes())
    f.write(array.astype(array.dtype.newbyteorder("<")).tobytes())


def expected_result(values, offsets, ks):
    return np.array([np.sort(values[start:end])[k - 1]
                     for start, end, k in zip(offsets[:-1], offsets[1:], ks)], dtype=np.uint32)


def save_futhark_binary(values, offsets, ks, path):
    """Write <path> (Futhark input) and <path minus .in>.out (expected output)."""
    path = Path(path)
    with path.open("wb") as f:
        write_futhark_array(f, ks.astype(np.int32), "i32")
        write_futhark_array(f, np.diff(offsets).astype(np.int32), "i32")
        write_futhark_array(f, values.astype(np.uint32), "u32")
    with path.with_suffix(".out").open("wb") as f:
        write_futhark_array(f, expected_result(values, offsets, ks), "u32")

FUTHARK_DATA = Path(__file__).resolve().parent / "futhark-data"


def dataset_name(m, n, distribution, k_mode, value_mode, seed):
    return f"m{m}-n{n}-{distribution}-{k_mode}-{value_mode}-s{seed}"


def write_futhark_dataset(m, n, distribution, k_mode, value_mode, seed):
    values, offsets, ks = generate_data(m, n, distribution == "irregular", k_mode, value_mode, seed)
    output = FUTHARK_DATA / f"{dataset_name(m, n, distribution, k_mode, value_mode, seed)}.in"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_futhark_binary(values, offsets, ks, output)
    print(f"Saved {len(values)} elements in {len(ks)} segments to {output} (+ .out)")


def write_futhark_all():
    """(Re)generate every dataset referenced in rank_search_k.fut."""
    fut = Path(__file__).resolve().parents[1] / "rank_search_k.fut"
    pattern = re.compile(r"@ benchmarks/futhark-data/m(\d+)-n(\d+)-(regular|irregular)-"
                         r"(min|max|median|random)-(full|segment|segment10|small)-s(\d+)\.in")
    for m, n, distribution, k_mode, value_mode, seed in sorted(set(pattern.findall(fut.read_text()))):
        write_futhark_dataset(int(m), int(n), distribution, k_mode, value_mode, int(seed))


def save_data(values, offsets, ks, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    values.astype("<u4").tofile(path / "values.bin")
    offsets.astype("<i4").tofile(path / "offsets.bin")
    ks.astype("<i4").tofile(path / "ks.bin")

    def write_array(f, array, dtype):
        f.write("[")
        for start in range(0, len(array), 10000):
            chunk = array[start:start + 10000]
            if start:
                f.write(", ")
            f.write(", ".join(f"{int(x)}{dtype}" for x in chunk))
        f.write("]\n")

    with (path / "input.in").open("w") as f:
        write_array(f, ks, "i32")
        write_array(f, np.diff(offsets), "i32")
        write_array(f, values, "u32")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--distribution", choices=["regular", "irregular"], default="regular")
    parser.add_argument("--k-mode", choices=["min", "max", "median", "random"], default="random")
    parser.add_argument("--value-mode", choices=["full", "segment", "segment10", "small"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--futhark", action="store_true", default=False,
                        help="only write a Futhark binary .in (+ expected .out) to benchmarks/futhark-data/")
    parser.add_argument("--futhark-all", action="store_true", default=False,
                        help="write all datasets referenced in rank_search_k.fut to benchmarks/futhark-data/")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.futhark_all:
        write_futhark_all()
        raise SystemExit
    if args.m is None or args.n is None:
        parser.error("the following arguments are required: --m, --n")
    if args.m < 1 or args.n < 1:
        parser.error("segments and size must be positive")
    if not args.futhark and args.output is None:
        parser.error("the following arguments are required: --output")

    if args.futhark:
        write_futhark_dataset(args.m, args.n, args.distribution, args.k_mode, args.value_mode, args.seed)
        raise SystemExit

    values, offsets, ks = generate_data(args.m, args.n, args.distribution == "irregular", args.k_mode, args.value_mode, args.seed)

    save_data(values, offsets, ks, args.output)


    metadata = {
        "m": args.m,
        "n": args.n,
        "total_elements": len(values),
        "distribution": args.distribution,
        "k_mode": args.k_mode,
        "value_mode": args.value_mode,
        "seed": args.seed
    }

    with (Path(args.output) / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved {len(values)} elements in {len(ks)} segments to {args.output}")
