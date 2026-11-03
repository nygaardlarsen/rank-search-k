# Overview

- `rank_search_k.fut` – Futhark implementation 
- `cub-baseline/` – CUB segmented radix sort + select implementation 
- `benchmarks/run.py` – generates data, runs CUB and the Futhark version, and checks every result against a NumPy sort. This requires Python 3 with `numpy` and `pandas`.

## Run (from the repo root)

```sh
python3 benchmarks/run.py --m 1 64 1024 --n 256 4096 16384 \
  --k-mode min max median random --value-mode full small --seeds 42 43 44 --print
```

Each case prints one line:
`<case> | <variant>: CUB=<us> us, Futhark=<us> us, speedup=<CUB/Futhark>x, VALIDATED`.
If any result is wrong, the script stops with an error.
Times are the mean of 400 runs after a warm-up, in microseconds.

## Futhark test/bench on precomputed datasets

```sh
futhark test --backend=cuda rank_search_k.fut
futhark bench --backend=cuda rank_search_k.fut
```

The datasets listed in `rank_search_k.fut` and their expected outputs (NumPy sort) are included in
`benchmarks/futhark-data/`; This validation uses a smaller dataset, to skip dataset generation