# Rank-k search on GPU

- `rank_search_k.fut` – Futhark implementation (median-of-3 pivot, one 2m-bin histogram).
- `cub-baseline/` – CUB segmented radix sort + select implementation 
- `benchmarks/run.py` – generates data, runs CUB and the Futhark version, and checks every result against a NumPy sort.

## Requirements

A CUDA GPU with `nvcc` and `futhark` on `PATH`, and Python 3 with `numpy` and `pandas`.

## Run (from the repo root)

```sh
python3 benchmarks/run.py --m 1 64 1024 --n 256 4096 16384 \
  --k-mode min max median random --value-mode full small --seeds 42 43 44 --print
```

Each case prints one line:
`<case> | <variant>: CUB=<us> us, Futhark=<us> us, speedup=<CUB/Futhark>x, VALIDATED`.
If any result is wrong, the script stops with an error.
Times are the mean of 400 runs after a warm-up, in microseconds.
