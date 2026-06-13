# flash2: FlashAttention-2 Implementation

This repository contains a Triton-based FlashAttention-style implementation, with utility scripts for benchmarking and correctness validation on CUDA GPUs.

The current workspace includes:

- `triton/flash2-triton.py` — Triton attention kernel implementation
- `triton/benchmark_flash2.py` — benchmark runner for forward/backward performance
- `triton/check_correctness_flash2.py` — numerical correctness checker against PyTorch
- `triton/requirements.txt` — Python dependency list
- `results/` — placeholder directory for generated output or visualizations

## Repository Structure

```text
.
├── README.md
├── .gitignore
├── results/
└── triton/
    ├── benchmark_flash2.py
    ├── check_correctness_flash2.py
    ├── flash2-triton.py
    └── requirements.txt
```

## Requirements

Install the Python dependencies from `triton/requirements.txt`:

```bash
pip install -r triton/requirements.txt
```

This project requires:

- Python
- CUDA-compatible GPU
- PyTorch
- Triton

## Files

### `triton/flash2-triton.py`

Contains the Triton implementation of the FlashAttention-style forward and backward kernels and the `TritonAttention` autograd wrapper.

### `triton/benchmark_flash2.py`

Performance benchmark script for the Triton kernel. It:

- prints readable terminal tables
- measures latency using CUDA events
- reports mean, median, min, max, and estimated TFLOPS
- compares against PyTorch SDPA when available
- shows Triton-only results if SDPA benchmarking is unavailable

### `triton/check_correctness_flash2.py`

Numerical correctness checker against a PyTorch reference implementation. It validates:

- forward output
- `dQ`
- `dK`
- `dV`

Defaults are conservative to fit memory-limited GPUs.

## Running Benchmarks

```bash
python triton/benchmark_flash2.py
```

Causal benchmark example:

```bash
python triton/benchmark_flash2.py --causal --seq-lens 1024,2048 --head-dims 64
```

Larger benchmark example:

```bash
python triton/benchmark_flash2.py --batch 8 --heads 16 --seq-lens 4096 --head-dims 64
```

## Running Correctness Checks

```bash
python triton/check_correctness_flash2.py
```

Safer minimal run:

```bash
python triton/check_correctness_flash2.py --batch 1 --heads 1 --seq-lens 128 --head-dims 64
```

Extended correctness sweep:

```bash
python triton/check_correctness_flash2.py --seq-lens 128,256,512 --head-dims 64 --causal-modes false,true
```

## Kaggle Notes

For Kaggle or other limited-memory environments:

- start correctness checks with small sequence lengths such as `128` or `256`
- avoid running dense PyTorch reference attention at very large sequence lengths
- benchmark Triton separately from correctness checks
- expect some GPU-dependent Triton autotuning differences

Recommended order on Kaggle:

1. Install dependencies
2. Run `check_correctness_flash2.py` on small shapes
3. Run `benchmark_flash2.py` on target benchmark shapes

## Output Placeholders

Add benchmark or correctness visuals to `results/` later, for example:

```markdown
![Benchmark Output](results/benchmark.png)
```

```markdown
![Correctness Output](results/correctness.png)
```
