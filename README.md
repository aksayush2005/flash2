# FlashAttention-2 Triton Implementation

A high-performance FlashAttention-style implementation built with Triton and PyTorch. This repository provides optimized forward and backward attention kernels, benchmarking utilities, and numerical correctness validation against PyTorch reference implementations.

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

## Features

* Triton-based FlashAttention-style forward and backward kernels
* Custom PyTorch autograd integration
* Numerical correctness validation against PyTorch reference attention
* Performance benchmarking with CUDA event timing
* Support for causal and non-causal attention
* Configurable batch sizes, sequence lengths, head counts, and head dimensions

## Requirements

Install dependencies using:

```bash
pip install -r triton/requirements.txt
```

### Prerequisites

* Python 3.10+
* CUDA-compatible GPU
* PyTorch
* Triton

## Components

### `triton/flash2-triton.py`

Core implementation containing:

* Forward attention kernel
* Backward attention kernels
* Triton kernel launch logic
* `TritonAttention` autograd wrapper

### `triton/benchmark_flash2.py`

Benchmarking utility that:

* Measures latency using CUDA events
* Reports mean, median, minimum, and maximum execution times
* Estimates throughput in TFLOPS
* Compares against PyTorch Scaled Dot Product Attention (SDPA) when available
* Falls back to Triton-only benchmarking when SDPA is unavailable

### `triton/check_correctness_flash2.py`

Validation script that compares Triton outputs against PyTorch reference implementations for:

* Forward pass outputs
* Query gradients (`dQ`)
* Key gradients (`dK`)
* Value gradients (`dV`)

Default settings are intentionally conservative to accommodate memory-constrained GPUs.

## Running Benchmarks

Basic benchmark:

```bash
python triton/benchmark_flash2.py
```

Causal attention benchmark:

```bash
python triton/benchmark_flash2.py --causal --seq-lens 1024,2048 --head-dims 64
```

Larger-scale benchmark:

```bash
python triton/benchmark_flash2.py --batch 8 --heads 16 --seq-lens 4096 --head-dims 64
```

## Running Correctness Validation

Default validation:

```bash
python triton/check_correctness_flash2.py
```

Minimal validation configuration:

```bash
python triton/check_correctness_flash2.py --batch 1 --heads 1 --seq-lens 128 --head-dims 64
```

Extended validation sweep:

```bash
python triton/check_correctness_flash2.py --seq-lens 128,256,512 --head-dims 64 --causal-modes false,true
```

## Kaggle Recommendations

For Kaggle or other resource-constrained environments:

* Begin with sequence lengths of `128` or `256`
* Run correctness checks before large-scale benchmarks
* Avoid dense PyTorch reference attention for very large sequence lengths
* Benchmark Triton kernels independently when GPU memory is limited
* Expect minor autotuning differences across GPU architectures

Recommended workflow:

1. Install dependencies
2. Run correctness validation on small problem sizes
3. Benchmark on target sequence lengths and head dimensions

## Results

Benchmark outputs, performance analyses, and validation results can be found in the `results/` directory.

## License

This project is provided for research, experimentation, and educational purposes.

