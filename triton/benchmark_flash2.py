import argparse
import gc
import importlib.util
import math
import statistics
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
KERNEL_PATH = ROOT / "flash2-triton.py"

def load_triton_attention():
    spec = importlib.util.spec_from_file_location("flash2_triton_module", KERNEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TritonAttention


def format_ms(value_ms):
    return f"{value_ms:8.3f} ms"


def format_tflops(value):
    return f"{value:8.2f}"


def format_mem_gb(value):
    return f"{value:7.2f} GB"


def print_rule(char="=", width=108):
    print(char * width)


def print_title(text, width=108):
    print_rule("=", width)
    print(text.center(width))
    print_rule("=", width)


def print_kv(label, value, width=108):
    left = f"{label}: {value}"
    print(left if len(left) <= width else left[: width - 3] + "...")


def render_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def sep(char="-"):
        return "+" + "+".join(char * (w + 2) for w in widths) + "+"

    print(sep("-"))
    print(
        "| "
        + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(headers))
        + " |"
    )
    print(sep("="))
    for row in rows:
        print(
            "| "
            + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))
            + " |"
        )
    print(sep("-"))


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def make_inputs(batch, heads, seq_len, head_dim, dtype, device, requires_grad):
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def attention_flops(batch, heads, seq_len, head_dim, causal, direction):
    factor = 0.5 if causal else 1.0
    base = 4.0 * batch * heads * (seq_len**2) * head_dim * factor
    if direction == "fwd":
        return base
    if direction == "bwd":
        return base * 2.5
    if direction == "fwd+bwd":
        return base * 3.5
    raise ValueError(f"Unknown direction: {direction}")


def benchmark_cuda_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def benchmark_triton(TritonAttention, batch, heads, seq_len, head_dim, causal, dtype, warmup, iters):
    scale = 1.0 / math.sqrt(head_dim)

    def run_forward():
        q, k, v = make_inputs(batch, heads, seq_len, head_dim, dtype, "cuda", False)
        TritonAttention.apply(q, k, v, causal, scale)

    def run_backward():
        q, k, v = make_inputs(batch, heads, seq_len, head_dim, dtype, "cuda", True)
        out = TritonAttention.apply(q, k, v, causal, scale)
        grad = torch.randn_like(out)
        out.backward(grad)

    fwd_times = benchmark_cuda_ms(run_forward, warmup, iters)
    torch.cuda.empty_cache()
    gc.collect()
    bwd_times = benchmark_cuda_ms(run_backward, warmup, iters)
    return fwd_times, bwd_times


def benchmark_sdpa(batch, heads, seq_len, head_dim, causal, dtype, warmup, iters):
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return None

    scale = 1.0 / math.sqrt(head_dim)

    def run_forward():
        q, k, v = make_inputs(batch, heads, seq_len, head_dim, dtype, "cuda", False)
        torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale
        )

    def run_backward():
        q, k, v = make_inputs(batch, heads, seq_len, head_dim, dtype, "cuda", True)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale
        )
        grad = torch.randn_like(out)
        out.backward(grad)

    fwd_times = benchmark_cuda_ms(run_forward, warmup, iters)
    torch.cuda.empty_cache()
    gc.collect()
    bwd_times = benchmark_cuda_ms(run_backward, warmup, iters)
    return fwd_times, bwd_times


def try_benchmark_sdpa(batch, heads, seq_len, head_dim, causal, dtype, warmup, iters):
    try:
        return benchmark_sdpa(batch, heads, seq_len, head_dim, causal, dtype, warmup, iters), None
    except Exception as exc:
        return None, exc


def summarize(times_ms):
    return {
        "mean": statistics.mean(times_ms),
        "median": statistics.median(times_ms),
        "min": min(times_ms),
        "max": max(times_ms),
    }


def build_row(name, direction, times_ms, flops):
    stats = summarize(times_ms)
    mean_s = stats["mean"] / 1000.0
    tflops = flops / mean_s / 1e12
    return [
        name,
        direction,
        format_ms(stats["mean"]),
        format_ms(stats["median"]),
        format_ms(stats["min"]),
        format_ms(stats["max"]),
        format_tflops(tflops),
    ]


def dtype_from_name(name):
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton FlashAttention kernels.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-lens", type=str, default="1024,2048,4096")
    parser.add_argument("--head-dims", type=str, default="64")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. Run this script on a CUDA machine.")

    dtype = dtype_from_name(args.dtype)
    seq_lens = parse_int_list(args.seq_lens)
    head_dims = parse_int_list(args.head_dims)
    TritonAttention = load_triton_attention()

    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    cuda_version = torch.version.cuda or "unknown"

    print_title("FlashAttention Triton Benchmark")
    print_kv("GPU", device_name)
    print_kv("CUDA", cuda_version)
    print_kv("PyTorch", torch.__version__)
    print_kv("dtype", args.dtype)
    print_kv("causal", args.causal)
    print_kv("batch", args.batch)
    print_kv("heads", args.heads)
    print_kv("seq_lens", seq_lens)
    print_kv("head_dims", head_dims)
    print_kv("warmup", args.warmup)
    print_kv("iters", args.iters)
    print_rule()

    overall_start = time.perf_counter()

    for head_dim in head_dims:
        for seq_len in seq_lens:
            print_title(f"Case: B={args.batch} H={args.heads} S={seq_len} D={head_dim}")

            rows = []
            fwd_times, bwd_times = benchmark_triton(
                TritonAttention,
                args.batch,
                args.heads,
                seq_len,
                head_dim,
                args.causal,
                dtype,
                args.warmup,
                args.iters,
            )
            rows.append(
                build_row(
                    "triton",
                    "forward",
                    fwd_times,
                    attention_flops(args.batch, args.heads, seq_len, head_dim, args.causal, "fwd"),
                )
            )
            rows.append(
                build_row(
                    "triton",
                    "backward",
                    bwd_times,
                    attention_flops(args.batch, args.heads, seq_len, head_dim, args.causal, "bwd"),
                )
            )

            sdpa_result, sdpa_error = try_benchmark_sdpa(
                args.batch,
                args.heads,
                seq_len,
                head_dim,
                args.causal,
                dtype,
                args.warmup,
                args.iters,
            )
            if sdpa_result is not None:
                sdpa_fwd, sdpa_bwd = sdpa_result
                rows.append(
                    build_row(
                        "torch_sdpa",
                        "forward",
                        sdpa_fwd,
                        attention_flops(
                            args.batch, args.heads, seq_len, head_dim, args.causal, "fwd"
                        ),
                    )
                )
                rows.append(
                    build_row(
                        "torch_sdpa",
                        "backward",
                        sdpa_bwd,
                        attention_flops(
                            args.batch, args.heads, seq_len, head_dim, args.causal, "bwd"
                        ),
                    )
                )

            render_table(
                ["kernel", "pass", "mean", "median", "min", "max", "TFLOPS"],
                rows,
            )
            if sdpa_error is not None:
                print_kv(
                    "sdpa_compare",
                    f"skipped due to error: {type(sdpa_error).__name__}: {sdpa_error}",
                )

            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            print_kv("peak_allocated", format_mem_gb(peak_mem))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()
            print_rule()

    total_s = time.perf_counter() - overall_start
    print_kv("total_runtime", f"{total_s:.2f} s")
    print_rule("=")


if __name__ == "__main__":
    main()
