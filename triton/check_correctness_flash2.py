import argparse
import gc
import importlib.util
import math
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


def print_rule(char="=", width=112):
    print(char * width)


def print_title(text, width=112):
    print_rule("=", width)
    print(text.center(width))
    print_rule("=", width)


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


def dtype_from_name(name):
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_case(TritonAttention, batch, heads, seq_len, head_dim, causal, dtype, atol, rtol, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    scale = 1.0 / math.sqrt(head_dim)
    shape = (batch, heads, seq_len, head_dim)

    q_ref = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    k_ref = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    v_ref = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)

    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)
    dO = torch.randn(shape, device="cuda", dtype=dtype)

    mask = None
    if causal:
        mask = torch.tril(torch.ones((seq_len, seq_len), device="cuda", dtype=torch.bool))

    ref_scores = torch.matmul(q_ref, k_ref.transpose(2, 3)) * scale
    if causal:
        ref_scores = ref_scores.masked_fill(~mask, float("-inf"))
    ref_probs = torch.softmax(ref_scores.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(ref_probs, v_ref)
    ref_out.backward(dO)

    tri_out = TritonAttention.apply(q_tri, k_tri, v_tri, causal, scale)
    tri_out.backward(dO)

    checks = [
        ("forward", ref_out, tri_out),
        ("dQ", q_ref.grad, q_tri.grad),
        ("dK", k_ref.grad, k_tri.grad),
        ("dV", v_ref.grad, v_tri.grad),
    ]

    rows = []
    all_passed = True
    for name, ref_tensor, tri_tensor in checks:
        passed = torch.allclose(ref_tensor, tri_tensor, atol=atol, rtol=rtol)
        diff = max_abs_diff(ref_tensor, tri_tensor)
        rows.append(
            [
                name,
                "PASS" if passed else "FAIL",
                f"{diff:.6e}",
                str(tuple(ref_tensor.shape)),
            ]
        )
        all_passed = all_passed and passed

    del q_ref, k_ref, v_ref, q_tri, k_tri, v_tri, dO, ref_scores, ref_probs, ref_out, tri_out
    cleanup()
    return all_passed, rows


def main():
    parser = argparse.ArgumentParser(description="Correctness checker for Triton FlashAttention.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq-lens", type=str, default="128,256")
    parser.add_argument("--head-dims", type=str, default="64")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--causal-modes", type=str, default="false,true")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. Run this script on a CUDA machine.")

    dtype = dtype_from_name(args.dtype)
    seq_lens = parse_int_list(args.seq_lens)
    head_dims = parse_int_list(args.head_dims)
    causal_modes = []
    for value in args.causal_modes.split(","):
        value = value.strip().lower()
        if value not in {"true", "false"}:
            raise ValueError(f"Invalid causal mode: {value}")
        causal_modes.append(value == "true")

    TritonAttention = load_triton_attention()

    print_title("FlashAttention Triton Correctness Check")
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"PyTorch: {torch.__version__}")
    print(f"dtype: {args.dtype}")
    print(f"batch: {args.batch}")
    print(f"heads: {args.heads}")
    print(f"seq_lens: {seq_lens}")
    print(f"head_dims: {head_dims}")
    print(f"causal_modes: {causal_modes}")
    print(f"atol: {args.atol}")
    print(f"rtol: {args.rtol}")
    print_rule()

    summary_rows = []
    overall_pass = True

    for causal in causal_modes:
        for head_dim in head_dims:
            for seq_len in seq_lens:
                case_label = f"B={args.batch} H={args.heads} S={seq_len} D={head_dim} causal={causal}"
                print_title(case_label)
                passed, rows = run_case(
                    TritonAttention,
                    args.batch,
                    args.heads,
                    seq_len,
                    head_dim,
                    causal,
                    dtype,
                    args.atol,
                    args.rtol,
                    args.seed,
                )
                render_table(["check", "status", "max_abs_diff", "shape"], rows)
                summary_rows.append([case_label, "PASS" if passed else "FAIL"])
                overall_pass = overall_pass and passed
                print_rule()

    print_title("Summary")
    render_table(["case", "result"], summary_rows)
    print(f"overall_result: {'PASS' if overall_pass else 'FAIL'}")
    print_rule("=")


if __name__ == "__main__":
    main()
