# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Cross-device benchmark: Xe-Forge baseline vs the installed finegrained_fp8 kernel.

Same as benchmark.py, except the "optimized" side is no longer loaded from the
local delivery file — it is the kernel exported by the ``finegrained_fp8``
package (``w8a8_block_dynamic_fp8_matmul_grouped``), i.e. the upstream
kernels-community build that already received the ported optimization. This lets
you confirm the in-tree package reproduces the standalone optimized kernel's
speedup and correctness.

Usage:
    python benchmark_finegrained.py                       # all variants, auto device
    python benchmark_finegrained.py --device cuda         # force CUDA (A100)
    python benchmark_finegrained.py --variant gate_up-decode
    python benchmark_finegrained.py --warmup 25 --iters 100
"""

import argparse
import os
import sys
import time

import torch
from kernels import get_kernel

HERE = os.path.dirname(os.path.abspath(__file__))

# Make the in-tree finegrained_fp8 package importable. Adjust if the
# kernels-community checkout lives elsewhere.
_FINEGRAINED_TORCH_EXT = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "..", "kernels-community", "finegrained-fp8", "torch-ext")
)
if os.path.isdir(_FINEGRAINED_TORCH_EXT) and _FINEGRAINED_TORCH_EXT not in sys.path:
    sys.path.insert(0, _FINEGRAINED_TORCH_EXT)

import finegrained_fp8  # noqa: E402  (path set up above)

# BASELINE is the remote published build fetched via get_kernel; no local file.
# get_kernel now requires an explicit version= or revision=.
BASELINE = get_kernel("kernels-community/finegrained-fp8", revision="main")
# OPTIMIZED is the in-tree package kernel. The block-dynamic grouped wrapper
# lives in the ``grouped`` submodule (the package ``__init__`` only re-exports
# the ``matmul_grouped`` dispatcher), so reference it there directly.
OPTIMIZED = finegrained_fp8.grouped

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN = torch.finfo(FP8_DTYPE).min


def make_experts_fp8(num_experts, out_features, in_features, block_n, block_k, device):
    """Per-expert block-wise FP8 weights ``(E, N, K)`` + scales ``(E, N//bn, K//bk)``."""
    assert out_features % block_n == 0 and in_features % block_k == 0
    rt, ct = out_features // block_n, in_features // block_k
    weight_q = torch.empty(num_experts, out_features, in_features, dtype=FP8_DTYPE, device=device)
    scales = torch.empty(num_experts, rt, ct, dtype=torch.float32, device=device)
    for e in range(num_experts):
        w = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
        reshaped = w.reshape(rt, block_n, ct, block_k)
        max_abs = reshaped.abs().amax(dim=(1, 3))
        safe = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
        scale = FP8_MAX / safe
        wq = (reshaped * scale[:, None, :, None]).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
        weight_q[e] = wq.reshape(out_features, in_features)
        scales[e] = (1.0 / scale).to(torch.float32)
    return weight_q.contiguous(), scales.contiguous()


def make_routed_inputs(S, E, K, top_k, dtype, device, seed: int = 0):
    """Build ``(selected_hidden_states (S, K), expert_ids (S,))`` for batched MoE."""
    assert S % top_k == 0, f"S ({S}) must be divisible by top_k ({top_k})"
    g = torch.Generator(device=device).manual_seed(seed)
    num_tokens = S // top_k
    hidden_states = torch.randn(num_tokens, K, dtype=dtype, device=device, generator=g)
    top_k_index = torch.randint(0, E, (num_tokens, top_k), device=device, generator=g)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    expert_ids = top_k_index.reshape(-1).to(torch.int32)
    selected = hidden_states[token_idx].contiguous()
    return selected, expert_ids


def prepare_grouped(a, expert_ids, num_experts):
    """Sort rows by expert; return ``(a_sorted, expert_ids_sorted, offsets, tokens_per_expert)``."""
    perm = torch.argsort(expert_ids)
    a_sorted = a[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]
    tokens_per_expert = torch.histc(
        expert_ids_sorted.float(), bins=num_experts, min=0, max=num_experts - 1
    ).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)
    return a_sorted, expert_ids_sorted, offsets, tokens_per_expert


# DeepSeek V4 (E=256, top_k=6) + Qwen3 (E=128, top_k=8) presets. N/K per proj.
VARIANTS = {
    # name:                  (S,    E,   N,    K,    top_k)
    "gate_up-decode":        (192,  256, 4096, 4096, 6),
    "down-decode":           (192,  256, 4096, 2048, 6),
    "gate_up-prefill":       (1536, 256, 4096, 4096, 6),
    "down-prefill":          (1536, 256, 4096, 2048, 6),
    "qwen3-gate_up-decode":  (256,  128, 1536, 2048, 8),
    "qwen3-down-decode":     (256,  128, 2048, 768,  8),
    "qwen3-gate_up-prefill": (1024, 128, 1536, 2048, 8),
    "qwen3-down-prefill":    (1024, 128, 2048, 768,  8),
}
BLOCK_N = BLOCK_K = 128


def detect_device(requested=None):
    if requested:
        return torch.device(requested)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("No XPU or CUDA device available")


def device_sync(device):
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "synchronize"):
        backend.synchronize()


def print_autotune_winners():
    """Dump the autotuned winning config per (N, K, BLOCK_SIZE_M) key.

    Use these to narrow ``get_block_dynamic_fp8_grouped_configs`` to the configs
    that actually win for the benchmarked shapes (the autotune ``key`` collapses
    the 8 variants to 4 distinct (N, K) keys, so the winner set is tiny). Each
    line is ``key -> Config(...)`` with the chosen BLOCK_SIZE_N / num_warps /
    num_stages / grf_mode.
    """
    kern = getattr(
        finegrained_fp8.grouped,
        "w8a8_block_dynamic_fp8_matmul_grouped_kernel",
        None,
    )
    cache = getattr(kern, "cache", None)
    if not cache:
        return
    print("\nAutotune winners (key=(N, K, BLOCK_SIZE_M) -> config):")
    for key, cfg in cache.items():
        print(f"  {key} -> {cfg}")


def build_inputs(S, E, N, K, top_k, device, seed=123):
    """Shared inputs for both kernels, built locally so the weight/activation
    layout matches exactly what the kernels expect."""
    torch.manual_seed(seed)
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "manual_seed_all"):
        backend.manual_seed_all(seed)
    B, Bs = make_experts_fp8(E, N, K, BLOCK_N, BLOCK_K, device)
    a, expert_ids = make_routed_inputs(S, E, K, top_k, torch.bfloat16, device)
    a_sorted, _, offsets, tokens_per_expert = prepare_grouped(a, expert_ids, E)
    return a_sorted, B, Bs, offsets, tokens_per_expert


def reference(a_sorted, B, Bs, offsets, tokens_per_expert, E, block_n, block_k):
    """High-precision dequantized grouped matmul (fp32) for correctness.

    Dequantizes one expert at a time to avoid materializing the full
    (E, N, K) fp32 weight tensor (which is ~17 GB for DeepSeek V4 and OOMs).
    """
    S, K = a_sorted.shape
    _, N, _ = B.shape
    out = torch.zeros(S, N, dtype=torch.float32, device=a_sorted.device)
    starts = torch.cat([torch.zeros(1, device=offsets.device, dtype=offsets.dtype), offsets[:-1]])
    a32 = a_sorted.to(torch.float32)
    for e in range(E):
        s0, s1 = int(starts[e]), int(offsets[e])
        if s1 <= s0:
            continue
        # Per-expert dequant: (N, K) fp8 * (N//bn, K//bk) scale expanded.
        we = B[e].to(torch.float32)
        se = Bs[e].to(torch.float32)
        se = se.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
        we = we * se  # (N, K) real weights for this expert
        out[s0:s1] = a32[s0:s1] @ we.T
        del we, se
    return out


def bench(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    device_sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    device_sync(device)
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds


def run_variant(name, dims, device, warmup, iters, rtol, atol):
    S, E, N, K, top_k = dims
    a_sorted, B, Bs, offsets, tpe = build_inputs(S, E, N, K, top_k, device)
    bsz = [BLOCK_N, BLOCK_K]

    base_fn = lambda: BASELINE.w8a8_block_fp8_matmul_grouped(
        a_sorted, B, Bs, offsets, tpe, bsz
    )
    opt_fn = lambda: OPTIMIZED.w8a8_block_dynamic_fp8_matmul_grouped(
        a_sorted, B, Bs, offsets, tpe, bsz, output_dtype=torch.bfloat16
    )

    out_base = base_fn()
    out_opt = opt_fn()
    ref = reference(a_sorted, B, Bs, offsets, tpe, E, BLOCK_N, BLOCK_K)

    def rel_err(x):
        x = x.to(torch.float32)
        return (x - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)

    eb, eo = rel_err(out_base), rel_err(out_opt)
    agree = torch.allclose(out_base.float(), out_opt.float(), rtol=rtol, atol=atol)

    us_base = bench(base_fn, warmup, iters, device)
    us_opt = bench(opt_fn, warmup, iters, device)
    speedup = us_base / us_opt if us_opt > 0 else float("nan")

    flop = 2 * S * N * K
    tf_base = flop / (us_base * 1e-6) / 1e12
    tf_opt = flop / (us_opt * 1e-6) / 1e12

    print(
        f"{name:24s} | base {us_base:9.2f}us ({tf_base:6.2f} TF) "
        f"| opt {us_opt:9.2f}us ({tf_opt:6.2f} TF) "
        f"| speedup {speedup:5.2f}x "
        f"| relerr base/opt {eb:.2e}/{eo:.2e} | agree {agree}"
    )
    return speedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda | xpu (auto-detect if unset)")
    ap.add_argument("--variant", default=None, help="single variant name (see --list)")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rtol", type=float, default=5e-2)
    ap.add_argument("--atol", type=float, default=5e-2)
    ap.add_argument("--list", action="store_true", help="list variants and exit")
    args = ap.parse_args()

    if args.list:
        for k, v in VARIANTS.items():
            print(f"{k:24s} S={v[0]} E={v[1]} N={v[2]} K={v[3]} top_k={v[4]}")
        return

    device = detect_device(args.device)
    print(f"Device: {device}  (torch {torch.__version__})")
    print(f"Optimized kernel: finegrained_fp8 @ {os.path.dirname(finegrained_fp8.__file__)}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)} "
              f"sm{torch.cuda.get_device_capability(device)[0]}{torch.cuda.get_device_capability(device)[1]}")
    print()

    variants = {args.variant: VARIANTS[args.variant]} if args.variant else VARIANTS
    speedups = []
    for name, dims in variants.items():
        try:
            speedups.append(run_variant(name, dims, device, args.warmup, args.iters, args.rtol, args.atol))
        except Exception as e:  # keep going across variants
            print(f"{name:24s} | FAILED: {type(e).__name__}: {e}")
        finally:
            # Free the ~4 GB per-variant weights before building the next one.
            backend = getattr(torch, device.type, None)
            if backend is not None and hasattr(backend, "empty_cache"):
                backend.empty_cache()

    if speedups:
        geo = torch.tensor(speedups).log().mean().exp().item()
        print(f"\nGeomean speedup: {geo:.3f}x over {len(speedups)} variant(s)")

    print_autotune_winners()


if __name__ == "__main__":
    main()
