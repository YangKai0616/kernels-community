import torch

from ._ops import ops

from . import layers


def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.silu_and_mul(out, x)
    return out


def mul_and_silu(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.mul_and_silu(out, x)
    return out


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.gelu_and_mul(out, x)
    return out


def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.gelu_tanh_and_mul(out, x)
    return out


def fatrelu_and_mul(out: torch.Tensor, x: torch.Tensor, threshold: float = 0.0) -> None:
    ops.fatrelu_and_mul(out, x, threshold)
    return out


def gelu(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.gelu(out, x)
    return out

def silu(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.silu(out, x)
    return out


def gelu_tanh(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.gelu_tanh(out, x)
    return out


def gelu_fast(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.gelu_fast(out, x)
    return out


def gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.gelu_new(out, x)
    return out


def gelu_quick(out: torch.Tensor, x: torch.Tensor) -> None:
    ops.gelu_quick(out, x)
    return out


__all__ = [
    "silu_and_mul",
    "mul_and_silu",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "fatrelu_and_mul",
    "gelu_fast",
    "gelu_new",
    "gelu_quick",
    "gelu_tanh",
    "silu",
    "gelu",
    "layers",
]
