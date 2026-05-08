"""
问题1的 PyTorch 参考实现。

使用示例：
    from triton_learning.reference_ops import create_problem_tensors, run_reference_pipeline

说明：
    所有 Step 都先以这里的 PyTorch 结果作为数值正确性的权威参考。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from triton_learning.problem_spec import ProblemShape


@dataclass(frozen=True)
class ProblemTensors:
    """保存问题1用到的四个输入矩阵。"""

    x: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor


def compute_lora_down(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """算子1：Y = X @ A。"""
    return x @ a


def compute_lora_expand(y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """算子2：Z = Y @ B。"""
    return y @ b


def compute_main_matmul(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """算子3：W = X @ C。"""
    return x @ c


def compute_output_add(w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """最终输出：O = W + Z。"""
    return w + z


def run_reference_pipeline(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """完整串行参考实现：O = X@C + (X@A)@B。"""
    y = compute_lora_down(x, a)
    z = compute_lora_expand(y, b)
    w = compute_main_matmul(x, c)
    return compute_output_add(w, z)


def create_problem_tensors(
    shape: ProblemShape,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> ProblemTensors:
    """按统一形状生成实验输入矩阵。"""
    torch.manual_seed(seed)
    return ProblemTensors(
        x=torch.randn((shape.m, shape.h), device=device, dtype=dtype),
        a=torch.randn((shape.h, shape.r), device=device, dtype=dtype),
        b=torch.randn((shape.r, shape.n), device=device, dtype=dtype),
        c=torch.randn((shape.h, shape.n), device=device, dtype=dtype),
    )
