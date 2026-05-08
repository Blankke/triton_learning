"""
问题1的统一形状与 dtype 配置。

使用示例：
    from triton_learning.problem_spec import DEFAULT_GATEUP_PROBLEM, resolve_dtype
    dtype = resolve_dtype(DEFAULT_GATEUP_PROBLEM.dtype_name)

说明：
    本模块只收录 `问题1.md` 与 `docs/` 中已经明确的 baseline 场景：
    gateup_proj, batch size=64, hidden_size=4096, output_size=28672, rank=8。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProblemShape:
    """统一描述问题1里一组矩阵乘实验的形状。"""

    m: int = 64
    h: int = 4096
    n: int = 28672
    r: int = 8
    dtype_name: str = "fp16"


DEFAULT_GATEUP_PROBLEM = ProblemShape()


def resolve_dtype(name: str) -> torch.dtype:
    """把命令行里的字符串 dtype 转为 PyTorch dtype。"""
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"未知 dtype：{name}")
