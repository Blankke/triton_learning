"""
sharing 实验：按连续 pid range 静态分配 worker 的单 kernel 融合方案。

使用示例：
    from sharing.range_fusion import build_half_split_schedule, triton_range_fused_down_main
    schedule = build_half_split_schedule(m=64, r=2048, n=28672, device=x.device)
    y, w = triton_range_fused_down_main(x, a, c, schedule)

说明：
    这里保留 `pid range` 的实验口径，但 range 的长度不再按“worker 均分”定义，
    而是严格按两个算子的 tile 数来切：
        - 实验1：
            if pid < num_tiles_1:
                do op1
            else:
                do op3
        - 实验2：
            先把 op1 tiles 与 op3 tiles 各自切成两段，再做
            op1 / op3 / op1 / op3 的 range 交错排列。
    因此这里默认会 launch `num_tiles_1 + num_tiles_3` 个 program，
    让 `pid` 本身就对应老师讨论的那种连续 range 语义。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fusion.scheme2_horizontal_fusion import device_num_sms
from triton_learning.kernels.matmul import matmul_autotune_configs


DEFAULT_CONSTRUCTED_R = 2048
REFERENCE_BLOCK_M = 16
REFERENCE_BLOCK_N = 128
MAX_SUPPORTED_RANGES = 4
OP1_NAME = "op1"
OP3_NAME = "op3"
_OP1_KIND = 1
_OP3_KIND = 3


@dataclass(frozen=True)
class RangeSchedule:
    """描述一次 sharing 实验的静态 pid range 分配结果。"""

    name: str
    num_sms: int
    launched_workers: int
    op1_tiles: int
    op3_tiles: int
    op1_workers: int
    op3_workers: int
    range_ops: tuple[str, ...]
    range_sizes: tuple[int, ...]
    op_worker_bases: tuple[int, ...]


@dataclass(frozen=True)
class RangeSegment:
    """描述一段连续 pid range 的显式边界与含义。"""

    range_index: int
    op_name: str
    pid_start: int
    pid_end: int
    op_tile_start: int
    op_tile_end: int
    range_size: int
    op_worker_base: int


def sharing_tile_count(m: int, n: int) -> int:
    """按 sharing 实验统一采用的 16x128 参考块大小估算 tile 数。"""
    return triton.cdiv(m, REFERENCE_BLOCK_M) * triton.cdiv(n, REFERENCE_BLOCK_N)


def sharing_tile_counts(m: int, r: int, n: int) -> tuple[int, int]:
    """返回 op1(X@A) 与 op3(X@C) 的参考 tile 数。"""
    return sharing_tile_count(m, r), sharing_tile_count(m, n)


def _check_inputs(x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """检查输入，并统一转为 contiguous。"""
    if x.ndim != 2 or a.ndim != 2 or c.ndim != 2:
        raise ValueError("sharing fused kernel 只支持二维矩阵。")
    if x.shape[1] != a.shape[0] or x.shape[1] != c.shape[0]:
        raise ValueError(f"矩阵形状不匹配：x={tuple(x.shape)}, a={tuple(a.shape)}, c={tuple(c.shape)}")
    if not (x.is_cuda and a.is_cuda and c.is_cuda):
        raise ValueError("sharing fused kernel 需要 CUDA tensor。")
    if len({x.dtype, a.dtype, c.dtype}) != 1:
        raise ValueError("x、a、c 的 dtype 必须一致。")
    return x.contiguous(), a.contiguous(), c.contiguous()


def _split_evenly(total: int, parts: int) -> tuple[int, ...]:
    """把总 worker 数尽量平均切成若干连续 range。"""
    if parts <= 0:
        raise ValueError("parts 必须大于 0。")
    base = total // parts
    remainder = total % parts
    return tuple(base + (1 if idx < remainder else 0) for idx in range(parts))


def _resolve_launch_workers(m: int, r: int, n: int, device: torch.device, num_workers: int) -> tuple[int, int, int, int]:
    """
    推导 sharing 实验真正 launch 的 program 数。

    sharing 实验的语义是“按 tile 数切 pid range”，所以默认必须 launch 全部 tiles。
    如果显式传入 `num_workers`，也要求它与 `total_tiles` 完全一致，避免语义跑偏。
    """
    if num_workers < 0:
        raise ValueError("num_workers 不能为负数。")
    num_sms = device_num_sms(device)
    op1_tiles, op3_tiles = sharing_tile_counts(m, r, n)
    total_tiles = op1_tiles + op3_tiles
    launched_workers = total_tiles if num_workers == 0 else num_workers
    if launched_workers != total_tiles:
        raise ValueError(
            "sharing 实验要求 launched workers 与 total_tiles 完全一致，"
            f"当前 total_tiles={total_tiles}, num_workers={launched_workers}。"
        )
    return num_sms, launched_workers, op1_tiles, op3_tiles


def _validate_range_ops(range_ops: Sequence[str]) -> tuple[str, ...]:
    """规范化 range 的算子标签。"""
    normalized_ops: list[str] = []
    for op_name in range_ops:
        if op_name not in (OP1_NAME, OP3_NAME):
            raise ValueError(f"不支持的 range op：{op_name}")
        normalized_ops.append(op_name)
    if not normalized_ops:
        raise ValueError("至少需要一个 pid range。")
    if len(normalized_ops) > MAX_SUPPORTED_RANGES:
        raise ValueError(f"最多只支持 {MAX_SUPPORTED_RANGES} 个 pid range。")
    return tuple(normalized_ops)


def _compute_op_worker_bases(range_ops: Sequence[str], range_sizes: Sequence[int]) -> tuple[tuple[int, ...], int, int]:
    """
    计算每个 range 在自己所属算子 worker 池里的起始偏移。

    例如实验2中：
        range1(op1), range2(op3), range3(op1), range4(op3)
    那么 range3 的 `op_worker_base` 就等于 range1 的 worker 数，
    这样两个 op1 range 才会共同覆盖 op1 的 tile 空间，而不是重复计算。
    """
    op1_workers = 0
    op3_workers = 0
    bases: list[int] = []
    for op_name, range_size in zip(range_ops, range_sizes, strict=True):
        if range_size < 0:
            raise ValueError("range_size 不能为负数。")
        if op_name == OP1_NAME:
            bases.append(op1_workers)
            op1_workers += range_size
        else:
            bases.append(op3_workers)
            op3_workers += range_size
    return tuple(bases), op1_workers, op3_workers


def make_range_schedule(
    name: str,
    m: int,
    r: int,
    n: int,
    device: torch.device,
    *,
    range_ops: Sequence[str],
    range_sizes: Sequence[int] | None = None,
    num_workers: int = 0,
) -> RangeSchedule:
    """按实验定义构造静态 worker range schedule。"""
    normalized_ops = _validate_range_ops(range_ops)
    num_sms, launched_workers, op1_tiles, op3_tiles = _resolve_launch_workers(m, r, n, device, num_workers)
    resolved_sizes = tuple(range_sizes) if range_sizes is not None else _split_evenly(launched_workers, len(normalized_ops))
    if len(resolved_sizes) != len(normalized_ops):
        raise ValueError("range_ops 与 range_sizes 的长度必须一致。")
    if sum(resolved_sizes) != launched_workers:
        raise ValueError(f"range_sizes 之和必须等于 launched_workers={launched_workers}。")

    op_worker_bases, op1_workers, op3_workers = _compute_op_worker_bases(normalized_ops, resolved_sizes)
    if op1_workers <= 0 or op3_workers <= 0:
        raise ValueError("sharing 实验要求 op1 与 op3 都至少分到一个 worker。")

    return RangeSchedule(
        name=name,
        num_sms=num_sms,
        launched_workers=launched_workers,
        op1_tiles=op1_tiles,
        op3_tiles=op3_tiles,
        op1_workers=op1_workers,
        op3_workers=op3_workers,
        range_ops=normalized_ops,
        range_sizes=resolved_sizes,
        op_worker_bases=op_worker_bases,
    )


def build_half_split_schedule(
    m: int,
    r: int,
    n: int,
    device: torch.device,
    *,
    num_workers: int = 0,
) -> RangeSchedule:
    """实验1：按 tile 数切 range，前一段 pid 做 op1，后一段 pid 做 op3。"""
    _, _, op1_tiles, op3_tiles = _resolve_launch_workers(m, r, n, device, num_workers)
    return make_range_schedule(
        "half_split",
        m,
        r,
        n,
        device,
        range_ops=(OP1_NAME, OP3_NAME),
        range_sizes=(op1_tiles, op3_tiles),
        num_workers=num_workers,
    )


def build_interleaved_schedule(
    m: int,
    r: int,
    n: int,
    device: torch.device,
    *,
    num_workers: int = 0,
) -> RangeSchedule:
    """实验2：按 tile 数比例切四段 range，交错做 op1 / op3 / op1 / op3。"""
    _, _, op1_tiles, op3_tiles = _resolve_launch_workers(m, r, n, device, num_workers)
    op1_split = _split_evenly(op1_tiles, 2)
    op3_split = _split_evenly(op3_tiles, 2)
    return make_range_schedule(
        "interleaved_quarters",
        m,
        r,
        n,
        device,
        range_ops=(OP1_NAME, OP3_NAME, OP1_NAME, OP3_NAME),
        range_sizes=(op1_split[0], op3_split[0], op1_split[1], op3_split[1]),
        num_workers=num_workers,
    )


def format_schedule(schedule: RangeSchedule) -> str:
    """把 schedule 格式化成便于日志打印的摘要。"""
    parts = [
        f"{op_name}:{range_size}(base={base})"
        for op_name, range_size, base in zip(
            schedule.range_ops,
            schedule.range_sizes,
            schedule.op_worker_bases,
            strict=True,
        )
    ]
    return " | ".join(parts)


def schedule_segments(schedule: RangeSchedule) -> tuple[RangeSegment, ...]:
    """把 schedule 展开成带显式 pid 边界的连续 range 描述。"""
    segments: list[RangeSegment] = []
    pid_cursor = 0
    for range_index, (op_name, range_size, op_worker_base) in enumerate(
        zip(schedule.range_ops, schedule.range_sizes, schedule.op_worker_bases, strict=True),
        start=1,
    ):
        pid_start = pid_cursor
        pid_end = pid_start + range_size
        op_tile_start = op_worker_base
        op_tile_end = op_worker_base + range_size
        segments.append(
            RangeSegment(
                range_index=range_index,
                op_name=op_name,
                pid_start=pid_start,
                pid_end=pid_end,
                op_tile_start=op_tile_start,
                op_tile_end=op_tile_end,
                range_size=range_size,
                op_worker_base=op_worker_base,
            )
        )
        pid_cursor = pid_end
    return tuple(segments)


def schedule_logic_lines(schedule: RangeSchedule) -> tuple[str, ...]:
    """生成更接近老师表述方式的伪代码日志。"""
    lines: list[str] = []
    segments = schedule_segments(schedule)
    for index, segment in enumerate(segments):
        condition = f"pid < {segment.pid_end}" if index == 0 else f"pid < {segment.pid_end}"
        prefix = "if" if index == 0 else "elif"
        lines.append(f"{prefix} {condition}: do {segment.op_name}")
    lines.append("else: unreachable")
    return tuple(lines)


@triton.jit
def _grouped_pid_from_linear_tile(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_SIZE_M: tl.constexpr,
):
    """把线性 tile id 映射回 Step 2 风格的 grouped ordering 坐标。"""
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _compute_gemm_tile(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    M,
    K,
    N_OUT,
    stride_lm,
    stride_lk,
    stride_rk,
    stride_rn,
    stride_om,
    stride_on,
    pid_m,
    pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """复用统一 GEMM 微内核，计算一个输出 tile。"""
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    lhs_ptrs = lhs_ptr + offs_m[:, None] * stride_lm + offs_k[None, :] * stride_lk
    rhs_ptrs = rhs_ptr + offs_k[:, None] * stride_rk + offs_n[None, :] * stride_rn
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        lhs_tile = tl.load(
            lhs_ptrs,
            mask=(offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K),
            other=0.0,
        )
        rhs_tile = tl.load(
            rhs_ptrs,
            mask=((k_start + offs_k[:, None]) < K) & (offs_n[None, :] < N_OUT),
            other=0.0,
        )
        acc += tl.dot(lhs_tile, rhs_tile)
        lhs_ptrs += BLOCK_SIZE_K * stride_lk
        rhs_ptrs += BLOCK_SIZE_K * stride_rk

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N_OUT))


@triton.autotune(
    configs=matmul_autotune_configs(),
    key=["M", "H", "R", "N", "NUM_RANGES"],
)
@triton.jit
def _range_fused_kernel(
    x_ptr,
    a_ptr,
    c_ptr,
    y_ptr,
    w_ptr,
    M,
    H,
    R,
    N,
    stride_xm,
    stride_xh,
    stride_ah,
    stride_ar,
    stride_ch,
    stride_cn,
    stride_ym,
    stride_yr,
    stride_wm,
    stride_wn,
    RANGE0_KIND,
    RANGE0_SIZE,
    RANGE0_BASE,
    RANGE1_KIND,
    RANGE1_SIZE,
    RANGE1_BASE,
    RANGE2_KIND,
    RANGE2_SIZE,
    RANGE2_BASE,
    RANGE3_KIND,
    RANGE3_SIZE,
    RANGE3_BASE,
    TOTAL_OP1_WORKERS,
    TOTAL_OP3_WORKERS,
    NUM_RANGES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    静态 range-scheduling fused kernel。

    每个 worker 先根据自己的 pid 落在哪个 range 来决定归属的算子，
    再以“同类 worker 总数”为 stride，扫描该算子的所有 tile。
    """
    worker_id = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n_op1 = tl.cdiv(R, BLOCK_SIZE_N)
    num_pid_n_op3 = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles_op1 = num_pid_m * num_pid_n_op1
    num_tiles_op3 = num_pid_m * num_pid_n_op3

    range0_end = RANGE0_SIZE
    range1_end = range0_end + RANGE1_SIZE
    range2_end = range1_end + RANGE2_SIZE

    selected_kind = RANGE0_KIND
    selected_local_rank = worker_id
    selected_base = RANGE0_BASE

    if NUM_RANGES > 1:
        if worker_id >= range0_end:
            selected_kind = RANGE1_KIND
            selected_local_rank = worker_id - range0_end
            selected_base = RANGE1_BASE
    if NUM_RANGES > 2:
        if worker_id >= range1_end:
            selected_kind = RANGE2_KIND
            selected_local_rank = worker_id - range1_end
            selected_base = RANGE2_BASE
    if NUM_RANGES > 3:
        if worker_id >= range2_end:
            selected_kind = RANGE3_KIND
            selected_local_rank = worker_id - range2_end
            selected_base = RANGE3_BASE

    if selected_kind == _OP1_KIND:
        op1_worker_rank = selected_base + selected_local_rank
        tile_id = op1_worker_rank
        while tile_id < num_tiles_op1:
            pid_m_op1, pid_n_op1 = _grouped_pid_from_linear_tile(tile_id, num_pid_m, num_pid_n_op1, GROUP_SIZE_M)
            _compute_gemm_tile(
                x_ptr,
                a_ptr,
                y_ptr,
                M,
                H,
                R,
                stride_xm,
                stride_xh,
                stride_ah,
                stride_ar,
                stride_ym,
                stride_yr,
                pid_m_op1,
                pid_n_op1,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
            )
            tile_id += TOTAL_OP1_WORKERS
        return

    op3_worker_rank = selected_base + selected_local_rank
    tile_id = op3_worker_rank
    while tile_id < num_tiles_op3:
        pid_m_op3, pid_n_op3 = _grouped_pid_from_linear_tile(tile_id, num_pid_m, num_pid_n_op3, GROUP_SIZE_M)
        _compute_gemm_tile(
            x_ptr,
            c_ptr,
            w_ptr,
            M,
            H,
            N,
            stride_xm,
            stride_xh,
            stride_ch,
            stride_cn,
            stride_wm,
            stride_wn,
            pid_m_op3,
            pid_n_op3,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )
        tile_id += TOTAL_OP3_WORKERS


def triton_range_fused_down_main(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    schedule: RangeSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    """启动 sharing 单 kernel：按给定 pid range schedule 同时计算 Y=X@A 与 W=X@C。"""
    x, a, c = _check_inputs(x, a, c)
    if len(schedule.range_ops) not in (2, 4):
        raise ValueError("当前 sharing fused kernel 只支持 2 段或 4 段 range。")

    m, h = x.shape
    _, r = a.shape
    _, n = c.shape
    y = torch.empty((m, r), device=x.device, dtype=x.dtype)
    w = torch.empty((m, n), device=x.device, dtype=x.dtype)

    padded_ops = list(schedule.range_ops) + [OP3_NAME] * (MAX_SUPPORTED_RANGES - len(schedule.range_ops))
    padded_sizes = list(schedule.range_sizes) + [0] * (MAX_SUPPORTED_RANGES - len(schedule.range_sizes))
    padded_bases = list(schedule.op_worker_bases) + [0] * (MAX_SUPPORTED_RANGES - len(schedule.op_worker_bases))
    padded_kinds = [_OP1_KIND if op_name == OP1_NAME else _OP3_KIND for op_name in padded_ops]

    grid = (schedule.launched_workers,)
    _range_fused_kernel[grid](
        x,
        a,
        c,
        y,
        w,
        m,
        h,
        r,
        n,
        x.stride(0),
        x.stride(1),
        a.stride(0),
        a.stride(1),
        c.stride(0),
        c.stride(1),
        y.stride(0),
        y.stride(1),
        w.stride(0),
        w.stride(1),
        padded_kinds[0],
        padded_sizes[0],
        padded_bases[0],
        padded_kinds[1],
        padded_sizes[1],
        padded_bases[1],
        padded_kinds[2],
        padded_sizes[2],
        padded_bases[2],
        padded_kinds[3],
        padded_sizes[3],
        padded_bases[3],
        schedule.op1_workers,
        schedule.op3_workers,
        NUM_RANGES=len(schedule.range_ops),
    )
    return y, w
