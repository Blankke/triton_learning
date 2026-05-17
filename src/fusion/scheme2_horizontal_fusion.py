"""
方案2：single-kernel horizontal fusion。

使用示例：
    from fusion.scheme2_horizontal_fusion import triton_horizontal_fused_down_main
    y, w = triton_horizontal_fused_down_main(x, a, c, variant="static_pid")
    y, w = triton_horizontal_fused_down_main(x, a, c, variant="grouped_persistent")

说明：
    本文件同时保留两个单 kernel 口径：
        1. static_pid：
           直接按 pid 区间分工：
               pid < num_tiles_down      -> 计算 Y = X @ A 的一个 tile
               pid >= num_tiles_down     -> 计算 W = X @ C 的一个 tile
        2. grouped_persistent：
           用 persistent workers 反复处理多个 grouped tiles：
               - 前若干 worker 先做 down tiles
               - down 做完后，这些 worker 回流到自己预留的 main chunk 条带
               - main tiles 继续沿用 Step 2 的 grouped ordering 骨架
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from triton_learning.kernels.matmul import matmul_autotune_configs


STATIC_PID_VARIANT = "static_pid"
GROUPED_PERSISTENT_VARIANT = "grouped_persistent"
DEFAULT_PERSISTENT_CHUNK_SIZE = 4

REFERENCE_BLOCK_M_DOWN = 16
REFERENCE_BLOCK_N_DOWN = 16
REFERENCE_BLOCK_M_MAIN = 16
REFERENCE_BLOCK_N_MAIN = 128


@dataclass(frozen=True)
class GroupedPersistentScheduler:
    """保存 grouped persistent 版本的调度超参数与参考 worker 划分。"""

    num_sms: int
    launched_workers: int
    requested_num_down_workers: int
    effective_num_down_workers: int
    chunk_size: int
    num_main_chunks: int


def _device_index(device: torch.device) -> int:
    """把 torch.device 归一化为可查询属性的 CUDA device index。"""
    if device.type != "cuda":
        raise ValueError("方案2只支持 CUDA device。")
    return torch.cuda.current_device() if device.index is None else device.index


def device_num_sms(device: torch.device) -> int:
    """返回当前 CUDA device 的 SM 数。"""
    return torch.cuda.get_device_properties(_device_index(device)).multi_processor_count


def down_tile_count(m: int, r: int) -> int:
    """按题目分析时采用的参考块大小估算算子1 program 数。"""
    return triton.cdiv(m, REFERENCE_BLOCK_M_DOWN) * triton.cdiv(r, REFERENCE_BLOCK_N_DOWN)


def main_tile_count(m: int, n: int) -> int:
    """按题目分析时采用的参考块大小估算算子3 program 数。"""
    return triton.cdiv(m, REFERENCE_BLOCK_M_MAIN) * triton.cdiv(n, REFERENCE_BLOCK_N_MAIN)


def resolve_grouped_persistent_scheduler(
    m: int,
    r: int,
    n: int,
    device: torch.device,
    *,
    num_down_workers: int = 0,
    chunk_size: int = DEFAULT_PERSISTENT_CHUNK_SIZE,
) -> GroupedPersistentScheduler:
    """
    推导 grouped persistent 版本的调度超参数。

    约定：
        - `num_down_workers=0` 表示自动推导
        - `chunk_size` 是每个 worker 每轮处理的连续 main grouped tiles 数
    """
    if num_down_workers < 0:
        raise ValueError("num_down_workers 不能为负数。")
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")

    num_sms = device_num_sms(device)
    tiles_down = down_tile_count(m, r)
    tiles_main = main_tile_count(m, n)
    total_tiles = tiles_down + tiles_main
    launched_workers = max(1, min(num_sms, total_tiles))
    effective_num_down_workers = min(
        tiles_down,
        launched_workers,
        tiles_down if num_down_workers == 0 else num_down_workers,
    )
    num_main_chunks = triton.cdiv(tiles_main, chunk_size)
    return GroupedPersistentScheduler(
        num_sms=num_sms,
        launched_workers=launched_workers,
        requested_num_down_workers=num_down_workers,
        effective_num_down_workers=effective_num_down_workers,
        chunk_size=chunk_size,
        num_main_chunks=num_main_chunks,
    )


@triton.jit
def _grouped_pid_from_linear_tile(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_SIZE_M: tl.constexpr,
):
    """把线性 tile_id 映射回 Step 2 风格的 grouped ordering 坐标。"""
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
    key=["M", "H", "R", "N"],
)
@triton.jit
def _horizontal_fused_static_pid_kernel(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """静态 pid 分区版：一个 program 只负责一个 down tile 或 main tile。"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    num_pid_n_down = tl.cdiv(R, BLOCK_SIZE_N)
    num_tiles_down = num_pid_m * num_pid_n_down
    if pid < num_tiles_down:
        pid_m_down, pid_n_down = _grouped_pid_from_linear_tile(pid, num_pid_m, num_pid_n_down, GROUP_SIZE_M)
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
            pid_m_down,
            pid_n_down,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )
        return

    pid_main = pid - num_tiles_down
    num_pid_n_main = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m_main, pid_n_main = _grouped_pid_from_linear_tile(pid_main, num_pid_m, num_pid_n_main, GROUP_SIZE_M)
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
        pid_m_main,
        pid_n_main,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )


@triton.autotune(
    configs=matmul_autotune_configs(),
    key=["M", "H", "R", "N"],
)
@triton.jit
def _horizontal_fused_grouped_persistent_kernel(
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
    NUM_DOWN_WORKERS,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    grouped persistent 版：
        - worker_id < NUM_DOWN_WORKERS 的 worker 先处理 down tiles
        - down 做完后，这些 worker 回流到自己预留的 main chunk 条带
        - main 侧继续使用 Step 2 的 grouped ordering 映射
    """
    worker_id = tl.program_id(axis=0)
    num_workers = tl.num_programs(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n_down = tl.cdiv(R, BLOCK_SIZE_N)
    num_pid_n_main = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles_down = num_pid_m * num_pid_n_down
    num_tiles_main = num_pid_m * num_pid_n_main
    num_main_chunks = tl.cdiv(num_tiles_main, CHUNK_SIZE)
    effective_num_down_workers = tl.minimum(NUM_DOWN_WORKERS, num_workers)
    effective_num_down_workers = tl.minimum(effective_num_down_workers, num_tiles_down)

    # 前若干 worker 先做 skinny 的 down tiles；做完后不退出，而是继续回流到 main。
    if worker_id < effective_num_down_workers:
        down_tile_id = worker_id
        while down_tile_id < num_tiles_down:
            pid_m_down, pid_n_down = _grouped_pid_from_linear_tile(
                down_tile_id,
                num_pid_m,
                num_pid_n_down,
                GROUP_SIZE_M,
            )
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
                pid_m_down,
                pid_n_down,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
            )
            down_tile_id += effective_num_down_workers

    # main 侧按预留 chunk 条带做确定性分工，避免引入 atomic work stealing。
    main_chunk_id = worker_id
    while main_chunk_id < num_main_chunks:
        main_tile_start = main_chunk_id * CHUNK_SIZE
        for chunk_offset in range(CHUNK_SIZE):
            main_tile_id = main_tile_start + chunk_offset
            if main_tile_id < num_tiles_main:
                pid_m_main, pid_n_main = _grouped_pid_from_linear_tile(
                    main_tile_id,
                    num_pid_m,
                    num_pid_n_main,
                    GROUP_SIZE_M,
                )
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
                    pid_m_main,
                    pid_n_main,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                )
        main_chunk_id += num_workers


def _check_inputs(x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """检查输入，并统一转为 contiguous。"""
    if x.ndim != 2 or a.ndim != 2 or c.ndim != 2:
        raise ValueError("方案2只支持二维矩阵。")
    if x.shape[1] != a.shape[0] or x.shape[1] != c.shape[0]:
        raise ValueError(f"矩阵形状不匹配：x={tuple(x.shape)}, a={tuple(a.shape)}, c={tuple(c.shape)}")
    if not (x.is_cuda and a.is_cuda and c.is_cuda):
        raise ValueError("方案2需要 CUDA tensor。")
    if len({x.dtype, a.dtype, c.dtype}) != 1:
        raise ValueError("x、a、c 的 dtype 必须一致。")
    return x.contiguous(), a.contiguous(), c.contiguous()


def triton_horizontal_fused_down_main_static(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """方案2静态版：一个 Triton kernel 通过 pid 区间同时产生 Y 和 W。"""
    x, a, c = _check_inputs(x, a, c)
    m, h = x.shape
    _, r = a.shape
    _, n = c.shape
    y = torch.empty((m, r), device=x.device, dtype=x.dtype)
    w = torch.empty((m, n), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(r, meta["BLOCK_SIZE_N"])
        + triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )

    _horizontal_fused_static_pid_kernel[grid](
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
    )
    return y, w


def triton_horizontal_fused_down_main_grouped_persistent(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    *,
    num_down_workers: int = 0,
    chunk_size: int = DEFAULT_PERSISTENT_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """方案2 persistent 版：grouped ordering + worker 条带回流。"""
    x, a, c = _check_inputs(x, a, c)
    m, h = x.shape
    _, r = a.shape
    _, n = c.shape
    schedule = resolve_grouped_persistent_scheduler(
        m,
        r,
        n,
        x.device,
        num_down_workers=num_down_workers,
        chunk_size=chunk_size,
    )
    y = torch.empty((m, r), device=x.device, dtype=x.dtype)
    w = torch.empty((m, n), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        min(
            schedule.num_sms,
            triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(r, meta["BLOCK_SIZE_N"])
            + triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        ),
    )

    _horizontal_fused_grouped_persistent_kernel[grid](
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
        schedule.effective_num_down_workers,
        CHUNK_SIZE=schedule.chunk_size,
    )
    return y, w


def triton_horizontal_fused_down_main(
    x: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    *,
    variant: str = STATIC_PID_VARIANT,
    num_down_workers: int = 0,
    chunk_size: int = DEFAULT_PERSISTENT_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    统一的方案2 wrapper。

    参数：
        variant:
            - `static_pid`
            - `grouped_persistent`
    """
    if variant == STATIC_PID_VARIANT:
        return triton_horizontal_fused_down_main_static(x, a, c)
    if variant == GROUPED_PERSISTENT_VARIANT:
        return triton_horizontal_fused_down_main_grouped_persistent(
            x,
            a,
            c,
            num_down_workers=num_down_workers,
            chunk_size=chunk_size,
        )
    raise ValueError(f"不支持的方案2 variant: {variant}")
