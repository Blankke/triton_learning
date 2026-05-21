"""
sharing 实验1：前半 worker 做 op1，后半 worker 做 op3。

使用示例：
    python -m sharing.bench_experiment1_half_split
    python -m sharing.bench_experiment1_half_split --num-workers 108

说明：
    本实验默认直接使用构造 workload：
        M=64, H=4096, N=28672, r=2048
    对比方法固定为：
        1. baseline：op1 单独 kernel + op3 单独 kernel
        2. stream overlap：op1 / op3 放在不同 CUDA stream 中并发 launch
        3. single fused：前半 worker 做 op1，后半 worker 做 op3
        4. physical concat：X @ [A, C]
"""

from __future__ import annotations

from sharing.benchmarks import SharingExperimentSpec, run_sharing_benchmark
from sharing.range_fusion import build_half_split_schedule


def main() -> None:
    """执行 sharing 实验1 benchmark。"""
    run_sharing_benchmark(
        SharingExperimentSpec(
            experiment_name="sharing_experiment1_half_split",
            description="sharing 实验1：前半 worker 做 op1，后半 worker 做 op3。",
            default_output="output/sharing/benchmarks/experiment1_half_split.csv",
            schedule_builder=lambda m, r, n, device, num_workers: build_half_split_schedule(
                m,
                r,
                n,
                device,
                num_workers=num_workers,
            ),
            profile_title_prefix="sharing/experiment1",
        )
    )


if __name__ == "__main__":
    main()

