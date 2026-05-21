"""
sharing 实验2：四段 pid range 交错做 op1 / op3 / op1 / op3。

使用示例：
    python -m sharing.bench_experiment2_interleaved
    python -m sharing.bench_experiment2_interleaved --num-workers 108

说明：
    本实验默认直接使用构造 workload：
        M=64, H=4096, N=28672, r=2048
    目的不是验证真实参数下谁最快，而是验证：
        不同 pid range 的连续排列方式，是否会影响第一波 block 的 SM 映射和并发效果。
"""

from __future__ import annotations

from sharing.benchmarks import SharingExperimentSpec, run_sharing_benchmark
from sharing.range_fusion import build_interleaved_schedule


def main() -> None:
    """执行 sharing 实验2 benchmark。"""
    run_sharing_benchmark(
        SharingExperimentSpec(
            experiment_name="sharing_experiment2_interleaved",
            description="sharing 实验2：四段 pid range 交错做 op1 / op3 / op1 / op3。",
            default_output="output/sharing/benchmarks/experiment2_interleaved.csv",
            schedule_builder=lambda m, r, n, device, num_workers: build_interleaved_schedule(
                m,
                r,
                n,
                device,
                num_workers=num_workers,
            ),
            profile_title_prefix="sharing/experiment2",
        )
    )


if __name__ == "__main__":
    main()

