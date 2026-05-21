"""
sharing 实验包。

使用示例：
    python -m sharing.bench_experiment1_half_split
    python -m sharing.bench_experiment2_interleaved

说明：
    本目录专门放“按 pid range 分配 worker”的构造性实验。
    默认不再使用真实 workload 的 `r=8`，而是直接把 `r` 提升到 `2048`，
    用来验证 pid range 排列是否会影响 block-to-SM 映射与并发效果。
"""

