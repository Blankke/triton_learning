# Triton Learning

本项目用于完成「问题1：Triton 矩阵乘算子融合」的学习、复现与实验整理。

## 目录说明

- `src/triton_learning`：Step 1 / Step 2 / Step 3 的基础学习与单 LoRA 融合实验
- `src/fusion`：三种“算子1 + 算子3”融合方案
- `src/sharing`：sharing 方向的两组调度实验
- `scripts/`：统一的可复现运行脚本
- `问题1.md`：题目约束、官方源码入口与参考资源

## 实现细节

### Step 1：baseline

- 入口：`python -m triton_learning.bench_step1_baseline`
- 实现：按 `Y = X @ A`、`Z = Y @ B`、`W = X @ C`、`O = W + Z` 的串行流程测量各子算子与全流程耗时
- 输出：`output/triton_learning/benchmarks/step1_baseline.csv`

### Step 2：Triton 主干 GEMM

- 入口：`python -m triton_learning.bench_step2_triton_matmul`
- 实现：复用 Triton GEMM 教程思路，只关注 `W = X @ C`，与 PyTorch/cuBLAS 对比性能和误差
- 输出：`output/triton_learning/benchmarks/step2_triton_vs_cublas.csv`

### Step 3：单 LoRA fused expand

- 入口：`python -m triton_learning.bench_step3_fused_expand`
- 实现：先算 `Y = X @ A`，再用单个 Triton kernel 融合 `X @ C` 与 `Y @ B`，验证单 adapter 场景的全流程融合可行性
- 输出：`output/triton_learning/benchmarks/step3_fused_expand.csv`

### 方案1：spatial sharing

- 入口：`python -m fusion.bench_scheme1_spatial_sharing`
- 实现：复用两次 Triton GEMM，在两个 CUDA stream 中并发 launch，作为空分复用近似
- 输出：`output/fusion/benchmarks/fusion_scheme1_spatial_sharing.csv`

### 方案2：horizontal fusion

- 入口：`python -m fusion.bench_scheme2_horizontal_fusion`
- 实现：将 down/main tiles 放进单次 launch，提供 `static_pid` 与 `grouped_persistent` 两个官方风格调度口径
- 输出：`output/fusion/benchmarks/fusion_scheme2_horizontal_fusion.csv`

### 方案3：column concat

- 入口：`python -m fusion.bench_scheme3_column_concat`
- 实现：将 `[A, C]` 做物理或逻辑拼接，复用宽矩阵 GEMM 完成 `Y/W` pair
- 输出：`output/fusion/benchmarks/fusion_scheme3_column_concat.csv`

### sharing 实验

- 主入口：`python -m sharing.bench_five_way_comparison`
- 实现：把真正不同的五种方法放到同一张表里比较
- 五种口径：
  - `baseline`
  - `stream_overlap`
  - `single_fused_half_split`
  - `single_fused_interleaved`
  - `physical_concat`
- 两种 fused 调度分别对应：
  - `single_fused_half_split`：前半 pid range 做 op1，后半 pid range 做 op3
  - `single_fused_interleaved`：四段 pid range 交错做 op1 / op3 / op1 / op3
- benchmark 输出：
  - `output/sharing/benchmarks/sharing_five_way_comparison.csv`
- profiling 输出：
  - `output/sharing/nsys/*.nsys-rep`
  - `output/sharing/ncu/*.ncu-rep`

## 参考学习资料

- 题目约束、官方源码链接与参考资源统一见 `问题1.md`
- `docs/` 中保留本项目的学习笔记与图示

## 复现步骤

先激活项目虚拟环境：

```bash
source scripts/activate_local_venv.sh
which python
python -c 'import sys; print(sys.executable)'
```

### 基础学习链路

```bash
bash scripts/run_step1_baseline.sh
bash scripts/run_step2_triton_matmul.sh
bash scripts/run_step3_fused_expand.sh
```

### 三种融合方案 benchmark

```bash
bash scripts/run_scheme1_spatial_sharing.sh
bash scripts/run_scheme2_horizontal_fusion.sh
bash scripts/run_scheme3_column_concat.sh
```

或直接一键复现：

```bash
bash scripts/reproduce_all_fusion_benchmarks.sh
```

### fusion profiling

```bash
bash scripts/profile_suite.sh
TOOL=nsys bash scripts/profile_suite.sh scheme1
TOOL=ncu VARIANT=physical_precat bash scripts/profile_suite.sh scheme3
```

profiling 输出目录：

- `output/fusion/nsys/`
- `output/fusion/ncu/`

### sharing benchmark + profiling

```bash
bash scripts/run_sharing_benchmark.sh
bash scripts/run_sharing_all.sh
RUN_PROFILE=0 bash scripts/run_sharing_all.sh
TOOL=nsys bash scripts/run_sharing_all.sh single_fused_half_split
TOOL=ncu bash scripts/run_sharing_all.sh physical_concat
```

如只想单独跑 sharing profiling，也可以直接执行：

```bash
bash scripts/profile_sharing.sh
TOOL=nsys bash scripts/profile_sharing.sh baseline
TOOL=ncu bash scripts/profile_sharing.sh single_fused_interleaved
```

## 输出目录规范

运行完成后，结果会按源码域分别落盘：

- `output/triton_learning/benchmarks/`
- `output/fusion/benchmarks/`
- `output/fusion/nsys/`
- `output/fusion/ncu/`
- `output/sharing/benchmarks/`
- `output/sharing/nsys/`
- `output/sharing/ncu/`

历史 `outputs/` 目录中的旧结果暂时保留，不作为新的默认输出位置。
