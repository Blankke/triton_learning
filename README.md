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
- stream overlap 修复：
  - 双 stream launch 现在统一走 `fusion.scheme1_spatial_sharing.launch_triton_matmul_pair`
  - 并发路径改成“显式 CUDA event 边界 + 持久化 stream/event 复用”，避免 profiling 专用路径和 benchmark 路径分叉
  - 这样在远端 A100 上用 `nsys` 看 `stream_overlap` 时，更容易直接看到两个 kernel 的真实并发关系
- profiling 修复：
  - sharing 统一改为新的场景脚本 `scripts/run_sharing_profile_cases.sh`
  - `ncu` 默认使用 `--set full`，并额外导出 `raw` / `details` 页面，补齐之前缺失的 detail 页面指标
  - `nsys` 输出按场景拆到不同目录，真实场景和构造场景不会再混在一起
  - 默认入口现在会完整执行“两场景 x 五方法 x nsys+ncu”，不再只抓单一方法
  - `GPU_METRICS_DEVICE=all` 会正确映射到 `nsys --gpu-metrics-devices=all`
- 两种 fused 调度分别对应：
  - `single_fused_half_split`：前半 pid range 做 op1，后半 pid range 做 op3
  - `single_fused_interleaved`：四段 pid range 交错做 op1 / op3 / op1 / op3
- 场景预设：
  - `real_r8`：`M=64, H=4096, N=28672, R=8`
  - `constructed_50_blocks`：`M=64, H=4096, N=1664, R=1664`，按当前 sharing 代码使用的 `16x128` 参考口径，`op1/op3` 都约 `52 blocks`
- benchmark 输出：
  - `output/sharing/scenarios/<scenario>/benchmarks/sharing_five_way_comparison.csv`
- profiling 输出：
  - `output/sharing/scenarios/<scenario>/nsys/*.nsys-rep`
  - `output/sharing/scenarios/<scenario>/ncu/*.ncu-rep`
  - `output/sharing/scenarios/<scenario>/ncu/*_raw.csv`
  - `output/sharing/scenarios/<scenario>/ncu/*_details.csv`

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
bash scripts/run_sharing_profile_cases.sh
SCENARIO=real_r8 bash scripts/run_sharing_profile_cases.sh
SCENARIO=constructed_50_blocks PROFILE_METHOD=physical_concat bash scripts/run_sharing_profile_cases.sh
RUN_BENCH=0 TOOL=nsys bash scripts/run_sharing_profile_cases.sh
RUN_BENCH=0 TOOL=ncu NCU_SET=full bash scripts/run_sharing_profile_cases.sh
GPU_METRICS_DEVICE=all bash scripts/run_sharing_profile_cases.sh
```

默认脚本会依次执行两个场景：

```bash
bash scripts/run_sharing_profile_cases.sh
```

其中：

- `real_r8` 会复现真实 `r=8` 场景
- `constructed_50_blocks` 会复现“两个 op 都约 50 block”的构造场景
- `PROFILE_METHOD` 默认是 `all`，会依次执行 `baseline / stream_overlap / single_fused_half_split / single_fused_interleaved / physical_concat`
- `ncu` 默认使用 `--set full`，会同时保留 `.ncu-rep` 与导出的 `details/raw` 页面 CSV
- `GPU_METRICS_DEVICE=all bash scripts/run_sharing_profile_cases.sh` 会在两种场景下都开启 `nsys` GPU metrics 采集

## 输出目录规范

运行完成后，结果会按源码域分别落盘：

- `output/triton_learning/benchmarks/`
- `output/fusion/benchmarks/`
- `output/fusion/nsys/`
- `output/fusion/ncu/`
- `output/sharing/scenarios/real_r8/`
- `output/sharing/scenarios/constructed_50_blocks/`

历史 `outputs/` 目录中的旧结果暂时保留，不作为新的默认输出位置。
