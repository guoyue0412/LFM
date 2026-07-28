# 2026-07-28 QBlade GPU 数据生成移交

## 当前边界

本轮严格按用户要求串行处理：先完成“单工况 QBlade GPU 运行与验收”，再处理 1.5 kg 飞行器参数及公开资料对齐，最后才继续批量补数、训练、配平、CMA-ES 和论文。当前仍停留在第一项；不得启动训练、扫参、CMA-ES，也不得更新论文数值结论。

## 已确认的数据状态

- 历史目录：`/home/gy/gy_2026/graduation/LFM/data_for_train/data_lhs`，999 个 pkl。
- pkl 内容中的唯一几何 ID 为 `0..999`，共 1000 个。
- `geom 1..998` 各有完整 42 工况；`geom 0` 仅 1/42，`geom 999` 仅 2/42。
- 唯一工况覆盖为 `41919/42000 = 99.8071%`。
- `geom 0` 的 cp8 无法从 22 截面可靠恢复；`geom 999` 可恢复。
- 正式新数据 manifest：
  `/home/gy/gy_2026/graduation/runs/datafreeze_20260728_348b7c2_inputs/supplement_manifest_20x63_seed20260727.json`
- manifest SHA-256：
  `131a510e1245b0dc285644edd219176d12c5dade4a1594dda8b688240d8eda6b`
- manifest 含 20 个新几何，每个 63 工况：42 个 base、15 个工况内插、6 个 `ANGLE=89 deg` 的 alpha=1 检测点。

## Git 与运行版本

### 父仓库

- 工作目录：`/Users/guoyue/Documents/graduation/codex-surrogate-ood-id`
- 分支：`codex/surrogate-ood-id-execution`
- 已推送到 GitHub 的稳定提交：`c3cb7d46991ad0dcd43d0f42f729b6582957373f`
- 本地后续提交：`b8922552a5a4f270a2b3110e4fca6e55c8edb76e`
- `b892255` 已把 gitlink 指向下述 QBlade 修复提交，并加入键格式回归测试。其后的 handoff 提交将 launcher 默认 `EXPECTED_SUBMODULE_COMMIT` 同步为 `f3e5674...`。

### QBlade 子模块

- 子模块路径：`code/Propeller_project-main`
- 物理基线提交：`c042826847dfe2ff1a3c708e5b3acbf3d987e972`
- wrapper 键修复提交：`f3e56740b83aa00a36f4d56380b095c5b741437f`
- 子模块分支：`codex/gpu-manifest-condition-key`
- `f3e5674` 只修改直接工况调用的结果键，不改变桨叶、尾迹、时间步或求解物理参数。
- 该子模块分支已通过 GitHub SSH 远程推送，可供父仓库 gitlink 解析。

## 3660 当前运行状态（2026-07-28 17:00 CST 刷新）

仍在运行且必须保留：

- `geom 0` CPU repair：父 PID `2024645`，worker `2024693`，10 OMP 线程。
- `geom 999` CPU repair：父 PID `2024646`，worker `2024697`，10 OMP 线程。
- 两者已运行约 22.5 小时，尚未写出整几何 pkl；旧 runner 不支持逐工况检查点，中断会损失当前整几何进度。

已停止：

- 无正式 manifest/Git 溯源的旧 `geom 1000` CPU 任务，原 PID `2026332/2026334/2026358`。

GPU 当前空闲，约 401 MiB 基础显存占用。

## 第一次 GPU 单点实测

正式部署目录：

`/home/gy/gy_2026/graduation/runs/LFM_datafreeze_20260728_c3cb7d4`

执行工况：

`geom_id=1000, RPM=4000, WIND=10 m/s, ANGLE=89 deg`

已确认：

- QBlade 枚举到 `d1: OpenCL 3.0 CUDA NVIDIA GeForce RTX 4090`。
- 计算进程使用约 384 MiB 显存；GPU 利用率呈脉冲式，采样峰值约 45%。
- worker 同时使用约 9--10 个 CPU 核；当时 `geom 0/999` 两个 CPU repair 仍在运行，因此存在 CPU 资源竞争。
- 1000 步计算壁钟时间为 `753.9 s = 12.6 min`。
- 该速度约对应 42 工况 8.8 小时，不支持“42 工况约 1 小时”的假设。
- 不得为追求一小时目标直接减少时间步或改回激进尾迹截断；历史记录显示短时间步会严重影响侧向力/俯仰力矩，且 `bbaff14` 与正式物理基线 `c042826` 的尾迹参数不同。

第一次运行最终失败且没有正式 pkl：

```text
[worker 2064849] geom 1000 FAIL: RuntimeError:
absent persisted QBlade result for (4000.0, 10.0, 89.0)
```

根因不是 GPU 求解失败。QBlade 完成了全部 1000 步，但 `run_one_simulation(RPM=..., WIND_SPEED=..., ANGLE=...)` 把结果写入固定键 `Base_simulation`；manifest runner 按规范键 `RPM4000.0_Wind10.0_Angle89.0` 查找，所以拒收。

修复位于子模块提交 `f3e5674`：直接工况调用现在用 lossless float 格式生成并持久化规范键。对应测试为 `code/tests/test_qblade_condition_key.py`。修复后本地联合测试曾达到 92 项通过；交接者仍须在最终 launcher SHA 修正后重新执行完整测试。

## 已修复的生成链路问题

- manifest 模式支持精确 `geom_id`、工况 split 和 condition key 子集。
- GPU 只允许 `workers=1`，拒绝已知不安全的多进程 GPU 路径。
- 每个成功工况原子写 checkpoint，崩溃后可按精确工况续跑。
- 所有新结果同时记录父仓库和 QBlade 子模块完整提交哈希。
- 运行前要求主仓库干净、submodule HEAD 等于 gitlink、manifest SHA 正确。
- alpha1、smoke、formal 共用同一个 pkl/checkpoint 目录。
- launcher 不包含训练或扫参入口。

## 接手后的严格执行顺序

1. 确认 launcher 默认 `EXPECTED_SUBMODULE_COMMIT=f3e56740b83aa00a36f4d56380b095c5b741437f`。
2. 运行：

   ```bash
   PYTHONPATH=code /Users/guoyue/anaconda3/bin/python -m pytest \
     code/tests/test_supplement_manifest.py \
     code/tests/test_run_data_gen_manifest.py \
     code/tests/test_audit_and_freeze_pkl.py \
     code/tests/test_run_supplement_3660.py \
     code/tests/test_qblade_condition_key.py -q
   bash -n code/scripts/run_supplement_3660.sh
   git diff --check
   ```

3. 测试会在子模块中产生两个未跟踪 `__pycache__/*.pyc`；测试后删除它们并确认工作树干净。
4. 子模块分支已推送；提交并推送父仓库 handoff/launcher 修复即可。
5. 用 Git bundle 部署新的提交专属 3660 worktree；不要修改远程脏目录 `/home/gy/gy_2026/graduation/LFM`，也不要复用名称中仍为 `c3cb7d4` 的旧 worktree。
6. 只重跑上述同一个 GPU 工况。输出必须进入新版本对应的统一正式 pkl 目录。
7. 验收该 checkpoint/pkl：规范 condition key、非空 DataFrame、有限数值、`geom_id/cp8/sections/category/split/manifest SHA/父仓库 SHA/子模块 SHA` 全部精确匹配。
8. 单点验收通过后，才依次运行 geom 1000 的其余五个 `ANGLE=89 deg` 工况；仍保持单 worker、工况串行。
9. 六个 alpha=1 点通过后，再评估 42 工况实际产能和是否需要单独的物理一致加速实验。
10. 完成第1项后，才开始第2项：基于公开可靠资料把飞行器质量改为 1.5 kg，并逐项核对臂长、参考面积、阻力系数和惯量。当前尚未完成该网络资料对齐。

## 禁止事项

- 不得把第一次GPU运行记作成功样本；它没有生成可接受pkl。
- 不得把 `sigma_mix` 解释为绝对误差上界。
- 不得启动训练、扫参、配平或CMA-ES，直到数据冻结完成。
- 不得用旧 `geom1000` bootstrap 输出替代正式 manifest 数据。
- 不得停止 `geom0/999` repair，除非它们完成或用户明确接受丢失当前整几何进度。
- 不得只按文件数判断数据完整性；必须按 pkl 内 `geom_id + condition key` 审计。
