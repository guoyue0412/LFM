# 2026-07-28 数据冻结前审计快照

本文件记录数据补齐仍在运行时的只读审计证据，不是最终数据冻结验收，也不得作为训练输入或论文数值结论来源。

## 版本与输入

- 审计代码提交：`348b7c23f4c6e7df6972c24574b4acb84ec33ef4`
- 历史 pkl：`/home/gy/gy_2026/graduation/LFM/data_for_train/data_lhs`，999 个文件
- 预运行补充目录：`/home/gy/gy_2026/graduation/LFM/data_for_train/data_supplement_20260727_bootstrap`，审计时 0 个 pkl
- 历史几何源：`geometries_lhs_1000.npy`，SHA-256 `c7a8b156777a79a72a69207d68f8b3b1c217596a5e115f863eface806669d076`
- 一行一几何的 44 截面 CSV：SHA-256 `f6100d7f541e361092566970513633e3217dc6935859baf9115bd7c444f53a73`
- 正式 20×63 清单：SHA-256 `131a510e1245b0dc285644edd219176d12c5dade4a1594dda8b688240d8eda6b`；相同输入和种子二次生成后逐字节一致
- 远程 acceptance JSON：`/home/gy/gy_2026/graduation/runs/audit_snapshot_20260728_348b7c2/acceptance_snapshot-20260728-348b7c2.json`
- acceptance JSON SHA-256：`816ae46e94a58e9486b05adbca8ebd49eaa24cdd0d4ceed6b1a76b19447d3830`

## 历史基准网格审计

- 观察到唯一历史几何 ID 1000 个，即 `0..999`。
- 998 个几何具有完整的 42 个唯一基准工况，对应几何 `1..998`。
- 几何 0 仅有 `(RPM, WIND, ANGLE)=(4000,10,85)`，缺 41 个工况。
- 几何 999 仅有 `(4000,10,85)` 和 `(5000,10,82)`，缺 40 个工况。
- 已存在的唯一基准工况键为 `41919/42000`，键覆盖率为 `99.8071%`；完整几何覆盖率为 `998/1000=99.8%`。
- 重复候选记录 336 条，影响几何 34--40 的 294 个逻辑工况键。审计器均按最新文件修改时间保留一个候选，并把每项选择写入 `duplicate_decisions`；当前没有未解决的重复冲突。
- 不可读 pkl、pkl 内外几何 ID 冲突、补充元数据错误和记录级排除均为 0。

## 控制点与表示资格

- 几何 0 的 44 截面无法在当前固定 B-Spline 节点下无损恢复 8D 控制点，最大重建误差为 `5.449712`，超过 `1e-10` 容差。
- 当前完整且同时具备 11D/47D 表示的历史几何为 998 个，即 `1..998`。
- 几何 999 补齐后可进入 11D/47D 公平比较集合；几何 0 即使补齐也只能作为 47D-only 历史样本，除非找到其原始 cp8 溯源证据。

## 新几何与当前运行状态

- 审计时正式 20 个新几何尚无可验收的 manifest-provenance pkl，因此 1260 个声明工况均未计入冻结集。
- 2026-07-28 12:00 CST，几何 0、999 和几何 1000 的三个旧启动器任务仍在运行；系统负载约 28.5（24 逻辑核），GPU 约 3%，日志未发现失败。
- 旧启动器只在整几何完成后写 pkl，因此当前尚无中间文件。正式批次将在经过复核的条件级原子检查点实现上运行。

## 配平角度与阻力模型的前置诊断

以下仅是符号和量级诊断，最终判断仍以 alpha=1 预检、单点 QBlade 原始量对照和重训后的代理模型为准。

- 使用 UAV-positive alias 重新聚合的 41,916 条完整历史样本中，`H` 全部为负，范围为 `[-0.514919,-0.137489] N`，均值为 `-0.303142 N`。当前 `trim_solver.py` 却使用 `-H cos(alpha)` 组装前向力，因此会把负的面内阻力转换成正向推进力；竖向力中的 `+H sin(alpha)` 也与“带符号 H”定义相反。
- 以平均每桨 `H=-0.303142 N` 估算，四桨面内力约为 `-1.213 N`。当前符号会把它当作约 `+1.213 N` 的前向力，这足以把配平角度推向训练域下边界。
- 当前 cross-flow 机身模型在 10 m/s、alpha=1--8 deg 给出约 `1.70--1.79 N` 阻力；论文中的 `Cd*A=1.10*0.0517` 给出 `3.483 N`。两者相差约一倍，必须做敏感性对照，现阶段不能宣称任一模型已被验证。
- 历史正确符号数据的单桨推力范围为 `1.550941--7.040686 N`。四桨最大约 `28.16 N`，低于代码默认 3.5 kg 对应的 `34.34 N` 重力，因此默认质量在现有 RPM 范围内没有稳态竖向配平余量。Iris 1.5 kg 假设必须在正式配平前单独核验并记录来源。
- 忽略旋翼力随姿态和转速的变化，仅作数量级估算时，1.5 kg、当前机身阻力和带符号面内力对应的倾角约 11 deg，而非 1 deg；论文阻力假设会进一步增大所需倾角。因此 alpha=1 样本适合作为条件 OOD/符号连续性检查，不应在未校准前直接改成配平域内训练结论。
- NASA 多旋翼风洞数据表明裸机身阻力随姿态变化，且完整飞行器存在旋翼—机身/旋翼—旋翼干扰；文中给出的某 SOLO 工况在约 6.1 m/s 时配平俯仰约 6 deg。Theys 与 De Schutter 的不同球形机身试验在 12--13 m/s 附近出现约 17--25 deg 俯仰。这些结果只能说明“所需倾角高度依赖机身与推进系统”，不能直接移植为本机参数。

一手参考：

- Russell et al., *Wind Tunnel and Hover Performance Test Results for Multicopter UAS Vehicles*, NASA/TM-2016-219310: https://ntrs.nasa.gov/api/citations/20160007399/downloads/20160007399.pdf
- Altamirano and McCrink, *Investigation of Longitudinal Aero-Propulsive Interactions of a Small Quadrotor UAS*: https://ntrs.nasa.gov/api/citations/20205010614/downloads/AIAA_SciTech21_GA_v7.pdf
- Theys and De Schutter, *Forward flight tests of a quadcopter UAV with various spherical body diameters*: https://doi.org/10.1177/1756829320923565

## 使用限制

本快照生成的训练 CSV 是审计器行为验证产物。由于几何 0、999 未补齐且新几何未生成，它们不得用于四模型重训、扫参、配平、CMA-ES 或论文结果更新。最终使用资格必须由补齐后的新一轮 acceptance JSON 决定。
