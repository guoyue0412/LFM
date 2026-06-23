# Graduation — 螺旋桨气动外形优化（硕士毕业课题）

> **作者**: 郭跃 (24S153206) · 哈工大深圳 · 机器人与先进制造学院 · 能源动力
> **导师**: 何晓舟 教授
> **答辩节点**: 2026.09 — 2026.11
> **课题**: 基于代理模型的无人机螺旋桨前飞工况气动外形优化

---

## 仓库结构（monorepo）

```
graduation/                          ← 单一 git 仓库
├── code/                            ← 代码与数据 (原 LFM 仓库, 保留完整 commit 历史)
│   ├── optimization_v2/             ← V2 管线: LLFVW + MoE + 配平 + CMA-ES
│   ├── ppo_optimize/                ← V1 PPO/CMA-ES + 硬约束
│   ├── data_for_train/              ← LHS 几何 + 处理后数据
│   ├── sweep_results/               ← 4 个完整 sweep 结果
│   ├── scripts/                     ← QBlade SIL 仿真启动脚本
│   ├── Propeller_project-main/      ← submodule (Linux 版 QBlade)
│   ├── docs/                        ← 4 份设计文档 (architecture/method/moe/flowchart)
│   └── *.py  CLAUDE.md  README.md   ← V1 顶层脚本与项目说明
│
├── thesis/                          ← LaTeX 学位论文
│   ├── midterm-thesis/              ← 中期论文 (已编译, thesis.pdf 1.9 MB)
│   │   ├── figures/                 ← 含 improvement_matrix / ood_sigma_failure / hyperparam 等图
│   │   └── plot_style.py            ← 统一图模板 (paper / ppt / a4)
│   ├── opening-defence/             ← 开题答辩
│   ├── hitszthesis-reference/       ← hitszthesis 模板参考
│   └── ...                          ← 备份与参考论文
│
├── archive/                         ← 历史临时材料 (PPTX / HTML 中期评审用)
│
├── README.md                        ← 本文档
├── .gitignore                       ← 顶层忽略规则 (大数据/编译产物)
└── .git/                            ← 唯一 git 仓库根 (前身: guoyue0412/LFM.git)
```

---

## 快速上手

### 代码
```bash
cd code
conda create -n LFM python=3.10 && conda activate LFM
pip install -r requirements.txt

# V2 主管线
python -m optimization_v2.run --step train
python -m optimization_v2.tools.sweep_moe --arch moe --loss lift \
    --data ./data_for_train/processed_lhs/norm/data_normalized.npz \
    --out ./sweep_results/my_run --epochs 1500 --n-trials 16
```

### LaTeX 论文
```bash
cd thesis/midterm-thesis
latexmk -xelatex thesis.tex          # 生成 thesis.pdf
```

### 答辩图制作（统一模板）
```bash
cd thesis/midterm-thesis/figures
python3 improvement_matrix.py --style ppt    # PPT 16:9 / 300 DPI
python3 ood_sigma_failure.py --style a4      # 论文嵌入 / textwidth
python3 hyperparam_heatmap.py --style paper  # 报告 / 答辩讲稿
```

---

## 关键中期成果

| 维度 | 中期结果 |
|---|---|
| LLFVW 数据 | 366 几何 × 42 工况 (3660 GPU 持续生成, 目标 1000) |
| 代理模型选型 | **MLP+LIFT (CP 11d)** — ECE 平均降 35%, 与 sections 输入精度持平 |
| OOD 验证 | σ 不能单独防 OOD → 三层防御 (硬约束 + Mahalanobis + in-dist σ) |
| 中期论文 | `thesis/midterm-thesis/thesis.pdf` (1.9 MB, 含 3 张新增对比图) |

---

## 仓库历史

- **2026-06-23**: 合并 `LFM` (代码) + `latex-thesis` (论文) 为单一 `graduation` 仓库,
  保留 LFM 全部 git 历史 (50+ commits)。
- 远程仓库: 拟改名为 `guoyue0412/graduation.git` (原 `LFM.git`)。

详见 `code/RESEARCH_LOG.md` 了解 V1 → V4 演进过程。
