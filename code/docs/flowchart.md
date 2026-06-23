# 项目技术路线流程图

> 复制下面任意一段 Mermaid 代码到 GitHub / VSCode / Typora / draw.io / mermaid.live 即可渲染为 PNG/SVG 用于答辩 PPT。

---

## A. 端到端总览(适合答辩首页)

```mermaid
flowchart LR
    A1["LHS 1000 几何采样<br>baseline ± 30%"] --> A2["QBlade SIL<br>LLFVW 仿真<br>1000 时间步"]
    A2 --> A3["data_lhs/*.pkl<br>106+ 几何 × 42 工况"]
    A3 --> B1["扁平化<br>convert_pkl_to_dataset"]
    B1 --> B2["归一化决策<br>analyze_and_normalize"]
    B2 --> C1["MoE 代理模型<br>(LIFT 异方差)"]
    C1 --> C2["Sweep 32 trials<br>1500 epoch"]
    C2 --> C3["最优模型<br>μ, σ ∈ R^4"]
    C3 --> D1["CMA-ES 优化<br>min P_elec/V<br>+ λ_U·σ"]
    D1 --> D2["最优 8 CP 几何"]
    D2 --> E1["QBlade 实测验证"]
    C3 -.σ 高几何.-> AL["Active Learning<br>主动补点"]
    AL -.回流.-> A2

    style A1 fill:#e8f1ff
    style A2 fill:#e8f1ff
    style A3 fill:#e8f1ff
    style B1 fill:#fff2cd
    style B2 fill:#fff2cd
    style C1 fill:#d6ffd6
    style C2 fill:#d6ffd6
    style C3 fill:#d6ffd6
    style D1 fill:#ffd6e6
    style D2 fill:#ffd6e6
    style E1 fill:#ffe6cc
    style AL fill:#f5d6ff
```

---

## B. 数据流(从 pkl 到训练矩阵)

```mermaid
flowchart TD
    P["data_lhs/*.pkl<br>{geometry_idx: {<br>  geometry: (22,3),<br>  RPM_Wind_Angle: DataFrame(120, 26),<br>  ...<br>}}"] --> SCAN[递归扫描 + 解析]
    SCAN --> AGG["每工况末段 120 步均值<br>Thrust, Thrust_z, M_y, Torque"]
    AGG --> RNM["字段重命名<br>Thrust→T, Thrust_z→H,<br>Torque→Q"]
    RNM --> FLAT["扁平表 (N, 53)<br>RPM, WIND, ANGLE,<br>chord_0..21, twist_0..21,<br>T, H, M_y, Q, geom_idx"]
    FLAT --> CSV["dataset_v2_<date>.csv"]
    FLAT --> XLSX["dataset_v2_<date>.xlsx"]

    CSV --> AN[逐列统计 + 决策]
    AN --> DEC{决策树}
    DEC -->|std<1e-12| SK[skip]
    DEC -->|skew>2 & min>0| LG["log1p + standard"]
    DEC -->|"|skew|>1.5 或 |kurt|>8"| Q[quantile→normal]
    DEC -->|默认| ST[standard]
    SK --> NORM[HybridNormalizer]
    LG --> NORM
    Q --> NORM
    ST --> NORM
    NORM --> NPZ["data_normalized.npz<br>(N, 51) ≈ N(0,1)"]
    NORM --> JSON["decisions.json"]
    NORM --> PNG1["distribution_grid.png<br>qq_outputs.png"]

    style P fill:#e8f1ff
    style NPZ fill:#d6ffd6
    style JSON fill:#d6ffd6
    style PNG1 fill:#d6ffd6
```

---

## C. MoE 模型结构(LIFT 异方差)

```mermaid
flowchart LR
    X["X ∈ R^47<br>RPM, WIND, ANGLE,<br>chord_0..21, twist_0..21"] --> SE["Shared Encoder<br>47 → d_s → d_s<br>GELU"]
    SE -->|h| G["Gate<br>d_s → 64 → K<br>softmax"]
    SE -->|h| E1["Expert 1<br>d_s → hidden → 8"]
    SE -->|h| E2["Expert 2"]
    SE -->|h| EK["Expert K<br>K ∈ {2,4,6,8}"]
    G -->|"π ∈ Δ^K"| MIX
    E1 -->|"(μ_1, log σ²_1) ∈ R^4 × R^4"| MIX["Mixture 层<br>μ = Σ π_k μ_k<br>σ² = Σ π_k σ²_k<br>     + Σ π_k (μ_k - μ)²"]
    E2 --> MIX
    EK --> MIX
    MIX --> OUT["μ ∈ R^4 (T, H, M_y, Q)<br>σ ∈ R^4 (不确定度)<br>π ∈ Δ^K (路由)"]

    style SE fill:#e8f1ff
    style G fill:#fff2cd
    style E1 fill:#fff2cd
    style E2 fill:#fff2cd
    style EK fill:#fff2cd
    style MIX fill:#d6ffd6
    style OUT fill:#ffd6e6
```

---

## D. 训练循环

```mermaid
sequenceDiagram
    autonumber
    participant DATA as 数据集
    participant M as MoE 模型
    participant L as 异方差 NLL Loss
    participant O as Adam + Plateau
    participant DIAG as 诊断
    participant B as Best 缓存

    DATA->>M: X_batch (B, 47)
    M->>M: Shared → K Experts + Gate
    M->>L: (μ, σ, π), y_batch
    Note over L: L = (y-μ)²·e^(-log σ²) + log σ²<br>     + λ_b·Balance - λ_s·H(π)
    L->>O: loss
    O->>M: clip_grad_norm(1.0) + step
    M->>DATA: forward(X_test)
    M->>DIAG: R², MAE per col
    DIAG->>O: scheduler.step(test_loss)
    alt test_loss < best
        DIAG->>B: 保存 state_dict
    end
    Note over DATA,B: 循环 1500 epoch
    B->>DIAG: 加载最优 state
    DIAG->>DIAG: 出图: 对角 / 残差 / σ-RMSE / reliability / gate
```

---

## E. 不确定度产出与下游耦合

```mermaid
flowchart LR
    G["候选几何<br>8 CP"] -->|"cp_to_sections<br>(B-Spline)"| X["47 维输入"]
    X --> MoE["MoE 推理"]
    MoE --> MU["μ:<br>T, H, M_y, Q"]
    MoE --> SI["σ:<br>T, H, M_y, Q"]
    MU --> TRIM["TrimSolver V2<br>解 (α, Ω_1, Ω_2)<br>使 Fx=Fz=M_y=0"]
    TRIM --> POW["P_elec = 2·Q·Ω / η<br>η = 0.80"]
    POW --> OBJ["目标:<br>J = P_elec/V<br>+ λ_R·|R|²<br>+ λ_U·Σ w_j·σ_j<br>+ λ_G·smoothness"]
    SI --> OBJ
    OBJ --> CMA["CMA-ES<br>popsize=16<br>maxiter=200"]
    CMA --> BEST["最优 CP"]

    SI -.σ_j > σ_max=0.1.-> AL["Active Learning<br>记录候选"]
    AL -->|每轮 N 个| QB["QBlade 实测"]
    QB --> NEW["新 pkl"]
    NEW -.合并.-> MoE

    style MoE fill:#d6ffd6
    style OBJ fill:#fff2cd
    style CMA fill:#ffd6e6
    style BEST fill:#ffd6e6
    style AL fill:#f5d6ff
```

---

## F. 双机仿真分工(系统层)

```mermaid
flowchart TB
    LHS["geometries_lhs_1000.npy<br>(1000, 22, 3)"]

    subgraph A["3660 工作站 (RTX 3060)"]
        LHS -->|"start-idx=710<br>end-idx=1000"| S3["run_data_gen_parallel<br>--device GPU<br>--workers 1"]
        S3 --> P3["data_lhs/b{0000..0289}.pkl<br>geom 710-999"]
    end

    subgraph B["7920 工作站 (RTX A4000)"]
        LHS -->|"start-idx=0<br>end-idx=710"| S7["run_data_gen_parallel<br>--device GPU<br>--workers 1"]
        S7 --> P7["data_lhs/b{0000..0708}.pkl<br>geom 0-709"]
    end

    P3 -.rsync via Tailscale.-> POOL["统一 data_lhs 池"]
    P7 -.同步.-> POOL
    POOL --> TRAIN["MoE 训练"]

    style A fill:#e8f1ff
    style B fill:#fff2cd
    style POOL fill:#d6ffd6
```

---

## 使用提示

1. **导出 PNG/SVG**: 复制 mermaid 代码到 https://mermaid.live → 右上角下载。
2. **批量渲染**: VSCode 装 `Markdown Preview Mermaid Support` 后,任意 .md 文件按 `Ctrl/Cmd+K V` 可看预览,右键导出。
3. **答辩 PPT 用法**: 总览图 A 放第 1 页(技术路线); B / C 放数据/模型章节; D / E 放训练/优化章节; F 用于工程实现部分。
