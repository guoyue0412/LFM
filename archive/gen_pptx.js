const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "郭跃";
pres.title = "基于代理模型的复杂风场内无人机螺旋桨气动外形优化 — 中期汇报";

// ── 配色 ──
const C = {
  navy:    "0B1D3A",
  navyLt:  "132B50",
  teal:    "00BFA5",
  tealDim: "0D3D38",
  orange:  "FF6D00",
  orangeDim: "3D2400",
  white:   "FFFFFF",
  gray:    "90A4AE",
  ice:     "E3F2FD",
  red:     "FF5252",
  green:   "69F0AE",
  yellow:  "FFD740",
  blue:    "448AFF",
  dark:    "0A1628",
};

// ── 工具函数 ──
const mkShadow = () => ({ type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.18 });

function addSlideNumber(slide, num, total) {
  slide.addText(`${num} / ${total}`, {
    x: 8.8, y: 5.2, w: 1, h: 0.35, fontSize: 10, color: C.gray, align: "right",
  });
}

const TOTAL = 13;

// ════════════════════════════════════════════════════════════════
// SLIDE 0 — 封面
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  // 装饰圆
  s.addShape(pres.shapes.OVAL, { x: 7.5, y: -1.5, w: 5, h: 5, fill: { color: C.teal, transparency: 92 } });

  s.addText("哈尔滨工业大学（深圳） · 硕士学位论文中期检查", {
    x: 0.8, y: 1.0, w: 8.4, h: 0.4, fontSize: 12, color: C.teal, align: "center",
  });
  s.addText("基于代理模型的复杂风场内", {
    x: 0.8, y: 1.5, w: 8.4, h: 0.6, fontSize: 30, bold: true, color: C.white, align: "center", valign: "middle", margin: 0,
  });
  s.addText("无人机螺旋桨气动外形优化", {
    x: 0.8, y: 2.1, w: 8.4, h: 0.6, fontSize: 30, bold: true, color: C.teal, align: "center", valign: "middle", margin: 0,
  });

  s.addText("LLFVW 高保真仿真 → MoE 代理模型 → CMA-ES 配平约束优化", {
    x: 1.5, y: 2.85, w: 7, h: 0.4, fontSize: 13, color: C.gray, align: "center",
  });

  s.addText([
    { text: "答辩人：郭跃  |  学号：24S153206\n", options: { breakLine: true } },
    { text: "指导教师：何晓舟 教授  |  机器人与先进制造学院 · 能源动力\n", options: { breakLine: true } },
    { text: "深圳市科技重大专项 KJZD20230923115210021\n", options: { breakLine: true } },
    { text: "2026 年 6 月", options: { color: C.teal, bold: true } },
  ], { x: 1.5, y: 3.4, w: 7, h: 1.4, fontSize: 11, color: C.gray, align: "center", lineSpacingMultiple: 1.5 });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 1 — 汇报提纲
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  addSlideNumber(s, 1, TOTAL);

  s.addText("汇报提纲", { x: 0.7, y: 0.4, w: 8, h: 0.7, fontSize: 30, bold: true, color: C.white, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.05, w: 0.6, h: 0.04, fill: { color: C.teal } });

  const items = [
    ["1", "研究背景与开题方案回顾", "问题定义、四项研究任务、原计划技术路线"],
    ["2", "已完成工作（一）：仿真平台", "B-Spline 参数化、LLFVW 气动模型、UIUC 验证"],
    ["3", "已完成工作（二）：代理模型与优化", "DNN→MoE 演进、V1→V4 迭代、OOD 发现"],
    ["4", "已完成工作（三）：辅助模块", "Mann 风场建模、CFD 验证模板"],
    ["5", "当前进展与数据扩展", "1000 几何 LLFVW 仿真、GPU 加速"],
    ["6", "进度对比与后续计划", "开题 vs 实际、方法演变、剩余工作"],
  ];

  items.forEach((item, i) => {
    const row = Math.floor(i / 2);
    const col = i % 2;
    const x = 0.7 + col * 4.5;
    const y = 1.35 + row * 1.35;

    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 4.2, h: 1.15, fill: { color: C.navyLt }, shadow: mkShadow() });
    s.addText(item[0], { x: x + 0.15, y: y + 0.12, w: 0.45, h: 0.45, fontSize: 18, bold: true, color: C.teal, align: "center", valign: "middle" });
    s.addText(item[1], { x: x + 0.65, y: y + 0.12, w: 3.3, h: 0.4, fontSize: 13, bold: true, color: C.white, margin: 0 });
    s.addText(item[2], { x: x + 0.65, y: y + 0.55, w: 3.3, h: 0.45, fontSize: 10, color: C.gray, margin: 0 });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 2 — 研究背景
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  addSlideNumber(s, 2, TOTAL);

  s.addText("研究背景", { x: 0.7, y: 0.3, w: 3, h: 0.5, fontSize: 11, color: C.teal, margin: 0 });
  s.addText("复杂风场内无人机螺旋桨气动外形优化", { x: 0.7, y: 0.7, w: 9, h: 0.6, fontSize: 24, bold: true, color: C.white, margin: 0 });

  // 左栏 — 问题
  const problems = [
    "1  城市低空湍流环境，前飞时桨叶承受非轴向来流，效率远低于悬停设计点",
    "2  传统 BEMT/CFD 耗时大，难以高效探索参数空间",
    "3  前后桨差速配平引入强耦合约束，现有研究多忽略",
    "4  已有文献集中于悬停优化，前飞巡航工况优化稀缺",
  ];
  problems.forEach((p, i) => {
    s.addText(p, { x: 0.7, y: 1.5 + i * 0.5, w: 4.4, h: 0.45, fontSize: 11, color: C.ice, margin: 0 });
  });

  // 优化目标
  s.addText("优化目标：min Pelec/V", { x: 0.7, y: 3.65, w: 4.4, h: 0.35, fontSize: 14, bold: true, color: C.teal, margin: 0 });
  s.addText("等价于最大化 Breguet 航程", { x: 0.7, y: 3.98, w: 4.4, h: 0.3, fontSize: 10, color: C.gray, margin: 0 });

  // 右栏 — 开题四任务
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 1.45, w: 4.2, h: 1.6, fill: { color: C.navyLt }, shadow: mkShadow() });
  s.addText("开题四项研究任务", { x: 5.6, y: 1.5, w: 3.8, h: 0.35, fontSize: 13, bold: true, color: C.teal, margin: 0 });
  const tasks = ["T1  LLFVW 高保真仿真数据生成", "T2  DNN 气动代理模型构建", "T3  PPO 在线调参 NSGA-II 多目标优化", "T4  CFD 高保真验证与可视化分析"];
  tasks.forEach((t, i) => {
    s.addText(t, { x: 5.6, y: 1.88 + i * 0.27, w: 3.8, h: 0.25, fontSize: 10, color: C.ice, margin: 0 });
  });

  // 设计变量
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 3.25, w: 4.2, h: 0.85, fill: { color: C.navyLt }, shadow: mkShadow() });
  s.addText("设计变量", { x: 5.6, y: 3.3, w: 3.8, h: 0.3, fontSize: 12, bold: true, color: C.white, margin: 0 });
  s.addText("8 个 B-Spline 控制点 (4 chord + 4 twist)\ndegree=3 → 22 径向截面  |  APC 10×7 基准桨", {
    x: 5.6, y: 3.6, w: 3.8, h: 0.45, fontSize: 10, color: C.gray, margin: 0,
  });

  // 无人机平台
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 4.3, w: 4.2, h: 0.6, fill: { color: C.navyLt }, shadow: mkShadow() });
  s.addText("无人机平台：3.5 kg · 10\" 桨\nVcruise=10 m/s · 前后对称 X 构型", {
    x: 5.6, y: 4.33, w: 3.8, h: 0.52, fontSize: 10, color: C.gray, margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 3 — 方案演变
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  addSlideNumber(s, 3, TOTAL);

  s.addText("方案演变", { x: 0.7, y: 0.3, w: 3, h: 0.4, fontSize: 11, color: C.blue, margin: 0 });
  s.addText("开题技术路线 → 实际演进", { x: 0.7, y: 0.65, w: 9, h: 0.5, fontSize: 24, bold: true, color: C.white, margin: 0 });

  // 开题方案（删除线用灰色+标注）
  s.addText("开题方案（2024.09）", { x: 0.7, y: 1.25, w: 4, h: 0.3, fontSize: 10, color: C.gray, margin: 0 });
  const oldBoxes = ["LLFVW 仿真\n100 几何", "单 MLP DNN\n代理模型", "PPO+NSGA-II\n多目标优化", "Star-CCM+ CFD\n高保真验证"];
  oldBoxes.forEach((b, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.7 + i * 2.35, y: 1.55, w: 2.0, h: 0.7, fill: { color: C.navyLt }, line: { color: C.gray, width: 1, dashType: "dash" } });
    s.addText(b, { x: 0.7 + i * 2.35, y: 1.55, w: 2.0, h: 0.7, fontSize: 9, color: C.gray, align: "center", valign: "middle" });
    if (i < 3) s.addText("→", { x: 2.55 + i * 2.35, y: 1.7, w: 0.5, h: 0.35, fontSize: 14, color: C.gray, align: "center" });
  });

  // 箭头
  s.addText("↓  发现 OOD 外推陷阱 → 方法迭代  ↓", { x: 1.5, y: 2.35, w: 7, h: 0.35, fontSize: 12, color: C.orange, align: "center", bold: true });

  // 实际方案
  s.addText("实际方案（V2 管线）", { x: 0.7, y: 2.75, w: 4, h: 0.3, fontSize: 10, color: C.teal, margin: 0 });
  const newBoxes = ["LLFVW 仿真\n1000 几何×42工况", "MoE 代理模型\nμ + σ 不确定度", "CMA-ES+配平约束\n三轴平衡+OOD防护", "QBlade LLFVW\n回验+可视化"];
  newBoxes.forEach((b, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.7 + i * 2.35, y: 3.05, w: 2.0, h: 0.7, fill: { color: C.tealDim }, line: { color: C.teal, width: 1 } });
    s.addText(b, { x: 0.7 + i * 2.35, y: 3.05, w: 2.0, h: 0.7, fontSize: 9, color: C.teal, align: "center", valign: "middle" });
    if (i < 3) s.addText("→", { x: 2.55 + i * 2.35, y: 3.2, w: 0.5, h: 0.35, fontSize: 14, color: C.teal, align: "center" });
  });

  // 三项变更
  const changes = [
    ["变更一：代理模型升级", "单 MLP → MoE (K=4)\n+异方差NLL不确定度\nMy从解析估算→直接预测"],
    ["变更二：优化算法", "PPO+NSGA-II → CMA-ES\nPPO FM=0.899 OOD不可控\nV4 FM=0.947 全部可信"],
    ["变更三：验证策略", "CFD终验 → QBlade回验优先\n与仿真数据同源,快速闭环\nCFD作辅助高保真验证"],
  ];
  changes.forEach((c, i) => {
    const x = 0.7 + i * 3.15;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 3.9, w: 2.9, h: 1.2, fill: { color: C.orangeDim }, line: { color: C.orange, width: 1 } });
    s.addText(c[0], { x: x + 0.1, y: 3.95, w: 2.7, h: 0.3, fontSize: 10, bold: true, color: C.orange, margin: 0 });
    s.addText(c[1], { x: x + 0.1, y: 4.22, w: 2.7, h: 0.82, fontSize: 9, color: C.ice, margin: 0 });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 4 — 仿真平台
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  addSlideNumber(s, 4, TOTAL);

  s.addText("已完成工作（一）", { x: 0.7, y: 0.3, w: 4, h: 0.4, fontSize: 11, color: C.teal, margin: 0 });
  s.addText("LLFVW 仿真平台搭建", { x: 0.7, y: 0.65, w: 8, h: 0.5, fontSize: 24, bold: true, color: C.white, margin: 0 });
  s.addText("对应开题任务 T1", { x: 6.5, y: 0.75, w: 2, h: 0.3, fontSize: 10, color: C.green, margin: 0 });

  // 左栏
  const leftCards = [
    ["B-Spline 参数化建模", "8控制点(4 chord+4 twist), degree=3\n→22径向截面→三维螺旋桨几何\nAPC 10×7 拟合验证通过"],
    ["LLFVW 气动理论", "Biot-Savart诱导速度+无穿透边界\nN×N线性方程→Kutta-Joukowski\n1000时间步+后120步周期平均"],
    ["自动化仿真框架", "Python驱动QBlade SIL (.so/.dll)\nUbuntu(GPU)+Windows跨平台\n批量.sim生成→仿真→提取→入库"],
  ];
  leftCards.forEach((c, i) => {
    const y = 1.3 + i * 1.3;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y, w: 4.4, h: 1.1, fill: { color: C.navyLt }, shadow: mkShadow() });
    s.addText(c[0], { x: 0.85, y: y + 0.08, w: 4.1, h: 0.3, fontSize: 12, bold: true, color: C.white, margin: 0 });
    s.addText(c[1], { x: 0.85, y: y + 0.38, w: 4.1, h: 0.65, fontSize: 9, color: C.gray, margin: 0 });
  });

  // 右栏 — UIUC验证
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 1.3, w: 4.2, h: 2.0, fill: { color: C.tealDim }, line: { color: C.teal, width: 1 } });
  s.addText("UIUC 两级验证 (Dantsker 2022)", { x: 5.55, y: 1.35, w: 3.9, h: 0.3, fontSize: 12, bold: true, color: C.teal, margin: 0 });
  s.addText([
    { text: "翼型 (2D)：", options: { bold: true, color: C.teal, breakLine: false } },
    { text: "E374/NACA6409 · XFoil 251点\nRe=10⁵~3×10⁵ · 小攻角Cl,Cd一致\n", options: { color: C.ice, breakLine: true } },
    { text: "螺旋桨 (3D)：", options: { bold: true, color: C.teal, breakLine: false } },
    { text: "APC 10×7 · RPM∈[4007,6519]\nCT, CP, η 差异均 ≤10%\n140组工况与UIUC风洞对照", options: { color: C.ice } },
  ], { x: 5.55, y: 1.7, w: 3.9, h: 1.5, fontSize: 9, margin: 0 });

  // 数据规模
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 3.5, w: 4.2, h: 0.7, fill: { color: C.navyLt } });
  const stats = [["100", "V1几何"], ["→1000", "V2扩展"], ["42", "工况/几何"]];
  stats.forEach((st, i) => {
    s.addText(st[0], { x: 5.6 + i * 1.35, y: 3.55, w: 1.2, h: 0.35, fontSize: 20, bold: true, color: i === 1 ? C.orange : C.teal, align: "center" });
    s.addText(st[1], { x: 5.6 + i * 1.35, y: 3.9, w: 1.2, h: 0.25, fontSize: 9, color: C.gray, align: "center" });
  });

  // 网格收敛
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 4.4, w: 4.2, h: 0.85, fill: { color: C.navyLt }, shadow: mkShadow() });
  s.addText("网格收敛性验证", { x: 5.55, y: 4.45, w: 3.9, h: 0.25, fontSize: 11, bold: true, color: C.white, margin: 0 });
  s.addText("翼型：251点标准网格 · 螺旋桨：3°/步·1000步→收敛\n时间分辨率与网格无关性均满足", {
    x: 5.55, y: 4.72, w: 3.9, h: 0.45, fontSize: 9, color: C.gray, margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 5 — 代理模型 DNN→MoE
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  addSlideNumber(s, 5, TOTAL);

  s.addText("已完成工作（二）", { x: 0.7, y: 0.3, w: 4, h: 0.4, fontSize: 11, color: C.teal, margin: 0 });
  s.addText("代理模型：DNN → MoE 演进", { x: 0.7, y: 0.65, w: 8, h: 0.5, fontSize: 24, bold: true, color: C.white, margin: 0 });
  s.addText("对应开题任务 T2", { x: 6.5, y: 0.75, w: 2, h: 0.3, fontSize: 10, color: C.green, margin: 0 });

  // 左：V1 DNN
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.3, w: 4.4, h: 2.2, fill: { color: C.navyLt }, line: { color: C.gray, width: 1 } });
  s.addText("V1 实现：单 MLP DNN", { x: 0.85, y: 1.35, w: 4.1, h: 0.3, fontSize: 12, bold: true, color: C.gray, margin: 0 });
  s.addText(
    "输入 47 维 (3工况 + 22×chord + 22×twist)\n" +
    "隐藏层 [64, 128, 64] · ReLU · Dropout 0.001\n" +
    "输出：Fx, Fy, Fz, Torque\n" +
    "MSE损失 · Adam · CosineAnnealingLR · 5000 epochs\n" +
    "100几何 × 72工况 = 7200 样本 (train:test=9:1)\n" +
    "Fx MAPE 1.03% · Fz 1.50% · Torque 0.99%",
    { x: 0.85, y: 1.7, w: 4.1, h: 1.7, fontSize: 9, color: C.ice, margin: 0 }
  );

  // 左下：问题
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 3.7, w: 4.4, h: 1.5, fill: { color: C.orangeDim }, line: { color: C.orange, width: 1 } });
  s.addText("问题：分布内精度高 ≠ 外推可靠", { x: 0.85, y: 3.75, w: 4.1, h: 0.3, fontSize: 11, bold: true, color: C.orange, margin: 0 });
  s.addText(
    "CMA-ES无约束 → FM=1.004（物理不可能）\n" +
    "twist 100%截面OOD, chord 68%截面OOD\n" +
    "RPM外推 +1455 (超出[4000,6500])\n" +
    "PPO同样受影响：FM=0.899 但OOD不可控\n" +
    "→ 核心问题不在优化算法，而在模型缺乏可信度感知",
    { x: 0.85, y: 4.08, w: 4.1, h: 1.05, fontSize: 9, color: C.ice, margin: 0 }
  );

  // 右：V2 MoE
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 1.3, w: 4.2, h: 2.7, fill: { color: C.tealDim }, line: { color: C.teal, width: 1 } });
  s.addText("实际方案：MoE 代理模型 (V2)", { x: 5.55, y: 1.35, w: 3.9, h: 0.3, fontSize: 12, bold: true, color: C.teal, margin: 0 });
  s.addText(
    "Shared Encoder 47→64 (BatchNorm+ReLU)\n" +
    "Gating Network 64→K (softmax权重)\n" +
    "K=4 Experts 各64→128→8 [μ₄+log_σ₄]\n" +
    "输出：[T,H,My,Q] + [σT,σH,σMy,σQ]\n\n" +
    "相比DNN的核心改进：\n" +
    "+ 异方差NLL损失→自动学习预测不确定度\n" +
    "+ My直接预测（不再解析估算）\n" +
    "+ 混合方差 σ²=Σπk(σk²+μk²)−μ²\n" +
    "+ Balance+熵正则化防expert坍缩\n" +
    "+ 输入维度11→47（含22截面分布）",
    { x: 5.55, y: 1.7, w: 3.9, h: 2.2, fontSize: 9, color: C.ice, margin: 0 }
  );

  // 右下：为什么MoE — 与左侧"问题"框对齐
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 3.7, w: 4.2, h: 1.5, fill: { color: C.navyLt }, shadow: mkShadow() });
  s.addText("为什么选 MoE 而非 MC-Dropout/Ensemble？", { x: 5.55, y: 3.75, w: 3.9, h: 0.25, fontSize: 10, bold: true, color: C.white, margin: 0 });
  s.addText(
    "单次前向同时获得μ+σ (CMA-ES 16×200次评估)\n" +
    "Gating天然适配多工况分区\n" +
    "异方差NLL学出输入相关σ\n" +
    "Ensemble需N次前向,MC-Dropout需T次采样",
    { x: 5.55, y: 4.05, w: 3.9, h: 1.0, fontSize: 9, color: C.gray, margin: 0 }
  );
}

// ════════════════════════════════════════════════════════════════
// SLIDE 6 — V1→V4 迭代
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  addSlideNumber(s, 6, TOTAL);

  s.addText("关键发现", { x: 0.7, y: 0.3, w: 3, h: 0.4, fontSize: 11, color: C.orange, margin: 0 });
  s.addText("V1→V4 迭代：从虚假最优到可信优化", { x: 0.7, y: 0.65, w: 9, h: 0.5, fontSize: 24, bold: true, color: C.white, margin: 0 });

  // 表格
  const tblRows = [
    [
      { text: "版本", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
      { text: "优化器", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
      { text: "约束策略", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
      { text: "FM", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
      { text: "功率(W)", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
      { text: "几何OOD", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
      { text: "可信度", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
    ],
    [
      { text: "V1", options: { bold: true } }, "CMA-ES", "无约束",
      { text: "1.004", options: { color: C.red } }, "342",
      { text: "37/44 (84%)", options: { color: C.red } },
      { text: "★", options: { color: C.red } },
    ],
    [
      { text: "V1", options: { bold: true } }, "PPO", "无约束",
      { text: "0.899", options: { color: C.red } }, "382",
      { text: "OOD", options: { color: C.red } },
      { text: "★", options: { color: C.red } },
    ],
    [
      { text: "V3", options: { bold: true } }, "CMA-ES", "OOD软惩罚",
      { text: "0.941", options: { color: C.yellow } }, "365",
      { text: "15/44 (34%)", options: { color: C.yellow } },
      { text: "★★", options: { color: C.yellow } },
    ],
    [
      { text: "V4", options: { bold: true, color: C.teal } }, "CMA-ES", "硬约束(数据驱动)",
      { text: "0.947", options: { bold: true, color: C.teal } },
      { text: "363", options: { bold: true } },
      { text: "0/44", options: { bold: true, color: C.teal } },
      { text: "★★★", options: { color: C.teal } },
    ],
  ];

  s.addTable(tblRows, {
    x: 0.5, y: 1.25, w: 9.0,
    fontSize: 10, color: C.ice,
    border: { pt: 0.5, color: "2A3A5A" },
    colW: [0.6, 0.9, 1.5, 0.8, 0.8, 1.3, 0.8],
    rowH: [0.35, 0.3, 0.3, 0.3, 0.35],
  });

  // V4最优解
  s.addText("V4最优解：FM=0.9469 · P=362.7W · chord CP=[0.0136, 0.0438, 0.0198, 0.0048] · twist CP=[37.4°, 13.5°, 11.3°, 8.5°]", {
    x: 0.7, y: 3.2, w: 8.6, h: 0.3, fontSize: 9, color: C.gray,
  });

  // 两个卡片
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 3.6, w: 4.4, h: 1.6, fill: { color: C.navyLt }, line: { color: C.red, width: 1 } });
  s.addText("OOD 外推陷阱（核心发现）", { x: 0.85, y: 3.65, w: 4.1, h: 0.25, fontSize: 11, bold: true, color: C.red, margin: 0 });
  s.addText(
    "现象：CMA-ES V1 FM=1.004（物理不可能值）\n" +
    "诊断：twist 22/22截面OOD, chord 15/22截面OOD\n" +
    "根因：MLP分布内MAPE 1%,但OOD外推不可靠\n" +
    "对比：PPO也FM=0.899 OOD不可控→问题不在优化算法\n" +
    "→ 需要代理模型具备不确定度感知能力",
    { x: 0.85, y: 3.95, w: 4.1, h: 1.15, fontSize: 9, color: C.ice, margin: 0 }
  );

  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 3.6, w: 4.2, h: 1.6, fill: { color: C.tealDim }, line: { color: C.teal, width: 1 } });
  s.addText("配平约束求解器（开题无此模块）", { x: 5.55, y: 3.65, w: 3.9, h: 0.25, fontSize: 11, bold: true, color: C.teal, margin: 0 });
  s.addText(
    "三轴平衡：Fx=0, Fz=mg, My=0\n" +
    "求解变量：[α, Ω₁, Ω₂] (迎角+前后桨转速)\n" +
    "V1收敛率仅3% → V2多初始值×6 → 100%收敛\n" +
    "My由MoE直接预测 (非解析km·T·R·μ)\n" +
    "RPM∈[4000,6500] · α∈[2°,8°]",
    { x: 5.55, y: 3.95, w: 3.9, h: 1.15, fontSize: 9, color: C.ice, margin: 0 }
  );
}

// ════════════════════════════════════════════════════════════════
// SLIDE 7 — 辅助模块
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  addSlideNumber(s, 7, TOTAL);

  s.addText("已完成工作（三）", { x: 0.7, y: 0.3, w: 4, h: 0.4, fontSize: 11, color: C.teal, margin: 0 });
  s.addText("辅助模块：风场建模 + CFD 验证", { x: 0.7, y: 0.65, w: 9, h: 0.5, fontSize: 24, bold: true, color: C.white, margin: 0 });

  // Mann风场
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.3, w: 4.4, h: 3.5, fill: { color: C.tealDim }, line: { color: C.teal, width: 1 } });
  s.addText("Mann 复杂风场建模", { x: 0.85, y: 1.35, w: 4.1, h: 0.3, fontSize: 13, bold: true, color: C.teal, margin: 0 });
  s.addText("基于 Mann 模型的三维湍流风场生成\n频谱法：谱表示+随机相位构造", {
    x: 0.85, y: 1.7, w: 4.1, h: 0.5, fontSize: 10, color: C.ice, margin: 0,
  });

  const mannParams = [
    ["风场盒子", "20m×1m×1m, 高度10m"],
    ["网格点数", "631×256×256"],
    ["积分尺度", "L_MANN = 15m"],
    ["各向异性", "Γ = 3.9"],
    ["αε参数", "0.0660"],
    ["参考风速", "12 m/s"],
    ["IEC标准", "61400, NTM, TI=16"],
    ["方差缩放", "(1.10,0.80,0.48)"],
  ];
  const mannTbl = [
    [
      { text: "参数", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
      { text: "值", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
    ],
    ...mannParams.map(r => [r[0], r[1]]),
  ];
  s.addTable(mannTbl, {
    x: 0.85, y: 2.3, w: 4.0, fontSize: 9, color: C.ice,
    border: { pt: 0.5, color: "2A3A5A" },
    colW: [1.2, 2.8], rowH: [0.25, ...Array(8).fill(0.22)],
  });

  // CFD验证
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 1.3, w: 4.2, h: 2.4, fill: { color: C.tealDim }, line: { color: C.teal, width: 1 } });
  s.addText("CFD 验证模板 (Star-CCM+)", { x: 5.55, y: 1.35, w: 3.9, h: 0.3, fontSize: 13, bold: true, color: C.teal, margin: 0 });
  s.addText("对应开题任务 T4：高保真验证与可视化", { x: 5.55, y: 1.65, w: 3.9, h: 0.25, fontSize: 9, color: C.gray, margin: 0 });

  const cfdParams = [
    ["计算域", "入口/出口10D, 总宽25D"],
    ["旋转域", "直径1.2D, 宽0.4D"],
    ["网格", "~700万, 阻塞比2.5%"],
    ["湍流模型", "SST k-ω"],
    ["y+范围", "0~1 (满足要求)"],
    ["精度", "CP,CT,η误差≤5%"],
  ];
  const cfdTbl = [
    [
      { text: "参数", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
      { text: "值", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
    ],
    ...cfdParams.map(r => [r[0], r[1]]),
  ];
  s.addTable(cfdTbl, {
    x: 5.55, y: 1.95, w: 3.9, fontSize: 9, color: C.ice,
    border: { pt: 0.5, color: "2A3A5A" },
    colW: [1.2, 2.7], rowH: [0.25, ...Array(6).fill(0.22)],
  });

  // 验证策略调整
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 3.9, w: 4.2, h: 0.9, fill: { color: C.navyLt }, shadow: mkShadow() });
  s.addText("验证策略调整", { x: 5.55, y: 3.95, w: 3.9, h: 0.25, fontSize: 11, bold: true, color: C.white, margin: 0 });
  s.addText("开题：CFD为主要终验手段\n实际：QBlade LLFVW回验优先(同源闭环)\nCFD仍保留为辅助高保真验证,模板已就绪", {
    x: 5.55, y: 4.22, w: 3.9, h: 0.55, fontSize: 9, color: C.gray, margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 8 — 数据扩展
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  addSlideNumber(s, 8, TOTAL);

  s.addText("当前进展", { x: 0.7, y: 0.3, w: 3, h: 0.4, fontSize: 11, color: C.orange, margin: 0 });
  s.addText("LLFVW 仿真数据扩展 — 100 → 1000 几何", { x: 0.7, y: 0.65, w: 9, h: 0.5, fontSize: 24, bold: true, color: C.white, margin: 0 });

  // 大数字
  const bigNums = [["1000", "LHS采样几何"], ["42", "工况/几何"], ["42K", "目标样本量"]];
  bigNums.forEach((n, i) => {
    s.addText(n[0], { x: 0.7 + i * 1.6, y: 1.25, w: 1.4, h: 0.5, fontSize: 30, bold: true, color: C.teal, align: "center" });
    s.addText(n[1], { x: 0.7 + i * 1.6, y: 1.72, w: 1.4, h: 0.25, fontSize: 9, color: C.gray, align: "center" });
  });

  // 仿真配置表
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 2.1, w: 4.4, h: 2.1, fill: { color: C.navyLt }, shadow: mkShadow() });
  s.addText("仿真配置", { x: 0.85, y: 2.15, w: 4.1, h: 0.25, fontSize: 12, bold: true, color: C.white, margin: 0 });
  const simCfg = [
    ["仿真器", "QBlade CE v2.0.8.6 LLFVW (SIL)"],
    ["部署", "Linux .so · RTX 3060 GPU (OpenCL)"],
    ["时间步", "1000步, 取后120步均值"],
    ["工况", "6 RPM × 7 Angle, Wind=10 m/s"],
    ["采样", "LHS (d=8, seed=42), baseline±30%"],
    ["落盘", "batch-size=1 (逐几何即时保存pkl)"],
  ];
  simCfg.forEach((r, i) => {
    s.addText(r[0], { x: 0.85, y: 2.45 + i * 0.27, w: 0.9, h: 0.25, fontSize: 9, bold: true, color: C.teal, margin: 0 });
    s.addText(r[1], { x: 1.8, y: 2.45 + i * 0.27, w: 3.1, h: 0.25, fontSize: 9, color: C.ice, margin: 0 });
  });

  // GPU加速
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 4.35, w: 4.4, h: 0.9, fill: { color: C.navyLt }, shadow: mkShadow() });
  s.addText("GPU 加速：3.5h → 1.1h/几何 (3.2× 提速)", { x: 0.85, y: 4.4, w: 4.1, h: 0.25, fontSize: 11, bold: true, color: C.teal, margin: 0 });
  s.addText("停止CPU并行后GPU独占资源,吞吐从~6/天提升到~21/天", { x: 0.85, y: 4.7, w: 4.1, h: 0.4, fontSize: 9, color: C.gray, margin: 0 });

  // 右栏 — 时间预估
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 1.25, w: 4.2, h: 1.8, fill: { color: C.orangeDim }, line: { color: C.orange, width: 1 } });
  s.addText("时间预估", { x: 5.55, y: 1.3, w: 3.9, h: 0.3, fontSize: 12, bold: true, color: C.orange, margin: 0 });
  const timeline = [
    ["当前速度", "~1.1 h/几何 · ~21几何/天"],
    ["geom 710-999", "~13天 → 约6/20完成"],
    ["geom 0-709", "~33天 → 约7/23完成"],
    ["全量1000", "预计7月下旬"],
  ];
  timeline.forEach((r, i) => {
    const isBold = i === 3;
    s.addText(r[0], { x: 5.55, y: 1.65 + i * 0.35, w: 1.4, h: 0.3, fontSize: 10, bold: isBold, color: isBold ? C.orange : C.ice, margin: 0 });
    s.addText(r[1], { x: 7.0, y: 1.65 + i * 0.35, w: 2.5, h: 0.3, fontSize: 10, bold: isBold, color: isBold ? C.orange : C.gray, margin: 0 });
  });

  // 可提前启动
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 3.2, w: 4.2, h: 1.0, fill: { color: C.navyLt }, shadow: mkShadow() });
  s.addText("可提前启动的工作", { x: 5.55, y: 3.25, w: 3.9, h: 0.25, fontSize: 11, bold: true, color: C.white, margin: 0 });
  s.addText("300+几何完成后即可启动MoE初步训练\n验证架构+调参,数据补齐后增量微调\n→ 不必等全量数据,节省2-3周", {
    x: 5.55, y: 3.55, w: 3.9, h: 0.6, fontSize: 9, color: C.gray, margin: 0,
  });

  // 进度条
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 4.35, w: 4.2, h: 0.9, fill: { color: C.navyLt }, shadow: mkShadow() });
  s.addText("进度概览 (2026-06-07)", { x: 5.55, y: 4.4, w: 3.9, h: 0.2, fontSize: 10, bold: true, color: C.white, margin: 0 });
  // 进度条形状
  s.addShape(pres.shapes.RECTANGLE, { x: 5.55, y: 4.7, w: 3.8, h: 0.12, fill: { color: C.navyLt }, line: { color: "2A3A5A", width: 1 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.55, y: 4.7, w: 0.15, h: 0.12, fill: { color: C.teal } });
  s.addText("GPU geom 710-999 进行中   ·   总进度 ~2%", { x: 5.55, y: 4.85, w: 3.8, h: 0.25, fontSize: 8, color: C.gray, margin: 0 });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 9 — 进度对比
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  addSlideNumber(s, 9, TOTAL);

  s.addText("进度对比", { x: 0.7, y: 0.3, w: 3, h: 0.4, fontSize: 11, color: C.blue, margin: 0 });
  s.addText("开题进度 vs 实际执行", { x: 0.7, y: 0.65, w: 9, h: 0.5, fontSize: 24, bold: true, color: C.white, margin: 0 });

  const header = [
    { text: "时间段", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
    { text: "开题计划", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
    { text: "实际执行", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
    { text: "状态", options: { bold: true, color: C.teal, fill: { color: C.navyLt } } },
  ];

  const rows = [
    ["2024.09—12", "气动建模+仿真平台", "B-Spline+UIUC验证+自动化框架", { text: "按计划", options: { color: C.green } }],
    ["2025.01—04", "DNN代理模型", "100几何DNN+V1 pipeline·发现OOD", { text: "按计划", options: { color: C.green } }],
    ["2025.05—08", "PPO调参NSGA-II", "V1→V4迭代·MoE+配平方案设计", { text: "方法调整", options: { color: C.blue } }],
    ["2025.09—11", "复杂风场扩展", "Mann风场完成·V2代码实现", { text: "并行推进", options: { color: C.blue } }],
    ["2025.12—26.04", "外形优化与鲁棒", "QBlade Linux GPU·1000几何启动", { text: "调整顺序", options: { color: C.blue } }],
    ["2026.05—06", "CFD验证", "CFD模板就绪·GPU数据生成中", { text: "进行中", options: { color: C.orange } }],
    ["2026.07—08", "结果整理", "MoE训练+CMA-ES优化+回验", { text: "待执行", options: { color: C.gray } }],
    ["2026.09—11", "论文与答辩", "论文撰写与答辩准备", { text: "待执行", options: { color: C.gray } }],
  ];

  s.addTable([header, ...rows], {
    x: 0.4, y: 1.2, w: 9.2,
    fontSize: 9, color: C.ice,
    border: { pt: 0.5, color: "2A3A5A" },
    colW: [1.3, 1.7, 3.2, 0.9],
    rowH: [0.32, ...Array(8).fill(0.38)],
    autoPage: false,
  });

  // 总结
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.5, w: 9.0, h: 0.6, fill: { color: "0D2040" }, line: { color: C.blue, width: 1 } });
  s.addText(
    "阶段1-2按计划完成；阶段3-5因OOD发现调整方法路线(PPO+NSGA-II→CMA-ES+配平, DNN→MoE)，工作量增加但方法更扎实；CFD模板已提前就绪；论文时间节点未变。",
    { x: 0.65, y: 4.55, w: 8.5, h: 0.5, fontSize: 9, color: C.ice, margin: 0 }
  );
}

// ════════════════════════════════════════════════════════════════
// SLIDE 10 — 后续计划
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  addSlideNumber(s, 10, TOTAL);

  s.addText("后续计划", { x: 0.7, y: 0.3, w: 3, h: 0.4, fontSize: 11, color: C.orange, margin: 0 });
  s.addText("剩余工作路线图", { x: 0.7, y: 0.65, w: 9, h: 0.5, fontSize: 24, bold: true, color: C.white, margin: 0 });

  // 时间线
  const tl = [
    [C.orange, "2026.06 — 进行中", "1000几何LLFVW数据生成", "GPU ~21几何/天 · 预计7月下旬全量"],
    [C.gray, "2026.07 上旬", "data.py转换 + MoE训练", "pkl→CSV · K-fold验证 · 300+几何先行"],
    [C.gray, "2026.07 下旬", "CMA-ES优化闭环", "不确定度引导主动补点 · 迭代收敛"],
    [C.gray, "2026.08", "QBlade回验 + CFD辅助", "最优几何全工况回验 · Star-CCM+流场"],
    [C.gray, "2026.08—09", "结果分析", "基线vs优化桨对比 · 灵敏度分析"],
    [C.gray, "2026.09—11", "论文撰写与答辩", "方法论+实验结果+结论展望 · 5-6万字"],
  ];

  // 画时间线
  s.addShape(pres.shapes.RECTANGLE, { x: 1.05, y: 1.35, w: 0.03, h: 3.7, fill: { color: "2A3A5A" } });

  tl.forEach((item, i) => {
    const y = 1.3 + i * 0.62;
    // 圆点
    s.addShape(pres.shapes.OVAL, { x: 0.93, y: y + 0.07, w: 0.25, h: 0.25, fill: { color: item[0] } });
    s.addText(item[1], { x: 1.35, y, w: 1.5, h: 0.2, fontSize: 8, color: C.gray, margin: 0 });
    s.addText(item[2], { x: 1.35, y: y + 0.2, w: 3.3, h: 0.2, fontSize: 11, bold: true, color: C.white, margin: 0 });
    s.addText(item[3], { x: 1.35, y: y + 0.4, w: 3.3, h: 0.2, fontSize: 9, color: C.gray, margin: 0 });
  });

  // 右栏 — 里程碑
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 1.3, w: 4.2, h: 2.2, fill: { color: C.tealDim }, line: { color: C.teal, width: 1 } });
  s.addText("近期关键里程碑", { x: 5.55, y: 1.35, w: 3.9, h: 0.3, fontSize: 12, bold: true, color: C.teal, margin: 0 });
  const milestones = [
    ["✓", C.green, "LHS 1000几何采样设计"],
    ["✓", C.green, "QBlade Linux GPU 部署"],
    ["✓", C.green, "V2 管线代码完成 (MoE+Trim+CMA-ES)"],
    ["✓", C.green, "batch-size=1 防丢失策略"],
    ["◐", C.orange, "全量数据生成 (进行中)"],
    ["○", C.gray, "pkl→CSV 批量转换"],
    ["○", C.gray, "MoE 真实数据训练"],
    ["○", C.gray, "CMA-ES 优化 + 主动补点"],
  ];
  milestones.forEach((m, i) => {
    s.addText(m[0], { x: 5.55, y: 1.7 + i * 0.22, w: 0.3, h: 0.2, fontSize: 10, color: m[1], align: "center", margin: 0 });
    s.addText(m[2], { x: 5.85, y: 1.7 + i * 0.22, w: 3.6, h: 0.2, fontSize: 9, color: C.ice, margin: 0 });
  });

  // 风险
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 3.7, w: 4.2, h: 1.5, fill: { color: C.navyLt }, line: { color: C.orange, width: 1 } });
  s.addText("风险与应对", { x: 5.55, y: 3.75, w: 3.9, h: 0.25, fontSize: 11, bold: true, color: C.orange, margin: 0 });
  const risks = [
    "1  数据周期长(~47天)：300+子集先行训练",
    "2  MoE过拟合：K-fold+早停+balance正则",
    "3  边界解持续：扩大CP范围+主动补点",
    "4  LLFVW与CFD偏差：多保真点验证+校准",
  ];
  risks.forEach((r, i) => {
    s.addText(r, { x: 5.55, y: 4.05 + i * 0.27, w: 3.9, h: 0.25, fontSize: 9, color: C.ice, margin: 0 });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 11 — 创新点
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  addSlideNumber(s, 11, TOTAL);

  s.addText("总结", { x: 0.7, y: 0.3, w: 3, h: 0.4, fontSize: 11, color: C.teal, margin: 0 });
  s.addText("研究创新点与预期成果", { x: 0.7, y: 0.65, w: 9, h: 0.5, fontSize: 24, bold: true, color: C.white, margin: 0 });

  const innovations = [
    ["1", "MoE代理模型+不确定度量化",
     "相对现有：Wu(2024)等使用单MLP/Kriging,无可信度输出\n" +
     "本文：K=4混合专家+异方差NLL→伴随输出σ\n" +
     "证据：V1 MLP在OOD区域给出FM=1.004虚假预测"],
    ["2", "前飞差速配平嵌套优化",
     "相对现有：Klimczyk(2022)等仅优化悬停,不涉配平\n" +
     "本文：CMA-ES每次评估嵌套三轴配平[α,Ω₁,Ω₂]\n" +
     "证据：多初始值×6→100%收敛,物理可实现"],
    ["3", "OOD感知+主动补点闭环",
     "相对现有：Tao(2019)等假设模型全域可靠\n" +
     "本文：实证OOD陷阱(84%截面外推)+硬约束+σ补点\n" +
     "闭环：优化→σ超阈值→补采LLFVW→重训→再优化"],
  ];

  innovations.forEach((inn, i) => {
    const x = 0.5 + i * 3.15;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.3, w: 2.95, h: 2.35, fill: { color: C.tealDim }, line: { color: C.teal, width: 1 } });
    s.addText(inn[0], { x: x + 0.1, y: 1.35, w: 0.35, h: 0.35, fontSize: 18, bold: true, color: C.teal, align: "center", valign: "middle" });
    s.addText(inn[1], { x: x + 0.5, y: 1.4, w: 2.3, h: 0.3, fontSize: 11, bold: true, color: C.teal, margin: 0 });
    s.addText(inn[2], { x: x + 0.1, y: 1.8, w: 2.75, h: 1.75, fontSize: 8.5, color: C.ice, margin: 0 });
  });

  // 预期成果
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.8, w: 9.0, h: 1.4, fill: { color: C.navyLt }, shadow: mkShadow() });
  s.addText("预期成果 — 对应开题四项目标", { x: 0.7, y: 3.85, w: 8.6, h: 0.3, fontSize: 12, bold: true, color: C.white, align: "center", margin: 0 });

  const deliverables = [
    ["1", "优化平台搭建", "LLFVW+MoE+CMA-ES\n全栈自动化", "代码完成", C.green],
    ["2", "优化螺旋桨", "APC 10×7 基准桨\nVcruise=10m/s最优", "7月执行", C.orange],
    ["3", "验证与可视化", "QBlade回验+Star-CCM+\n流场对比·模板就绪", "8月执行", C.orange],
    ["4", "42K仿真数据集", "1000几何×42工况\nLHS d=8全覆盖", "生成中", C.orange],
  ];

  deliverables.forEach((d, i) => {
    const x = 0.7 + i * 2.2;
    s.addText(d[0], { x, y: 4.2, w: 0.3, h: 0.3, fontSize: 16, bold: true, color: C.teal, align: "center" });
    s.addText(d[1], { x: x + 0.35, y: 4.2, w: 1.5, h: 0.25, fontSize: 10, bold: true, color: C.white, margin: 0 });
    s.addText(d[2], { x: x + 0.35, y: 4.45, w: 1.6, h: 0.4, fontSize: 8, color: C.gray, margin: 0 });
    s.addText(d[3], { x: x + 0.35, y: 4.85, w: 1.0, h: 0.2, fontSize: 8, bold: true, color: d[4] });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 12 — 结尾
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.OVAL, { x: 7.5, y: -1.5, w: 5, h: 5, fill: { color: C.teal, transparency: 92 } });

  s.addText("感谢各位老师 · 批评指正", {
    x: 1, y: 1.8, w: 8, h: 1.0, fontSize: 36, bold: true, color: C.white, align: "center", valign: "middle",
  });

  s.addText("敬请提问", { x: 1, y: 2.9, w: 8, h: 0.5, fontSize: 16, color: C.teal, align: "center" });

  s.addText([
    { text: "郭跃  |  指导教师：何晓舟 教授\n", options: { breakLine: true } },
    { text: "哈尔滨工业大学（深圳）  |  2026 年 6 月", options: {} },
  ], { x: 1, y: 3.7, w: 8, h: 0.8, fontSize: 11, color: C.gray, align: "center" });
}

// ── 输出 ──
const outPath = "/Users/guoyue/gy_2026/graduation/midterm_review.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("PPTX saved to: " + outPath);
}).catch(err => {
  console.error("Error:", err);
});
