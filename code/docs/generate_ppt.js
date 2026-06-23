const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Guo Yue";
pres.title = "LFM 螺旋桨气动优化系统";

// 配色方案 - Ocean Gradient 深色主题
const C = {
  navy: "0A1628",
  deepBlue: "0E2A4E",
  teal: "0891B2",
  cyan: "22D3EE",
  white: "FFFFFF",
  lightGray: "E2E8F0",
  gray: "94A3B8",
  dark: "1E293B",
  card: "162032",
  accent: "F59E0B",
};

const makeShadow = () => ({ type: "outer", blur: 8, offset: 3, color: "000000", opacity: 0.3, angle: 135 });

// ═══════════════════════════════════════════════
// Slide 1: 封面
// ═══════════════════════════════════════════════
let s1 = pres.addSlide();
s1.background = { color: C.navy };
// 顶部装饰线
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.teal } });
// 底部装饰线
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.565, w: 10, h: 0.06, fill: { color: C.teal } });

s1.addText("LFM", {
  x: 0.8, y: 1.0, w: 8.4, h: 1.2,
  fontSize: 60, fontFace: "Arial Black", color: C.cyan,
  bold: true, align: "left", margin: 0,
});
s1.addText("螺旋桨气动优化全栈系统", {
  x: 0.8, y: 2.1, w: 8.4, h: 0.8,
  fontSize: 28, fontFace: "Arial", color: C.white,
  align: "left", margin: 0,
});
s1.addText("DNN 代理模型 + 四旋翼配平 + RL/进化优化 + QBlade 验证", {
  x: 0.8, y: 2.9, w: 8.4, h: 0.6,
  fontSize: 16, fontFace: "Arial", color: C.gray,
  align: "left", margin: 0,
});

// 统计数据 - 三个数据卡片
const stats = [
  { num: "47", label: "代码文件" },
  { num: "788", label: "符号节点" },
  { num: "1,292", label: "依赖边" },
];
stats.forEach((st, i) => {
  const x = 0.8 + i * 2.8;
  s1.addShape(pres.shapes.RECTANGLE, {
    x, y: 3.8, w: 2.4, h: 1.2,
    fill: { color: C.deepBlue },
    line: { color: C.teal, width: 1 },
  });
  s1.addText(st.num, {
    x, y: 3.85, w: 2.4, h: 0.7,
    fontSize: 32, fontFace: "Arial Black", color: C.cyan,
    bold: true, align: "center", valign: "middle", margin: 0,
  });
  s1.addText(st.label, {
    x, y: 4.5, w: 2.4, h: 0.4,
    fontSize: 13, fontFace: "Arial", color: C.gray,
    align: "center", valign: "middle", margin: 0,
  });
});

s1.addText("郭越  |  2026", {
  x: 0.8, y: 5.1, w: 8.4, h: 0.35,
  fontSize: 12, fontFace: "Arial", color: C.gray,
  align: "left", margin: 0,
});

// ═══════════════════════════════════════════════
// Slide 2: 系统总览 - 四阶段流水线
// ═══════════════════════════════════════════════
let s2 = pres.addSlide();
s2.background = { color: C.white };

s2.addText("系统总览：四阶段流水线", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: C.dark,
  bold: true, align: "left", margin: 0,
});
s2.addText("从 QBlade 仿真到优化验证的完整闭环", {
  x: 0.5, y: 0.95, w: 9, h: 0.4,
  fontSize: 14, fontFace: "Arial", color: C.gray,
  align: "left", margin: 0,
});

// 四个阶段卡片
const stages = [
  { title: "QBlade 仿真", desc: "LLFVW 涡格法\n多工况扫描\npkl 数据输出", color: "0D9488", files: "SIMULATION\nrun_data_gen.py" },
  { title: "DNN 代理模型", desc: "V1: MLP [64,128,64]\nV2: MoE 混合专家\n不确定度量化", color: "0891B2", files: "train.py\nmoe_model.py" },
  { title: "优化求解", desc: "CMA-ES 进化策略\nPPO 强化学习\n硬约束边界", color: "6366F1", files: "cma_optimizer.py\ntrain_ppo.py" },
  { title: "配平验证", desc: "3 方程非线性配平\n多初始值鲁棒求解\n功率/航程评估", color: "8B5CF6", files: "trim_solver.py\nevaluate.py" },
];

stages.forEach((st, i) => {
  const x = 0.35 + i * 2.4;
  const w = 2.15;
  // 卡片背景
  s2.addShape(pres.shapes.RECTANGLE, {
    x, y: 1.6, w, h: 3.4,
    fill: { color: C.white },
    line: { color: C.lightGray, width: 1 },
    shadow: makeShadow(),
  });
  // 顶部色条
  s2.addShape(pres.shapes.RECTANGLE, {
    x, y: 1.6, w, h: 0.08,
    fill: { color: st.color },
  });
  // 阶段编号
  s2.addShape(pres.shapes.OVAL, {
    x: x + 0.7, y: 1.85, w: 0.7, h: 0.7,
    fill: { color: st.color },
  });
  s2.addText(String(i + 1), {
    x: x + 0.7, y: 1.85, w: 0.7, h: 0.7,
    fontSize: 22, fontFace: "Arial Black", color: C.white,
    bold: true, align: "center", valign: "middle", margin: 0,
  });
  // 标题
  s2.addText(st.title, {
    x: x + 0.1, y: 2.65, w: w - 0.2, h: 0.45,
    fontSize: 14, fontFace: "Arial", color: C.dark,
    bold: true, align: "center", margin: 0,
  });
  // 描述
  s2.addText(st.desc, {
    x: x + 0.15, y: 3.1, w: w - 0.3, h: 1.1,
    fontSize: 11, fontFace: "Arial", color: C.gray,
    align: "left", margin: 0, lineSpacingMultiple: 1.3,
  });
  // 核心文件
  s2.addText(st.files, {
    x: x + 0.15, y: 4.2, w: w - 0.3, h: 0.6,
    fontSize: 10, fontFace: "Consolas", color: C.teal,
    align: "left", margin: 0, lineSpacingMultiple: 1.2,
  });
  // 箭头（除最后一个）
  if (i < 3) {
    s2.addText("→", {
      x: x + w - 0.05, y: 2.8, w: 0.35, h: 0.6,
      fontSize: 24, fontFace: "Arial", color: C.teal,
      bold: true, align: "center", valign: "middle", margin: 0,
    });
  }
});

// ═══════════════════════════════════════════════
// Slide 3: V1 vs V2 管线对比
// ═══════════════════════════════════════════════
let s3 = pres.addSlide();
s3.background = { color: C.navy };
s3.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.teal } });

s3.addText("V1 vs V2 管线对比", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: C.white,
  bold: true, align: "left", margin: 0,
});
s3.addText("两条完全独立的管线，无交叉依赖", {
  x: 0.5, y: 0.95, w: 9, h: 0.35,
  fontSize: 14, fontFace: "Arial", color: C.gray,
  align: "left", margin: 0,
});

// V1 卡片
s3.addShape(pres.shapes.RECTANGLE, {
  x: 0.4, y: 1.5, w: 4.3, h: 3.5,
  fill: { color: C.card },
  line: { color: "0D9488", width: 2 },
});
s3.addText("V1 管线", {
  x: 0.6, y: 1.6, w: 3.9, h: 0.5,
  fontSize: 20, fontFace: "Arial", color: "0D9488",
  bold: true, align: "left", margin: 0,
});
s3.addText([
  { text: "模型: ", options: { bold: true, color: C.lightGray, breakLine: false } },
  { text: "单 MLP DNN [64, 128, 64]", options: { color: C.gray, breakLine: true } },
  { text: "目标: ", options: { bold: true, color: C.lightGray, breakLine: false } },
  { text: "FM 品质因数最大化", options: { color: C.gray, breakLine: true } },
  { text: "配平: ", options: { bold: true, color: C.lightGray, breakLine: false } },
  { text: "M_p_y 解析估算", options: { color: C.gray, breakLine: true } },
  { text: "优化: ", options: { bold: true, color: C.lightGray, breakLine: false } },
  { text: "PPO + CMA-ES", options: { color: C.gray, breakLine: true } },
  { text: "影响: ", options: { bold: true, color: C.lightGray, breakLine: false } },
  { text: "19 符号 / 6 文件", options: { color: C.gray, breakLine: true } },
], {
  x: 0.6, y: 2.2, w: 3.9, h: 2.5,
  fontSize: 13, fontFace: "Arial", lineSpacingMultiple: 1.6, margin: 0,
});

// 数据流
s3.addText("config → data → train → adapter → trim → ppo_optimize", {
  x: 0.6, y: 4.4, w: 3.9, h: 0.4,
  fontSize: 10, fontFace: "Consolas", color: C.teal,
  align: "left", margin: 0,
});

// V2 卡片
s3.addShape(pres.shapes.RECTANGLE, {
  x: 5.3, y: 1.5, w: 4.3, h: 3.5,
  fill: { color: C.card },
  line: { color: C.accent, width: 2 },
});
s3.addText("V2 管线 (毕业方案)", {
  x: 5.5, y: 1.6, w: 3.9, h: 0.5,
  fontSize: 20, fontFace: "Arial", color: C.accent,
  bold: true, align: "left", margin: 0,
});
s3.addText([
  { text: "模型: ", options: { bold: true, color: C.lightGray, breakLine: false } },
  { text: "LLFVW-MoE 混合专家", options: { color: C.gray, breakLine: true } },
  { text: "目标: ", options: { bold: true, color: C.lightGray, breakLine: false } },
  { text: "min P_elec/V 单位距离能耗", options: { color: C.gray, breakLine: true } },
  { text: "配平: ", options: { bold: true, color: C.lightGray, breakLine: false } },
  { text: "M_p_y 直接由 MoE 预测", options: { color: C.gray, breakLine: true } },
  { text: "优化: ", options: { bold: true, color: C.lightGray, breakLine: false } },
  { text: "CMA-ES + 不确定度惩罚", options: { color: C.gray, breakLine: true } },
  { text: "影响: ", options: { bold: true, color: C.lightGray, breakLine: false } },
  { text: "15 符号 / 4 文件", options: { color: C.gray, breakLine: true } },
], {
  x: 5.5, y: 2.2, w: 3.9, h: 2.5,
  fontSize: 13, fontFace: "Arial", lineSpacingMultiple: 1.6, margin: 0,
});
s3.addText("run.py → data_loader → train_moe → cma → trim → eval", {
  x: 5.5, y: 4.4, w: 3.9, h: 0.4,
  fontSize: 10, fontFace: "Consolas", color: C.accent,
  align: "left", margin: 0,
});

// ═══════════════════════════════════════════════
// Slide 4: 模块架构
// ═══════════════════════════════════════════════
let s4 = pres.addSlide();
s4.background = { color: C.white };

s4.addText("模块架构与符号统计", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: C.dark,
  bold: true, align: "left", margin: 0,
});

// 表格
const tableHeader = [
  { text: "模块", options: { fill: { color: C.deepBlue }, color: C.white, bold: true, fontSize: 13, fontFace: "Arial" } },
  { text: "文件数", options: { fill: { color: C.deepBlue }, color: C.white, bold: true, fontSize: 13, fontFace: "Arial", align: "center" } },
  { text: "符号数", options: { fill: { color: C.deepBlue }, color: C.white, bold: true, fontSize: 13, fontFace: "Arial", align: "center" } },
  { text: "职责", options: { fill: { color: C.deepBlue }, color: C.white, bold: true, fontSize: 13, fontFace: "Arial" } },
];

const rowStyle = (alt) => ({ fill: { color: alt ? "F8FAFC" : C.white }, fontSize: 12, fontFace: "Arial", color: C.dark });
const centerStyle = (alt) => ({ fill: { color: alt ? "F8FAFC" : C.white }, fontSize: 12, fontFace: "Arial", color: C.dark, align: "center" });

const tableRows = [
  [
    { text: "根目录", options: rowStyle(false) },
    { text: "11", options: centerStyle(false) },
    { text: "234", options: centerStyle(false) },
    { text: "V1 管线：DNN 训练 / 配平 / 评估", options: rowStyle(false) },
  ],
  [
    { text: "optimization_v2/", options: rowStyle(true) },
    { text: "15", options: centerStyle(true) },
    { text: "199", options: centerStyle(true) },
    { text: "V2 管线（毕业方案）：MoE + CMA-ES", options: rowStyle(true) },
  ],
  [
    { text: "ppo_optimize/", options: rowStyle(false) },
    { text: "6", options: centerStyle(false) },
    { text: "102", options: centerStyle(false) },
    { text: "PPO / CMA-ES 优化器", options: rowStyle(false) },
  ],
  [
    { text: "Propeller_project-main/", options: rowStyle(true) },
    { text: "8", options: centerStyle(true) },
    { text: "165", options: centerStyle(true) },
    { text: "QBlade 仿真引擎（Linux submodule）", options: rowStyle(true) },
  ],
  [
    { text: "scripts/", options: rowStyle(false) },
    { text: "2", options: centerStyle(false) },
    { text: "40", options: centerStyle(false) },
    { text: "Linux 数据采集 wrapper", options: rowStyle(false) },
  ],
];

s4.addTable([tableHeader, ...tableRows], {
  x: 0.5, y: 1.2, w: 9, h: 2.8,
  border: { pt: 0.5, color: C.lightGray },
  colW: [2.2, 1.0, 1.0, 4.8],
  rowH: [0.45, 0.42, 0.42, 0.42, 0.42, 0.42],
});

// 右侧饼图
s4.addChart(pres.charts.DOUGHNUT, [{
  name: "符号分布",
  labels: ["根目录", "optimization_v2", "ppo_optimize", "Propeller", "scripts"],
  values: [234, 199, 102, 165, 40],
}], {
  x: 5.5, y: 3.5, w: 4, h: 2.0,
  showPercent: true,
  showTitle: false,
  showLegend: true,
  legendPos: "b",
  legendFontSize: 9,
  chartColors: ["0D9488", "F59E0B", "6366F1", "0891B2", "94A3B8"],
});

s4.addText("符号节点分布", {
  x: 5.5, y: 3.2, w: 4, h: 0.35,
  fontSize: 13, fontFace: "Arial", color: C.dark,
  bold: true, align: "center", margin: 0,
});

// 左侧节点类型
s4.addText("节点类型分布", {
  x: 0.5, y: 3.2, w: 4.5, h: 0.35,
  fontSize: 13, fontFace: "Arial", color: C.dark,
  bold: true, align: "left", margin: 0,
});
s4.addChart(pres.charts.BAR, [{
  name: "数量",
  labels: ["import", "variable", "function", "method", "class"],
  values: [283, 186, 158, 89, 25],
}], {
  x: 0.3, y: 3.5, w: 4.8, h: 2.0,
  barDir: "bar",
  showValue: true,
  showLegend: false,
  chartColors: ["0891B2"],
  catAxisLabelColor: C.dark,
  valAxisLabelColor: C.gray,
  valGridLine: { color: C.lightGray, size: 0.5 },
  catGridLine: { style: "none" },
  dataLabelColor: C.dark,
  dataLabelPosition: "outEnd",
  chartArea: { fill: { color: C.white } },
});

// ═══════════════════════════════════════════════
// Slide 5: 核心类影响分析
// ═══════════════════════════════════════════════
let s5 = pres.addSlide();
s5.background = { color: C.navy };
s5.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.teal } });

s5.addText("核心类影响分析", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: C.white,
  bold: true, align: "left", margin: 0,
});
s5.addText("CodeGraph impact 分析 — 修改核心类的波及范围", {
  x: 0.5, y: 0.95, w: 9, h: 0.35,
  fontSize: 14, fontFace: "Arial", color: C.gray,
  align: "left", margin: 0,
});

// PropellerPredictor
const classes = [
  {
    name: "PropellerPredictor",
    pipe: "V1",
    files: 6, symbols: 19,
    affects: ["train.py", "adapter.py", "eval.py", "sweep.py", "run_trim.py", "optimize_trim.py"],
    color: "0D9488",
  },
  {
    name: "MoEPredictor",
    pipe: "V2",
    files: 4, symbols: 15,
    affects: ["moe_model.py", "train_moe.py", "run.py", "eval_ppo.py"],
    color: C.accent,
  },
  {
    name: "SIMULATION",
    pipe: "数据采集",
    files: 2, symbols: 10,
    affects: ["simulation.py", "run_data_gen.py"],
    color: "6366F1",
  },
];

classes.forEach((cls, i) => {
  const x = 0.35 + i * 3.15;
  const w = 2.9;
  s5.addShape(pres.shapes.RECTANGLE, {
    x, y: 1.55, w, h: 3.5,
    fill: { color: C.card },
    line: { color: cls.color, width: 2 },
  });

  // 类名
  s5.addText(cls.name, {
    x: x + 0.15, y: 1.65, w: w - 0.3, h: 0.45,
    fontSize: 16, fontFace: "Consolas", color: cls.color,
    bold: true, align: "left", margin: 0,
  });
  s5.addText(cls.pipe, {
    x: x + 0.15, y: 2.1, w: w - 0.3, h: 0.3,
    fontSize: 11, fontFace: "Arial", color: C.gray,
    align: "left", margin: 0,
  });

  // 数字指标
  s5.addText(String(cls.files), {
    x: x + 0.15, y: 2.5, w: 1.1, h: 0.6,
    fontSize: 28, fontFace: "Arial Black", color: cls.color,
    bold: true, align: "center", valign: "middle", margin: 0,
  });
  s5.addText("文件", {
    x: x + 0.15, y: 3.05, w: 1.1, h: 0.3,
    fontSize: 11, fontFace: "Arial", color: C.gray,
    align: "center", margin: 0,
  });
  s5.addText(String(cls.symbols), {
    x: x + 1.4, y: 2.5, w: 1.2, h: 0.6,
    fontSize: 28, fontFace: "Arial Black", color: cls.color,
    bold: true, align: "center", valign: "middle", margin: 0,
  });
  s5.addText("符号", {
    x: x + 1.4, y: 3.05, w: 1.2, h: 0.3,
    fontSize: 11, fontFace: "Arial", color: C.gray,
    align: "center", margin: 0,
  });

  // 影响文件列表
  s5.addText(cls.affects.join("\n"), {
    x: x + 0.15, y: 3.45, w: w - 0.3, h: 1.4,
    fontSize: 10, fontFace: "Consolas", color: C.lightGray,
    align: "left", margin: 0, lineSpacingMultiple: 1.3,
  });
});

// ═══════════════════════════════════════════════
// Slide 6: 技术栈
// ═══════════════════════════════════════════════
let s6 = pres.addSlide();
s6.background = { color: C.white };

s6.addText("技术栈与工具链", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: C.dark,
  bold: true, align: "left", margin: 0,
});

const techGroups = [
  {
    title: "仿真引擎",
    items: ["QBlade CE 2.0.8.6", "LLFVW 涡格法", "Linux SIL 接口", "ctypes DLL 桥接"],
    color: "0D9488",
  },
  {
    title: "深度学习",
    items: ["PyTorch 2.12", "MLP / MoE 架构", "BatchNorm + Dropout", "异方差 NLL 损失"],
    color: "0891B2",
  },
  {
    title: "优化算法",
    items: ["CMA-ES (cma 4.4)", "PPO (SB3 2.8)", "Gymnasium 环境", "硬约束边界系统"],
    color: "6366F1",
  },
  {
    title: "科学计算",
    items: ["NumPy / SciPy", "scikit-learn", "pandas / openpyxl", "matplotlib 论文级"],
    color: "8B5CF6",
  },
];

techGroups.forEach((g, i) => {
  const x = 0.35 + i * 2.4;
  const w = 2.15;
  s6.addShape(pres.shapes.RECTANGLE, {
    x, y: 1.2, w, h: 3.6,
    fill: { color: C.white },
    line: { color: C.lightGray, width: 1 },
    shadow: makeShadow(),
  });
  s6.addShape(pres.shapes.RECTANGLE, {
    x, y: 1.2, w, h: 0.08,
    fill: { color: g.color },
  });
  s6.addText(g.title, {
    x: x + 0.15, y: 1.45, w: w - 0.3, h: 0.5,
    fontSize: 16, fontFace: "Arial", color: C.dark,
    bold: true, align: "left", margin: 0,
  });

  const textArr = g.items.map((item, j) => ({
    text: item,
    options: { bullet: true, breakLine: j < g.items.length - 1, color: C.gray, fontSize: 12, fontFace: "Arial" },
  }));
  s6.addText(textArr, {
    x: x + 0.15, y: 2.1, w: w - 0.3, h: 2.4,
    margin: 0, paraSpaceAfter: 6,
  });
});

// 底部语言分布
s6.addText("语言分布：Python 41 文件  |  Objective-C 4  |  C 2", {
  x: 0.5, y: 5.0, w: 9, h: 0.4,
  fontSize: 12, fontFace: "Arial", color: C.gray,
  align: "center", margin: 0,
});

// ═══════════════════════════════════════════════
// Slide 7: 设计变量与优化目标
// ═══════════════════════════════════════════════
let s7 = pres.addSlide();
s7.background = { color: C.navy };
s7.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.teal } });

s7.addText("设计变量与优化目标", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: C.white,
  bold: true, align: "left", margin: 0,
});

// 左侧 - 设计变量
s7.addShape(pres.shapes.RECTANGLE, {
  x: 0.4, y: 1.3, w: 4.3, h: 3.8,
  fill: { color: C.card },
  line: { color: C.teal, width: 1 },
});
s7.addText("设计变量", {
  x: 0.6, y: 1.4, w: 3.9, h: 0.5,
  fontSize: 18, fontFace: "Arial", color: C.cyan,
  bold: true, align: "left", margin: 0,
});
s7.addText([
  { text: "8 个 B 样条控制点", options: { bold: true, color: C.white, fontSize: 15, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "4 个 chord 控制点 (弦长分布)", options: { color: C.lightGray, fontSize: 13, breakLine: true } },
  { text: "4 个 twist 控制点 (扭转角分布)", options: { color: C.lightGray, fontSize: 13, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "B-Spline 插值 → 22 截面几何", options: { color: C.gray, fontSize: 12, breakLine: true } },
  { text: "数据驱动边界 CP_BOUNDS_DATA", options: { color: C.gray, fontSize: 12, breakLine: true } },
  { text: "100 个训练几何 + 5% 边距", options: { color: C.gray, fontSize: 12, breakLine: true } },
], {
  x: 0.6, y: 2.0, w: 3.9, h: 2.8,
  margin: 0, lineSpacingMultiple: 1.4,
});

// 右侧 - 优化目标
s7.addShape(pres.shapes.RECTANGLE, {
  x: 5.3, y: 1.3, w: 4.3, h: 3.8,
  fill: { color: C.card },
  line: { color: C.accent, width: 1 },
});
s7.addText("优化目标", {
  x: 5.5, y: 1.4, w: 3.9, h: 0.5,
  fontSize: 18, fontFace: "Arial", color: C.accent,
  bold: true, align: "left", margin: 0,
});
s7.addText([
  { text: "V1: max FM = mg·V / P_total", options: { bold: true, color: C.white, fontSize: 14, breakLine: true } },
  { text: "品质因数最大化", options: { color: C.gray, fontSize: 12, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "V2: min P_elec / V", options: { bold: true, color: C.white, fontSize: 14, breakLine: true } },
  { text: "单位距离电气能耗最小化", options: { color: C.gray, fontSize: 12, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "约束条件:", options: { bold: true, color: C.lightGray, fontSize: 13, breakLine: true } },
  { text: "• RPM ∈ [4000, 6500]", options: { color: C.gray, fontSize: 12, breakLine: true } },
  { text: "• ANGLE ∈ [82°, 88°]", options: { color: C.gray, fontSize: 12, breakLine: true } },
  { text: "• 配平收敛 + 不确定度惩罚", options: { color: C.gray, fontSize: 12, breakLine: true } },
], {
  x: 5.5, y: 2.0, w: 3.9, h: 2.8,
  margin: 0, lineSpacingMultiple: 1.4,
});

// ═══════════════════════════════════════════════
// Slide 8: 结尾
// ═══════════════════════════════════════════════
let s8 = pres.addSlide();
s8.background = { color: C.navy };
s8.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.teal } });
s8.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.565, w: 10, h: 0.06, fill: { color: C.teal } });

s8.addText("Thank You", {
  x: 0.5, y: 1.5, w: 9, h: 1.0,
  fontSize: 48, fontFace: "Arial Black", color: C.cyan,
  bold: true, align: "center", valign: "middle", margin: 0,
});
s8.addText("LFM 螺旋桨气动优化全栈系统", {
  x: 0.5, y: 2.5, w: 9, h: 0.6,
  fontSize: 20, fontFace: "Arial", color: C.white,
  align: "center", margin: 0,
});
s8.addText("CodeGraph: 47 files · 788 nodes · 1,292 edges", {
  x: 0.5, y: 3.3, w: 9, h: 0.4,
  fontSize: 14, fontFace: "Consolas", color: C.gray,
  align: "center", margin: 0,
});

// 底部联系信息
s8.addText("郭越  |  guoguoyue315@gmail.com  |  2026", {
  x: 0.5, y: 4.5, w: 9, h: 0.4,
  fontSize: 12, fontFace: "Arial", color: C.gray,
  align: "center", margin: 0,
});

// 输出
const outPath = "/Users/guoyue/gy_2026/graduation/LFM/docs/LFM_项目介绍.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("✅ PPT 生成完成：" + outPath);
});
