#!/usr/bin/env bash
# ============================================================
# hithesis/hithesisartplus 本地编译环境一键补全
# 适用：macOS + BasicTeX 2026
# 用法：sudo bash install-tex-deps.sh
# ============================================================

set -e

if [ "$EUID" -ne 0 ]; then
  echo "需要 sudo 运行：sudo bash $0"
  exit 1
fi

echo "===> 1) 更新 tlmgr 自身"
tlmgr update --self

echo "===> 2) 安装 hithesis 直接缺失的宏包"
# 来自源码扫描 + 之前编译报错
DIRECT_PKGS=(
  ntheorem        # hithesis.sty 必需
  changepage
  environ
  glossaries
  layouts
  lipsum
  multirow
  newtxmath
  siunitx
  subfigure
  tikzpagenodes
  varwidth
  tex-gyre        # TeX Gyre 基础西文字体族
  tex-gyre-math   # 配套数学字体
  newtx           # TeXGyreTermesX 扩展字体（hithesisbook \setmainfont 必需）
  bigfoot         # perpage.sty（hithesis 脚注每页重新编号）
  gbt7714         # GB/T 7714-2015 中文参考文献样式
  placeins        # 浮动体屏障 \FloatBarrier
  ccaption        # 续表/续图标题
  splitidx        # 分索引（hithesisbook 内部使用）
  titlesec        # 章节标题格式化
  tabu            # 增强表格
  enumitem        # 增强 enumerate
  mdframed        # 带框文字
  threeparttable  # 三段式表格
  makecell        # 表格单元格控制
  array           # 表格列样式
  xstring         # 字符串处理
  xpatch          # 命令补丁
  zhnumber        # 中文数字
  xpatch          # xparse 命令补丁
)
tlmgr install "${DIRECT_PKGS[@]}"

echo "===> 3) 安装中文论文常用集合（一次到位，避免后续逐个补包）"
COLLECTIONS=(
  collection-latexrecommended    # latex 常用宏包
  collection-fontsrecommended    # 常用字体
  collection-xetex               # XeLaTeX 引擎与扩展
  collection-langchinese         # 中文支持（ctex、xeCJK、fandol 等）
  collection-mathscience         # 数学/科学常用宏包
  collection-pictures            # tikz、pgf 等绘图
  collection-bibtexextra         # 参考文献扩展（含 hithesis.bst 依赖）
)
tlmgr install "${COLLECTIONS[@]}" || echo "（部分集合已存在，正常）"

echo "===> 4) 刷新字体与文件名数据库"
mktexlsr || true
fc-cache -fv 2>/dev/null | tail -3 || true

echo "===> 5) 验证关键包"
MISSING=0
for pkg in ntheorem ctex xeCJK fandol newtxmath ${DIRECT_PKGS[@]}; do
  if ! kpsewhich ${pkg}.sty >/dev/null 2>&1 && ! kpsewhich ${pkg}.cls >/dev/null 2>&1; then
    echo "  ✗ 仍然缺失: $pkg"
    MISSING=$((MISSING+1))
  fi
done

if [ $MISSING -eq 0 ]; then
  echo ""
  echo "✓ 环境准备完成，现在可以本地编译："
  echo "  cd ~/gy_2026/latex-thesis/midterm-report"
  echo "  latexmk -xelatex report.tex"
else
  echo ""
  echo "⚠️  仍有 $MISSING 个包缺失，请检查网络或手动安装。"
fi
