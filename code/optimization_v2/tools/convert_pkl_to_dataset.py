"""pkl → V2 训练数据集转换器

用法:
    python -m optimization_v2.tools.convert_pkl_to_dataset \
        --src ./data_for_train/data_lhs \
        --dst ./data_for_train/processed_lhs \
        --formats csv xlsx parquet

输入: data_lhs/ 目录下的 SIMULATION dump (格式B)
    {
      "geometry_<idx>": {
          "geometry": ndarray(22, 3),
          "RPM<v>_Wind<v>_Angle<v>": DataFrame[1000 timesteps, ...],
          ...
      },
      ...
    }

输出: 每条样本一行的扁平表，列结构:
    [RPM, WIND, ANGLE, chord_0..21, twist_0..21, T, H, My, Q,
     geom_idx, source_file]

聚合策略: 优先使用 pinned QBlade wrapper 写入的 UAV-positive aliases；
          仅有 raw QBlade 列时，对末段 N 步均值显式取负。
"""

import argparse
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


CASE_PATTERN = re.compile(r"RPM([\d.]+)_Wind([\d.]+)_Angle([\d.]+)")
N_SECTIONS = 22
CHORD_COLS = [f"chord_{i}" for i in range(N_SECTIONS)]
TWIST_COLS = [f"twist_{i}" for i in range(N_SECTIONS)]
SECTION_COLS = CHORD_COLS + TWIST_COLS
CONDITION_COLS = ["RPM", "WIND", "ANGLE"]
OUTPUT_COLS = ["T", "H", "My", "Q"]

QBlade_ALIAS_SPEC = {
    "T": "THRUST",
    "H": "THRUST_Z",
    "My": "MY",
    "Q": "TORQUE",
}
QBlade_RAW_SPEC = {
    "T": "Thrust",
    "H": "Thrust_z",
    "My": "My",
    "Q": "Torque",
}


def aggregate_timeseries(df: pd.DataFrame, last_n: int) -> tuple[dict, str]:
    """Extract four UAV-positive outputs without mixing sign conventions.

    The pinned wrapper persists all four upper-case aliases after applying the
    UAV-positive sign convention.  If any alias is present, all four are
    required.  Legacy frames containing only the four raw QBlade columns are
    converted using the wrapper's exact ``-mean(raw)`` rule.
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or last_n <= 0:
        return {}, "invalid_timeseries"
    tail = df.iloc[-last_n:] if len(df) >= last_n else df
    alias_columns = set(QBlade_ALIAS_SPEC.values())
    present_aliases = alias_columns.intersection(df.columns)
    if present_aliases:
        if present_aliases != alias_columns:
            return {}, "invalid_partial_qblade_aliases"
        out = {
            target: float(pd.to_numeric(tail[column], errors="coerce").iloc[-1])
            for target, column in QBlade_ALIAS_SPEC.items()
        }
        if not all(np.isfinite(value) for value in out.values()):
            return {}, "invalid_qblade_uav_positive_aliases"
        return out, "qblade_uav_positive_aliases"

    raw_columns = set(QBlade_RAW_SPEC.values())
    if not raw_columns.issubset(df.columns):
        return {}, "invalid_missing_output_columns"
    out = {
        target: -float(pd.to_numeric(tail[column], errors="coerce").mean())
        for target, column in QBlade_RAW_SPEC.items()
    }
    if not all(np.isfinite(value) for value in out.values()):
        return {}, "invalid_legacy_raw_outputs"
    return out, "legacy_raw_converted"


def _aggregate_timeseries(df: pd.DataFrame, last_n: int) -> dict:
    """Backward-compatible value-only wrapper used by older callers."""
    return aggregate_timeseries(df, last_n)[0]


def _split_geometry(geometry: np.ndarray) -> dict:
    """geometry ndarray (22, 3 或类似) → {chord_0..21, twist_0..21}。

    约定: 第 1 列为 chord, 第 2 列为 twist (与既有 pipeline 一致)。
    """
    g = np.asarray(geometry)
    if g.ndim != 2 or g.shape[0] < N_SECTIONS:
        return {}
    # 一些 pkl 把 geometry 存成 (22, 3): [radius, chord, twist]
    if g.shape[1] >= 3:
        chord = g[:N_SECTIONS, 1]
        twist = g[:N_SECTIONS, 2]
    elif g.shape[1] == 2:
        chord = g[:N_SECTIONS, 0]
        twist = g[:N_SECTIONS, 1]
    else:
        return {}
    out = {f"chord_{i}": float(chord[i]) for i in range(N_SECTIONS)}
    out.update({f"twist_{i}": float(twist[i]) for i in range(N_SECTIONS)})
    return out


def _extract_geom_idx(geo_key: str, fname: str) -> int:
    """从 'geometry_<n>' 或文件名末尾的 b<NNNN> 提取索引,失败返回 -1。"""
    m = re.search(r"geometry_(\d+)", geo_key)
    if m:
        return int(m.group(1))
    m = re.search(r"b(\d{4})", fname)
    if m:
        return int(m.group(1))
    return -1


def convert_directory(src_dir: Path, last_n: int, verbose: bool) -> pd.DataFrame:
    """扫描 src_dir 所有 pkl, 返回扁平 DataFrame。"""
    pkl_files = sorted(p for p in src_dir.rglob("*.pkl") if p.name.lower().startswith(("data", "sample_")))
    if not pkl_files:
        raise FileNotFoundError(f"{src_dir} 下无 data*.pkl / sample_*.pkl")

    records = []
    skipped = []
    n_geom_ok = 0
    n_geom_seen = 0

    for fpath in pkl_files:
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            skipped.append({"file": fpath.name, "case": "-", "reason": f"读取失败: {e}"})
            continue

        if not isinstance(data, dict):
            skipped.append({"file": fpath.name, "case": "-", "reason": f"非 dict: {type(data).__name__}"})
            continue

        # 兼容 格式B (含 geometry_<n> 键) + 格式C (顶层即工况)
        if any(k.startswith("geometry") for k in data.keys()):
            geo_items = [(k, v) for k, v in data.items() if isinstance(v, dict)]
        else:
            geo_items = [(f"geometry_{_extract_geom_idx('', fpath.name)}", data)]

        for geo_key, geo_val in geo_items:
            n_geom_seen += 1
            geom_idx = _extract_geom_idx(geo_key, fpath.name)
            geometry = geo_val.get("geometry")
            geom_cols = _split_geometry(geometry) if geometry is not None else {}
            if not geom_cols:
                skipped.append({"file": fpath.name, "case": geo_key, "reason": "几何缺失或形状异常"})
                continue

            cases_added = 0
            for case_key, case_df in geo_val.items():
                m = CASE_PATTERN.match(case_key)
                if not m:
                    continue
                if not isinstance(case_df, pd.DataFrame):
                    skipped.append({"file": fpath.name, "case": case_key, "reason": "非 DataFrame"})
                    continue
                rpm, wind, angle = [float(x) for x in m.groups()]
                out_cols, output_sign_source = aggregate_timeseries(case_df, last_n)
                if not out_cols:
                    skipped.append({"file": fpath.name, "case": case_key, "reason": "时序无效/缺列"})
                    continue
                row = {
                    "RPM": rpm, "WIND": wind, "ANGLE": angle,
                    **geom_cols, **out_cols,
                    "geom_idx": geom_idx,
                    "geom_id": geom_idx,
                    "output_sign_source": output_sign_source,
                    "source_file": fpath.name,
                }
                records.append(row)
                cases_added += 1

            if cases_added > 0:
                n_geom_ok += 1

    if not records:
        raise ValueError(f"扫描 {len(pkl_files)} 个 pkl 后未提取到有效行")

    df = pd.DataFrame(records)
    # 强制列顺序
    fixed = CONDITION_COLS + SECTION_COLS + OUTPUT_COLS + [
        "geom_idx", "geom_id", "output_sign_source", "source_file"
    ]
    df = df[fixed]

    if verbose:
        print(f"[扫描] pkl 文件: {len(pkl_files)}")
        print(f"[扫描] 几何样本: {n_geom_seen} 见到 / {n_geom_ok} 成功")
        print(f"[扫描] 工况记录: {len(df)} 条")
        print(f"[扫描] 跳过: {len(skipped)} 条")
    return df, skipped


def save_dataset(df: pd.DataFrame, dst_dir: Path, formats: list, tag: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = dst_dir / f"dataset_{tag}"

    if "csv" in formats:
        path = base.with_suffix(".csv")
        df.to_csv(path, index=False)
        print(f"[落盘] CSV   → {path}  ({path.stat().st_size/1024:.1f} KB)")

    if "xlsx" in formats:
        path = base.with_suffix(".xlsx")
        # xlsx 不支持过宽,主表 + meta sheet
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df.drop(columns=["source_file"]).to_excel(w, sheet_name="data", index=False)
            meta = pd.DataFrame({
                "n_rows": [len(df)],
                "n_geom": [df["geom_idx"].nunique()],
                "rpm_range": [f"{df.RPM.min()}~{df.RPM.max()}"],
                "angle_range": [f"{df.ANGLE.min()}~{df.ANGLE.max()}"],
                "wind_range": [f"{df.WIND.min()}~{df.WIND.max()}"],
            })
            meta.to_excel(w, sheet_name="meta", index=False)
        print(f"[落盘] XLSX  → {path}  ({path.stat().st_size/1024:.1f} KB)")

    if "parquet" in formats:
        path = base.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        print(f"[落盘] PARQ  → {path}  ({path.stat().st_size/1024:.1f} KB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="pkl 数据目录, 如 data_for_train/data_lhs")
    p.add_argument("--dst", required=True, help="输出目录, 如 data_for_train/processed_lhs")
    p.add_argument("--formats", nargs="+", default=["csv", "xlsx"],
                   choices=["csv", "xlsx", "parquet"])
    p.add_argument("--last-n", type=int, default=120,
                   help="时序末段平均的步数 (默认 120, 对应 1000 步的最后 12%)")
    p.add_argument("--tag", default=None, help="输出文件名标签, 默认用时间戳")
    args = p.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    tag = args.tag or time.strftime("v2_%Y%m%d")

    t0 = time.time()
    df, skipped = convert_directory(src, args.last_n, verbose=True)

    save_dataset(df, dst, args.formats, tag)

    if skipped:
        log = dst / f"skipped_{tag}.csv"
        pd.DataFrame(skipped).to_csv(log, index=False, encoding="utf-8-sig")
        print(f"[落盘] 跳过日志 → {log}")

    print(f"[done] 共 {len(df)} 行, 耗时 {time.time()-t0:.1f}s")
    print(f"       唯一几何 {df['geom_idx'].nunique()} 个, 工况/几何 ≈ {len(df)/max(1,df['geom_idx'].nunique()):.1f}")


if __name__ == "__main__":
    main()
