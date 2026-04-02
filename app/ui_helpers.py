from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def window_output_frame(meta: pd.DataFrame, window_features: pd.DataFrame, micro: pd.Series, macro: pd.Series) -> pd.DataFrame:
    out = meta.copy().reset_index(drop=True)
    out["micro_state"] = micro.values
    out["macro_state"] = macro.values

    mean_cols = [c for c in window_features.columns if c.endswith("_mean")]
    for col in mean_cols:
        out[col] = window_features[col].values

    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    out["duration_hours"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 3600.0
    return out


def normalize_input_columns(df: pd.DataFrame, station_col: str, timestamp_col: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    if station_col not in df.columns and station_col.lower() in col_map:
        df = df.rename(columns={col_map[station_col.lower()]: station_col})
    if timestamp_col not in df.columns and timestamp_col.lower() in col_map:
        df = df.rename(columns={col_map[timestamp_col.lower()]: timestamp_col})
    def _key(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", s).upper()
    key_map = {_key(c): c for c in df.columns}
    station_key = _key(station_col)
    time_key = _key(timestamp_col)
    if station_col not in df.columns and station_key in key_map:
        df = df.rename(columns={key_map[station_key]: station_col})
    if timestamp_col not in df.columns and time_key in key_map:
        df = df.rename(columns={key_map[time_key]: timestamp_col})
    if station_col not in df.columns:
        df[station_col] = "STATION_0"
    return df


def top_feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["WIND_SPEED", "WAVE_HGT", "SEA_LVL_PRES", "AIR_TEMP", "SWELL_HGT", "WAVE_PERIOD"]
    selected = []
    for p in preferred:
        cols = [c for c in df.columns if c.startswith(f"{p}_") and c.endswith("_mean")]
        if cols:
            selected.append(sorted(cols)[0])
    return selected[:8]


def build_regime_notes(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = top_feature_columns(df)
    if not feature_cols:
        return pd.DataFrame()

    grp = df.groupby("macro_state")
    stats = grp[feature_cols].mean()
    stats["avg_duration_hr"] = grp["duration_hours"].mean()

    note_rows = []
    for rid, row in stats.iterrows():
        hints = []
        for col in feature_cols[:3]:
            hints.append(f"{col}: {row[col]:.2f}")
        note_rows.append(
            {
                "macro_state": int(rid),
                "avg_duration_hr": float(row["avg_duration_hr"]),
                "profile_hint": " | ".join(hints),
                "interpretation": f"Marine regime {int(rid)}",
            }
        )
    return pd.DataFrame(note_rows).sort_values("macro_state")


def first_mean_col(df: pd.DataFrame, prefix: str) -> str | None:
    cols = [c for c in df.columns if c.startswith(f"{prefix}_") and c.endswith("_mean")]
    return sorted(cols)[0] if cols else None


def risk_snapshot(df: pd.DataFrame) -> tuple[str, str]:
    wave_col = first_mean_col(df, "WAVE_HGT")
    wind_col = first_mean_col(df, "WIND_SPEED")
    if wave_col is None and wind_col is None:
        return "Unknown", "No WAVE_HGT/WIND_SPEED mean features available."

    recent = df.tail(max(20, min(len(df), 120)))
    score = 0.0
    details = []
    if wave_col is not None:
        wave_val = float(recent[wave_col].mean())
        score += wave_val
        details.append(f"wave={wave_val:.2f}")
    if wind_col is not None:
        wind_val = float(recent[wind_col].mean())
        score += wind_val / 15.0
        details.append(f"wind={wind_val:.2f}")

    if score >= 4.5:
        return "High", "Rough/storm tendency in recent windows (" + ", ".join(details) + ")."
    if score >= 2.5:
        return "Medium", "Moderate marine variability (" + ", ".join(details) + ")."
    return "Low", "Relatively calm marine behavior (" + ", ".join(details) + ")."


def _normalize_score(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    lo = float(vals.min())
    hi = float(vals.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-9:
        return pd.Series([0.0] * len(series), index=series.index)
    return (vals - lo) / (hi - lo)


def macro_severity_map(df: pd.DataFrame) -> pd.DataFrame:
    wave_col = first_mean_col(df, "WAVE_HGT")
    wind_col = first_mean_col(df, "WIND_SPEED")
    if wave_col is None and wind_col is None:
        return pd.DataFrame()

    grp = df.groupby("macro_state")
    summary = pd.DataFrame(index=grp.size().index)
    if wave_col is not None:
        summary["wave_mean"] = grp[wave_col].mean()
    if wind_col is not None:
        summary["wind_mean"] = grp[wind_col].mean()

    score = pd.Series(0.0, index=summary.index)
    if "wave_mean" in summary.columns:
        score += 0.7 * _normalize_score(summary["wave_mean"])
    if "wind_mean" in summary.columns:
        score += 0.3 * _normalize_score(summary["wind_mean"])
    summary["severity_score"] = score

    def _level(val: float) -> str:
        if val >= 0.66:
            return "High"
        if val >= 0.33:
            return "Medium"
        return "Low"

    summary["severity_level"] = summary["severity_score"].apply(_level)
    summary = summary.reset_index().rename(columns={"index": "macro_state"})
    return summary.sort_values("severity_score", ascending=False)


def next_macro_probabilities(
    hmm_model: object,
    current_micro: int,
    macro_mapping: dict[int, int] | None,
) -> pd.DataFrame:
    trans = np.asarray(getattr(hmm_model, "transmat_", np.array([])), dtype=float)
    if trans.ndim != 2 or trans.shape[0] == 0:
        return pd.DataFrame()
    if current_micro < 0 or current_micro >= trans.shape[0]:
        return pd.DataFrame()

    p_next_micro = trans[current_micro]
    if macro_mapping is None:
        probs = {int(i): float(p_next_micro[i]) for i in range(len(p_next_micro))}
    else:
        probs: dict[int, float] = {}
        for micro_id, prob in enumerate(p_next_micro):
            macro_id = int(macro_mapping.get(int(micro_id), int(micro_id)))
            probs[macro_id] = probs.get(macro_id, 0.0) + float(prob)

    out = pd.DataFrame(
        [{"macro_state": int(k), "probability": float(v)} for k, v in probs.items()]
    ).sort_values("probability", ascending=False)
    return out.reset_index(drop=True)


def infer_macro_names(df: pd.DataFrame) -> dict[int, str]:
    if "macro_state" not in df.columns or df.empty:
        return {}
    macro_ids = sorted(int(v) for v in df["macro_state"].dropna().unique())
    if not macro_ids:
        return {}

    wave_col = first_mean_col(df, "WAVE_HGT")
    wind_col = first_mean_col(df, "WIND_SPEED")
    pres_col = first_mean_col(df, "SEA_LVL_PRES")

    grp = df.groupby("macro_state")
    summary = pd.DataFrame(index=macro_ids)
    if wave_col is not None:
        summary["wave"] = grp[wave_col].mean()
    if wind_col is not None:
        summary["wind"] = grp[wind_col].mean()
    if pres_col is not None:
        summary["pres"] = grp[pres_col].mean()

    names: dict[int, str] = {}
    for rid, row in summary.iterrows():
        label = f"Regime {int(rid)}"
        if "wave" in row and row["wave"] == summary["wave"].max():
            label = "Storm-like"
        elif "wind" in row and row["wind"] == summary["wind"].max():
            label = "Windy"
        elif "pres" in row and row["pres"] == summary["pres"].min():
            label = "Low-pressure"
        names[int(rid)] = label
    return names


def station_early_warning(
    df: pd.DataFrame,
    station_col: str,
    hmm_model: object,
    macro_mapping: dict[int, int] | None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    wave_col = first_mean_col(df, "WAVE_HGT")
    wind_col = first_mean_col(df, "WIND_SPEED")
    if wave_col is None and wind_col is None:
        return pd.DataFrame()

    severity = macro_severity_map(df)
    if severity.empty:
        return pd.DataFrame()
    severity_map = severity.set_index("macro_state")
    high_macros = set(severity_map[severity_map["severity_level"] == "High"].index.tolist())

    if "end_time" not in df.columns:
        return pd.DataFrame()

    idx = df.groupby(station_col)["end_time"].idxmax()
    latest = df.loc[idx].copy()
    latest["macro_state"] = latest["macro_state"].astype(int)
    latest["current_severity"] = latest["macro_state"].map(severity_map["severity_score"]).fillna(0.0)
    latest["current_level"] = latest["macro_state"].map(severity_map["severity_level"]).fillna("Unknown")

    prob_high_next = []
    for _, row in latest.iterrows():
        cur_micro = int(row.get("micro_state", -1))
        probs = next_macro_probabilities(hmm_model, cur_micro, macro_mapping)
        if probs.empty or not high_macros:
            prob_high_next.append(0.0)
        else:
            prob_high_next.append(float(probs[probs["macro_state"].isin(high_macros)]["probability"].sum()))
    latest["prob_high_next"] = prob_high_next

    latest["risk_score"] = 0.7 * latest["current_severity"] + 0.3 * latest["prob_high_next"]

    def _risk_level(val: float) -> str:
        if val >= 0.66:
            return "High"
        if val >= 0.33:
            return "Medium"
        return "Low"

    latest["risk_level"] = latest["risk_score"].apply(_risk_level)
    if wave_col is not None:
        latest["wave_mean"] = latest[wave_col]
    if wind_col is not None:
        latest["wind_mean"] = latest[wind_col]

    def _explain(row: pd.Series) -> str:
        parts = [f"regime={int(row['macro_state'])} ({row['current_level']})"]
        if wave_col is not None:
            parts.append(f"wave={float(row['wave_mean']):.2f}")
        if wind_col is not None:
            parts.append(f"wind={float(row['wind_mean']):.2f}")
        parts.append(f"next-high-prob={float(row['prob_high_next']):.2f}")
        return " | ".join(parts)

    latest["explanation"] = latest.apply(_explain, axis=1)
    keep_cols = [station_col, "end_time", "macro_state", "risk_level", "risk_score", "prob_high_next", "explanation"]
    if wave_col is not None:
        keep_cols.insert(4, "wave_mean")
    if wind_col is not None:
        keep_cols.insert(5 if wave_col is not None else 4, "wind_mean")

    return latest[keep_cols].sort_values(["risk_score", "prob_high_next"], ascending=False)


def monthly_regime_shares(
    df: pd.DataFrame, station_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty or "end_time" not in df.columns or "macro_state_name" not in df.columns:
        empty = pd.DataFrame()
        return empty, empty, empty

    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["end_time"], errors="coerce").dt.to_period("M").astype(str)
    tmp = tmp.dropna(subset=["month"])
    if tmp.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    counts = tmp.groupby(["month", "macro_state_name"]).size().reset_index(name="count")
    counts["share"] = counts["count"] / counts.groupby("month")["count"].transform("sum")
    share_pivot = counts.pivot(index="month", columns="macro_state_name", values="share").fillna(0.0)
    share_pivot = (share_pivot * 100.0).round(2)

    dominant_idx = counts.groupby("month")["share"].idxmax()
    dominant = counts.loc[dominant_idx].copy()
    dominant = dominant.rename(columns={"share": "dominant_share"}).sort_values("month")

    station_counts = tmp.groupby([station_col, "month", "macro_state_name"]).size().reset_index(name="count")
    station_counts["share"] = station_counts["count"] / station_counts.groupby([station_col, "month"])["count"].transform("sum")
    station_idx = station_counts.groupby([station_col, "month"])["share"].idxmax()
    station_dominant = station_counts.loc[station_idx].copy().sort_values([station_col, "month"])
    station_dominant = station_dominant.rename(columns={"share": "dominant_share"})

    return share_pivot, dominant, station_dominant


def operational_planning_summary(
    df: pd.DataFrame,
    station_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "end_time" not in df.columns or "macro_state" not in df.columns:
        empty = pd.DataFrame()
        return empty, empty

    severity = macro_severity_map(df)
    if severity.empty:
        empty = pd.DataFrame()
        return empty, empty
    severity_map = severity.set_index("macro_state")

    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["end_time"], errors="coerce").dt.to_period("M").astype(str)
    tmp = tmp.dropna(subset=["month"])
    if tmp.empty:
        empty = pd.DataFrame()
        return empty, empty

    tmp["severity_level"] = tmp["macro_state"].map(severity_map["severity_level"]).fillna("Unknown")
    tmp["severity_score"] = tmp["macro_state"].map(severity_map["severity_score"]).fillna(0.0)

    overall = (
        tmp.groupby("month")
        .agg(
            low_share=("severity_level", lambda s: float((s == "Low").mean())),
            med_share=("severity_level", lambda s: float((s == "Medium").mean())),
            high_share=("severity_level", lambda s: float((s == "High").mean())),
            avg_severity=("severity_score", "mean"),
            windows=("severity_level", "size"),
        )
        .reset_index()
    )
    overall["recommended"] = (overall["low_share"] >= 0.6).map({True: "Yes", False: "No"})
    overall["low_share"] = (overall["low_share"] * 100.0).round(2)
    overall["med_share"] = (overall["med_share"] * 100.0).round(2)
    overall["high_share"] = (overall["high_share"] * 100.0).round(2)
    overall["avg_severity"] = overall["avg_severity"].round(3)

    by_station = (
        tmp.groupby([station_col, "month"])
        .agg(
            low_share=("severity_level", lambda s: float((s == "Low").mean())),
            high_share=("severity_level", lambda s: float((s == "High").mean())),
            avg_severity=("severity_score", "mean"),
            windows=("severity_level", "size"),
        )
        .reset_index()
    )
    by_station["low_share"] = (by_station["low_share"] * 100.0).round(2)
    by_station["high_share"] = (by_station["high_share"] * 100.0).round(2)
    by_station["avg_severity"] = by_station["avg_severity"].round(3)
    by_station["recommended"] = (by_station["low_share"] >= 60.0) & (by_station["high_share"] <= 15.0)
    by_station["recommended"] = by_station["recommended"].map({True: "Yes", False: "No"})

    return overall.sort_values("month"), by_station.sort_values([station_col, "month"])


def sensor_health_report(
    df: pd.DataFrame,
    station_col: str,
    timestamp_col: str,
    numeric_columns: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    cols = [c for c in numeric_columns if c in df.columns]
    if not cols:
        return pd.DataFrame()

    rows = []
    for station, grp in df.groupby(station_col, sort=False):
        g = grp.sort_values(timestamp_col)
        total = len(g)
        if total == 0:
            continue

        missing_rate = float(g[cols].isna().mean().mean())

        flat_scores = []
        for c in cols:
            series = g[c].astype(float)
            diffs = series.diff()
            valid = diffs.notna().sum()
            if valid == 0:
                flat_scores.append(0.0)
            else:
                flat_scores.append(float((diffs == 0).sum() / valid))
        flatline_rate = float(np.mean(flat_scores)) if flat_scores else 0.0

        spike_scores = []
        for c in cols:
            series = g[c].astype(float)
            med = float(series.median())
            mad = float(np.median(np.abs(series - med)))
            if not np.isfinite(mad) or mad <= 1e-9:
                spike_scores.append(0.0)
                continue
            z = np.abs(series - med) / (1.4826 * mad)
            spike_scores.append(float((z > 6.0).mean()))
        spike_rate = float(np.mean(spike_scores)) if spike_scores else 0.0

        gap_rate = 0.0
        if timestamp_col in g.columns:
            t = pd.to_datetime(g[timestamp_col], errors="coerce").dropna()
            if len(t) >= 3:
                deltas = t.sort_values().diff().dropna().dt.total_seconds()
                med_gap = float(deltas.median()) if not deltas.empty else 0.0
                if med_gap > 0:
                    gap_rate = float((deltas > 2.5 * med_gap).mean())

        score = 1.0 - (0.45 * missing_rate + 0.25 * flatline_rate + 0.2 * spike_rate + 0.1 * gap_rate)
        score = float(max(0.0, min(1.0, score)))

        if score >= 0.75:
            status = "Good"
        elif score >= 0.5:
            status = "Warning"
        else:
            status = "Critical"

        rows.append(
            {
                station_col: station,
                "rows": total,
                "missing_rate": round(missing_rate, 4),
                "flatline_rate": round(flatline_rate, 4),
                "spike_rate": round(spike_rate, 4),
                "gap_rate": round(gap_rate, 4),
                "health_score": round(score, 4),
                "status": status,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["health_score", "missing_rate"], ascending=[True, False])
