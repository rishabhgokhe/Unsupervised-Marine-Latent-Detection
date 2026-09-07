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
        for col in feature_cols[:4]:
            val = float(row[col])
            # Scale NOAA tenths if applicable for intuitive reading
            if "SEA_LVL_PRES" in col and val > 2000.0:
                val = val / 10.0
                hints.append(f"Pressure: {val:.1f} hPa")
            elif "WIND_SPEED" in col and val > 30.0:
                val = val / 10.0
                hints.append(f"Wind: {val:.1f} m/s")
            elif "AIR_TEMP" in col and val > 50.0:
                val = val / 10.0
                hints.append(f"Air Temp: {val:.1f} °C")
            elif "WAVE_HGT" in col:
                hints.append(f"Wave: {val:.2f} m")
            else:
                short_name = col.split("_s")[0] if "_s" in col else col
                hints.append(f"{short_name}: {val:.2f}")

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


def scale_telemetry_val(val: float | int | None, param_type: str, col_max: float | None = None) -> float:
    if val is None or not np.isfinite(val):
        return 0.0
    v = float(val)
    if param_type == "wind":
        return v / 10.0 if (col_max is not None and col_max > 25.0) or v > 20.0 else v
    if param_type == "wave":
        return v / 10.0 if (col_max is not None and col_max > 12.0) or v > 10.0 else v
    if param_type == "pressure":
        return v / 10.0 if v > 2000.0 else v
    return v


def risk_snapshot(df: pd.DataFrame) -> tuple[str, str]:
    wave_col = first_mean_col(df, "WAVE_HGT")
    wind_col = first_mean_col(df, "WIND_SPEED")
    pres_col = first_mean_col(df, "SEA_LVL_PRES")

    if wave_col is None and wind_col is None and pres_col is None:
        return "Unknown", "No environmental telemetry available."

    recent = df.tail(max(20, min(len(df), 120)))
    score = 0.0
    details = []

    w_max = df[wind_col].max() if wind_col and wind_col in df.columns else 0.0
    wv_max = df[wave_col].max() if wave_col and wave_col in df.columns else 0.0

    if wind_col is not None and wind_col in recent.columns:
        valid_wind = recent[recent[wind_col] > 0][wind_col]
        raw_wind = float(valid_wind.mean()) if not valid_wind.empty else 0.0
        wind_display = scale_telemetry_val(raw_wind, "wind", w_max)
        w_sev = float(np.clip(wind_display / 18.0, 0.0, 1.0) ** 1.2)
        score += w_sev * 1.5
        details.append(f"wind={wind_display:.1f} m/s")

    if wave_col is not None and wave_col in recent.columns and recent[wave_col].max() > 0.05:
        valid_wave = recent[recent[wave_col] > 0][wave_col]
        wave_val = float(valid_wave.mean()) if not valid_wave.empty else 0.0
        wave_display = scale_telemetry_val(wave_val, "wave", wv_max)
        wv_sev = float(np.clip(wave_display / 3.5, 0.0, 1.0))
        score += wv_sev * 1.5
        details.append(f"wave={wave_display:.1f} m")

    if pres_col is not None and pres_col in recent.columns:
        valid_pres = recent[recent[pres_col] > 5000][pres_col]
        if not valid_pres.empty:
            raw_pres = float(valid_pres.mean())
            pres_display = scale_telemetry_val(raw_pres, "pressure")
            if pres_display < 1008.0:
                score += float(np.clip((1008.0 - pres_display) / 20.0, 0.0, 1.0))
                details.append(f"pres={pres_display:.1f} hPa")

    if score >= 1.5:
        return "High", "Rough/storm conditions in recent windows (" + ", ".join(details) + ")."
    if score >= 0.7:
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
    pres_col = first_mean_col(df, "SEA_LVL_PRES")
    if wave_col is None and wind_col is None and pres_col is None:
        return pd.DataFrame()

    w_max = df[wind_col].max() if wind_col and wind_col in df.columns else 0.0
    wv_max = df[wave_col].max() if wave_col and wave_col in df.columns else 0.0

    macro_states = sorted(int(v) for v in df["macro_state"].dropna().unique())
    macro_rows = []
    for rid in macro_states:
        sub = df[df["macro_state"] == rid]
        w = scale_telemetry_val(sub[wind_col].mean(), "wind", w_max) if wind_col and wind_col in sub.columns else 0.0
        wv = scale_telemetry_val(sub[wave_col].mean(), "wave", wv_max) if wave_col and wave_col in sub.columns else 0.0
        p_valid = sub[sub[pres_col] > 5000][pres_col] if pres_col and pres_col in sub.columns else pd.Series(dtype=float)
        p = scale_telemetry_val(p_valid.mean(), "pressure") if not p_valid.empty else 1013.25

        w_sev = float(np.clip(w / 18.0, 0.0, 1.0) ** 1.2) if w > 0 else 0.0
        wv_sev = float(np.clip(wv / 4.0, 0.0, 1.0)) if wv > 0.05 else 0.0
        p_sev = float(np.clip((1008.0 - p) / 25.0, 0.0, 1.0)) if p < 1008.0 else 0.0

        score = float(np.clip(0.55 * w_sev + 0.35 * wv_sev + 0.10 * p_sev, 0.0, 1.0))
        level = "High" if score >= 0.60 else ("Medium" if score >= 0.30 else "Low")
        macro_rows.append({
            "macro_state": int(rid),
            "wind_mean": round(w, 2),
            "wave_mean": round(wv, 2),
            "pres_mean": round(p, 1),
            "severity_score": round(score, 3),
            "severity_level": level,
        })

    res = pd.DataFrame(macro_rows)
    return res.sort_values("severity_score", ascending=False).reset_index(drop=True)


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

    w_max = df[wind_col].max() if wind_col and wind_col in df.columns else 0.0
    wv_max = df[wave_col].max() if wave_col and wave_col in df.columns else 0.0

    names: dict[int, str] = {}
    for rid in macro_ids:
        sub = df[df["macro_state"] == rid]
        w = scale_telemetry_val(sub[wind_col].mean(), "wind", w_max) if wind_col and wind_col in sub.columns else 0.0
        wv = scale_telemetry_val(sub[wave_col].mean(), "wave", wv_max) if wave_col and wave_col in sub.columns else 0.0
        p_valid = sub[sub[pres_col] > 5000][pres_col] if pres_col and pres_col in sub.columns else pd.Series(dtype=float)
        p = scale_telemetry_val(p_valid.mean(), "pressure") if not p_valid.empty else 1013.25

        if w >= 14.0 or wv >= 3.0:
            names[rid] = "Storm / Rough Sea"
        elif w >= 7.0 or wv >= 1.5:
            names[rid] = "Moderate / Swell"
        elif p < 1005.0 and p > 850.0:
            names[rid] = "Low-Pressure Front"
        elif w <= 0.2 and wv <= 0.05 and p >= 1020.0:
            names[rid] = "Calm / High-Pressure"
        elif w <= 0.2 and wv <= 0.05:
            names[rid] = "Calm / Sparse Telemetry"
        else:
            names[rid] = "Calm / Fair Sea"

    counts = {}
    for rid, name in names.items():
        counts[name] = counts.get(name, 0) + 1
    if any(c > 1 for c in counts.values()):
        for rid in macro_ids:
            nm = names[rid]
            if counts[nm] > 1:
                names[rid] = f"{nm} (Regime {rid})"

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
    pres_col = first_mean_col(df, "SEA_LVL_PRES")
    if wave_col is None and wind_col is None and pres_col is None:
        return pd.DataFrame()

    severity = macro_severity_map(df)
    if severity.empty:
        return pd.DataFrame()
    severity_map = severity.set_index("macro_state")
    high_macros = set(severity[severity["severity_level"] == "High"]["macro_state"].tolist())
    med_macros = set(severity[severity["severity_level"] == "Medium"]["macro_state"].tolist())

    if "end_time" not in df.columns:
        return pd.DataFrame()

    idx = df.groupby(station_col)["end_time"].idxmax()
    latest = df.loc[idx].copy()
    latest["macro_state"] = latest["macro_state"].astype(int)
    latest["current_severity"] = latest["macro_state"].map(severity_map["severity_score"]).fillna(0.0)
    latest["current_level"] = latest["macro_state"].map(severity_map["severity_level"]).fillna("Low")

    w_max = df[wind_col].max() if wind_col and wind_col in df.columns else 0.0
    wv_max = df[wave_col].max() if wave_col and wave_col in df.columns else 0.0

    recon_col = "reconstruction_error" if "reconstruction_error" in df.columns else None
    recon_p90 = float(df[recon_col].quantile(0.90)) if recon_col else None

    warn_rows = []
    for _, row in latest.iterrows():
        st_id = row[station_col]
        rid = int(row["macro_state"])
        cur_micro = int(row.get("micro_state", -1))

        w = scale_telemetry_val(row.get(wind_col, 0.0), "wind", w_max) if wind_col else 0.0
        wv = scale_telemetry_val(row.get(wave_col, 0.0), "wave", wv_max) if wave_col else 0.0
        raw_p = row.get(pres_col, 0.0) if pres_col else 0.0
        p = scale_telemetry_val(raw_p, "pressure") if raw_p > 5000.0 else 1013.25

        w_sev = float(np.clip(w / 18.0, 0.0, 1.0) ** 1.2) if w > 0 else 0.0
        wv_sev = float(np.clip(wv / 4.0, 0.0, 1.0)) if wv > 0.05 else 0.0
        p_sev = float(np.clip((1008.0 - p) / 25.0, 0.0, 1.0)) if p < 1008.0 else 0.0
        phys_sev = float(np.clip(0.60 * w_sev + 0.30 * wv_sev + 0.10 * p_sev, 0.0, 1.0))

        probs = next_macro_probabilities(hmm_model, cur_micro, macro_mapping)
        p_high = 0.0
        if not probs.empty and high_macros:
            p_high = float(probs[probs["macro_state"].isin(high_macros)]["probability"].sum())
        elif not probs.empty and med_macros:
            p_high = 0.4 * float(probs[probs["macro_state"].isin(med_macros)]["probability"].sum())

        anom = 0.0
        if recon_col and recon_p90 and recon_p90 > 0:
            r_val = float(row.get(recon_col, 0.0))
            if r_val > recon_p90:
                anom = float(np.clip((r_val - recon_p90) / (2.0 * recon_p90), 0.0, 0.20))

        regime_sev = float(severity_map.loc[rid, "severity_score"]) if rid in severity_map.index else 0.0
        risk = float(np.clip(0.50 * phys_sev + 0.25 * regime_sev + 0.20 * p_high + anom, 0.0, 1.0))
        risk_lvl = "High" if risk >= 0.60 else ("Medium" if risk >= 0.30 else "Low")

        parts = [f"regime={rid} ({row['current_level']})"]
        if wave_col is not None:
            parts.append(f"wave={wv:.2f} m")
        if wind_col is not None:
            parts.append(f"wind={w:.1f} m/s")
        if p < 1008.0:
            parts.append(f"pres={p:.1f} hPa")
        parts.append(f"next-high-prob={p_high:.2f}")

        warn_rows.append({
            station_col: st_id,
            "end_time": row["end_time"],
            "macro_state": rid,
            "risk_level": risk_lvl,
            "wave_mean": round(wv, 2),
            "wind_mean": round(w, 2),
            "risk_score": round(risk, 3),
            "prob_high_next": round(p_high, 3),
            "explanation": " | ".join(parts),
        })

    result_df = pd.DataFrame(warn_rows)
    keep_cols = [station_col, "end_time", "macro_state", "risk_level", "risk_score", "prob_high_next", "explanation"]
    if wave_col is not None:
        keep_cols.insert(4, "wave_mean")
    if wind_col is not None:
        keep_cols.insert(5 if wave_col is not None else 4, "wind_mean")

    return result_df[keep_cols].sort_values(["risk_score", "prob_high_next"], ascending=False).reset_index(drop=True)


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

    st_col = station_col if station_col in tmp.columns else ("station" if "station" in tmp.columns else "STATION")

    counts = tmp.groupby(["month", "macro_state_name"]).size().reset_index(name="count")
    counts["share"] = counts["count"] / counts.groupby("month")["count"].transform("sum")
    share_pivot = counts.pivot(index="month", columns="macro_state_name", values="share").fillna(0.0)
    share_pivot = (share_pivot * 100.0).round(2)

    dominant_idx = counts.groupby("month")["share"].idxmax()
    dominant = counts.loc[dominant_idx].copy()
    dominant = dominant.rename(columns={"share": "dominant_share"}).sort_values("month")

    if st_col in tmp.columns:
        station_counts = tmp.groupby([st_col, "month", "macro_state_name"]).size().reset_index(name="count")
        station_counts["share"] = station_counts["count"] / station_counts.groupby([st_col, "month"])["count"].transform("sum")
        station_idx = station_counts.groupby([st_col, "month"])["share"].idxmax()
        station_dominant = station_counts.loc[station_idx].copy().sort_values([st_col, "month"])
        station_dominant = station_dominant.rename(columns={"share": "dominant_share"})
    else:
        station_dominant = pd.DataFrame()

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

    st_col = station_col if station_col in tmp.columns else ("station" if "station" in tmp.columns else "STATION")

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

    if st_col in tmp.columns:
        by_station = (
            tmp.groupby([st_col, "month"])
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
        by_station = by_station.sort_values([st_col, "month"])
    else:
        by_station = pd.DataFrame()

    return overall.sort_values("month"), by_station


def sensor_health_report(
    df: pd.DataFrame,
    station_col: str,
    timestamp_col: str,
    numeric_columns: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    st_col = station_col if station_col in df.columns else ("station" if "station" in df.columns else "STATION")
    cols = [c for c in numeric_columns if c in df.columns]
    if not cols:
        return pd.DataFrame()

    rows = []
    for station, grp in df.groupby(st_col, sort=False):
        g = grp.sort_values(timestamp_col)
        total = len(g)
        if total == 0:
            continue

        # Only evaluate sensors installed on this buoy (having at least 1 reading)
        active_cols = [c for c in cols if g[c].notna().sum() > 0]
        if not active_cols:
            active_cols = cols

        missing_rate = float(g[active_cols].isna().mean().mean())

        flat_scores = []
        for c in active_cols:
            series = g[c].dropna().astype(float)
            if len(series) < 2:
                continue
            diffs = series.diff().dropna()
            flat_scores.append(float((diffs == 0).mean()))
        flatline_rate = float(np.mean(flat_scores)) if flat_scores else 0.0

        spike_scores = []
        for c in active_cols:
            series = g[c].dropna().astype(float)
            if len(series) < 5:
                continue
            med = float(series.median())
            mad = float(np.median(np.abs(series - med)))
            if not np.isfinite(mad) or mad <= 1e-6:
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
        elif score >= 0.50:
            status = "Warning"
        else:
            status = "Critical"

        rows.append(
            {
                st_col: station,
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


def compute_latent_pca(latent: np.ndarray) -> np.ndarray | None:
    """Projects latent embeddings (N, D) into 2D via PCA for manifold visualization."""
    if latent is None or len(latent) < 3 or latent.ndim != 2:
        return None
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(latent)
    except Exception:
        return None


def extract_station_coordinates(df: pd.DataFrame, station_col: str) -> pd.DataFrame:
    """
    Extracts geographical coordinates (latitude, longitude) for each station.
    Matches case-insensitively on station_col, LATITUDE, LONGITUDE.
    Returns a DataFrame with [station_col, 'latitude', 'longitude'].
    """
    if df.empty:
        return pd.DataFrame(columns=[station_col, "latitude", "longitude"])

    actual_st_col = None
    for c in [station_col, "station", "STATION"]:
        if c in df.columns:
            actual_st_col = c
            break
    if actual_st_col is None:
        col_map = {c.lower(): c for c in df.columns}
        actual_st_col = col_map.get(station_col.lower())

    if actual_st_col is None:
        return pd.DataFrame(columns=[station_col, "latitude", "longitude"])

    lat_col = next((c for c in df.columns if c.upper() in ["LATITUDE", "LAT"]), None)
    lon_col = next((c for c in df.columns if c.upper() in ["LONGITUDE", "LON", "LONG"]), None)

    if lat_col is None or lon_col is None:
        return pd.DataFrame(columns=[station_col, "latitude", "longitude"])

    valid = df.dropna(subset=[lat_col, lon_col])
    if valid.empty:
        return pd.DataFrame(columns=[station_col, "latitude", "longitude"])

    geo = (
        valid.groupby(actual_st_col)[[lat_col, lon_col]]
        .last()
        .reset_index()
        .rename(columns={actual_st_col: station_col, lat_col: "latitude", lon_col: "longitude"})
    )
    geo["latitude"] = pd.to_numeric(geo["latitude"], errors="coerce")
    geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
    geo = geo.dropna(subset=["latitude", "longitude"])
    geo = geo[(geo["latitude"].between(-90.0, 90.0)) & (geo["longitude"].between(-180.0, 180.0))]
    return geo.reset_index(drop=True)

