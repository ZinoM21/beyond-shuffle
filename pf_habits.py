from typing import Any, Dict, List, Tuple
import calendar

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from constants import (
    NUMERICAL_FEATURES_TO_CHECK,
    HABIT_MIN_WEEKS,
    HABIT_MIN_STREAMS_PER_SLOT,
    HABIT_FEATURE_ZSCORE_THRESHOLD,
    HABIT_MIN_NUM_FEATURES,
    HABIT_MAX_SLOTS_PER_SCHEMA,
    HABIT_TOP_PLATFORMS,
    HABIT_AUDIO_CLUSTER_K,
    HABIT_BEHAVIORAL_FEATURES,
    HABIT_BEHAVIORAL_ZSCORE_THRESHOLD,
    HABIT_MIN_DEVICE_SHARE,
)


def slot_key_builders() -> Dict[str, Any]:
    builders: Dict[str, Any] = {}

    def ensure_hour(out: pd.DataFrame) -> pd.DataFrame:
        if "hour" not in out.columns:
            out["hour"] = (
                out.index.hour
                if isinstance(out.index, pd.DatetimeIndex)
                else pd.to_datetime(out["datetime"]).dt.hour
            )
        return out

    def dow_hour(d: pd.DataFrame) -> pd.DataFrame:
        out = ensure_hour(d.copy())
        out["_slot"] = list(zip(out["day_of_week"], out["hour"]))
        return out

    builders["dow_hour"] = dow_hour

    def dow_hour_platform(d: pd.DataFrame) -> pd.DataFrame:
        out = dow_hour(d)
        if "platform_group" in out.columns:
            out["_slot"] = list(
                zip(out["day_of_week"], out["hour"], out["platform_group"])
            )
        elif "platform" in out.columns:
            top = (
                out["platform"].value_counts().head(HABIT_TOP_PLATFORMS).index.tolist()
            )
            out["_platform_lim"] = out["platform"].where(
                out["platform"].isin(top), other="Other"
            )
            out["_slot"] = list(
                zip(out["day_of_week"], out["hour"], out["_platform_lim"])
            )
        return out

    builders["dow_hour_platform"] = dow_hour_platform

    def season_hour(d: pd.DataFrame) -> pd.DataFrame:
        out = ensure_hour(d.copy())
        out["_slot"] = list(
            zip(
                out.get("season", pd.Series(index=out.index, dtype=object)), out["hour"]
            )
        )
        return out

    builders["season_hour"] = season_hour

    def dow_platform(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        if "platform_group" in out.columns:
            out["_slot"] = list(zip(out["day_of_week"], out["platform_group"]))
        elif "platform" in out.columns:
            top = (
                out["platform"].value_counts().head(HABIT_TOP_PLATFORMS).index.tolist()
            )
            out["_platform_lim"] = out["platform"].where(
                out["platform"].isin(top), other="Other"
            )
            out["_slot"] = list(zip(out["day_of_week"], out["_platform_lim"]))
        return out

    builders["dow_platform"] = dow_platform

    def month_hour(d: pd.DataFrame) -> pd.DataFrame:
        out = ensure_hour(d.copy())
        out["_slot"] = list(
            zip(out.get("month", pd.Series(index=out.index, dtype=int)), out["hour"])
        )
        return out

    builders["month_hour"] = month_hour

    def country_hour(d: pd.DataFrame) -> pd.DataFrame:
        out = ensure_hour(d.copy())
        if "country" in out.columns:
            valid = out["country"].notna() & (out["country"] != "ZZ")
            out = out[valid]
            out["_slot"] = list(zip(out["country"], out["hour"]))
        return out

    builders["country_hour"] = country_hour

    def dow_country(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        if "country" in out.columns:
            valid = out["country"].notna() & (out["country"] != "ZZ")
            out = out[valid]
            out["_slot"] = list(zip(out["day_of_week"], out["country"]))
        return out

    builders["dow_country"] = dow_country

    def hour_platform(d: pd.DataFrame) -> pd.DataFrame:
        out = ensure_hour(d.copy())
        if "platform_group" in out.columns:
            out["_slot"] = list(zip(out["hour"], out["platform_group"]))
        elif "platform" in out.columns:
            top = (
                out["platform"].value_counts().head(HABIT_TOP_PLATFORMS).index.tolist()
            )
            out["_platform_lim"] = out["platform"].where(
                out["platform"].isin(top), other="Other"
            )
            out["_slot"] = list(zip(out["hour"], out["_platform_lim"]))
        return out

    builders["hour_platform"] = hour_platform

    return builders


def compute_slot_feature_stats(df: pd.DataFrame, slot_col: str) -> pd.DataFrame:
    num_cols = [f for f in NUMERICAL_FEATURES_TO_CHECK if f in df.columns]
    beh_cols = [f for f in HABIT_BEHAVIORAL_FEATURES if f in df.columns]
    overall = df[num_cols + beh_cols].agg(["mean", "std"]).T

    df = df.copy()
    df["_week"] = (
        df.index.isocalendar().week
        if isinstance(df.index, pd.DatetimeIndex)
        else pd.to_datetime(df["datetime"]).dt.isocalendar().week
    ).astype(int)
    gb = df.groupby(slot_col)
    slot_counts = gb.size().rename("count")
    week_counts = gb["_week"].nunique().rename("weeks")
    slot_means = gb[num_cols + beh_cols].mean()

    for feat in num_cols + beh_cols:
        mu = overall.loc[feat, "mean"] if feat in overall.index else 0.0
        sigma = (
            overall.loc[feat, "std"]
            if feat in overall.index and overall.loc[feat, "std"] not in (0, None)
            else 0.0
        )
        slot_means[f"{feat}_z"] = (slot_means[feat] - mu) / sigma if sigma else 0.0

    stats = (
        pd.concat([slot_means, slot_counts, week_counts], axis=1)
        .reset_index()
        .rename(columns={slot_col: "_slot"})
    )
    return stats


def select_habit_slots(stats: pd.DataFrame) -> List[Tuple]:
    if stats.empty:
        return []

    def deviating_feature_count(row: pd.Series) -> int:
        count = 0
        for feat in NUMERICAL_FEATURES_TO_CHECK:
            z = row.get(f"{feat}_z", 0.0)
            if pd.notna(z) and abs(z) >= HABIT_FEATURE_ZSCORE_THRESHOLD:
                count += 1
        for feat in HABIT_BEHAVIORAL_FEATURES:
            z = row.get(f"{feat}_z", 0.0)
            if pd.notna(z) and abs(z) >= HABIT_BEHAVIORAL_ZSCORE_THRESHOLD:
                count += 1
        return count

    stats = stats.copy()
    stats["_num_dev_feats"] = stats.apply(deviating_feature_count, axis=1)
    eligible = stats[
        (stats["count"] >= HABIT_MIN_STREAMS_PER_SLOT)
        & (stats["weeks"] >= HABIT_MIN_WEEKS)
        & (stats["_num_dev_feats"] >= HABIT_MIN_NUM_FEATURES)
    ]
    eligible = eligible.sort_values(
        ["_num_dev_feats", "count"], ascending=[False, False]
    )
    return eligible["_slot"].head(HABIT_MAX_SLOTS_PER_SCHEMA).tolist()


def cluster_audio_profiles(df: pd.DataFrame) -> pd.Series:
    features = [f for f in NUMERICAL_FEATURES_TO_CHECK if f in df.columns]
    if len(features) < 2:
        return pd.Series(data=np.full(len(df), -1, dtype=int), index=df.index)
    mask = df[features].notna().all(axis=1)
    if mask.sum() < HABIT_AUDIO_CLUSTER_K * 3:
        return pd.Series(data=np.full(len(df), -1, dtype=int), index=df.index)
    X = df.loc[mask, features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=HABIT_AUDIO_CLUSTER_K, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    labels_full = np.full(len(df), -1, dtype=int)
    labels_full[np.flatnonzero(mask.to_numpy())] = labels.astype(int)
    return pd.Series(labels_full, index=df.index)


def format_slot_name(schema: str, slot_key: Tuple) -> str:
    if schema == "dow_hour":
        dow, hour = slot_key
        return f"{str(dow)} at {int(hour):02d}:00"
    if schema == "dow_hour_platform":
        dow, hour, platform = slot_key
        if str(platform).lower() == "other":
            return f"{str(dow)} at {int(hour):02d}:00"
        return f"{str(dow)} at {int(hour):02d}:00 on {platform}"
    if schema == "season_hour":
        season, hour = slot_key
        return f"{season} at {int(hour):02d}:00"
    if schema == "dow_platform":
        dow, platform = slot_key
        if str(platform).lower() == "other":
            return f"{str(dow)}"
        return f"{str(dow)} on {platform}"
    if schema == "month_hour":
        month, hour = slot_key
        month_name = (
            calendar.month_name[int(month)]
            if pd.notna(month) and int(month) in range(1, 13)
            else str(month)
        )
        return f"{month_name} at {int(hour):02d}:00"
    if schema == "country_hour":
        country, hour = slot_key
        return f"{country} at {int(hour):02d}:00"
    if schema == "dow_country":
        dow, country = slot_key
        return f"{str(dow)} in {country}"
    if schema == "hour_platform":
        hour, platform = slot_key
        if str(platform).lower() == "other":
            return f"{int(hour):02d}:00"
        return f"{int(hour):02d}:00 on {platform}"
    return str(slot_key)


def refine_slot_name_with_device(df_slot: pd.DataFrame, base_name: str) -> str:
    name = base_name
    device_col = (
        "platform_group"
        if "platform_group" in df_slot.columns
        else ("platform" if "platform" in df_slot.columns else None)
    )
    if device_col is None:
        return name
    counts = df_slot[device_col].value_counts(normalize=True)
    if counts.empty:
        return name
    top_device = counts.index[0]
    top_share = float(counts.iloc[0])
    if top_share >= HABIT_MIN_DEVICE_SHARE:
        if str(top_device).lower() == "other":
            if len(counts) > 1:
                second_device = counts.index[1]
                second_share = float(counts.iloc[1])
                if (
                    second_share >= HABIT_MIN_DEVICE_SHARE / 2
                    and str(second_device).lower() != "other"
                ):
                    return (
                        name
                        if f" on {second_device}" in name
                        else f"{name} on {second_device}"
                    )
            return name
        return name if f" on {top_device}" in name else f"{name} on {top_device}"
    return name
