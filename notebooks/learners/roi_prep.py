import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Tuple


@dataclass
class PrepConfig:
    """Configuration for converting the raw ROI dataframe to an analysis-ready format."""

    subject_col: str = "subject_code"
    session_id_col: str = "session_id"  # expects YYYYMMDDHHMM as int/str
    condition_col: str = "condition"  # e.g., Learning / Professional / Control
    learner_flag_col: str = "learner"  # boolean fallback if condition is missing
    group_col: str = "group"  # e.g., Climbing / Bjj / ...
    age_col: str = "age_at_scan"
    sex_col: str = "sex"
    value_col: str = "value"  # outcome for this ROI
    # Role labels used downstream
    role_labels: Tuple[str, str, str, str] = ("Learner", "Athlete", "Control", "Other")
    # Whether to restrict output to Learner/Athlete/Control only
    restrict_roles: bool = True
    # Minimum scans per subject for inclusion (set to 1 to keep everyone)
    min_scans_per_subject: int = 1
    # If True, baseline-center time so baseline per subject is zero
    center_time_by_subject: bool = True


def _parse_session_id(x) -> Optional[pd.Timestamp]:
    """Parse session_id formatted like YYYYMMDDHHMM (int or str) to pandas Timestamp."""
    if pd.isna(x):
        return pd.NaT
    try:
        s = str(int(x))
    except Exception:
        s = str(x).strip()
    if len(s) == 10:  # YYYYMMDDHH (pad minutes)
        s = s + "00"
    try:
        return pd.Timestamp(datetime.strptime(s, "%Y%m%d%H%M"))
    except Exception:
        return pd.NaT


def _infer_role(
    row: pd.Series, condition_col: str, learner_flag_col: str, labels: Tuple[str, str, str, str]
) -> str:
    """Return one of (Learner, Athlete, Control, Other) based on condition/learner flag."""
    learner_label, athlete_label, control_label, other_label = labels
    cond = str(row.get(condition_col, "")).strip()
    if cond == "Learning":
        return learner_label
    if cond == "Professional":
        return athlete_label
    if cond == "Control":
        return athlete_label
    # fallback via boolean flag
    if learner_flag_col in row and bool(row[learner_flag_col]):
        return learner_label
    return other_label


def _infer_sport(group_val: object) -> str:
    """Map 'group' to sport category."""
    g = str(group_val).strip().lower()
    return g


def prepare_roi_longitudinal(df: pd.DataFrame, config: PrepConfig = PrepConfig()) -> pd.DataFrame:
    """
    Convert a raw single-ROI dataframe to an analysis-ready long format.

    Output columns:
        subject_id : str
        scan_dt    : pandas.Timestamp
        time_days  : float (days since first scan for that subject)
        time_months: float (months since first scan for that subject)
        role       : {'Learner','Athlete','Control','Other'}
        sport      : {'Climb','BJJ','None'}
        age_at_scan: float
        sex        : original coding (recommended to treat as categorical)
        y          : outcome value (from config.value_col)

    Filtering:
        - If config.restrict_roles: keep only Learner/Athlete/Control
        - Keep subjects with >= config.min_scans_per_subject scans (after prior filtering)

    Notes:
        - Time is baseline-centered per subject if config.center_time_by_subject=True
        - session_id is parsed as YYYYMMDDHHMM; if unparsable, rows are dropped
    """
    d = df.copy()

    # Make required columns visible with standard names
    missing = [
        c
        for c in [config.subject_col, config.session_id_col, config.value_col]
        if c not in d.columns
    ]
    if missing:
        raise KeyError(f"Missing required columns in input df: {missing}")

    d["subject_code"] = d[config.subject_col].astype(str)
    d["scan_dt"] = d[config.session_id_col].apply(_parse_session_id)

    # Drop rows without parseable timestamp or outcome
    d = d.dropna(subset=["subject_code", "scan_dt", config.value_col]).copy()

    # Time since baseline
    if config.center_time_by_subject:
        first_dt = d.groupby("subject_code")["scan_dt"].transform("min")
        d["time_days"] = (d["scan_dt"] - first_dt).dt.days.astype(float)
    else:
        # absolute time in days since the earliest date in the dataset
        global_min = d["scan_dt"].min()
        d["time_days"] = (d["scan_dt"] - global_min).dt.days.astype(float)
    d["time_months"] = d["time_days"] / 30.4375

    # Role & sport
    d["condition"] = d.apply(
        _infer_role,
        axis=1,
        condition_col=config.condition_col,
        learner_flag_col=config.learner_flag_col,
        labels=config.role_labels,
    )
    d["sport"] = (
        d[config.group_col].apply(_infer_sport) if config.group_col in d.columns else "None"
    )

    # Covariates (if present)
    if config.age_col in d.columns:
        d["age_at_scan"] = pd.to_numeric(d[config.age_col], errors="coerce")
    else:
        d["age_at_scan"] = np.nan

    if config.sex_col in d.columns:
        d["sex"] = d[config.sex_col]
    else:
        d["sex"] = np.nan

    # Outcome
    d["y"] = pd.to_numeric(d[config.value_col], errors="coerce")

    # --- Tag pre/middle/post per subject ---
    d = d.sort_values(["subject_code", "scan_dt"]).copy()
    d["tp_index"] = d.groupby("subject_code").cumcount()  # 0,1,2,...
    d["tp_count"] = d.groupby("subject_code")["scan_dt"].transform("size")  # total scans

    # Label logic:
    # - 1 scan  -> 'pre'
    # - 2 scans -> indices 0='pre', 1='post'
    # - >=3     -> 0='pre', last='post', others='middle'
    d["tp_label"] = np.where(
        d["tp_count"] == 1,
        "pre",
        np.where(
            d["tp_index"] == 0,
            "pre",
            np.where(d["tp_index"] == d["tp_count"] - 1, "post", "middle"),
        ),
    )
    d["tp_label"] = pd.Categorical(
        d["tp_label"], categories=["pre", "middle", "post"], ordered=True
    )

    # Keep relevant columns
    keep = [
        "subject_code",
        "scan_dt",
        "tp_label",
        "time_days",
        "time_months",
        "condition",
        "sport",
        "age_at_scan",
        "sex",
        "tiv",
        "y",
    ]
    tidy = d[[i for i in keep if i in d.columns]].dropna(subset=["y", "scan_dt"]).copy()

    # Restrict roles if desired
    if config.restrict_roles:
        wanted = set(config.role_labels[:3])  # Learner, Athlete, Control
        tidy = tidy[tidy["condition"].isin(wanted)].copy()

    # Minimum scans per subject
    if config.min_scans_per_subject > 1:
        cnt = tidy.groupby("subject_code").size()
        good_subjects = set(cnt[cnt >= config.min_scans_per_subject].index)
        tidy = tidy[tidy["subject_code"].isin(good_subjects)].copy()

    # Sort for convenience
    tidy = tidy.sort_values(["subject_code", "scan_dt"]).reset_index(drop=True)
    return tidy


def prepost_ancova_table(
    tidy: pd.DataFrame,
    id_col: str = "subject_code",
    time_col: str = "time_months",
    y_col: str = "y",
    keep: str = "last",
    carry_cols: Iterable[str] = ("condition", "sport", "age_at_scan", "sex"),
) -> pd.DataFrame:
    """
    Build an ANCOVA-style table: y_post ~ y_pre + covariates.
    Keeps subjects with at least 2 timepoints. Uses the earliest as pre and
    either the latest ('keep'='last') or the 2nd ('keep'='second') as post.

    Returns columns: id_col, y_pre, y_post, time_post_months, <carry_cols...>
    """
    if keep not in {"last", "second"}:
        raise ValueError("keep must be 'last' or 'second'.")

    g = tidy.sort_values([id_col, time_col]).groupby(id_col, as_index=False)
    first = g.nth(0)
    post = g.nth(-1) if keep == "last" else g.nth(1)

    have_prepost = first.merge(post, on=id_col, suffixes=("_pre", "_post"))
    # Require distinct times
    have_prepost = have_prepost[
        have_prepost[f"{time_col}_post"] > have_prepost[f"{time_col}_pre"]
    ].copy()

    out = pd.DataFrame(
        {
            id_col: have_prepost[id_col],
            "y_pre": have_prepost[f"{y_col}_pre"],
            "y_post": have_prepost[f"{y_col}_post"],
            "time_post_months": have_prepost[f"{time_col}_post"],
        }
    )

    # Carry covariates from the post row (common in ANCOVA)
    for c in carry_cols:
        c_post = f"{c}_post"
        if c_post in have_prepost.columns:
            out[c] = have_prepost[c_post]
        elif c in have_prepost.columns:
            # fallback if suffix didn't exist (constant across time)
            out[c] = have_prepost[c]
        else:
            out[c] = np.nan

    return out


def delta_change_table(
    tidy: pd.DataFrame,
    id_col: str = "subject_code",
    time_col: str = "time_months",
    y_col: str = "y",
    carry_cols: Iterable[str] = ("condition", "sport", "age_at_scan", "sex"),
) -> pd.DataFrame:
    """
    Build a change-score table: delta = y_post - y_pre.
    Uses first and last timepoints per subject.

    Returns: id_col, y_pre, y_post, delta, months_between, <carry_cols...> (from post)
    """
    g = tidy.sort_values([id_col, time_col]).groupby(id_col, as_index=False)
    pre = g.nth(0)
    post = g.nth(-1)

    have_prepost = pre.merge(post, on=id_col, suffixes=("_pre", "_post"))
    have_prepost = have_prepost[
        have_prepost[f"{time_col}_post"] > have_prepost[f"{time_col}_pre"]
    ].copy()

    out = pd.DataFrame(
        {
            id_col: have_prepost[id_col],
            "y_pre": have_prepost[f"{y_col}_pre"],
            "y_post": have_prepost[f"{y_col}_post"],
            "delta": have_prepost[f"{y_col}_post"] - have_prepost[f"{y_col}_pre"],
            "months_between": have_prepost[f"{time_col}_post"] - have_prepost[f"{time_col}_pre"],
        }
    )
    # Carry covariates from post
    for c in carry_cols:
        c_post = f"{c}_post"
        if c_post in have_prepost.columns:
            out[c] = have_prepost[c_post]
        elif c in have_prepost.columns:
            out[c] = have_prepost[c]
        else:
            out[c] = np.nan

    return out
