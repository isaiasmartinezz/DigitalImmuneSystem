import pathlib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- CONFIG ----------

DEFAULT_WW_PATH = "data/synthetic_wastewater.csv"
DEFAULT_HOSP_PATH = "data/synthetic_hospital.csv"
DEFAULT_ENV_PATH = "data/synthetic_environment.csv"

DATE_COL = "date"
WW_COL = "synthetic_viral_load"
HOSP_COL = "synthetic_admissions"
ENV_COL = "synthetic_env"

# Human-readable names for plotting / UI
WW_LABEL = "Wastewater viral load"
HOSP_LABEL = "Hospital admissions"
ENV_LABEL = "Air-quality index"

WW_Z_LABEL = "Wastewater z-score"
HOSP_Z_LABEL = "Hospital z-score"
ENV_Z_LABEL = "Environment z-score"
FUSED_LABEL = "Fused risk score"


# ---------- HELPERS ----------

def load_default_csv(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        st.error(f"Default file not found: {p}")
        return pd.DataFrame()
    return pd.read_csv(p)


def infer_or_select_date_col(df: pd.DataFrame, label: str) -> str:
    """
    Try to guess the date column; if ambiguous, ask the user.
    """
    candidates = [c for c in df.columns if "date" in c.lower()]
    if len(candidates) == 1:
        date_col = candidates[0]
    elif DATE_COL in df.columns:
        date_col = DATE_COL
    else:
        date_col = st.selectbox(
            f"Select date column for {label}",
            options=df.columns,
            key=f"{label}_datecol",
        )
    return date_col


def compute_rolling_z(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    s = df[col].astype(float)
    mean = s.rolling(window, min_periods=max(3, window // 2)).mean()
    std = s.rolling(window, min_periods=max(3, window // 2)).std()
    z = (s - mean) / std
    return z


def fuse_risk(df: pd.DataFrame, ww_col: str, hosp_col: str, env_col: str,
              ww_w: float, hosp_w: float, env_w: float) -> pd.Series:
    total = ww_w + hosp_w + env_w
    if total <= 0:
        ww_w = hosp_w = env_w = 0.0
    else:
        ww_w /= total
        hosp_w /= total
        env_w /= total

    return (
        ww_w * df[ww_col].fillna(0.0)
        + hosp_w * df[hosp_col].fillna(0.0)
        + env_w * df[env_col].fillna(0.0)
    )


def label_alert_tier(risk: float, med: float, high: float) -> str:
    if risk >= high:
        return "High"
    elif risk >= med:
        return "Medium"
    elif risk >= 0:
        return "Low"
    else:
        return "Baseline"


# ---------- APP ----------

def main():
    st.set_page_config(
        page_title="Digital Immune System Dashboard",
        layout="wide",
    )

    st.title("🛡️ Digital Immune System Dashboard")
    st.write(
        "Upload or use default synthetic datasets for **wastewater**, "
        "**hospital admissions**, and **environment (air quality)**. "
        "The dashboard will compute rolling z-scores, a fused risk index, "
        "and recommended alert tiers."
    )

    # ---- SIDEBAR: DATA INPUT ----
    st.sidebar.header("1. Data sources")

    use_defaults = st.sidebar.checkbox(
        "Use default synthetic CSVs from /data",
        value=True,
    )

    if use_defaults:
        ww_df = load_default_csv(DEFAULT_WW_PATH)
        hosp_df = load_default_csv(DEFAULT_HOSP_PATH)
        env_df = load_default_csv(DEFAULT_ENV_PATH)
        st.sidebar.success("Loaded default synthetic datasets.")
    else:
        ww_file = st.sidebar.file_uploader("Wastewater CSV", type="csv")
        hosp_file = st.sidebar.file_uploader("Hospital CSV", type="csv")
        env_file = st.sidebar.file_uploader("Environment CSV", type="csv")

        ww_df = pd.read_csv(ww_file) if ww_file is not None else pd.DataFrame()
        hosp_df = pd.read_csv(hosp_file) if hosp_file is not None else pd.DataFrame()
        env_df = pd.read_csv(env_file) if env_file is not None else pd.DataFrame()

    if ww_df.empty or hosp_df.empty or env_df.empty:
        st.warning("Waiting for all three datasets. Upload files or enable defaults.")
        st.stop()

    # ---- SIDEBAR: PARAMETERS ----
    st.sidebar.header("2. Detection parameters")

    window = st.sidebar.slider("Rolling window (days)", 7, 60, 28, step=1)

    ww_weight = st.sidebar.slider("Weight: Wastewater", 0.0, 1.0, 0.4, 0.05)
    hosp_weight = st.sidebar.slider("Weight: Hospital", 0.0, 1.0, 0.4, 0.05)
    env_weight = st.sidebar.slider("Weight: Environment", 0.0, 1.0, 0.2, 0.05)

    medium_thresh = st.sidebar.slider(
        "Medium alert threshold (z-score)", 0.0, 4.0, 1.5, 0.1
    )
    high_thresh = st.sidebar.slider(
        "High alert threshold (z-score)", 1.0, 6.0, 3.0, 0.1
    )

    # ---- PREP DATA ----
    # Ensure all have a date column
    ww_date_col = infer_or_select_date_col(ww_df, "wastewater")
    hosp_date_col = infer_or_select_date_col(hosp_df, "hospital")
    env_date_col = infer_or_select_date_col(env_df, "environment")

    ww_df[DATE_COL] = pd.to_datetime(ww_df[ww_date_col]).dt.date
    hosp_df[DATE_COL] = pd.to_datetime(hosp_df[hosp_date_col]).dt.date
    env_df[DATE_COL] = pd.to_datetime(env_df[env_date_col]).dt.date

    # Try to find value columns if not exact names
    def pick_value_col(df, preferred, label):
        if preferred in df.columns:
            return preferred
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            st.error(f"No numeric column found for {label}.")
            st.stop()
        if len(numeric_cols) == 1:
            return numeric_cols[0]
        return st.sidebar.selectbox(
            f"Choose value column for {label}",
            options=numeric_cols,
            key=f"{label}_valcol",
        )

    ww_val_col = pick_value_col(ww_df, WW_COL, "wastewater")
    hosp_val_col = pick_value_col(hosp_df, HOSP_COL, "hospital")
    env_val_col = pick_value_col(env_df, ENV_COL, "environment")

    # Merge on date
    merged = (
        ww_df[[DATE_COL, ww_val_col]]
        .merge(hosp_df[[DATE_COL, hosp_val_col]], on=DATE_COL, how="inner")
        .merge(env_df[[DATE_COL, env_val_col]], on=DATE_COL, how="inner")
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    merged.rename(
        columns={
            ww_val_col: WW_COL,
            hosp_val_col: HOSP_COL,
            env_val_col: ENV_COL,
        },
        inplace=True,
    )

    # Compute z-scores
    merged["ww_z"] = compute_rolling_z(merged, WW_COL, window)
    merged["hosp_z"] = compute_rolling_z(merged, HOSP_COL, window)
    merged["env_z"] = compute_rolling_z(merged, ENV_COL, window)

    # Compute fused risk
    merged["fused_risk"] = fuse_risk(
        merged, "ww_z", "hosp_z", "env_z",
        ww_weight, hosp_weight, env_weight
    )

    # Alert tiers
    merged["alert_tier"] = merged["fused_risk"].apply(
        lambda r: label_alert_tier(r, med=medium_thresh, high=high_thresh)
    )
    merged["is_alert"] = merged["alert_tier"].isin(["Medium", "High"])

    # ---- TOP SUMMARY ----
    st.subheader("Summary")

    total_days = len(merged)
    num_high = (merged["alert_tier"] == "High").sum()
    num_med = (merged["alert_tier"] == "Medium").sum()
    num_low = (merged["alert_tier"] == "Low").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Days in dataset", total_days)
    col2.metric("High alert days", num_high)
    col3.metric("Medium alert days", num_med)
    col4.metric("Low alert days", num_low)

    # Prepare plotting dataframe with human-friendly labels
    trend_df = merged.copy()
    trend_df["date_ts"] = pd.to_datetime(trend_df[DATE_COL])

    plot_df = trend_df.rename(
        columns={
            WW_COL: WW_LABEL,
            HOSP_COL: HOSP_LABEL,
            ENV_COL: ENV_LABEL,
            "ww_z": WW_Z_LABEL,
            "hosp_z": HOSP_Z_LABEL,
            "env_z": ENV_Z_LABEL,
            "fused_risk": FUSED_LABEL,
        }
    )

    # ---- RAW SIGNALS: THREE SEPARATE GRAPHS ----
    st.subheader("Trend lines – raw signals")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.caption(WW_LABEL)
        st.line_chart(
            plot_df.set_index("date_ts")[[WW_LABEL]]
        )

    with r2:
        st.caption(HOSP_LABEL)
        st.line_chart(
            plot_df.set_index("date_ts")[[HOSP_LABEL]]
        )

    with r3:
        st.caption(ENV_LABEL)
        st.line_chart(
            plot_df.set_index("date_ts")[[ENV_LABEL]]
        )

    # ---- STANDARDIZED SIGNALS + FUSED RISK ----
    st.subheader("Standardized signals (z-scores) and fused risk")

    st.line_chart(
        plot_df.set_index("date_ts")[
            [WW_Z_LABEL, HOSP_Z_LABEL, ENV_Z_LABEL, FUSED_LABEL]
        ]
    )

    # ---- ALERT VIEW ----
    st.subheader("Alert tiers over time")

    alert_view = plot_df[[DATE_COL, FUSED_LABEL, "alert_tier"]].copy()
    alert_view["date_ts"] = pd.to_datetime(alert_view[DATE_COL])

    st.dataframe(alert_view.tail(20))

    st.download_button(
        "Download fused & alert data as CSV",
        data=merged.to_csv(index=False),
        file_name="fused_alerts.csv",
        mime="text/csv",
    )

    st.write(
        "Interpretation:\n"
        "- **Raw signal charts** show each underlying data stream separately: "
        "wastewater viral load, hospital admissions, and air-quality index.\n"
        "- **z-scores** show how many standard deviations above/below the recent "
        "baseline each signal is.\n"
        "- **Fused risk score** is a weighted combination of wastewater, hospital, "
        "and environment z-scores.\n"
        "- **Alert tiers** are based on how large the fused risk is compared to the "
        "thresholds you set in the sidebar."
    )


if __name__ == "__main__":
    main()
