# -*- coding: utf-8 -*-
"""
Water Injection RMT Dashboard with Facility Constraints and Facility Constrained

Inputs uploaded from the app sidebar:
1) Target CSV      -> WI_Target_RMT_All_Targets.csv
2) Facilities CSV  -> facilities.csv
3) Mapping CSV     -> mapping.csv

Facility Constrained logic:
- Agreed Target is already the minimum of RMT_OE and BMT constrained rate
- Therefore network-constrained IIC must be based on AGREED, not RMT_OE
- For each day and each well path:
    Well -> Manifold -> CPS
- Node request = sum(Agreed_mbd) of linked wells
- Node scale = min(1, capacity / requested)
- Final well IIC = Agreed_mbd * min(manifold scale, CPS scale)

Run:
    streamlit run WI_RMT_Facility_Dashboard.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="WI RMT + Facility Constrained Dashboard", layout="wide")

# ============================================================
# GLOBAL STYLE
# ============================================================
COLOR_MAP = {
    "Main Pay": "#1565C0",
    "Mishrif": "#EF6C00",
    "Upper Shale": "#2E7D32",
}

AREA_COLOR_MAP = {
    "North": "#8E24AA",
    "South": "#D81B60",
    "Unknown": "#546E7A",
}

SERIES_COLOR_MAP = {
    "RMT_OE_mbd": "#0D47A1",
    "Agreed_mbd": "#29B6F6",
    "IIC_mbd": "#E53935",
    "RMT OE (mbd)": "#0D47A1",
    "Agreed (mbd)": "#29B6F6",
    "Facility Constrained": "#E53935",
}

FONT_FAMILY = "Arial Black"
FONT_TITLE = 22
FONT_AXIS = 15
FONT_TICK = 12
FONT_LEGEND = 13
PLOT_TEMPLATE = "plotly_white"


def apply_layout_style(fig, title, x_title="Date", y_title="mbd", height=450):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(
            text=title,
            font=dict(size=FONT_TITLE, family=FONT_FAMILY, color="#111111"),
        ),
        xaxis=dict(
            title=x_title,
            title_font=dict(size=FONT_AXIS, family=FONT_FAMILY, color="#111111"),
            tickfont=dict(size=FONT_TICK, family="Arial", color="#222222"),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.12)",
            zeroline=False,
        ),
        yaxis=dict(
            title=y_title,
            title_font=dict(size=FONT_AXIS, family=FONT_FAMILY, color="#111111"),
            tickfont=dict(size=FONT_TICK, family="Arial", color="#222222"),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.12)",
            zeroline=False,
        ),
        legend=dict(
            font=dict(size=FONT_LEGEND, family="Arial", color="#111111"),
            title_font=dict(size=FONT_LEGEND, family=FONT_FAMILY, color="#111111"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(l=40, r=30, t=70, b=40),
    )
    return fig


def style_all_axes(fig):
    fig.update_xaxes(
        title_font=dict(size=FONT_AXIS, family=FONT_FAMILY, color="#111111"),
        tickfont=dict(size=FONT_TICK, family="Arial", color="#222222"),
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
    )
    fig.update_yaxes(
        title_font=dict(size=FONT_AXIS, family=FONT_FAMILY, color="#111111"),
        tickfont=dict(size=FONT_TICK, family="Arial", color="#222222"),
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        rangemode="tozero", 
    )
    fig.update_layout(
        title_font=dict(size=FONT_TITLE, family=FONT_FAMILY, color="#111111"),
        legend=dict(font=dict(size=FONT_LEGEND, family="Arial", color="#111111")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=30, t=70, b=40),
    )
    return fig


def thicken_lines(fig, width=3):
    for tr in fig.data:
        if hasattr(tr, "line"):
            tr.line.width = width
    return fig


def add_pct_comparisons(df, rmt_col="RMT_OE_mbd", agreed_col="Agreed_mbd", iic_col="IIC_mbd"):
    df["RMT_vs_Agreed_pct"] = np.where(
        df[agreed_col].abs() > 1e-9,
        100 * df[rmt_col] / df[agreed_col],
        np.nan,
    )
    df["IIC_vs_Agreed_pct"] = np.where(
        df[agreed_col].abs() > 1e-9,
        100 * df[iic_col] / df[agreed_col],
        np.nan,
    )
    df["IIC_vs_RMT_pct"] = np.where(
        df[rmt_col].abs() > 1e-9,
        100 * df[iic_col] / df[rmt_col],
        np.nan,
    )
    return df


def render_spotfire_trellis(df, node_col, cols_wrap=3):
    plot_df = df.melt(
        id_vars=["Date", node_col],
        value_vars=["Agreed_mbd", "RMT_OE_mbd", "IIC_mbd", "Capacity_mbd"],
        var_name="Series",
        value_name="mbd"
    ).dropna(subset=["mbd"]) 
    
    series_map = {
        "Agreed_mbd": "Agreed (Area)",
        "RMT_OE_mbd": "RMT OE (Line)",
        "IIC_mbd": "Facility Constrained (Line)",
        "Capacity_mbd": "Capacity (Line)",
    }
    plot_df["Series"] = plot_df["Series"].map(series_map)
    
    c_map = {
        "Agreed (Area)": "#29B6F6",     
        "RMT OE (Line)": "#43A047",     
        "Facility Constrained (Line)": "#111111",  
        "Capacity (Line)": "#E53935",   
    }

    fig = px.line(
        plot_df,
        x="Date",
        y="mbd",
        color="Series",
        facet_col=node_col,
        facet_col_wrap=cols_wrap,
        color_discrete_map=c_map,
        title=f"RMT vs Agreed vs Facility Constrained by {node_col}"
    )
    
    fig.update_yaxes(matches=None) 
    fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True, rangemode="tozero")) 
    
    for tr in fig.data:
        if tr.name == "Agreed (Area)":
            tr.fill = 'tozeroy'
            tr.line.width = 0 
        elif tr.name == "Facility Constrained (Line)":
            tr.line.width = 3
        else:
            tr.line.width = 2
            
    style_all_axes(fig)
    nrows = max(1, len(df[node_col].dropna().unique()) // cols_wrap + 1)
    fig.update_layout(height=max(450, 250 * nrows))
    return fig


# ============================================================
# HELPERS
# ============================================================
def infer_area(well_name: str) -> str:
    if pd.isna(well_name):
        return "Unknown"
    w = str(well_name).strip().upper()
    if w.startswith("RU-"):
        return "South"
    if w.startswith("R-"):
        return "North"
    return "Unknown"


def normalize_well_name(x):
    if pd.isna(x):
        return x
    return str(x).strip().upper()


def multiselect_with_all(label, options, default_all=True):
    options = list(pd.Series(options).dropna().astype(str).sort_values().unique())
    all_option = "All"
    default = [all_option] if default_all else []
    selected = st.multiselect(label, [all_option] + options, default=default)
    if all_option in selected or not selected:
        return options
    return selected


# ============================================================
# LOADERS
# ============================================================
@st.cache_data
def load_target_data(target_bytes: bytes):
    df = pd.read_csv(BytesIO(target_bytes))
    df.columns = df.columns.str.strip()

    required_cols = [
        "Well",
        "Reservoir",
        "RMT Publish Date",
        "RMT Rate+OE [bbl/day]",
        "Agreed Target [bbl/day]",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required target columns: {missing}")

    df["RMT Publish Date"] = pd.to_datetime(df["RMT Publish Date"], errors="coerce").dt.normalize()
    df["RMT Rate+OE [bbl/day]"] = pd.to_numeric(df["RMT Rate+OE [bbl/day]"], errors="coerce").fillna(0)
    df["Agreed Target [bbl/day]"] = pd.to_numeric(df["Agreed Target [bbl/day]"], errors="coerce").fillna(0)
    # Optional actual BMT well capacity; never substitute Agreed Target for capacity.
    for candidate in ["BMT Well Capacity [bbl/day]", "BMT Well Capacity", "Well Capacity [bbl/day]", "WELL_CAPACITY", "BMT_Well_Capacity"]:
        if candidate in df.columns:
            df["BMT_Capacity_mbd"] = pd.to_numeric(df[candidate], errors="coerce") / 1000.0
            break

    df = df.dropna(subset=["Well", "Reservoir", "RMT Publish Date"]).copy()
    df["Well_norm"] = df["Well"].apply(normalize_well_name)

    if "RMT Status" in df.columns:
        approved = df["RMT Status"].fillna("").astype(str).str.strip().str.upper()
        df = df[approved.eq("APPROVED")].copy()

    df = df.drop_duplicates(
        subset=[
            "Well_norm",
            "Reservoir",
            "RMT Publish Date",
            "RMT Rate+OE [bbl/day]",
            "Agreed Target [bbl/day]",
        ]
    ).copy()

    df = df.sort_values(["Well_norm", "RMT Publish Date"]).copy()
    return df


@st.cache_data
def load_mapping_data(mapping_bytes: bytes):
    mp = pd.read_csv(BytesIO(mapping_bytes))
    mp.columns = mp.columns.str.strip()

    required_cols = ["WELL_NAME", "Reservoir", "CPS", "Manifold", "Location"]
    missing = [c for c in required_cols if c not in mp.columns]
    if missing:
        raise ValueError(f"Missing required mapping columns: {missing}")

    mp["Well_norm"] = mp["WELL_NAME"].apply(normalize_well_name)
    mp["Area"] = mp["Location"].fillna("").astype(str).str.strip().replace({"": "Unknown"})
    mp["Area"] = mp["Area"].replace({"North": "North", "South": "South"})
    if "Status" not in mp.columns:
        mp["Status"] = "Unknown"

    mp = mp.drop_duplicates(subset=["Well_norm"], keep="first").copy()
    return mp


@st.cache_data
def load_facility_data(facilities_bytes: bytes):
    fc = pd.read_csv(BytesIO(facilities_bytes))
    fc.columns = fc.columns.str.strip()

    required_cols = ["FAC_NAME", "DATE", "CAP", "CAP-OE"]
    missing = [c for c in required_cols if c not in fc.columns]
    if missing:
        raise ValueError(f"Missing required facilities columns: {missing}")

    fc["DATE"] = pd.to_datetime(fc["DATE"], errors="coerce").dt.normalize()
    fc["CAP"] = pd.to_numeric(fc["CAP"], errors="coerce").fillna(0)
    fc["CAP-OE"] = pd.to_numeric(fc["CAP-OE"], errors="coerce").fillna(0)
    fc["FAC_NAME"] = fc["FAC_NAME"].astype(str).str.strip()

    fc = fc.dropna(subset=["FAC_NAME", "DATE"]).copy()
    fc = fc.sort_values(["FAC_NAME", "DATE"]).copy()
    return fc


# ============================================================
# BUILD DAILY WELL SERIES
# ============================================================
@st.cache_data
def build_daily_well_targets(target_df: pd.DataFrame):
    end_date = pd.Timestamp.today().normalize()

    daily_frames = []
    change_events = []

    for well, g in target_df.groupby("Well_norm", sort=False):
        g = g.sort_values("RMT Publish Date").copy()
        g = g.drop_duplicates(subset=["RMT Publish Date"], keep="last").copy()

        well_label = g["Well"].iloc[-1]
        reservoir = g["Reservoir"].iloc[-1]

        g["Prev_RMT"] = g["RMT Rate+OE [bbl/day]"].shift(1)
        g["Prev_Agreed"] = g["Agreed Target [bbl/day]"].shift(1)
        g["RMT_Change"] = g["RMT Rate+OE [bbl/day]"] - g["Prev_RMT"]
        g["Agreed_Change"] = g["Agreed Target [bbl/day]"] - g["Prev_Agreed"]
        g["Changed_Flag"] = (
            g["RMT_Change"].fillna(0).ne(0) | g["Agreed_Change"].fillna(0).ne(0)
        ).astype(int)

        change_events.append(g.copy())

        start_date = g["RMT Publish Date"].min()
        full_dates = pd.DataFrame({"Date": pd.date_range(start=start_date, end=end_date, freq="D")})

        daily_cols = ["RMT Publish Date", "RMT Rate+OE [bbl/day]", "Agreed Target [bbl/day]"]
        if "BMT_Capacity_mbd" in g.columns:
            daily_cols.append("BMT_Capacity_mbd")
        tmp = g[daily_cols].rename(columns={"RMT Publish Date": "Date"})

        full = full_dates.merge(tmp, on="Date", how="left")
        full["RMT Rate+OE [bbl/day]"] = full["RMT Rate+OE [bbl/day]"].ffill().fillna(0)
        full["Agreed Target [bbl/day]"] = full["Agreed Target [bbl/day]"].ffill().fillna(0)
        if "BMT_Capacity_mbd" in full.columns:
            full["BMT_Capacity_mbd"] = full["BMT_Capacity_mbd"].ffill()
        full["Well"] = well_label
        full["Well_norm"] = well
        full["Reservoir"] = reservoir

        daily_frames.append(full)

    daily_well = pd.concat(daily_frames, ignore_index=True)
    change_events_df = pd.concat(change_events, ignore_index=True)

    daily_well["RMT_OE_mbd"] = daily_well["RMT Rate+OE [bbl/day]"] / 1000.0
    daily_well["Agreed_mbd"] = daily_well["Agreed Target [bbl/day]"] / 1000.0

    change_events_df["Change_Date"] = change_events_df["RMT Publish Date"]
    change_events_df["RMT_Change_mbd"] = change_events_df["RMT_Change"] / 1000.0
    change_events_df["Agreed_Change_mbd"] = change_events_df["Agreed_Change"] / 1000.0

    return daily_well, change_events_df


# ============================================================
# FACILITY CAPACITIES - DAILY CONTINUOUS
# ============================================================
@st.cache_data
def build_daily_facility_capacity(fac_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp):
    out = []
    for fac, g in fac_df.groupby("FAC_NAME", sort=False):
        g = g.sort_values("DATE").copy()
        full_dates = pd.DataFrame({"Date": pd.date_range(start=start_date, end=end_date, freq="D")})
        tmp = g.rename(columns={"DATE": "Date"})
        full = full_dates.merge(tmp[["Date", "FAC_NAME", "CAP", "CAP-OE"]], on="Date", how="left")
        full["FAC_NAME"] = fac
        full["CAP"] = full["CAP"].ffill().fillna(0)
        full["CAP-OE"] = full["CAP-OE"].ffill().fillna(0)
        out.append(full)

    if not out:
        return pd.DataFrame(columns=["Date", "FAC_NAME", "CAP", "CAP-OE"])

    return pd.concat(out, ignore_index=True)


# ============================================================
# MERGE TARGET + MAPPING
# ============================================================
def merge_target_mapping(daily_well: pd.DataFrame, mapping_df: pd.DataFrame):
    cols = ["Well_norm", "WELL_NAME", "Reservoir", "CPS", "Manifold", "Area", "Status"]
    if "Pattern_Name" in mapping_df.columns:
        cols.append("Pattern_Name")

    df = daily_well.merge(mapping_df[cols], on="Well_norm", how="left", suffixes=("", "_map")).copy()

    df["CPS"] = df["CPS"].fillna("Unknown")
    df["Manifold"] = df["Manifold"].fillna("Unknown")
    df["Area"] = df["Area"].fillna(df["Well"].apply(infer_area))
    df["Status"] = df["Status"].fillna("Unknown")

    if "Reservoir_map" in df.columns:
        df["Reservoir"] = df["Reservoir_map"].fillna(df["Reservoir"])
        df = df.drop(columns=["Reservoir_map"])

    return df


# ============================================================
# Facility Constrained CALCULATION
# ============================================================
def calculate_iic(daily_well_map: pd.DataFrame, daily_facility_cap: pd.DataFrame, cap_col: str = "CAP"):
    """
    Facility Constrained logic:
    - Request entering the network is Agreed_mbd, not RMT_OE_mbd
    - Node request = sum(Agreed_mbd) for wells linked to that node
    - Scale(node) = min(1, node capacity / node request)
    - Well IIC = Agreed_mbd * min(scale of manifold, scale of CPS)
    """

    cap_daily = daily_facility_cap.rename(
        columns={"FAC_NAME": "Node", cap_col: "Node_Capacity"}
    )[["Date", "Node", "Node_Capacity"]].copy()

    req = daily_well_map.copy()
    req["Manifold"] = req["Manifold"].astype(str)
    req["CPS"] = req["CPS"].astype(str)

    manifold_req = (
        req.groupby(["Date", "Manifold"], as_index=False)["Agreed_mbd"]
        .sum()
        .rename(columns={"Manifold": "Node", "Agreed_mbd": "Requested_mbd"})
    )

    cps_req = (
        req.groupby(["Date", "CPS"], as_index=False)["Agreed_mbd"]
        .sum()
        .rename(columns={"CPS": "Node", "Agreed_mbd": "Requested_mbd"})
    )

    node_req = pd.concat([manifold_req, cps_req], ignore_index=True)
    node_req = node_req.groupby(["Date", "Node"], as_index=False)["Requested_mbd"].sum()

    node_perf = node_req.merge(cap_daily, on=["Date", "Node"], how="left")
    node_perf["Node_Capacity"] = node_perf["Node_Capacity"].fillna(np.inf)

    node_perf["Scale"] = np.where(
        node_perf["Requested_mbd"] > 1e-12,
        np.minimum(1.0, node_perf["Node_Capacity"] / node_perf["Requested_mbd"]),
        1.0,
    )

    node_perf["Utilization_pct"] = np.where(
        np.isfinite(node_perf["Node_Capacity"]) & (node_perf["Node_Capacity"] > 1e-12),
        100.0 * node_perf["Requested_mbd"] / node_perf["Node_Capacity"],
        np.nan,
    )

    manifold_scale = node_perf.rename(
        columns={
            "Node": "Manifold",
            "Scale": "Manifold_Scale",
            "Requested_mbd": "Manifold_Requested_mbd",
            "Node_Capacity": "Manifold_Capacity_mbd",
            "Utilization_pct": "Manifold_Util_pct",
        }
    )[["Date", "Manifold", "Manifold_Scale", "Manifold_Requested_mbd", "Manifold_Capacity_mbd", "Manifold_Util_pct"]]

    cps_scale = node_perf.rename(
        columns={
            "Node": "CPS",
            "Scale": "CPS_Scale",
            "Requested_mbd": "CPS_Requested_mbd",
            "Node_Capacity": "CPS_Capacity_mbd",
            "Utilization_pct": "CPS_Util_pct",
        }
    )[["Date", "CPS", "CPS_Scale", "CPS_Requested_mbd", "CPS_Capacity_mbd", "CPS_Util_pct"]]

    out = req.merge(manifold_scale, on=["Date", "Manifold"], how="left")
    out = out.merge(cps_scale, on=["Date", "CPS"], how="left")

    out["Manifold_Scale"] = out["Manifold_Scale"].fillna(1.0)
    out["CPS_Scale"] = out["CPS_Scale"].fillna(1.0)

    out["IIC_Scale"] = out[["Manifold_Scale", "CPS_Scale"]].min(axis=1)
    out["IIC_mbd"] = out["Agreed_mbd"] * out["IIC_Scale"]

    out["IIC_Loss_vs_Agreed_mbd"] = out["Agreed_mbd"] - out["IIC_mbd"]
    out["IIC_Loss_vs_RMT_mbd"] = out["RMT_OE_mbd"] - out["IIC_mbd"]

    out["Constraint_Node"] = np.where(
        out["Manifold_Scale"] <= out["CPS_Scale"], out["Manifold"], out["CPS"]
    )

    return out, node_perf


# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.title("💧 Water Injection RMT + Facility Constrained Dashboard")
st.caption("Constraint-aware injection dashboard with Area+Line Spotfire styles, auto-scaled Y-axes, and exact IIC tracking.")

with st.sidebar:
    st.header("Settings")
    target_file = st.file_uploader(
        "Upload Target CSV",
        type=["csv"],
        help="Required file containing well targets and RMT publish dates.",
        key="target_file",
    )

    facilities_file = st.file_uploader(
        "Upload Facilities CSV",
        type=["csv"],
        help="Required file containing CPS/manifold capacity profiles.",
        key="facilities_file",
    )

    mapping_file = st.file_uploader(
        "Upload Mapping CSV",
        type=["csv"],
        help="Required file mapping each well to reservoir, CPS, manifold and area.",
        key="mapping_file",
    )

    bmt_capacity_file = st.file_uploader(
        "Optional BMT Well Capacity CSV (required for prediction if not in Target CSV)",
        type=["csv"], key="bmt_cap_file",
        help="Columns: Well, BMT Well Capacity [bbl/day]; optionally Date. Units are bbl/day."
    )

    uploads_ready = all(
        uploaded_file is not None
        for uploaded_file in (target_file, facilities_file, mapping_file)
    )

    if uploads_ready:
        st.success("All three CSV files are ready.")
    else:
        st.info("Upload all three CSV files to start the dashboard.")

    cap_basis = st.selectbox("Capacity basis for constraints", ["CAP", "CAP-OE"], index=0)

    st.header("Font sizes")
    ui_title_size = st.number_input("Title font size", min_value=12, max_value=40, value=FONT_TITLE, step=1)
    ui_axis_size = st.number_input("Axis title font size", min_value=10, max_value=30, value=FONT_AXIS, step=1)
    ui_tick_size = st.number_input("Axis tick font size", min_value=8, max_value=24, value=FONT_TICK, step=1)
    ui_legend_size = st.number_input("Legend font size", min_value=8, max_value=24, value=FONT_LEGEND, step=1)

FONT_TITLE = ui_title_size
FONT_AXIS = ui_axis_size
FONT_TICK = ui_tick_size
FONT_LEGEND = ui_legend_size

# ============================================================
# LOAD + BUILD
# ============================================================
if not uploads_ready:
    st.info(
        "Please upload the Target, Facilities and Mapping CSV files "
        "from the sidebar to open the dashboard."
    )
    st.stop()

try:
    target_df = load_target_data(target_file.getvalue())
    mapping_df = load_mapping_data(mapping_file.getvalue())
    fac_df = load_facility_data(facilities_file.getvalue())

    daily_well, changes_df = build_daily_well_targets(target_df)
    daily_well = merge_target_mapping(daily_well, mapping_df)
    if bmt_capacity_file is not None:
        bc = pd.read_csv(BytesIO(bmt_capacity_file.getvalue()))
        bc.columns = bc.columns.str.strip()
        name_col = next((x for x in ["Well", "WELL_NAME", "WELL"] if x in bc.columns), None)
        cap_col_name = next((x for x in ["BMT Well Capacity [bbl/day]", "BMT Well Capacity", "Well Capacity [bbl/day]", "WELL_CAPACITY", "BMT_Well_Capacity"] if x in bc.columns), None)
        if not name_col or not cap_col_name:
            raise ValueError("BMT capacity CSV needs Well and BMT Well Capacity [bbl/day] columns.")
        bc["Well_norm"] = bc[name_col].apply(normalize_well_name)
        bc["BMT_upload_mbd"] = pd.to_numeric(bc[cap_col_name], errors="coerce") / 1000.0
        date_col = next((x for x in ["Date", "DATE", "RMT Publish Date"] if x in bc.columns), None)
        if date_col:
            bc["Date"] = pd.to_datetime(bc[date_col], errors="coerce").dt.normalize()
            bc = bc.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(["Well_norm", "Date"], keep="last")
            daily_well = daily_well.merge(bc[["Well_norm", "Date", "BMT_upload_mbd"]], on=["Well_norm", "Date"], how="left")
            daily_well = daily_well.sort_values(["Well_norm", "Date"])
            daily_well["BMT_upload_mbd"] = daily_well.groupby("Well_norm")["BMT_upload_mbd"].ffill()
        else:
            bc = bc.drop_duplicates("Well_norm", keep="last")
            daily_well = daily_well.merge(bc[["Well_norm", "BMT_upload_mbd"]], on="Well_norm", how="left")
        if "BMT_Capacity_mbd" in daily_well.columns:
            daily_well["BMT_Capacity_mbd"] = daily_well["BMT_upload_mbd"].combine_first(daily_well["BMT_Capacity_mbd"])
        else:
            daily_well["BMT_Capacity_mbd"] = daily_well["BMT_upload_mbd"]
        daily_well = daily_well.drop(columns=["BMT_upload_mbd"])

    global_start = min(daily_well["Date"].min(), fac_df["DATE"].min())
    global_end = max(daily_well["Date"].max(), fac_df["DATE"].max())
    daily_facility_cap = build_daily_facility_capacity(fac_df, global_start, global_end)

    daily_well_iic, node_perf = calculate_iic(daily_well, daily_facility_cap, cap_col=cap_basis)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ============================================================
# GLOBAL FILTERS
# ============================================================
min_date = daily_well_iic["Date"].min().date()
max_date = daily_well_iic["Date"].max().date()

with st.sidebar:
    st.header("Filters")

    date_range = st.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
    )

    selected_areas = multiselect_with_all("Area", daily_well_iic["Area"].unique())
    area_filtered = daily_well_iic[daily_well_iic["Area"].isin(selected_areas)]

    selected_reservoirs = multiselect_with_all("Reservoir", area_filtered["Reservoir"].unique())
    res_filtered = area_filtered[area_filtered["Reservoir"].isin(selected_reservoirs)]

    selected_cps = multiselect_with_all("CPS", res_filtered["CPS"].unique())
    cps_filtered = res_filtered[res_filtered["CPS"].isin(selected_cps)]

    selected_manifolds = multiselect_with_all("Manifold", cps_filtered["Manifold"].unique())
    man_filtered = cps_filtered[cps_filtered["Manifold"].isin(selected_manifolds)]

    selected_wells = multiselect_with_all("Well", man_filtered["Well"].unique(), default_all=True)

    st.header("Display options")
    show_markers = st.checkbox("Show markers", value=True)
    rolling_days = st.selectbox("Rolling average (days)", [1, 7, 14, 30], index=0)
    show_data_table = st.checkbox("Show filtered table", value=False)

start_date = pd.Timestamp(date_range[0])
end_date = pd.Timestamp(date_range[1])

filtered_well = daily_well_iic[
    (daily_well_iic["Date"] >= start_date)
    & (daily_well_iic["Date"] <= end_date)
    & (daily_well_iic["Area"].isin(selected_areas))
    & (daily_well_iic["Reservoir"].isin(selected_reservoirs))
    & (daily_well_iic["CPS"].isin(selected_cps))
    & (daily_well_iic["Manifold"].isin(selected_manifolds))
    & (daily_well_iic["Well"].isin(selected_wells))
].copy()

filtered_changes = changes_df[
    (changes_df["Change_Date"] >= start_date)
    & (changes_df["Change_Date"] <= end_date)
    & (changes_df["Well_norm"].isin(filtered_well["Well_norm"].unique()))
].copy()

filtered_node_perf = node_perf[
    (node_perf["Date"] >= start_date)
    & (node_perf["Date"] <= end_date)
].copy()

relevant_nodes = set(filtered_well["Manifold"].dropna().unique()).union(set(filtered_well["CPS"].dropna().unique()))
filtered_node_perf = filtered_node_perf[filtered_node_perf["Node"].isin(relevant_nodes)].copy()

if filtered_well.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

# ============================================================
# AGGREGATIONS
# ============================================================
f_res = (
    filtered_well.groupby(["Date", "Reservoir"], as_index=False)
    .agg({"RMT_OE_mbd": "sum", "Agreed_mbd": "sum", "IIC_mbd": "sum"})
)
f_res = add_pct_comparisons(f_res)

f_cps = (
    filtered_well.groupby(["Date", "CPS"], as_index=False)
    .agg({"RMT_OE_mbd": "sum", "Agreed_mbd": "sum", "IIC_mbd": "sum"})
)
cps_caps = filtered_node_perf[filtered_node_perf["Node"].isin(f_cps["CPS"].unique())][["Date", "Node", "Node_Capacity"]].rename(columns={"Node": "CPS", "Node_Capacity": "Capacity_mbd"})
f_cps = f_cps.merge(cps_caps, on=["Date", "CPS"], how="left")
f_cps["Capacity_mbd"] = f_cps["Capacity_mbd"].replace(np.inf, np.nan) 
f_cps = add_pct_comparisons(f_cps)

f_manifold = (
    filtered_well.groupby(["Date", "Manifold"], as_index=False)
    .agg({"RMT_OE_mbd": "sum", "Agreed_mbd": "sum", "IIC_mbd": "sum"})
)
man_caps = filtered_node_perf[filtered_node_perf["Node"].isin(f_manifold["Manifold"].unique())][["Date", "Node", "Node_Capacity"]].rename(columns={"Node": "Manifold", "Node_Capacity": "Capacity_mbd"})
f_manifold = f_manifold.merge(man_caps, on=["Date", "Manifold"], how="left")
f_manifold["Capacity_mbd"] = f_manifold["Capacity_mbd"].replace(np.inf, np.nan)
f_manifold = add_pct_comparisons(f_manifold)

f_field = (
    filtered_well.groupby("Date", as_index=False)
    .agg({"RMT_OE_mbd": "sum", "Agreed_mbd": "sum", "IIC_mbd": "sum"})
)
f_field = add_pct_comparisons(f_field)

if rolling_days > 1:
    for col in ["RMT_OE_mbd", "Agreed_mbd", "IIC_mbd", "RMT_vs_Agreed_pct", "IIC_vs_Agreed_pct", "IIC_vs_RMT_pct"]:
        f_field[col] = f_field[col].rolling(rolling_days, min_periods=1).mean()

snapshot_date = st.slider(
    "Snapshot date",
    min_value=min_date,
    max_value=max_date,
    value=max_date,
    format="YYYY-MM-DD",
)
snapshot_date = pd.Timestamp(snapshot_date)

snapshot = filtered_well[filtered_well["Date"] == snapshot_date].copy()

res_snapshot = (
    snapshot.groupby("Reservoir", as_index=False)
    .agg({"RMT_OE_mbd": "sum", "Agreed_mbd": "sum", "IIC_mbd": "sum", "Well": "nunique"})
    .rename(columns={"Well": "Well_Count"})
)
res_snapshot = add_pct_comparisons(res_snapshot)

base_snapshot = filtered_well[filtered_well["Date"] == start_date].copy()
base_res_snapshot = (
    base_snapshot.groupby("Reservoir", as_index=False)
    .agg({"RMT_OE_mbd": "sum", "Agreed_mbd": "sum", "IIC_mbd": "sum"})
    .rename(columns={
        "RMT_OE_mbd": "Base_RMT_OE_mbd",
        "Agreed_mbd": "Base_Agreed_mbd",
        "IIC_mbd": "Base_IIC_mbd",
    })
)

res_snapshot = res_snapshot.merge(base_res_snapshot, on="Reservoir", how="left")
for c in ["Base_RMT_OE_mbd", "Base_Agreed_mbd", "Base_IIC_mbd"]:
    res_snapshot[c] = res_snapshot[c].fillna(0)

res_snapshot["RMT_Change_Since_Start_mbd"] = res_snapshot["RMT_OE_mbd"] - res_snapshot["Base_RMT_OE_mbd"]
res_snapshot["Agreed_Change_Since_Start_mbd"] = res_snapshot["Agreed_mbd"] - res_snapshot["Base_Agreed_mbd"]
res_snapshot["IIC_Change_Since_Start_mbd"] = res_snapshot["IIC_mbd"] - res_snapshot["Base_IIC_mbd"]

res_snapshot["RMT_Change_Since_Start_pct"] = np.where(
    res_snapshot["Base_RMT_OE_mbd"].abs() > 1e-9,
    100 * res_snapshot["RMT_Change_Since_Start_mbd"] / res_snapshot["Base_RMT_OE_mbd"],
    np.nan,
)
res_snapshot["Agreed_Change_Since_Start_pct"] = np.where(
    res_snapshot["Base_Agreed_mbd"].abs() > 1e-9,
    100 * res_snapshot["Agreed_Change_Since_Start_mbd"] / res_snapshot["Base_Agreed_mbd"],
    np.nan,
)
res_snapshot["IIC_Change_Since_Start_pct"] = np.where(
    res_snapshot["Base_IIC_mbd"].abs() > 1e-9,
    100 * res_snapshot["IIC_Change_Since_Start_mbd"] / res_snapshot["Base_IIC_mbd"],
    np.nan,
)

node_snapshot = filtered_node_perf[filtered_node_perf["Date"] == snapshot_date].copy()
node_snapshot["Headroom_mbd"] = node_snapshot["Node_Capacity"] - node_snapshot["Requested_mbd"]
node_snapshot["Bottleneck_Flag"] = np.where(node_snapshot["Scale"] < 0.9999, "Constrained", "Free")

# Compute Facility Constrained Change by CPS
cps_base = base_snapshot.groupby("CPS", as_index=False)["IIC_mbd"].sum().rename(columns={"IIC_mbd": "Base_IIC_mbd"})
cps_snap = snapshot.groupby("CPS", as_index=False)["IIC_mbd"].sum()
cps_change = cps_snap.merge(cps_base, on="CPS", how="left").fillna(0)
cps_change["IIC_Change_mbd"] = cps_change["IIC_mbd"] - cps_change["Base_IIC_mbd"]
cps_change = cps_change[cps_change["CPS"] != "Unknown"] 

# Compute Facility Constrained Change by Manifold
man_base = base_snapshot.groupby("Manifold", as_index=False)["IIC_mbd"].sum().rename(columns={"IIC_mbd": "Base_IIC_mbd"})
man_snap = snapshot.groupby("Manifold", as_index=False)["IIC_mbd"].sum()
man_change = man_snap.merge(man_base, on="Manifold", how="left").fillna(0)
man_change["IIC_Change_mbd"] = man_change["IIC_mbd"] - man_change["Base_IIC_mbd"]
man_change = man_change[man_change["Manifold"] != "Unknown"]

# ============================================================
# KPIs
# ============================================================
# latest_field = f_field.sort_values("Date").iloc[-1]
# prev_field = f_field.sort_values("Date").iloc[-2] if len(f_field) > 1 else latest_field

# latest_rmt = latest_field["RMT_OE_mbd"]
# latest_agreed = latest_field["Agreed_mbd"]
# latest_iic = latest_field["IIC_mbd"]

# prev_rmt = prev_field["RMT_OE_mbd"]
# prev_agreed = prev_field["Agreed_mbd"]
# prev_iic = prev_field["IIC_mbd"]

# base_rmt = f_field.sort_values("Date").iloc[0]["RMT_OE_mbd"] if len(f_field) > 0 else latest_rmt
# base_agreed = f_field.sort_values("Date").iloc[0]["Agreed_mbd"] if len(f_field) > 0 else latest_agreed
# base_iic = f_field.sort_values("Date").iloc[0]["IIC_mbd"] if len(f_field) > 0 else latest_iic

# rmt_change_from_start_mbd = latest_rmt - base_rmt
# agreed_change_from_start_mbd = latest_agreed - base_agreed
# iic_change_from_start_mbd = latest_iic - base_iic

# latest_iic_vs_agreed_pct = latest_field["IIC_vs_Agreed_pct"]
# latest_iic_vs_rmt_pct = latest_field["IIC_vs_RMT_pct"]

# k1, k2, k3, k4, k5, k6, k7, k8, k9 = st.columns(9)
# k1.metric("RMT OE (mbd)", f"{latest_rmt:.1f}", f"{latest_rmt - prev_rmt:+.1f}")
# k2.metric("Agreed (mbd)", f"{latest_agreed:.1f}", f"{latest_agreed - prev_agreed:+.1f}")
# k3.metric("Final Facility Constrained", f"{latest_iic:.1f}", f"{latest_iic - prev_iic:+.1f}")
# k4.metric(f"RMT Δ vs {start_date.date()}", f"{rmt_change_from_start_mbd:+.1f}")
# k5.metric(f"Agreed Δ vs {start_date.date()}", f"{agreed_change_from_start_mbd:+.1f}")
# k6.metric(f"IIC Δ vs {start_date.date()}", f"{iic_change_from_start_mbd:+.1f}")
# k7.metric("IIC / Agreed (%)", f"{latest_iic_vs_agreed_pct:.1f}%" if pd.notna(latest_iic_vs_agreed_pct) else "n/a")
# k8.metric("IIC / RMT (%)", f"{latest_iic_vs_rmt_pct:.1f}%" if pd.notna(latest_iic_vs_rmt_pct) else "n/a")
# k9.metric("Wells in scope", f"{snapshot['Well'].nunique()}")

# ============================================================
# KPIs
# ============================================================
latest_field = f_field.sort_values("Date").iloc[-1]
prev_field = f_field.sort_values("Date").iloc[-2] if len(f_field) > 1 else latest_field

latest_rmt = latest_field["RMT_OE_mbd"]
latest_agreed = latest_field["Agreed_mbd"]
latest_iic = latest_field["IIC_mbd"]

prev_rmt = prev_field["RMT_OE_mbd"]
prev_agreed = prev_field["Agreed_mbd"]
prev_iic = prev_field["IIC_mbd"]

base_rmt = f_field.sort_values("Date").iloc[0]["RMT_OE_mbd"] if len(f_field) > 0 else latest_rmt
base_agreed = f_field.sort_values("Date").iloc[0]["Agreed_mbd"] if len(f_field) > 0 else latest_agreed
base_iic = f_field.sort_values("Date").iloc[0]["IIC_mbd"] if len(f_field) > 0 else latest_iic

rmt_change_from_start_mbd = latest_rmt - base_rmt
agreed_change_from_start_mbd = latest_agreed - base_agreed
iic_change_from_start_mbd = latest_iic - base_iic

# Calculate Percentage Change since Start Date
rmt_change_pct = (100 * rmt_change_from_start_mbd / base_rmt) if abs(base_rmt) > 1e-9 else np.nan
agreed_change_pct = (100 * agreed_change_from_start_mbd / base_agreed) if abs(base_agreed) > 1e-9 else np.nan
iic_change_pct = (100 * iic_change_from_start_mbd / base_iic) if abs(base_iic) > 1e-9 else np.nan

k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = st.columns(10)
k1.metric("RMT OE (mbd)", f"{latest_rmt:.1f}", f"{latest_rmt - prev_rmt:+.1f}")
k2.metric("Agreed (mbd)", f"{latest_agreed:.1f}", f"{latest_agreed - prev_agreed:+.1f}")
k3.metric("Final Facility Constrained", f"{latest_iic:.1f}", f"{latest_iic - prev_iic:+.1f}")
k4.metric("RMT Δ vs start", f"{rmt_change_from_start_mbd:+.1f}")
k5.metric("Agreed Δ vs start", f"{agreed_change_from_start_mbd:+.1f}")
k6.metric("IIC Δ vs start", f"{iic_change_from_start_mbd:+.1f}")
k7.metric("RMT_OE (%)", f"{rmt_change_pct:+.1f}%" if pd.notna(rmt_change_pct) else "n/a")
k8.metric("Agreed (%)", f"{agreed_change_pct:+.1f}%" if pd.notna(agreed_change_pct) else "n/a")
k9.metric("IIC (%)", f"{iic_change_pct:+.1f}%" if pd.notna(iic_change_pct) else "n/a")
k10.metric("Wells in scope", f"{snapshot['Well'].nunique()}")

# ============================================================
# TABS
# ============================================================
summary_tab, reservoir_tab, cps_tab, manifold_tab, facility_tab, change_tab, exports_tab, gap_tab, forecast_tab = st.tabs(
    ["Summary", "Reservoir", "CPS", "Manifold", "Facilities / IIC", "Changes", "Exports", "Capacity Gap", "Prediction / Gap"]
)

with summary_tab:
    st.markdown("### Trend view options")
    trend_scope = st.radio(
        "Show trend by",
        ["Field", "Reservoir", "Area", "Reservoir within Area"],
        horizontal=True,
    )

    c1, c2 = st.columns([1.8, 1.2])

    with c1:
        mode = "lines+markers" if show_markers else "lines"

        if trend_scope == "Field":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=f_field["Date"], y=f_field["RMT_OE_mbd"], mode=mode,
                name="RMT OE (mbd)",
                line=dict(color=SERIES_COLOR_MAP["RMT OE (mbd)"], width=3),
                marker=dict(color=SERIES_COLOR_MAP["RMT OE (mbd)"], size=7),
            ))
            fig.add_trace(go.Scatter(
                x=f_field["Date"], y=f_field["Agreed_mbd"], mode=mode,
                name="Agreed (mbd)",
                line=dict(color=SERIES_COLOR_MAP["Agreed (mbd)"], width=3, dash="dash"),
                marker=dict(color=SERIES_COLOR_MAP["Agreed (mbd)"], size=7),
            ))
            fig.add_trace(go.Scatter(
                x=f_field["Date"], y=f_field["IIC_mbd"], mode=mode,
                name="Facility Constrained",
                line=dict(color=SERIES_COLOR_MAP["Facility Constrained"], width=4),
                marker=dict(color=SERIES_COLOR_MAP["Facility Constrained"], size=7),
            ))
            apply_layout_style(fig, "Field Total RMT OE vs Agreed vs Facility Constrained", "Date", "mbd", 430)
            st.plotly_chart(fig, use_container_width=True, key="summary_main_trend_field")

        elif trend_scope == "Reservoir":
            plot_df = f_res.melt(
                id_vars=["Date", "Reservoir"],
                value_vars=["RMT_OE_mbd", "Agreed_mbd", "IIC_mbd"],
                var_name="Series",
                value_name="mbd",
            )
            fig = px.line(
                plot_df, x="Date", y="mbd", color="Reservoir", line_dash="Series",
                facet_col="Reservoir", facet_col_wrap=2, color_discrete_map=COLOR_MAP,
                title="Trellis by Reservoir: RMT vs Agreed vs Facility Constrained",
            )
            thicken_lines(fig, 3)
            style_all_axes(fig)
            fig.update_layout(height=max(500, 260 * max(1, plot_df["Reservoir"].nunique())))
            st.plotly_chart(fig, use_container_width=True, key="summary_main_trend_reservoir")

        elif trend_scope == "Area":
            area_daily = (
                filtered_well.groupby(["Date", "Area"], as_index=False)
                .agg({"RMT_OE_mbd": "sum", "Agreed_mbd": "sum", "IIC_mbd": "sum"})
            )
            plot_df = area_daily.melt(
                id_vars=["Date", "Area"],
                value_vars=["RMT_OE_mbd", "Agreed_mbd", "IIC_mbd"],
                var_name="Series",
                value_name="mbd",
            )
            fig = px.line(
                plot_df, x="Date", y="mbd", color="Area", line_dash="Series",
                facet_col="Area", facet_col_wrap=2, color_discrete_map=AREA_COLOR_MAP,
                title="Trellis by Area: RMT vs Agreed vs Facility Constrained",
            )
            thicken_lines(fig, 3)
            style_all_axes(fig)
            fig.update_layout(height=max(450, 280 * max(1, plot_df["Area"].nunique())))
            st.plotly_chart(fig, use_container_width=True, key="summary_main_trend_area")

        else:
            res_area_daily = (
                filtered_well.groupby(["Date", "Area", "Reservoir"], as_index=False)
                .agg({"RMT_OE_mbd": "sum", "Agreed_mbd": "sum", "IIC_mbd": "sum"})
            )
            plot_df = res_area_daily.melt(
                id_vars=["Date", "Area", "Reservoir"],
                value_vars=["RMT_OE_mbd", "Agreed_mbd", "IIC_mbd"],
                var_name="Series",
                value_name="mbd",
            )
            fig = px.line(
                plot_df, x="Date", y="mbd", color="Reservoir", line_dash="Series",
                facet_row="Area", facet_col="Reservoir", color_discrete_map=COLOR_MAP,
                title="Trellis by Reservoir and Area: RMT vs Agreed vs Facility Constrained",
            )
            thicken_lines(fig, 3)
            style_all_axes(fig)
            fig.update_layout(height=max(650, 280 * max(1, plot_df["Area"].nunique())))
            st.plotly_chart(fig, use_container_width=True, key="summary_main_trend_reservoir_area")

    with c2:
        fig = px.bar(
            res_snapshot.sort_values("IIC_mbd", ascending=False),
            x="Reservoir", y="IIC_mbd", color="Reservoir",
            color_discrete_map=COLOR_MAP, hover_data=["Agreed_mbd", "RMT_OE_mbd", "Well_Count"],
            title=f"Reservoir Facility Constrained at {snapshot_date.date()} (mbd)",
        )
        apply_layout_style(fig, f"Reservoir Facility Constrained at {snapshot_date.date()} (mbd)", "Reservoir", "mbd", 430)
        st.plotly_chart(fig, use_container_width=True, key="summary_iic_snapshot_mbd")

    c3, c4 = st.columns(2)

    with c3:
        fig = px.area(
            f_res, x="Date", y="IIC_mbd", color="Reservoir", color_discrete_map=COLOR_MAP,
            title="Reservoir Contribution to Facility Constrained",
        )
        style_all_axes(fig)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True, key="summary_reservoir_contribution_iic")

    with c4:
        # User explicitly requested this bottom-right chart to show Facility Constrained Change Since Start as Percentage
        fig = px.bar(
            res_snapshot.sort_values("IIC_Change_Since_Start_pct", ascending=False),
            x="Reservoir", y="IIC_Change_Since_Start_pct", color="Reservoir",
            color_discrete_map=COLOR_MAP, hover_data=["IIC_Change_Since_Start_mbd", "IIC_mbd", "Base_IIC_mbd"],
            title=f"Reservoir Facility Constrained Change Since {start_date.date()} (%)",
        )
        apply_layout_style(fig, f"Reservoir Facility Constrained Change Since {start_date.date()} (%)", "Reservoir", "%", 450)
        st.plotly_chart(fig, use_container_width=True, key="summary_iic_change_pct_bottom")

with reservoir_tab:
    reservoir_options = sorted(filtered_well["Reservoir"].dropna().unique())
    selected_res_detail = st.selectbox("Reservoir detail", reservoir_options)

    d = f_res[f_res["Reservoir"] == selected_res_detail].copy()
    res_color = COLOR_MAP.get(selected_res_detail, "#1565C0")
    mode = "lines+markers" if show_markers else "lines"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["Date"], y=d["RMT_OE_mbd"], mode=mode, name="RMT OE (mbd)",
        line=dict(color=res_color, width=4), marker=dict(color=res_color, size=7),
    ))
    fig.add_trace(go.Scatter(
        x=d["Date"], y=d["Agreed_mbd"], mode=mode, name="Agreed (mbd)",
        line=dict(color="#29B6F6", width=3, dash="dash"), marker=dict(color="#29B6F6", size=6),
    ))
    fig.add_trace(go.Scatter(
        x=d["Date"], y=d["IIC_mbd"], mode=mode, name="Facility Constrained",
        line=dict(color="#E53935", width=4), marker=dict(color="#E53935", size=7),
    ))
    apply_layout_style(fig, f"Reservoir: {selected_res_detail}", "Date", "mbd", 500)
    st.plotly_chart(fig, use_container_width=True, key="reservoir_detail_trend")

    c1, c2 = st.columns(2)
    with c1:
        pie_df = res_snapshot.copy()
        pie_df["Abs_IIC_Change_Since_Start_mbd"] = pie_df["IIC_Change_Since_Start_mbd"].abs()
        fig = px.pie(
            pie_df, names="Reservoir", values="Abs_IIC_Change_Since_Start_mbd", color="Reservoir",
            color_discrete_map=COLOR_MAP, title=f"Share of Net Facility Constrained Change Since {start_date.date()}",
        )
        style_all_axes(fig)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True, key="reservoir_iic_share_pie")

    with c2:
        fig = px.bar(
            res_snapshot.sort_values("IIC_Change_Since_Start_mbd", ascending=False),
            x="Reservoir", y="IIC_Change_Since_Start_mbd", color="Reservoir", color_discrete_map=COLOR_MAP,
            hover_data=["IIC_Change_Since_Start_pct", "IIC_mbd", "Agreed_mbd", "RMT_OE_mbd"],
            title=f"Net Facility Constrained Change Since {start_date.date()} (mbd)",
        )
        apply_layout_style(fig, f"Net Facility Constrained Change Since {start_date.date()} (mbd)", "Reservoir", "mbd", 450)
        st.plotly_chart(fig, use_container_width=True, key="reservoir_net_iic_change_bar")

with cps_tab:
    st.markdown("### CPS Level Constraints & Targets")
    
    f_cps_plot = f_cps[f_cps["CPS"] != "Unknown"].copy()
    fig = render_spotfire_trellis(f_cps_plot, node_col="CPS", cols_wrap=3)
    st.plotly_chart(fig, use_container_width=True, key="cps_spotfire_trend")

    c1, c2 = st.columns(2)
    with c1:
        cps_snapshot = snapshot.groupby("CPS", as_index=False)[["RMT_OE_mbd", "Agreed_mbd", "IIC_mbd"]].sum()
        cps_snapshot = add_pct_comparisons(cps_snapshot)
        cps_snapshot = cps_snapshot[cps_snapshot["CPS"] != "Unknown"] 
        fig = px.bar(
            cps_snapshot.sort_values("IIC_vs_Agreed_pct", ascending=False), 
            x="CPS", y="IIC_vs_Agreed_pct", color="CPS",
            title=f"Snapshot Facility Constrained / Agreed by CPS at {snapshot_date.date()} (%)",
        )
        apply_layout_style(fig, f"Snapshot Facility Constrained / Agreed by CPS at {snapshot_date.date()} (%)", "CPS", "%", 420)
        st.plotly_chart(fig, use_container_width=True, key="cps_snapshot_compliance_pct")

    with c2:
        cps_heat = snapshot[snapshot["CPS"] != "Unknown"].groupby(["CPS", "Reservoir"], as_index=False)["IIC_mbd"].sum()
        fig = px.density_heatmap(
            cps_heat, x="Reservoir", y="CPS", z="IIC_mbd", title="CPS vs Reservoir Facility Constrained Heatmap",
        )
        style_all_axes(fig)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True, key="cps_heatmap")

with manifold_tab:
    st.markdown("### Manifold Level Constraints & Targets")
    
    f_manifold_plot = f_manifold[f_manifold["Manifold"] != "Unknown"].copy()
    fig = render_spotfire_trellis(f_manifold_plot, node_col="Manifold", cols_wrap=4)
    st.plotly_chart(fig, use_container_width=True, key="manf_spotfire_trend")

    c1, c2 = st.columns(2)
    with c1:
        man_snapshot = snapshot.groupby("Manifold", as_index=False)[["RMT_OE_mbd", "Agreed_mbd", "IIC_mbd"]].sum()
        man_snapshot = add_pct_comparisons(man_snapshot)
        man_snapshot = man_snapshot[man_snapshot["Manifold"] != "Unknown"] 
        fig = px.bar(
            man_snapshot.sort_values("IIC_vs_Agreed_pct", ascending=False), 
            x="Manifold", y="IIC_vs_Agreed_pct", color="Manifold",
            title=f"Snapshot Facility Constrained / Agreed by Manifold at {snapshot_date.date()} (%)",
        )
        apply_layout_style(fig, f"Snapshot Facility Constrained / Agreed by Manifold at {snapshot_date.date()} (%)", "Manifold", "%", 420)
        st.plotly_chart(fig, use_container_width=True, key="manf_snapshot_compliance_pct")

    with c2:
        man_heat = snapshot[snapshot["Manifold"] != "Unknown"].groupby(["Manifold", "Reservoir"], as_index=False)["IIC_mbd"].sum()
        fig = px.density_heatmap(
            man_heat, x="Reservoir", y="Manifold", z="IIC_mbd", title="Manifold vs Reservoir Facility Constrained Heatmap",
        )
        style_all_axes(fig)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True, key="manf_heatmap")

with facility_tab:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            node_snapshot.sort_values("Utilization_pct", ascending=False),
            x="Node", y="Utilization_pct", color="Bottleneck_Flag",
            title=f"Facility Utilization vs Agreed at {snapshot_date.date()} (%)",
            color_discrete_map={"Constrained": "#E53935", "Free": "#43A047"},
            hover_data=["Requested_mbd", "Node_Capacity", "Headroom_mbd", "Scale"],
        )
        apply_layout_style(fig, f"Facility Utilization vs Agreed at {snapshot_date.date()} (%)", "Node", "%", 430)
        fig.add_hline(y=100, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True, key="facility_util_snapshot")

    with c2:
        top_nodes = node_snapshot.sort_values("Utilization_pct", ascending=False)["Node"].head(8).tolist()
        node_trend = filtered_node_perf[filtered_node_perf["Node"].isin(top_nodes)].copy()
        fig = px.line(
            node_trend, x="Date", y="Utilization_pct", color="Node", title="Top Facility Utilization Trends",
        )
        thicken_lines(fig, 3)
        style_all_axes(fig)
        fig.update_layout(height=430)
        fig.add_hline(y=100, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True, key="facility_util_trend")

    c3, c4 = st.columns(2)

    with c3:
        well_bottleneck = (
            snapshot.groupby("Constraint_Node", as_index=False)["IIC_Loss_vs_Agreed_mbd"]
            .sum().sort_values("IIC_Loss_vs_Agreed_mbd", ascending=False)
        )
        fig = px.bar(
            well_bottleneck, x="Constraint_Node", y="IIC_Loss_vs_Agreed_mbd",
            title=f"IIC Loss vs Agreed by Bottleneck Node at {snapshot_date.date()}",
            color_discrete_sequence=["#FB8C00"],
        )
        apply_layout_style(fig, f"IIC Loss vs Agreed by Bottleneck Node at {snapshot_date.date()}", "Node", "mbd", 430)
        st.plotly_chart(fig, use_container_width=True, key="facility_loss_by_node")

    with c4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=f_field["Date"], y=f_field["Agreed_mbd"], mode="lines", name="Agreed (mbd)",
            line=dict(color=SERIES_COLOR_MAP["Agreed_mbd"], width=3)
        ))
        fig.add_trace(go.Scatter(
            x=f_field["Date"], y=f_field["IIC_mbd"], mode="lines", name="Facility Constrained",
            line=dict(color=SERIES_COLOR_MAP["IIC_mbd"], width=4)
        ))
        apply_layout_style(fig, "Field Agreed vs Facility Constrained", "Date", "mbd", 430)
        st.plotly_chart(fig, use_container_width=True, key="facility_field_agreed_iic")

    st.subheader("Facility snapshot table")
    st.dataframe(
        node_snapshot.sort_values(["Utilization_pct", "Requested_mbd"], ascending=[False, False]),
        use_container_width=True, height=280,
    )

with change_tab:
    c1, c2 = st.columns(2)

    changes_timeline = (
        filtered_changes[filtered_changes["Changed_Flag"] == 1]
        .groupby("Change_Date", as_index=False)
        .agg(
            Change_Count=("Well_norm", "count"),
            Net_RMT_Change_mbd=("RMT_Change_mbd", "sum"),
            Net_Agreed_Change_mbd=("Agreed_Change_mbd", "sum"),
        )
    )

    with c1:
        fig = px.bar(
            changes_timeline, x="Change_Date", y="Change_Count", title="Daily Change Count",
            color_discrete_sequence=["#3949AB"],
        )
        apply_layout_style(fig, "Daily Change Count", "Date", "Count", 420)
        st.plotly_chart(fig, use_container_width=True, key="changes_daily_count")

    with c2:
        fig = px.bar(
            changes_timeline, x="Change_Date", y="Net_Agreed_Change_mbd", title="Daily Net Agreed Change (mbd)",
            color_discrete_sequence=["#00897B"],
        )
        apply_layout_style(fig, "Daily Net Agreed Change (mbd)", "Date", "mbd", 420)
        st.plotly_chart(fig, use_container_width=True, key="changes_daily_net")

    st.subheader("Top well changes")
    top_changes = filtered_changes[filtered_changes["Changed_Flag"] == 1].copy()
    top_changes = top_changes[
        ["Change_Date", "Well", "Reservoir", "RMT Rate+OE [bbl/day]", "Agreed Target [bbl/day]", "RMT_Change_mbd", "Agreed_Change_mbd"]
    ].sort_values(["Change_Date", "RMT_Change_mbd"], ascending=[False, True])

    st.dataframe(top_changes, use_container_width=True, height=360)

with exports_tab:
    st.subheader("CSV exports")

    exp_well = filtered_well.copy()
    # Well-level BMT constraint follows the gap-table sign convention:
    # negative values represent shortfall against RMT OE.
    # Historical Agreed Target already incorporates the well constraint.
    exp_well["BMT_Well_Constraint_mbd"] = exp_well["Agreed_mbd"] - exp_well["RMT_OE_mbd"]
    # Keep missing actual BMT capacities as NaN; never infer them from Agreed.
    if "BMT_Capacity_mbd" not in exp_well.columns:
        exp_well["BMT_Capacity_mbd"] = np.nan

    # One row per well, using its most recent date within the active filters.
    # RMT and Agreed originate from that same record, not a mixed-date maximum.
    latest_well_export = (
        exp_well.sort_values(["Date", "Well_norm"])
        .drop_duplicates(subset=["Well_norm"], keep="last")
        .copy()
    )
    latest_well_export = latest_well_export[[
        "Date", "Well", "Reservoir", "Area", "CPS", "Manifold",
        "RMT_OE_mbd", "Agreed_mbd", "BMT_Capacity_mbd", "BMT_Well_Constraint_mbd"
    ]].rename(columns={
        "RMT_OE_mbd": "RMT OE (mbd)",
        "Agreed_mbd": "Agreed Target (mbd)",
        "BMT_Capacity_mbd": "BMT Well Capacity (mbd)",
        "BMT_Well_Constraint_mbd": "BMT Well Constraint (mbd)",
    }).sort_values(["CPS", "Manifold", "Well"]).reset_index(drop=True)

    exp_res = f_res.copy()
    exp_cps = f_cps.copy()
    exp_manifold = f_manifold.copy()
    exp_field = f_field.copy()
    exp_changes = filtered_changes.copy()
    exp_res_change = res_snapshot.copy()
    exp_node = filtered_node_perf.copy()

    # --- NEW: Averages over selected date range ---
    # 1. Average by Well
    exp_well_avg = filtered_well.groupby(["Well", "CPS", "Manifold"], as_index=False)[
        ["RMT_OE_mbd", "Agreed_mbd", "IIC_mbd"]
    ].mean()

    # 2. Average Pivot by CPS & Manifold
    # First sum the daily well contributions per CPS/Manifold, then average across the selected dates
    daily_cps_man = filtered_well.groupby(["Date", "CPS", "Manifold"], as_index=False)[
        ["RMT_OE_mbd", "Agreed_mbd", "IIC_mbd"]
    ].sum()
    exp_pivot_avg = daily_cps_man.groupby(["CPS", "Manifold"], as_index=False)[
        ["RMT_OE_mbd", "Agreed_mbd", "IIC_mbd"]
    ].mean()
    # ----------------------------------------------

    st.download_button("Download filtered well daily CSV", exp_well.to_csv(index=False).encode("utf-8"), file_name="WI_filtered_well_daily_with_final_IIC.csv", mime="text/csv")
    st.download_button("Download latest well-by-well RMT / Agreed / Capacity CSV",
                       latest_well_export.to_csv(index=False).encode("utf-8"),
                       file_name="WI_latest_well_RMT_Agreed_BMT_Capacity.csv", mime="text/csv")
    st.download_button("Download filtered reservoir daily CSV", exp_res.to_csv(index=False).encode("utf-8"), file_name="WI_filtered_reservoir_daily_with_final_IIC.csv", mime="text/csv")
    st.download_button("Download filtered CPS daily CSV", exp_cps.to_csv(index=False).encode("utf-8"), file_name="WI_filtered_cps_daily_with_final_IIC.csv", mime="text/csv")
    st.download_button("Download filtered manifold daily CSV", exp_manifold.to_csv(index=False).encode("utf-8"), file_name="WI_filtered_manifold_daily_with_final_IIC.csv", mime="text/csv")
    st.download_button("Download filtered field daily CSV", exp_field.to_csv(index=False).encode("utf-8"), file_name="WI_filtered_field_daily_with_final_IIC.csv", mime="text/csv")
    st.download_button("Download facility performance CSV", exp_node.to_csv(index=False).encode("utf-8"), file_name="WI_facility_performance_vs_agreed.csv", mime="text/csv")
    st.download_button("Download reservoir snapshot / change CSV", exp_res_change.to_csv(index=False).encode("utf-8"), file_name="WI_reservoir_snapshot_change_with_final_IIC.csv", mime="text/csv")
    st.download_button("Download change events CSV", exp_changes.to_csv(index=False).encode("utf-8"), file_name="WI_filtered_change_events.csv", mime="text/csv")
    
    # New Download Buttons
    st.download_button("Download average by Well CSV", exp_well_avg.to_csv(index=False).encode("utf-8"), file_name="WI_average_by_well.csv", mime="text/csv")
    st.download_button("Download average by CPS & Manifold Pivot CSV", exp_pivot_avg.to_csv(index=False).encode("utf-8"), file_name="WI_average_by_cps_manifold.csv", mime="text/csv")

    st.info(
        "Use the download buttons above to save the filtered results. "
        "Browser-uploaded files do not expose a local target folder."
    )

# Global Missing Mappings Dataframe Extractor
missing_mappings = daily_well_iic[
    (daily_well_iic["Date"] == end_date) & 
    ((daily_well_iic["CPS"] == "Unknown") | (daily_well_iic["Manifold"] == "Unknown"))
][["Well", "Area", "Reservoir", "CPS", "Manifold", "Status"]].drop_duplicates()

st.markdown("---")

with st.expander("⚠️ Missing / Unknown Mappings", expanded=not missing_mappings.empty):
    if missing_mappings.empty:
        st.success("All active wells currently have mapped CPS and Manifold associations!")
    else:
        st.warning(f"Found {len(missing_mappings)} well(s) missing CPS or Manifold data in your mapping CSV.")
        st.dataframe(missing_mappings, use_container_width=True)

if show_data_table:
    st.subheader("Filtered daily well table")
    st.dataframe(filtered_well, use_container_width=True, height=350)

st.caption("Final logic implemented: Area plot for Agreed, Line plots for RMT (Green), Facility Constrained (Black), and Capacity (Red). Missing map tracking included.")

# ============================================================
# ADDITIONS ONLY: DATE-SPECIFIC GAP AND 6% WELL DECLINE FORECAST
# ============================================================
def capacity_gap_table(wells, is_average=False):
    """Decompose each well's shortfall; do not double-count cumulative gaps.

    Historical: Agreed = supplied approved target, IIC = original calculate_iic.
    Forecast: Agreed = min(RMT, declining BMT capacity), IIC = same network solver.
    The five gap metrics reproduce the screenshot's progression, with CPS total
    and MF-to-RMT measures shown as cumulative (not additive) columns.
    """
    if wells.empty:
        return pd.DataFrame()
    w = wells.copy()
    w["BMT_Capacity_mbd"] = w.get("BMT_Capacity_mbd", pd.Series(np.nan, index=w.index))
    w["Well_Gap_mbd"] = w["Agreed_mbd"] - w["RMT_OE_mbd"]
    # Scale information comes directly from the existing manifold/CPS solver.
    w["After_Manifold_mbd"] = w["Agreed_mbd"] * w["Manifold_Scale"]
    w["Manifold_Gap_vs_Agreed_mbd"] = w["After_Manifold_mbd"] - w["Agreed_mbd"]
    w["Manifold_Gap_vs_RMT_mbd"] = w["After_Manifold_mbd"] - w["RMT_OE_mbd"]
    w["CPS_Gap_vs_Agreed_mbd"] = w["IIC_mbd"] - w["After_Manifold_mbd"]
    w["CPS_Gap_vs_RMT_mbd"] = w["IIC_mbd"] - w["RMT_OE_mbd"]
    metrics = ["RMT_OE_mbd", "BMT_Capacity_mbd", "Agreed_mbd", "After_Manifold_mbd", "IIC_mbd",
               "Well_Gap_mbd", "Manifold_Gap_vs_Agreed_mbd", "Manifold_Gap_vs_RMT_mbd",
               "CPS_Gap_vs_Agreed_mbd", "CPS_Gap_vs_RMT_mbd"]
    # Aggregate by date and location first, THEN average across days; otherwise
    # variable well counts can distort period-average field / facility rates.
    day = w.groupby(["Date", "CPS", "Manifold", "Reservoir"], as_index=False)[metrics].sum(min_count=1)
    if is_average:
        out = day.groupby(["CPS", "Manifold", "Reservoir"], as_index=False)[metrics].mean()
    else:
        out = day.drop(columns=["Date"])
    return out.rename(columns={
        "RMT_OE_mbd": "RMT OE (mbd)",
        "BMT_Capacity_mbd": "BMT Well Capacity (mbd)",
        "Agreed_mbd": "Agreed Target (mbd)",
        "After_Manifold_mbd": "After Manifold (mbd)",
        "IIC_mbd": "Final IIC (mbd)",
        "Well_Gap_mbd": "Well Constraint to RMT (mbd)",
        "Manifold_Gap_vs_Agreed_mbd": "MF Constraint to Agreed (mbd)",
        "Manifold_Gap_vs_RMT_mbd": "MF Constraint to RMT (mbd)",
        "CPS_Gap_vs_Agreed_mbd": "CPS Incremental Constraint (mbd)",
        "CPS_Gap_vs_RMT_mbd": "Total Gap to RMT (mbd)",
    }).sort_values(["CPS", "Manifold", "Reservoir"])


def total_gap_row(well_data, avg=False):
    """Total calculated from well contributions so cross-group constraints aren't double-counted."""
    if well_data.empty:
        return pd.DataFrame()
    temp = well_data.copy()
    temp["After_Manifold_mbd"] = temp["Agreed_mbd"] * temp["Manifold_Scale"]
    temp["Well_Gap_mbd"] = temp["Agreed_mbd"] - temp["RMT_OE_mbd"]
    temp["MF_Inc_mbd"] = temp["After_Manifold_mbd"] - temp["Agreed_mbd"]
    temp["MF_Cum_mbd"] = temp["After_Manifold_mbd"] - temp["RMT_OE_mbd"]
    temp["CPS_Inc_mbd"] = temp["IIC_mbd"] - temp["After_Manifold_mbd"]
    temp["Total_Gap_mbd"] = temp["IIC_mbd"] - temp["RMT_OE_mbd"]
    cols = ["RMT_OE_mbd", "Agreed_mbd", "After_Manifold_mbd", "IIC_mbd", "Well_Gap_mbd", "MF_Inc_mbd", "MF_Cum_mbd", "CPS_Inc_mbd", "Total_Gap_mbd"]
    totals = temp.groupby("Date")[cols].sum().mean() if avg else temp[cols].sum()
    return pd.DataFrame([{
        "RMT OE (mbd)": totals["RMT_OE_mbd"],
        "Agreed Target (mbd)": totals["Agreed_mbd"],
        "After Manifold (mbd)": totals["After_Manifold_mbd"],
        "Final IIC (mbd)": totals["IIC_mbd"],
        "Well Constraint to RMT (mbd)": totals["Well_Gap_mbd"],
        "MF Constraint to Agreed (mbd)": totals["MF_Inc_mbd"],
        "MF Constraint to RMT (mbd)": totals["MF_Cum_mbd"],
        "CPS Incremental Constraint (mbd)": totals["CPS_Inc_mbd"],
        "Total Gap to RMT (mbd)": totals["Total_Gap_mbd"],
    }])


def extend_capacity_to_forecast(facilities, forecast_dates):
    """Last known facility capacity carried forward; no new projects assumed."""
    dates = pd.DatetimeIndex(forecast_dates)
    frames = []
    for node, group in facilities.groupby("FAC_NAME"):
        g = group.sort_values("DATE").drop_duplicates("DATE", keep="last")
        lookup = pd.merge_asof(pd.DataFrame({"Date": dates}),
                               g[["DATE", "CAP", "CAP-OE"]].rename(columns={"DATE": "Date"}),
                               on="Date", direction="backward")
        lookup["FAC_NAME"] = node
        # Missing earlier capacities stay NaN and are treated as unconstrained,
        # rather than making up a zero capacity before first observation.
        frames.append(lookup)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Date", "FAC_NAME", "CAP", "CAP-OE"])


with gap_tab:
    st.subheader("Capacity Gap Analysis — exact date")
    selected_gap_day = pd.Timestamp(st.date_input("Capacity gap date", value=max_date,
                                                  min_value=min_date, max_value=max_date,
                                                  key="capacity_gap_exact_date"))
    # Existing global filters apply, but the date here is independent of the trend slider.
    gap_source = daily_well_iic[
        (daily_well_iic["Date"] == selected_gap_day)
        & daily_well_iic["Area"].isin(selected_areas)
        & daily_well_iic["Reservoir"].isin(selected_reservoirs)
        & daily_well_iic["CPS"].isin(selected_cps)
        & daily_well_iic["Manifold"].isin(selected_manifolds)
        & daily_well_iic["Well"].isin(selected_wells)
    ].copy()
    if gap_source.empty:
        st.warning("No mapped well data on the chosen date with these filters.")
    else:
        gap_result = capacity_gap_table(gap_source)
        st.dataframe(gap_result.style.format(precision=1, na_rep="—"), use_container_width=True, hide_index=True)
        st.markdown("**Field total (mbd)**")
        st.dataframe(total_gap_row(gap_source).style.format(precision=1), use_container_width=True, hide_index=True)
        st.download_button("Download exact-date capacity gap CSV", gap_result.to_csv(index=False).encode("utf-8"),
                           file_name=f"capacity_gap_{selected_gap_day:%Y%m%d}.csv", mime="text/csv")
    st.caption("Negative gaps mean a shortfall. MF-to-RMT and Total-to-RMT are cumulative metrics, not additional losses. "
               "The manifold and CPS incremental losses form a non-overlapping waterfall from Agreed to final IIC.")

with forecast_tab:
    st.subheader("Well-by-well RMT prediction — BMT Well Capacity decline")
    st.caption("Fixed baseline RMT OE, 6% effective annual BMT well-capacity decline, no added wells. "
               "Forecast Agreed = min(RMT OE, predicted BMT Well Capacity). "
               "Existing network constraint logic is then rerun for every prediction day.")
    f0_col, f1_col = st.columns(2)
    with f0_col:
        pred_start = pd.Timestamp(st.date_input("Prediction start", value=max_date, min_value=min_date,
                                                key="prediction_start"))
    with f1_col:
        pred_end = pd.Timestamp(st.date_input("Prediction end", value=(pred_start + pd.DateOffset(years=1)).date(),
                                              min_value=pred_start.date(), key="prediction_end"))
    decline_pct = 6.0
    if (pred_end - pred_start).days > 3653:
        st.error("Choose a forecast of 10 years or less to keep the daily model responsive.")
    else:
        base = daily_well_iic[daily_well_iic["Date"] == pred_start].copy()
        base = base[
            base["Area"].isin(selected_areas) & base["Reservoir"].isin(selected_reservoirs)
            & base["CPS"].isin(selected_cps) & base["Manifold"].isin(selected_manifolds)
            & base["Well"].isin(selected_wells)
        ].copy()
        if "BMT_Capacity_mbd" not in base.columns or base["BMT_Capacity_mbd"].isna().any():
            st.warning("A real BMT Well Capacity is missing for one or more selected wells. "
                       "Provide it in your Target CSV or the optional sidebar BMT capacity CSV. "
                       "The forecast is not run using Agreed as an invented capacity.")
            if "BMT_Capacity_mbd" in base.columns:
                st.dataframe(base[base["BMT_Capacity_mbd"].isna()][["Well", "Reservoir", "CPS", "Manifold"]].drop_duplicates(),
                             use_container_width=True, hide_index=True)
        elif base.empty:
            st.warning("No wells on prediction start date.")
        else:
            future_dates = pd.date_range(pred_start, pred_end, freq="D")
            # Vectorized well-by-well daily forecast, baseline rate at prediction start.
            w = base.assign(_key=1).merge(pd.DataFrame({"Date": future_dates, "_key": 1}), on="_key", suffixes=("_baseline", ""))
            w = w.drop(columns=["_key", "Date_baseline"], errors="ignore")
            t = (w["Date"] - pred_start).dt.days / 365.25
            w["BMT_Capacity_mbd"] = w["BMT_Capacity_mbd"] * (1.0 - decline_pct / 100.0) ** t
            w["Agreed_mbd"] = np.minimum(w["RMT_OE_mbd"], w["BMT_Capacity_mbd"]).clip(lower=0)
            cap_future = extend_capacity_to_forecast(fac_df, future_dates)
            forecast_wells, forecast_nodes = calculate_iic(w, cap_future, cap_col=cap_basis)
            forecast_gaps = capacity_gap_table(forecast_wells, is_average=True)
            forecast_totals = total_gap_row(forecast_wells, avg=True)
            st.markdown("#### Average daily capacity gap across forecast period (mbd)")
            st.dataframe(forecast_gaps.style.format(precision=1, na_rep="—"), use_container_width=True, hide_index=True)
            st.markdown("**Field average daily total (mbd)**")
            st.dataframe(forecast_totals.style.format(precision=1), use_container_width=True, hide_index=True)
            series = forecast_wells.groupby("Date", as_index=False)[["RMT_OE_mbd", "BMT_Capacity_mbd", "Agreed_mbd", "IIC_mbd"]].sum()
            fig = px.line(series, x="Date", y=["RMT_OE_mbd", "BMT_Capacity_mbd", "Agreed_mbd", "IIC_mbd"],
                          title=f"RMT / BMT well capacity / agreed / IIC prediction — {decline_pct:.0f}% annual decline")
            apply_layout_style(fig, "Well-by-well rolled-up injection prediction", "Date", "mbd", 460)
            st.plotly_chart(fig, use_container_width=True, key="wi_prediction_trend")
            forecast_export = forecast_wells[["Date", "Well", "Reservoir", "Area", "CPS", "Manifold",
                                              "RMT_OE_mbd", "BMT_Capacity_mbd", "Agreed_mbd", "IIC_mbd",
                                              "Manifold_Scale", "CPS_Scale", "Constraint_Node"]]
            st.download_button("Download daily well-by-well prediction CSV", forecast_export.to_csv(index=False).encode("utf-8"),
                               file_name="WI_well_by_well_prediction.csv", mime="text/csv")
            st.download_button("Download average forecast capacity gap CSV", forecast_gaps.to_csv(index=False).encode("utf-8"),
                               file_name="WI_forecast_average_capacity_gap.csv", mime="text/csv")
            st.caption("Gap averages are daily rate averages, NOT cumulative volumes. Capacities remain at their "
                       "last observed settings; BMT well capacities decline from the selected baseline date.")
