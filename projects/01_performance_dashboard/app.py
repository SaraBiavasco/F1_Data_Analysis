#per farlo ripartire streamlit run F1_Data_Analysis/projects/01_performance_dashboard/app.pystreamlit run F1_Data_Analysis/projects/01_performance_dashboard/app.py
"""
F1 Performance Dashboard

1. Load a Formula 1 session with FastF1.
2. Store it in Streamlit session_state so widget changes do not reload the data.
3. Prepare reusable lap/sector datasets.
4. Compare two selected drivers for pace and tyres.
5. Analyse session-wide evolution and team sector performance.
6. Use telemetry X/Y coordinates to draw the circuit.

The code is organised as:
- configuration / styling
- helper functions
- session loading
- shared data preparation
- one Streamlit tab per analysis

"""

import streamlit as st
import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIGURATION
# =========================================================
# This must be called before creating any Streamlit elements.
# layout="wide" uses more horizontal space, which is useful for dashboards,
# side-by-side metric cards and large Plotly charts.
st.set_page_config(page_title="F1 Performance Dashboard", layout="wide")

# Streamlit gives us the functionality, but CSS lets us customise the visual style.
# Here we create a dark F1-inspired interface:
# - dark main background
# - dark sidebar
# - white/light-grey text
# - red accent colour for buttons and the selected tab
#
# unsafe_allow_html=True is required because Streamlit normally sanitises raw HTML/CSS.
st.markdown("""
<style>
.stApp {background-color:#0E1117;color:#F5F5F5;}
.block-container {padding-top:2rem;padding-bottom:3rem;max-width:1400px;}
header[data-testid="stHeader"] {background-color:#0E1117;}
section[data-testid="stSidebar"] {background-color:#151922;border-right:1px solid #2A2F3A;}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {color:#F5F5F5 !important;}
div[data-baseweb="select"] > div {background-color:#20252F !important;border-color:#343A46 !important;color:#F5F5F5 !important;}
div[data-baseweb="select"] span {color:#F5F5F5 !important;}
div[data-baseweb="select"] svg {fill:#F5F5F5 !important;}
div[role="listbox"], div[role="option"] {background-color:#20252F !important;color:#F5F5F5 !important;}
div[role="option"]:hover {background-color:#303642 !important;}
div.stButton > button,
section[data-testid="stSidebar"] button[kind="secondary"] {background-color:#E10600 !important;color:white !important;border:1px solid #E10600 !important;border-radius:8px;font-weight:600;}
div.stButton > button:hover,
section[data-testid="stSidebar"] button[kind="secondary"]:hover {background-color:#FF1E16 !important;border-color:#FF1E16 !important;}
h1,h2,h3,h4 {color:#F5F5F5 !important;}
p {color:#C5CAD3;}
h1 {font-weight:800 !important;letter-spacing:-1px;}
h2,h3 {font-weight:700 !important;}
div[data-testid="stMetric"] {background-color:#171B24;border:1px solid #2A2F3A;padding:18px;border-radius:12px;}
div[data-testid="stMetric"] * {color:#F5F5F5 !important;}
button[data-baseweb="tab"] {font-weight:600;}
button[data-baseweb="tab"] p {color:#B8BDC7 !important;}
button[data-baseweb="tab"][aria-selected="true"] {border-bottom:3px solid #E10600 !important;}
button[data-baseweb="tab"][aria-selected="true"] p {color:#E10600 !important;}
div[data-testid="stExpander"] {background-color:#151922;border-color:#2A2F3A;}
div[data-testid="stPlotlyChart"] {margin-top:.5rem;margin-bottom:1rem;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
# These dictionaries keep colours consistent across the whole dashboard.
# By defining them once, we avoid repeating colour codes in multiple plots.
#
# Sector convention used in this project:
# S1 = green, S2 = yellow, S3 = red.
SECTOR_COLORS = {"Sector 1": "#00C853", "Sector 2": "#FFD600", "Sector 3": "#E10600"}
# Standard tyre colours used throughout the dashboard.
# They are intentionally NOT team colours because here colour represents compound.
COMPOUND_COLORS = {
    "SOFT": "#E10600",
    "MEDIUM": "#FFD12E",
    "HARD": "#D9D9D9",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF"
}

# =========================================================
# HELPER FUNCTIONS
# =========================================================
# Helper functions keep repeated logic outside the main dashboard code.
# This makes the app easier to read and easier to maintain.

def format_lap_time(value):
    """Convert a FastF1 Timedelta into M:SS.mmm format."""
    # Some laps (for example pit/out laps or incomplete laps) have no valid time.
    # pd.isna() handles pandas/NumPy missing values safely.
    if pd.isna(value):
        return None

    # FastF1 stores lap times as pandas Timedelta objects.
    # We convert them to total seconds and then format them as M:SS.mmm.
    seconds = value.total_seconds()
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"


def apply_dark_plotly_style(fig):
    """Apply one consistent dark theme to every Plotly chart."""

    # Plotly charts are styled separately from Streamlit CSS.
    # This function ensures that chart background, titles, legends,
    # axis labels and tick labels are all readable on the dark dashboard.
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#F5F5F5"),
        title_font=dict(color="#FFFFFF", size=22),
        legend=dict(font=dict(color="#F5F5F5"), title_font=dict(color="#F5F5F5"))
    )
    fig.update_xaxes(
        title_font=dict(color="#F5F5F5"),
        tickfont=dict(color="#C5CAD3"),
        gridcolor="#2A2F3A"
    )
    fig.update_yaxes(
        title_font=dict(color="#F5F5F5"),
        tickfont=dict(color="#C5CAD3"),
        gridcolor="#2A2F3A"
    )
    return fig


def get_driver_team_color(driver, results, session):
    """Get the driver's team colour for the selected season/session."""

    # Filter the session results to find the selected driver.
    # We use Abbreviation because FastF1 normally stores drivers as VER, HAM, LEC, etc.
    driver_result = results[results["Abbreviation"] == driver]
    if driver_result.empty:
        return "#B0B0B0"

    # FastF1 sometimes already provides the team colour directly in session.results.
    # This is our preferred source because it automatically matches the selected season.
    if "TeamColor" in driver_result.columns:
        color = driver_result.iloc[0]["TeamColor"]
        if pd.notna(color):
            color = str(color)
            return color if color.startswith("#") else f"#{color}"

    # If TeamColor is not available, ask FastF1's plotting utility for the team colour.
    # The fallback grey prevents the application from failing if a colour cannot be found.
    try:
        team = driver_result.iloc[0]["TeamName"]
        return fastf1.plotting.get_team_color(team, session=session)
    except Exception:
        return "#B0B0B0"


def calculate_degradation(data, driver):
    """Estimate lap-time change per tyre lap using a simple linear regression."""

    # Keep only the selected driver's tyre age and lap time.
    # Missing values are removed because regression cannot use NaNs.
    driver_data = data[data["Driver"] == driver][["TyreLife", "LapTimeSeconds"]].dropna()
    # If the same tyre age appears more than once (for example across separate stints),
    # we take the mean lap time for that tyre age before fitting the trend.
    driver_data = driver_data.groupby("TyreLife", as_index=False)["LapTimeSeconds"].mean()

    # A straight line needs at least two points.
    if len(driver_data) < 2:
        return None
    # np.polyfit(..., 1) fits: LapTime = slope * TyreLife + intercept.
    # The slope is our estimated degradation rate in seconds per tyre lap.
    # Positive slope -> observed lap times become slower as the tyre ages.
    # Negative slope -> other effects (fuel burn, track evolution, traffic) dominate.
    slope, _ = np.polyfit(driver_data["TyreLife"], driver_data["LapTimeSeconds"], 1)
    return slope


def load_driver_telemetry(session, driver):
    """Return the selected driver's fastest lap and telemetry, when available."""

    # Telemetry is heavier than ordinary lap data, so we load it only when needed.
    # pick_drivers() selects the driver's laps; pick_fastest() selects their best valid lap.
    try:
        driver_laps = session.laps.pick_drivers(driver)
        fastest_lap = driver_laps.pick_fastest()
        if fastest_lap is None:
            return None, pd.DataFrame()

        # get_telemetry() combines car data and position data.
        # X and Y reconstruct the circuit shape; Speed is used for the speed map.
        telemetry = fastest_lap.get_telemetry().dropna(subset=["X", "Y", "Speed"]).copy()
        return fastest_lap, telemetry
    except Exception:
        return None, pd.DataFrame()


# =========================================================
# HEADER + SIDEBAR CONTROLS
# =========================================================
# The main page contains the analyses, while the sidebar contains user inputs.
st.title("F1 Performance Dashboard")
st.caption("Interactive Formula 1 performance analysis powered by FastF1.")

st.sidebar.title("F1 Dashboard")
st.sidebar.caption("Session & comparison controls")
st.sidebar.divider()
st.sidebar.subheader("Session")

# The user chooses the session to analyse.
# These values are only selections until "Load Session" is pressed.
season = st.sidebar.selectbox(
    "Season",
    [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010]
)
grand_prix = st.sidebar.selectbox(
    "Grand Prix",
    ["Monaco", "Monza", "Silverstone", "Spa", "Bahrain", "Suzuka", "Abu Dhabi"]
)
session_type = st.sidebar.selectbox("Session", ["Race", "Qualifying", "FP1", "FP2", "FP3"])
# Loading FastF1 data can take time, so we do it only after an explicit button click.
load_button = st.sidebar.button("Load Session", use_container_width=True)

if load_button:
    try:
        # get_session() creates the session object; session.load() downloads/loads
        # timing, results, laps, weather and telemetry references into that object.
        session = fastf1.get_session(season, grand_prix, session_type)
        session.load()

        # Streamlit reruns the whole script whenever a widget changes.
        # session_state preserves the loaded FastF1 session between those reruns,
        # so changing Driver 1/Driver 2 does NOT force us to reload the GP.
        st.session_state["session"] = session
        # Store the labels of the ACTUALLY loaded session.
        # This avoids showing "2025" if the user selects 2025 but has not clicked Load yet.
        st.session_state["session_info"] = {
            "season": season,
            "grand_prix": grand_prix,
            "session_type": session_type
        }
        st.success("Session loaded successfully!")
    except Exception as exc:
        st.error(f"Unable to load this session: {exc}")


# =========================================================
# MAIN DASHBOARD
# =========================================================
# Nothing below is displayed until a session has successfully been loaded.
if "session" in st.session_state:
    session = st.session_state["session"]
    session_info = st.session_state.get("session_info", {})

    # FastF1 gives us two central tables:
    # results -> one row per driver, with position/team information
    # laps    -> one row per lap, with timing, compound, sectors, tyre life, etc.
    #
    # We copy them so our transformations do not modify FastF1's original objects.
    results = session.results.copy()
    laps = session.laps.copy()
    # We keep two versions of LapTime:
    # - a readable string (e.g. 1:26.103) for tables/cards
    # - numeric seconds for calculations and charts.
    laps["LapTimeFormatted"] = laps["LapTime"].apply(format_lap_time)
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

    if session_info:
        st.markdown(
            f"### {session_info['grand_prix']} · "
            f"{session_info['season']} · "
            f"{session_info['session_type']}"
        )

    # =========================================================
    # SESSION OVERVIEW
    # =========================================================
    # High-level KPIs shown before the detailed analysis tabs.
    st.subheader("Session Overview")

    # Position can occasionally contain non-numeric/missing values,
    # so we safely coerce it to numeric before selecting P1.
    p1_rows = results[pd.to_numeric(results["Position"], errors="coerce") == 1]
    p1_driver = p1_rows.iloc[0]["Abbreviation"] if not p1_rows.empty else "N/A"

    # Only laps with an actual recorded time can be candidates for fastest lap.
    valid_laps = laps.dropna(subset=["LapTimeSeconds"])
    if not valid_laps.empty:
        fastest_lap_row = valid_laps.loc[valid_laps["LapTimeSeconds"].idxmin()]
        fastest_driver = fastest_lap_row["Driver"]
        fastest_time = fastest_lap_row["LapTimeFormatted"]
    else:
        fastest_driver, fastest_time = "N/A", "N/A"

    # "Recorded Laps" counts valid lap observations across ALL drivers,
    # not the number of laps in the race itself.
    number_of_drivers = results["Abbreviation"].dropna().nunique()
    total_recorded_laps = len(valid_laps)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("P1", p1_driver, border=True)

    with col2:
        st.metric("Fastest Lap", fastest_driver, border=True)
        st.caption(f"⏱ {fastest_time}")

    col3.metric("Drivers", number_of_drivers, border=True)
    col4.metric("Recorded Laps", total_recorded_laps, border=True)

    # =========================================================
    # DRIVER SELECTION + SHARED DRIVER DATA
    # =========================================================
    # The comparison tabs reuse the same two selected drivers.
    drivers = sorted(laps["Driver"].dropna().unique())

    st.sidebar.divider()
    st.sidebar.subheader("Driver Comparison")

    driver_1 = st.sidebar.selectbox("Driver 1", drivers, key="driver_1")
    driver_2 = st.sidebar.selectbox(
        "Driver 2",
        drivers,
        index=1 if len(drivers) > 1 else 0,
        key="driver_2"
    )

    # Driver lines use the colour of their team in the selected season.
    # If both drivers are teammates, both lines can have the same colour;
    # the legend/hover label still identifies them.
    driver_1_color = get_driver_team_color(driver_1, results, session)
    driver_2_color = get_driver_team_color(driver_2, results, session)

    # Extract all laps belonging to each selected driver.
    driver_1_laps = laps[laps["Driver"] == driver_1].copy()
    driver_2_laps = laps[laps["Driver"] == driver_2].copy()

    # Remove laps without a valid lap time before computing pace.
    clean_driver_1_laps = driver_1_laps.dropna(subset=["LapTimeSeconds"]).copy()
    clean_driver_2_laps = driver_2_laps.dropna(subset=["LapTimeSeconds"]).copy()

    # Very slow laps can come from pit stops, Safety Car, traffic or incidents.
    # We keep laps below 120% of each driver's median lap time.
    # This is a simple outlier filter, NOT an attempt to perfectly isolate "clean air" laps.
    if not clean_driver_1_laps.empty:
        median_1 = clean_driver_1_laps["LapTimeSeconds"].median()
        clean_driver_1_laps = clean_driver_1_laps[
            clean_driver_1_laps["LapTimeSeconds"] < median_1 * 1.20
        ]

    if not clean_driver_2_laps.empty:
        median_2 = clean_driver_2_laps["LapTimeSeconds"].median()
        clean_driver_2_laps = clean_driver_2_laps[
            clean_driver_2_laps["LapTimeSeconds"] < median_2 * 1.20
        ]

    # =========================================================
    # SHARED SECTOR DATA
    # =========================================================
    # Sector data is prepared once because both Team Sectors and Raw Data use it.
    # FastF1 stores sector times as Timedelta objects; calculations are easier in seconds.
    sector_data = laps[["Driver", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    sector_data["Sector1Seconds"] = sector_data["Sector1Time"].dt.total_seconds()
    sector_data["Sector2Seconds"] = sector_data["Sector2Time"].dt.total_seconds()
    sector_data["Sector3Seconds"] = sector_data["Sector3Time"].dt.total_seconds()

    # For each driver, take their personal best time in S1, S2 and S3.
    # These three best sectors do NOT need to come from the same lap.
    driver_sectors = (
        sector_data.groupby("Driver", as_index=False)
        .agg(
            BestS1=("Sector1Seconds", "min"),
            BestS2=("Sector2Seconds", "min"),
            BestS3=("Sector3Seconds", "min")
        )
    )
    # Theoretical Best = best S1 + best S2 + best S3.
    # It represents the lap the driver could theoretically produce if their
    # three personal-best sectors were combined into one lap.
    driver_sectors["TheoreticalBest"] = (
        driver_sectors["BestS1"]
        + driver_sectors["BestS2"]
        + driver_sectors["BestS3"]
    )

    # Add team information to each driver's sector summary.
    driver_team = (
        results[["Abbreviation", "TeamName"]]
        .rename(columns={"Abbreviation": "Driver"})
        .drop_duplicates("Driver")
    )
    driver_sectors = (
        driver_sectors.merge(driver_team, on="Driver", how="left")
        .sort_values("TheoreticalBest")
        .reset_index(drop=True)
    )

    # Each tab focuses on one analytical question while keeping the page compact.
    tab_race, tab_tyres, tab_track, tab_sectors, tab_circuit, tab_raw = st.tabs([
        "Race Pace",
        "Tyre Strategy",
        "Track Evolution",
        "Team Sectors",
        "Circuit View",
        "Raw Data"
    ])

    # =========================================================
    # RACE PACE
    # =========================================================
    with tab_race:
        # =====================================================
        # RACE PACE TAB
        # =====================================================
        # Compare the selected drivers' lap-time evolution across the session.
        st.subheader("Race Pace")
        st.caption(
            f"Comparison between {driver_1} and {driver_2}. "
            "Very slow laps are filtered to reduce the effect of pit stops and major outliers."
        )

        if clean_driver_1_laps.empty or clean_driver_2_laps.empty:
            st.warning("Not enough valid lap data for this driver comparison.")
        else:
            # Create an empty Plotly figure, then add one line per driver.
            fig_race = go.Figure()

            fig_race.add_trace(go.Scatter(
                x=clean_driver_1_laps["LapNumber"],
                y=clean_driver_1_laps["LapTimeSeconds"],
                mode="lines",
                name=driver_1,
                line=dict(color=driver_1_color, width=3),
                hovertemplate=(
                    f"<b>{driver_1}</b><br>"
                    "Lap: %{x:.0f}<br>"
                    "Lap Time: %{y:.3f} s"
                    "<extra></extra>"
                )
            ))

            fig_race.add_trace(go.Scatter(
                x=clean_driver_2_laps["LapNumber"],
                y=clean_driver_2_laps["LapTimeSeconds"],
                mode="lines",
                name=driver_2,
                line=dict(color=driver_2_color, width=3),
                hovertemplate=(
                    f"<b>{driver_2}</b><br>"
                    "Lap: %{x:.0f}<br>"
                    "Lap Time: %{y:.3f} s"
                    "<extra></extra>"
                )
            ))

            fig_race.update_layout(
                title="Race Pace Comparison",
                xaxis_title="Lap",
                yaxis_title="Lap Time (s)",
                height=500,
                hovermode="x unified",
                legend_title_text="Driver",
                margin=dict(l=20, r=20, t=60, b=20)
            )

            fig_race = apply_dark_plotly_style(fig_race)
            st.plotly_chart(fig_race, use_container_width=True)

            # Mean lap time is a simple pace summary after the outlier filter.
            # The absolute difference becomes the average pace gap.
            avg_pace_1 = clean_driver_1_laps["LapTimeSeconds"].mean()
            avg_pace_2 = clean_driver_2_laps["LapTimeSeconds"].mean()
            pace_gap = abs(avg_pace_1 - avg_pace_2)

            col1, col2, col3 = st.columns(3)
            col1.metric(f"{driver_1} Avg Pace", f"{avg_pace_1:.3f} s", border=True)
            col2.metric(f"{driver_2} Avg Pace", f"{avg_pace_2:.3f} s", border=True)
            col3.metric("Average Pace Gap", f"{pace_gap:.3f} s", border=True)

    # =========================================================
    # TYRE STRATEGY
    # =========================================================
    with tab_tyres:
        # =====================================================
        # TYRE STRATEGY TAB
        # =====================================================
        # This tab answers two questions:
        # 1) Which compounds/stints did each driver use?
        # 2) How did observed lap time change as tyre age increased?
        st.subheader("Tyre Strategy")

        # Keep only columns needed to reconstruct each driver's tyre strategy.
        driver_1_stints = driver_1_laps[
            ["Stint", "Compound", "LapNumber", "TyreLife", "LapTimeSeconds"]
        ].copy()

        driver_2_stints = driver_2_laps[
            ["Stint", "Compound", "LapNumber", "TyreLife", "LapTimeSeconds"]
        ].copy()

        # Group by Stint + Compound.
        # For each stint we record start lap, end lap and number of recorded laps.
        driver_1_summary = (
            driver_1_stints.groupby(["Stint", "Compound"], as_index=False)
            .agg(
                StartLap=("LapNumber", "min"),
                EndLap=("LapNumber", "max"),
                Laps=("LapNumber", "count")
            )
        )

        driver_2_summary = (
            driver_2_stints.groupby(["Stint", "Compound"], as_index=False)
            .agg(
                StartLap=("LapNumber", "min"),
                EndLap=("LapNumber", "max"),
                Laps=("LapNumber", "count")
            )
        )

        # Build a horizontal timeline.
        # Each bar begins at StartLap and its width equals the stint length.
        fig_strategy = go.Figure()

        # A compound may appear in several stints; this set prevents duplicate legend entries.
        shown_compounds = set()

        for driver, summary in [
            (driver_1, driver_1_summary),
            (driver_2, driver_2_summary)
        ]:
            for _, stint in summary.iterrows():
                compound = stint["Compound"]
                start_lap = stint["StartLap"]
                end_lap = stint["EndLap"]
                stint_length = stint["Laps"]

                fig_strategy.add_trace(go.Bar(
                    y=[driver],
                    x=[stint_length],
                    base=[start_lap - 1],
                    orientation="h",
                    name=compound,
                    marker=dict(
                        color=COMPOUND_COLORS.get(compound, "#888888"),
                        line=dict(color="#0E1117", width=1)
                    ),
                    legendgroup=compound,
                    showlegend=compound not in shown_compounds,
                    hovertemplate=(
                        f"<b>{driver}</b><br>"
                        f"Compound: {compound}<br>"
                        f"Laps: {int(start_lap)}–{int(end_lap)}<br>"
                        f"Stint length: {int(stint_length)} laps"
                        "<extra></extra>"
                    )
                ))

                shown_compounds.add(compound)

        fig_strategy.update_layout(
            title="Tyre Strategy Timeline",
            xaxis_title="Lap",
            yaxis_title="",
            barmode="overlay",
            height=350,
            legend_title_text="Compound",
            margin=dict(l=20, r=20, t=60, b=20)
        )

        fig_strategy.update_yaxes(
            categoryorder="array",
            categoryarray=[driver_2, driver_1]
        )

        fig_strategy = apply_dark_plotly_style(fig_strategy)
        st.plotly_chart(fig_strategy, use_container_width=True)

        # Keep raw stint tables available without permanently occupying the dashboard.
        with st.expander("View stint details"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**{driver_1}**")
                st.dataframe(
                    driver_1_summary,
                    use_container_width=True,
                    hide_index=True
                )

            with col2:
                st.write(f"**{driver_2}**")
                st.dataframe(
                    driver_2_summary,
                    use_container_width=True,
                    hide_index=True
                )

        # -----------------------------------------------------
        # TYRE DEGRADATION / OBSERVED TYRE-AGE TREND
        # -----------------------------------------------------
        st.subheader("Tyre Degradation")
        st.caption(
            "Lap time evolution as tyre age increases. "
            "The estimated trend also reflects fuel load, traffic and track evolution."
        )

        tyre_deg_1 = clean_driver_1_laps[
            ["TyreLife", "LapTimeSeconds", "Compound"]
        ].copy()

        tyre_deg_2 = clean_driver_2_laps[
            ["TyreLife", "LapTimeSeconds", "Compound"]
        ].copy()

        # Add the driver label and combine both drivers into one analysis table.
        tyre_degradation = pd.concat([
            tyre_deg_1.assign(Driver=driver_1),
            tyre_deg_2.assign(Driver=driver_2)
        ])

        # Only show compounds that actually appear for the selected drivers/session.
        available_compounds = sorted(
            tyre_degradation["Compound"].dropna().unique()
        )

        if available_compounds:
            selected_compound = st.selectbox(
                "Compound for degradation analysis",
                available_compounds,
                key="degradation_compound"
            )

            # Compare drivers only on the compound chosen by the user.
            compound_data = tyre_degradation[
                tyre_degradation["Compound"] == selected_compound
            ].copy()

            fig_degradation = go.Figure()
            driver_colors = {
                driver_1: driver_1_color,
                driver_2: driver_2_color
            }

            for driver in [driver_1, driver_2]:
                driver_deg = compound_data[
                    compound_data["Driver"] == driver
                ][["TyreLife", "LapTimeSeconds"]].dropna()

                # Multiple observations with the same TyreLife are averaged,
                # giving one point per tyre age in the chart.
                driver_deg = (
                    driver_deg.groupby("TyreLife", as_index=False)["LapTimeSeconds"]
                    .mean()
                    .sort_values("TyreLife")
                )

                fig_degradation.add_trace(go.Scatter(
                    x=driver_deg["TyreLife"],
                    y=driver_deg["LapTimeSeconds"],
                    mode="lines+markers",
                    name=driver,
                    line=dict(color=driver_colors[driver], width=2),
                    marker=dict(size=7),
                    hovertemplate=(
                        f"<b>{driver}</b><br>"
                        "Tyre Age: %{x:.0f} laps<br>"
                        "Lap Time: %{y:.3f} s"
                        "<extra></extra>"
                    )
                ))

            fig_degradation.update_layout(
                title=f"{selected_compound} Tyre Performance",
                xaxis_title="Tyre Age (laps)",
                yaxis_title="Lap Time (s)",
                height=500,
                hovermode="x unified",
                legend_title_text="Driver",
                margin=dict(l=20, r=20, t=60, b=20)
            )

            fig_degradation = apply_dark_plotly_style(fig_degradation)
            st.plotly_chart(fig_degradation, use_container_width=True)

            # The metric below is a regression slope, so it is an ESTIMATE.
            # It is influenced by tyre wear but also fuel load, traffic and track evolution.
            degradation_1 = calculate_degradation(compound_data, driver_1)
            degradation_2 = calculate_degradation(compound_data, driver_2)

            col1, col2 = st.columns(2)

            col1.metric(
                f"{driver_1} Estimated Degradation",
                (
                    f"{degradation_1:+.3f} s/lap"
                    if degradation_1 is not None
                    else "Not enough data"
                ),
                border=True
            )

            col2.metric(
                f"{driver_2} Estimated Degradation",
                (
                    f"{degradation_2:+.3f} s/lap"
                    if degradation_2 is not None
                    else "Not enough data"
                ),
                border=True
            )

        else:
            st.info("No tyre compound data available for this session.")

    # =========================================================
    # TRACK EVOLUTION
    # =========================================================
    with tab_track:
        # =====================================================
        # TRACK EVOLUTION TAB
        # =====================================================
        # We use ALL drivers, not only Driver 1 and Driver 2.
        # The aim is to see how the best achievable session performance evolves over time.
        st.subheader("Track Evolution")
        st.caption(
            "Calculated using all valid laps from all drivers. "
            "'Best Lap So Far' is the fastest lap achieved by any driver "
            "up to that point in the session."
        )

        # LapStartTime tells us WHEN each lap occurred in the session.
        # This is better than LapNumber because drivers can be on different lap counts.
        track_data = laps[
            ["LapStartTime", "LapTimeSeconds"]
        ].dropna().copy()

        if track_data.empty:
            st.warning(
                "No valid lap timing data available for track evolution."
            )

        else:
            # Convert session elapsed time into ordinary minutes for the x-axis.
            track_data["SessionMinute"] = (
                track_data["LapStartTime"].dt.total_seconds() / 60
            )

            # Remove very slow observations (pit laps, SC laps, etc.).
            median_lap = track_data["LapTimeSeconds"].median()

            track_data = track_data[
                track_data["LapTimeSeconds"] < median_lap * 1.15
            ]

            # Divide the session into 5-minute windows.
            # Example: minute 17 -> TimeBin 15.
            track_data["TimeBin"] = (
                track_data["SessionMinute"] // 5 * 5
            ).astype(int)

            # BestLapInWindow = fastest lap achieved in each 5-minute window.
            track_evolution = (
                track_data.groupby("TimeBin", as_index=False)["LapTimeSeconds"]
                .min()
                .rename(
                    columns={
                        "LapTimeSeconds": "BestLapInWindow"
                    }
                )
            )

            # cummin() creates a running record:
            # "What was the fastest lap seen anywhere in the session up to this point?"
            track_evolution["BestLapSoFar"] = (
                track_evolution["BestLapInWindow"].cummin()
            )

            fig_track = go.Figure()

            fig_track.add_trace(go.Scatter(
                x=track_evolution["TimeBin"],
                y=track_evolution["BestLapInWindow"],
                mode="lines+markers",
                name="Best Lap in Window",
                line=dict(color="#8A8A8A", width=2),
                marker=dict(size=7),
                hovertemplate=(
                    "<b>Best Lap in Window</b><br>"
                    "Session Minute: %{x}<br>"
                    "Lap Time: %{y:.3f} s"
                    "<extra></extra>"
                )
            ))

            fig_track.add_trace(go.Scatter(
                x=track_evolution["TimeBin"],
                y=track_evolution["BestLapSoFar"],
                mode="lines+markers",
                name="Best Lap So Far",
                line=dict(color="#E10600", width=4),
                marker=dict(size=8),
                hovertemplate=(
                    "<b>Best Lap So Far</b><br>"
                    "Session Minute: %{x}<br>"
                    "Lap Time: %{y:.3f} s"
                    "<extra></extra>"
                )
            ))

            fig_track.update_layout(
                title="Session Performance Evolution",
                xaxis_title="Session Time (minutes)",
                yaxis_title="Lap Time (s)",
                height=500,
                hovermode="x unified",
                legend_title_text="Metric",
                margin=dict(l=20, r=20, t=60, b=20)
            )

            fig_track = apply_dark_plotly_style(fig_track)
            st.plotly_chart(fig_track, use_container_width=True)

            # Compare the first available benchmark with the final session benchmark.
            # We call it "Session Best-Lap Improvement", not pure "track improvement",
            # because fuel, tyres and traffic also affect lap time.
            if len(track_evolution) >= 2:
                improvement = (
                    track_evolution.iloc[0]["BestLapSoFar"]
                    - track_evolution.iloc[-1]["BestLapSoFar"]
                )

                st.metric(
                    "Session Best-Lap Improvement",
                    f"{improvement:.3f} s",
                    border=True
                )

    # =========================================================
    # TEAM SECTORS
    # =========================================================
    with tab_sectors:
        # =====================================================
        # TEAM SECTOR ANALYSIS TAB
        # =====================================================
        # We first aggregate best sectors at driver level, then average the two drivers
        # to obtain a team-level profile for S1, S2 and S3.
        st.subheader("Team Sector Analysis")
        st.caption(
            "Best sector times are calculated from each driver's fastest valid sector. "
            "Team performance is the average of both drivers when both have valid data."
        )

        # Mean of the drivers' personal-best sector times.
        # The count columns tell us whether the mean uses 2 drivers or only 1.
        team_sectors = (
            driver_sectors.groupby("TeamName", as_index=False)
            .agg(
                TeamS1=("BestS1", "mean"),
                TeamS2=("BestS2", "mean"),
                TeamS3=("BestS3", "mean"),
                S1Drivers=("BestS1", "count"),
                S2Drivers=("BestS2", "count"),
                S3Drivers=("BestS3", "count")
            )
        )

        team_sectors["TeamTheoreticalBest"] = (
            team_sectors["TeamS1"]
            + team_sectors["TeamS2"]
            + team_sectors["TeamS3"]
        )

        team_sectors = (
            team_sectors.sort_values("TeamTheoreticalBest")
            .reset_index(drop=True)
        )

        # If one driver has no valid sector, the team average may actually use one driver only.
        # We warn the user instead of silently treating it as a normal two-driver average.
        incomplete_teams = team_sectors[
            (team_sectors["S1Drivers"] < 2)
            | (team_sectors["S2Drivers"] < 2)
            | (team_sectors["S3Drivers"] < 2)
        ]

        if not incomplete_teams.empty:
            st.warning(
                "Some teams have incomplete sector data because only one driver "
                "recorded a valid sector time. Interpret their team averages with caution."
            )

        # Gap-to-best is easier to interpret than raw sector time:
        # 0.000 = reference team in that sector; larger values = more time lost.
        team_sectors["S1Gap"] = (
            team_sectors["TeamS1"]
            - team_sectors["TeamS1"].min()
        )

        team_sectors["S2Gap"] = (
            team_sectors["TeamS2"]
            - team_sectors["TeamS2"].min()
        )

        team_sectors["S3Gap"] = (
            team_sectors["TeamS3"]
            - team_sectors["TeamS3"].min()
        )

        # For the "Best Sector" cards, prefer teams with two valid drivers.
        # If no team has complete data, we fall back to all available teams.
        complete_s1 = team_sectors[
            team_sectors["S1Drivers"] == 2
        ]

        complete_s2 = team_sectors[
            team_sectors["S2Drivers"] == 2
        ]

        complete_s3 = team_sectors[
            team_sectors["S3Drivers"] == 2
        ]

        s1_candidates = (
            complete_s1
            if not complete_s1.empty
            else team_sectors
        )

        s2_candidates = (
            complete_s2
            if not complete_s2.empty
            else team_sectors
        )

        s3_candidates = (
            complete_s3
            if not complete_s3.empty
            else team_sectors
        )

        best_s1_row = s1_candidates.loc[
            s1_candidates["TeamS1"].idxmin()
        ]

        best_s2_row = s2_candidates.loc[
            s2_candidates["TeamS2"].idxmin()
        ]

        best_s3_row = s3_candidates.loc[
            s3_candidates["TeamS3"].idxmin()
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Best Sector 1",
                best_s1_row["TeamName"],
                border=True
            )
            st.caption(f'{best_s1_row["TeamS1"]:.3f} s')

        with col2:
            st.metric(
                "Best Sector 2",
                best_s2_row["TeamName"],
                border=True
            )
            st.caption(f'{best_s2_row["TeamS2"]:.3f} s')

        with col3:
            st.metric(
                "Best Sector 3",
                best_s3_row["TeamName"],
                border=True
            )
            st.caption(f'{best_s3_row["TeamS3"]:.3f} s')

        # Grouped horizontal bars compare each team's gap in S1/S2/S3.
        # Sector colours match the Circuit View for visual consistency.
        fig_sectors = go.Figure()

        for sector, gap_column in [
            ("Sector 1", "S1Gap"),
            ("Sector 2", "S2Gap"),
            ("Sector 3", "S3Gap")
        ]:
            fig_sectors.add_trace(go.Bar(
                y=team_sectors["TeamName"],
                x=team_sectors[gap_column],
                name=sector,
                orientation="h",
                marker=dict(
                    color=SECTOR_COLORS[sector]
                ),
                hovertemplate=(
                    f"<b>%{{y}}</b><br>"
                    f"{sector} Gap: +%{{x:.3f}} s"
                    "<extra></extra>"
                )
            ))

        fig_sectors.update_layout(
            title="Sector Gap to Best Team",
            xaxis_title="Gap to Best (s)",
            yaxis_title="",
            barmode="group",
            height=600,
            legend_title_text="Sector",
            margin=dict(l=20, r=20, t=60, b=20)
        )

        fig_sectors.update_yaxes(
            autorange="reversed"
        )

        fig_sectors = apply_dark_plotly_style(fig_sectors)
        st.plotly_chart(
            fig_sectors,
            use_container_width=True
        )

        with st.expander(
            "View detailed sector data"
        ):
            st.write(
                "**Driver sector performance**"
            )

            st.dataframe(
                driver_sectors[
                    [
                        "Driver",
                        "TeamName",
                        "BestS1",
                        "BestS2",
                        "BestS3",
                        "TheoreticalBest"
                    ]
                ].round(3),
                use_container_width=True,
                hide_index=True
            )

            st.write(
                "**Team sector performance**"
            )

            st.dataframe(
                team_sectors[
                    [
                        "TeamName",
                        "TeamS1",
                        "TeamS2",
                        "TeamS3",
                        "TeamTheoreticalBest",
                        "S1Drivers",
                        "S2Drivers",
                        "S3Drivers"
                    ]
                ].round(3),
                use_container_width=True,
                hide_index=True
            )

    # =========================================================
    # CIRCUIT VIEW
    # =========================================================
    with tab_circuit:
        # =====================================================
        # CIRCUIT VIEW TAB
        # =====================================================
        # X/Y telemetry reconstructs the circuit shape.
        # The same fastest lap can then be coloured by speed or divided into sectors.
        st.subheader("Circuit View")
        st.caption(
            "Circuit visualization based on telemetry from the selected driver's fastest lap."
        )

        circuit_driver = st.selectbox(
            "Reference driver",
            drivers,
            key="circuit_driver"
        )

        circuit_mode = st.radio(
            "Visualization",
            ["Speed Map", "Sector Map"],
            horizontal=True,
            key="circuit_mode"
        )

        # First try to use telemetry from the currently loaded session.
        fastest_lap, telemetry = load_driver_telemetry(
            session,
            circuit_driver
        )

        telemetry_source = session_info.get(
            "session_type",
            "Current session"
        )

        # Some sessions (especially specific historical/event feeds) may not provide X/Y data.
        # In that case we use Qualifying from the SAME season and GP as a visual fallback.
        # The app explicitly informs the user when this happens.
        if telemetry.empty and session_info:
            # Cache the fallback Qualifying session in session_state.
            # Otherwise Streamlit would reload it every time the user changes a widget.
            fallback_key = (
                f"circuit_q_"
                f"{session_info['season']}_"
                f"{session_info['grand_prix']}"
            )

            try:
                if fallback_key not in st.session_state:
                    qualifying_session = fastf1.get_session(
                        session_info["season"],
                        session_info["grand_prix"],
                        "Q"
                    )

                    qualifying_session.load()

                    st.session_state[
                        fallback_key
                    ] = qualifying_session

                else:
                    qualifying_session = st.session_state[
                        fallback_key
                    ]

                fastest_lap, telemetry = (
                    load_driver_telemetry(
                        qualifying_session,
                        circuit_driver
                    )
                )

                if not telemetry.empty:
                    telemetry_source = "Qualifying"

            except Exception:
                telemetry = pd.DataFrame()

        if telemetry.empty or fastest_lap is None:
            st.warning(
                "Position telemetry is not available for this event/session, "
                "so the circuit map cannot be generated."
            )

        else:
            if telemetry_source != session_info.get(
                "session_type"
            ):
                st.info(
                    f"Position telemetry is not available for "
                    f"{session_info.get('session_type', 'this session')}. "
                    f"The circuit visualization therefore uses "
                    f"{telemetry_source} telemetry from the same Grand Prix."
                )

            # -------------------------------------------------
            # SPEED MAP
            # -------------------------------------------------
            # A dark line draws the circuit base; coloured telemetry points show speed.
            if circuit_mode == "Speed Map":
                fig_circuit = go.Figure()

                fig_circuit.add_trace(go.Scatter(
                    x=telemetry["X"],
                    y=telemetry["Y"],
                    mode="lines",
                    line=dict(
                        color="#202020",
                        width=10
                    ),
                    hoverinfo="skip",
                    showlegend=False
                ))

                # Scattergl uses WebGL and is efficient for many telemetry points.
                # The Turbo colour scale maps slow/fast sections to different colours.
                fig_circuit.add_trace(go.Scattergl(
                    x=telemetry["X"],
                    y=telemetry["Y"],
                    mode="markers",
                    marker=dict(
                        size=7,
                        color=telemetry["Speed"],
                        colorscale="Turbo",
                        showscale=True,
                        colorbar=dict(
                            title=dict(
                                text="Speed<br>(km/h)",
                                font=dict(
                                    color="#F5F5F5"
                                )
                            ),
                            tickfont=dict(
                                color="#F5F5F5"
                            ),
                            thickness=15
                        )
                    ),
                    customdata=telemetry[["Speed"]],
                    hovertemplate=(
                        "Speed: %{customdata[0]:.0f} km/h"
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))

                fig_circuit.add_trace(go.Scatter(
                    x=[telemetry.iloc[0]["X"]],
                    y=[telemetry.iloc[0]["Y"]],
                    mode="markers",
                    marker=dict(
                        size=13,
                        color="white",
                        line=dict(
                            color="black",
                            width=2
                        )
                    ),
                    name="Start / Finish",
                    hovertemplate=(
                        "<b>Start / Finish</b>"
                        "<extra></extra>"
                    )
                ))

                fig_circuit.update_layout(
                    title=(
                        f"{circuit_driver} "
                        "Fastest Lap — Speed Map"
                    ),
                    height=650,
                    xaxis=dict(
                        visible=False,
                        scaleanchor="y",
                        scaleratio=1
                    ),
                    yaxis=dict(
                        visible=False
                    ),
                    margin=dict(
                        l=20,
                        r=20,
                        t=60,
                        b=20
                    )
                )

                fig_circuit = apply_dark_plotly_style(
                    fig_circuit
                )

                st.plotly_chart(
                    fig_circuit,
                    use_container_width=True
                )

            else:
                # ---------------------------------------------
                # SECTOR MAP
                # ---------------------------------------------
                # Use the fastest lap's official sector times to assign each telemetry
                # point to S1, S2 or S3.
                sector_1_time = fastest_lap[
                    "Sector1Time"
                ]

                sector_2_time = fastest_lap[
                    "Sector2Time"
                ]

                if (
                    pd.isna(sector_1_time)
                    or pd.isna(sector_2_time)
                ):
                    st.warning(
                        "Sector timing data is not available for this lap."
                    )

                else:
                    # Convert telemetry timestamps to seconds elapsed since lap start.
                    telemetry["LapElapsedSeconds"] = (
                        telemetry["Time"]
                        - telemetry["Time"].iloc[0]
                    ).dt.total_seconds()

                    # S1 ends after Sector1Time seconds.
                    # S2 ends after Sector1Time + Sector2Time seconds.
                    s1_end = (
                        sector_1_time.total_seconds()
                    )

                    s2_end = (
                        s1_end
                        + sector_2_time.total_seconds()
                    )

                    # Default every telemetry point to S3, then overwrite the ranges
                    # that belong to S1 and S2.
                    telemetry["Sector"] = "Sector 3"

                    telemetry.loc[
                        telemetry["LapElapsedSeconds"]
                        <= s1_end,
                        "Sector"
                    ] = "Sector 1"

                    telemetry.loc[
                        (
                            telemetry["LapElapsedSeconds"]
                            > s1_end
                        )
                        & (
                            telemetry["LapElapsedSeconds"]
                            <= s2_end
                        ),
                        "Sector"
                    ] = "Sector 2"

                    fig_sector_map = go.Figure()

                    for sector in [
                        "Sector 1",
                        "Sector 2",
                        "Sector 3"
                    ]:
                        sector_tel = telemetry[
                            telemetry["Sector"]
                            == sector
                        ]

                        fig_sector_map.add_trace(
                            go.Scatter(
                                x=sector_tel["X"],
                                y=sector_tel["Y"],
                                mode="lines",
                                name=sector,
                                line=dict(
                                    color=SECTOR_COLORS[
                                        sector
                                    ],
                                    width=9
                                ),
                                hovertemplate=(
                                    f"<b>{sector}</b>"
                                    "<extra></extra>"
                                )
                            )
                        )

                    fig_sector_map.add_trace(
                        go.Scatter(
                            x=[
                                telemetry.iloc[0]["X"]
                            ],
                            y=[
                                telemetry.iloc[0]["Y"]
                            ],
                            mode="markers",
                            marker=dict(
                                size=13,
                                color="white",
                                line=dict(
                                    color="black",
                                    width=2
                                )
                            ),
                            name="Start / Finish"
                        )
                    )

                    fig_sector_map.update_layout(
                        title=dict(
                            text=(
                                f"{circuit_driver} "
                                "Fastest Lap — Sector Map"
                            ),
                            x=0.02,
                            xanchor="left"
                        ),
                        height=700,
                        xaxis=dict(
                            visible=False,
                            scaleanchor="y",
                            scaleratio=1
                        ),
                        yaxis=dict(
                            visible=False
                        ),
                        margin=dict(
                            l=20,
                            r=20,
                            t=80,
                            b=90
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.05,
                            xanchor="center",
                            x=0.5
                        )
                    )

                    fig_sector_map = (
                        apply_dark_plotly_style(
                            fig_sector_map
                        )
                    )

                    st.plotly_chart(
                        fig_sector_map,
                        use_container_width=True
                    )

            # Summary cards below the map help validate which lap/driver is visualised.
            fastest_seconds = (
                fastest_lap["LapTime"].total_seconds()
                if pd.notna(
                    fastest_lap["LapTime"]
                )
                else None
            )

            formatted_lap = (
                f"{int(fastest_seconds // 60)}:"
                f"{fastest_seconds % 60:06.3f}"
                if fastest_seconds is not None
                else "N/A"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Driver",
                circuit_driver,
                border=True
            )

            col2.metric(
                "Fastest Lap",
                formatted_lap,
                border=True
            )

            col3.metric(
                "Max Speed",
                f"{telemetry['Speed'].max():.0f} km/h",
                border=True
            )

    # =========================================================
    # RAW DATA
    # =========================================================
    with tab_raw:
        # =====================================================
        # RAW DATA TAB
        # =====================================================
        # These tables expose the underlying data used by the analyses.
        # They are hidden inside expanders so the main dashboard stays clean.
        st.subheader("Raw Data")
        st.caption(
            "Explore the underlying FastF1 data used throughout the dashboard."
        )

        with st.expander(
            "Session Results",
            expanded=True
        ):
            results_display = results[
                [
                    "Abbreviation",
                    "TeamName",
                    "Position"
                ]
            ].copy()

            results_display = (
                results_display.rename(
                    columns={
                        "Abbreviation": "Driver",
                        "TeamName": "Team"
                    }
                )
            )

            st.dataframe(
                results_display,
                use_container_width=True,
                hide_index=True
            )

        with st.expander("Lap Data"):
            lap_data_display = laps[
                [
                    "Driver",
                    "LapNumber",
                    "LapTimeFormatted",
                    "Compound",
                    "TyreLife",
                    "Stint"
                ]
            ].copy()

            lap_data_display = (
                lap_data_display.rename(
                    columns={
                        "LapNumber": "Lap",
                        "LapTimeFormatted": "Lap Time",
                        "TyreLife": "Tyre Age"
                    }
                )
            )

            st.dataframe(
                lap_data_display,
                use_container_width=True,
                hide_index=True
            )

        with st.expander("Sector Data"):
            sector_data_display = (
                driver_sectors[
                    [
                        "Driver",
                        "TeamName",
                        "BestS1",
                        "BestS2",
                        "BestS3",
                        "TheoreticalBest"
                    ]
                ].copy()
            )

            sector_data_display = (
                sector_data_display.rename(
                    columns={
                        "TeamName": "Team",
                        "BestS1": "Best S1 (s)",
                        "BestS2": "Best S2 (s)",
                        "BestS3": "Best S3 (s)",
                        "TheoreticalBest":
                            "Theoretical Best (s)"
                    }
                )
            )

            st.dataframe(
                sector_data_display.round(3),
                use_container_width=True,
                hide_index=True
            )

else:
    # Initial state: no FastF1 session has been loaded yet.
    st.info(
        "Choose a session from the sidebar "
        "and click **Load Session** to start."
    )
