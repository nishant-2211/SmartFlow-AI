import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Glass/gradient theme — transparent backgrounds so charts sit on the app's dark glass panels
_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter, sans-serif", size=12),
    title_font=dict(size=17, color="#e2e8f0", family="Inter, sans-serif"),
    legend=dict(
        bgcolor="rgba(255,255,255,0.04)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(color="#94a3b8"),
    ),
    margin=dict(l=40, r=40, t=60, b=40),
)

# Purple → Blue → Cyan → Green → Amber palette
_PALETTE = ["#7c3aed", "#2563eb", "#06b6d4", "#10b981", "#f59e0b"]

# Satisfied = cyan, dissatisfied = purple
_SAT_MAP = {
    "satisfied": "#06b6d4",
    "neutral or dissatisfied": "#7c3aed",
}


def _apply_dark(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_DARK_LAYOUT)
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
        tickfont=dict(color="#64748b"),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
        tickfont=dict(color="#64748b"),
    )
    return fig


# ─── Individual chart functions ───────────────────────────────────────────────

def get_satisfaction_chart(df: pd.DataFrame) -> go.Figure:
    """Pie chart — overall passenger satisfaction breakdown."""
    counts = df["satisfaction"].value_counts().reset_index()
    counts.columns = ["Satisfaction", "Count"]

    fig = px.pie(
        counts,
        names="Satisfaction",
        values="Count",
        title="Overall Passenger Satisfaction",
        color_discrete_sequence=["#06b6d4", "#7c3aed"],
        hole=0.48,
    )
    fig.update_traces(
        textinfo="percent+label",
        textfont_size=13,
        marker=dict(line=dict(color="rgba(0,0,0,0.4)", width=2)),
        pull=[0.03, 0],
    )
    return _apply_dark(fig)


def get_class_distribution(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart — satisfaction counts split by travel class."""
    grouped = (
        df.groupby(["Class", "satisfaction"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        grouped,
        x="Class",
        y="Count",
        color="satisfaction",
        barmode="group",
        title="Satisfaction by Travel Class",
        color_discrete_map={
            "satisfied": "#06b6d4",
            "neutral or dissatisfied": "#7c3aed",
        },
        text="Count",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(legend_title_text="Satisfaction")
    return _apply_dark(fig)


def get_delay_analysis(df: pd.DataFrame) -> go.Figure:
    """Box plot — departure and arrival delay distributions per satisfaction group."""
    melted = df.melt(
        id_vars=["satisfaction"],
        value_vars=["Departure Delay in Minutes", "Arrival Delay in Minutes"],
        var_name="Delay Type",
        value_name="Minutes",
    )
    # Cap extreme outliers for readability
    melted = melted[melted["Minutes"] <= 120]

    fig = px.box(
        melted,
        x="Delay Type",
        y="Minutes",
        color="satisfaction",
        title="Delay Distribution by Satisfaction (capped at 120 min)",
        color_discrete_map={
            "satisfied": "#06b6d4",
            "neutral or dissatisfied": "#7c3aed",
        },
    )
    fig.update_layout(legend_title_text="Satisfaction")
    return _apply_dark(fig)


def get_age_distribution(df: pd.DataFrame) -> go.Figure:
    """Overlapping histogram — age spread by satisfaction group."""
    fig = px.histogram(
        df,
        x="Age",
        color="satisfaction",
        nbins=30,
        barmode="overlay",
        opacity=0.75,
        title="Passenger Age Distribution by Satisfaction",
        color_discrete_map={
            "satisfied": "#06b6d4",
            "neutral or dissatisfied": "#7c3aed",
        },
    )
    fig.update_layout(legend_title_text="Satisfaction", bargap=0.05)
    return _apply_dark(fig)


def get_travel_type_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart — satisfaction split by type of travel."""
    grouped = (
        df.groupby(["Type of Travel", "satisfaction"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        grouped,
        x="Type of Travel",
        y="Count",
        color="satisfaction",
        barmode="group",
        title="Satisfaction by Type of Travel",
        color_discrete_map={
            "satisfied": "#06b6d4",
            "neutral or dissatisfied": "#7c3aed",
        },
        text="Count",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(legend_title_text="Satisfaction")
    return _apply_dark(fig)


# ─── Keyword router ───────────────────────────────────────────────────────────

# Maps keyword groups → chart function
_KEYWORD_MAP = [
    (["satisfaction", "satisfied", "happy", "overall", "outcome"],  get_satisfaction_chart),
    (["class", "business", "economy", "first"],                      get_class_distribution),
    (["delay", "late", "departure", "arrival", "on time"],           get_delay_analysis),
    (["age", "young", "old", "demographic"],                         get_age_distribution),
    (["travel type", "business travel", "personal", "leisure"],      get_travel_type_chart),
]


def get_chart_for_query(query: str, df: pd.DataFrame):
    """
    Auto-detect which chart best matches a natural-language query.

    Returns a Plotly figure, or None if no match is found.
    """
    q = query.lower()
    for keywords, chart_fn in _KEYWORD_MAP:
        if any(kw in q for kw in keywords):
            try:
                return chart_fn(df)
            except Exception:
                return None
    # Default fallback — satisfaction overview is always relevant
    try:
        return get_satisfaction_chart(df)
    except Exception:
        return None
