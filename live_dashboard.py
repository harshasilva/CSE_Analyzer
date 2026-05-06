from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table

DATA_DIR = Path(__file__).resolve().parent / "cse_output"
NEWS_PREFIX = "cse_news"
PREDICTION_FILE = "cse_prediction_results.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, low_memory=False)


def _read_json(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except Exception:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return pd.json_normalize(payload)
        except Exception:
            return pd.DataFrame()


def load_all_data() -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    if not DATA_DIR.exists():
        return data

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        data[csv_path.name] = _read_csv(csv_path)
    for json_path in sorted(DATA_DIR.glob("*.json")):
        data[json_path.name] = _read_json(json_path)
    return data


def build_news_df(data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, df in data_map.items():
        if not name.startswith(NEWS_PREFIX) or not name.endswith(".csv"):
            continue
        if df.empty:
            continue
        df = df.copy()
        df["_source_file"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_predictions_df(data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data_map.get(PREDICTION_FILE)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()


def ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "_year" in df.columns:
        return df
    date_cols = ["_parsed_date", "createdDate", "announcementDate", "publishedDate", "date"]
    for col in date_cols:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                df = df.copy()
                df["_year"] = parsed.dt.year
                return df
    return df


def text_search_mask(df: pd.DataFrame, keyword: str, columns: Iterable[str]) -> pd.Series:
    if not keyword:
        return pd.Series(True, index=df.index)
    keyword = str(keyword).strip()
    if not keyword:
        return pd.Series(True, index=df.index)

    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            mask = mask | df[col].fillna("").astype(str).str.contains(keyword, case=False, na=False)
    return mask


def filter_news(
    df: pd.DataFrame,
    year_value: int | None,
    keyword: str | None,
    company_value: str | None,
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    if df.empty:
        return df, notes

    df = ensure_year_column(df)
    if year_value and "_year" in df.columns:
        df = df[df["_year"] == year_value]
    elif year_value:
        notes.append("Year filter ignored (no _year column).")

    if company_value and "company" in df.columns:
        df = df[df["company"].fillna("").astype(str).str.contains(company_value, case=False, na=False)]
    elif company_value:
        notes.append("Company filter ignored (no company column).")

    text_cols = ["company", "subject", "description", "heading", "title", "remarks"]
    if keyword:
        mask = text_search_mask(df, keyword, text_cols)
        df = df[mask]

    return df, notes


def add_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "combined_score" not in df.columns:
        return df
    df = df.copy()
    df["combined_score"] = pd.to_numeric(df["combined_score"], errors="coerce")

    def score_to_risk(score: float | None) -> str:
        if score is None or pd.isna(score):
            return "unknown"
        if score >= 8:
            return "low"
        if score >= 4:
            return "medium"
        return "high"

    df["risk_level"] = df["combined_score"].apply(score_to_risk)
    return df


def filter_predictions(
    df: pd.DataFrame,
    name_value: str | None,
    risk_value: str | None,
    score_min: float | None,
    score_max: float | None,
    price_min: float | None,
    price_max: float | None,
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    if df.empty:
        return df, notes

    df = add_risk_level(df)

    if name_value and "company" in df.columns:
        df = df[df["company"].fillna("").astype(str).str.contains(name_value, case=False, na=False)]
    elif name_value:
        notes.append("Name filter ignored (no company column).")

    if risk_value and "risk_level" in df.columns:
        df = df[df["risk_level"] == risk_value]
    elif risk_value:
        notes.append("Risk filter ignored (no combined_score column).")

    if "combined_score" in df.columns:
        scores = pd.to_numeric(df["combined_score"], errors="coerce")
        if score_min is not None:
            df = df[scores >= score_min]
        if score_max is not None:
            df = df[scores <= score_max]
    elif score_min is not None or score_max is not None:
        notes.append("Score filter ignored (no combined_score column).")

    price_col = None
    for candidate in ["price", "close", "last_price"]:
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col:
        prices = pd.to_numeric(df[price_col], errors="coerce")
        if price_min is not None:
            df = df[prices >= price_min]
        if price_max is not None:
            df = df[prices <= price_max]
    elif price_min is not None or price_max is not None:
        notes.append("Price filter ignored (no price column).")

    return df, notes


def to_table_columns(df: pd.DataFrame) -> List[dict]:
    return [{"name": col, "id": col} for col in df.columns]


def file_options(data_map: Dict[str, pd.DataFrame]) -> List[dict]:
    return [{"label": name, "value": name} for name in sorted(data_map.keys())]


def prepare_file_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.head(500).copy()


def filter_file_df(df: pd.DataFrame, company_value: str | None) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    if df.empty:
        return df, notes
    if company_value and "company" in df.columns:
        df = df[df["company"].fillna("").astype(str).str.contains(company_value, case=False, na=False)]
    elif company_value:
        notes.append("Company filter ignored (no company column).")
    return df, notes


def build_company_figure(
    pred_df: pd.DataFrame, company_value: str | None
) -> Tuple[go.Figure, List[dict], str | None]:
    if pred_df.empty or "company" not in pred_df.columns:
        fig = go.Figure()
        fig.update_layout(
            height=300,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                {
                    "text": "No prediction data available",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14, "color": "#4a5a52"},
                }
            ],
        )
        return fig, [], None

    companies = sorted({str(c) for c in pred_df["company"].dropna().unique()})
    options = [{"label": company, "value": company} for company in companies]
    if company_value not in companies:
        company_value = companies[0] if companies else None

    row = pred_df[pred_df["company"] == company_value].head(1)
    if row.empty:
        return build_company_figure(pred_df.iloc[:0], company_value)

    row = row.iloc[0]
    metrics: List[dict] = []
    for label, key in [
        ("Combined", "combined_score"),
        ("Historical", "hist_score"),
        ("Live", "live_score"),
        ("Hist mentions", "hist_mentions"),
        ("Live mentions", "live_mentions"),
        ("Positive %", "pct_positive"),
        ("Negative %", "pct_negative"),
        ("Neutral %", "pct_neutral"),
    ]:
        if key in pred_df.columns:
            value = pd.to_numeric(row.get(key), errors="coerce")
            if pd.notna(value):
                metrics.append({"metric": label, "value": float(value)})

    fig = go.Figure()
    if metrics:
        fig.add_bar(
            x=[m["metric"] for m in metrics],
            y=[m["value"] for m in metrics],
            marker_color="#0f766e",
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    fig.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Sora", "color": "#0f1b17"},
        yaxis_title="",
        xaxis_title="",
        xaxis_tickangle=-20,
    )
    return fig, options, company_value


def build_insight_message(pred_df: pd.DataFrame) -> str:
    if pred_df.empty or "signal" not in pred_df.columns or "company" not in pred_df.columns:
        return ""

    strong = pred_df[pred_df["signal"] == "*** STRONG BUY ***"]
    if strong.empty:
        return "No strong signals right now. Keep monitoring upcoming announcements and score changes."

    names = strong["company"].dropna().astype(str).unique().tolist()
    names = names[:6]
    joined = ", ".join(names)
    return (
        "Informational only: the current signals highlight these shares as strong candidates to watch based on the "
        f"engine metrics: {joined}."
    )


APP_STYLE = """
:root {
    --paper: #eef3ef;
    --ink: #0f1b17;
    --muted: #4a5a52;
    --accent: #0f766e;
    --accent-strong: #134e4a;
    --surface: #f9fdfb;
    --line: #d7e3dd;
    --shadow: 0 18px 40px rgba(8, 24, 20, 0.12);
    --shadow-soft: 0 10px 24px rgba(8, 24, 20, 0.12);
    --highlight: #f6c453;
}

* { box-sizing: border-box; }

body {
  margin: 0;
    font-family: "Sora", "Segoe UI", sans-serif;
    background: radial-gradient(circle at top left, #eef7f1 0%, #e3efe8 45%, #d7e6de 100%);
  color: var(--ink);
}

.app-shell {
  min-height: 100vh;
  padding: 28px 22px 60px;
    background-image: linear-gradient(120deg, rgba(15, 118, 110, 0.12), transparent 45%),
        radial-gradient(circle at 80% 20%, rgba(15, 76, 71, 0.12), transparent 60%);
}

.hero {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 20px 22px;
    background: linear-gradient(135deg, #ffffff 0%, #f3fbf7 45%, #e8f3ee 100%);
    box-shadow: var(--shadow);
  animation: slideIn 600ms ease-out;
}

.hero-title {
    font-family: "Fraunces", "Georgia", serif;
    font-size: clamp(30px, 3.4vw, 40px);
  margin: 0 0 6px;
}

.hero-subtitle {
  color: var(--muted);
  margin: 0;
  font-size: 14px;
}

.controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.btn {
  border: none;
    background: linear-gradient(135deg, var(--accent), var(--accent-strong));
  color: white;
  padding: 10px 16px;
  border-radius: 999px;
  font-weight: 600;
  letter-spacing: 0.3px;
    cursor: pointer;
    box-shadow: 0 12px 24px rgba(15, 118, 110, 0.28);
  transition: transform 150ms ease, box-shadow 150ms ease;
}

.btn:hover { transform: translateY(-1px); box-shadow: 0 14px 26px rgba(180, 83, 9, 0.3); }
.btn:hover { box-shadow: 0 16px 30px rgba(15, 118, 110, 0.3); }

.chart-card {
    background: linear-gradient(135deg, #ffffff 0%, #f4fbf7 60%, #e7f4ee 100%);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px 16px 6px;
    box-shadow: var(--shadow-soft);
    margin-bottom: 16px;
}

.insight-banner {
    background: linear-gradient(120deg, rgba(15, 118, 110, 0.14), rgba(246, 196, 83, 0.22));
    border: 1px solid rgba(15, 118, 110, 0.18);
    border-radius: 16px;
    padding: 14px 16px;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 14px;
    box-shadow: var(--shadow-soft);
}

.chart-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}

.chart-title {
    margin: 0;
    font-size: 18px;
}

.chart-subtitle {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 13px;
}

.tabs {
  margin-top: 18px;
}

.tab {
  background: var(--surface);
  border-radius: 16px 16px 0 0;
  border: 1px solid var(--line);
  padding: 12px 16px;
  font-weight: 600;
    color: var(--muted);
}

.tab--selected {
    background: #e9f6f0;
    border-bottom: 2px solid var(--accent);
    color: var(--ink);
}

.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-top: none;
  border-radius: 0 0 18px 18px;
  padding: 18px 16px 24px;
  animation: fadeIn 500ms ease-out;
    box-shadow: var(--shadow-soft);
}

.filter-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin: 12px 0 10px;
}

.filter-row input,
.filter-row .Select-control,
.filter-row .Select-menu-outer,
.filter-row .Select-placeholder,
.filter-row .Select-input,
.filter-row .Select-value-label {
    font-family: "Sora", "Segoe UI", sans-serif;
}

.filter-row input,
.filter-row .Select-control {
    border-radius: 12px !important;
    border: 1px solid var(--line) !important;
    background: #fffdf9 !important;
    min-height: 38px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

.filter-row input:focus,
.filter-row .Select-control:hover {
    border-color: #5fb2a2 !important;
    box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.16);
}

.dash-table-container {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--line);
}

.dash-spreadsheet td,
.dash-spreadsheet th {
    border-color: #f1e7d7 !important;
}

.dash-spreadsheet tr:nth-child(even) td {
    background-color: #f2faf6 !important;
}

.dash-spreadsheet tr:hover td {
    background-color: #e4f4ed !important;
}

.note { color: #8a2d2d; margin-bottom: 6px; }
.count { margin-bottom: 10px; font-weight: 600; color: var(--muted); }

.count::before {
    content: "\25CF";
    color: var(--highlight);
    margin-right: 8px;
}

@media (max-width: 720px) {
    .hero { padding: 16px; }
    .filter-row { gap: 8px; }
    .tab { padding: 10px 12px; }
    .panel { padding: 16px 12px; }
    .chart-card { padding: 12px; }
}

@keyframes slideIn {
  from { transform: translateY(12px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
"""

app = Dash(__name__)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>CSE Analyzer Live Dashboard</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Sora:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
        """ + APP_STYLE + """
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="hero",
            children=[
                html.Div(
                    children=[
                        html.H1("CSE Analyzer Live Dashboard", className="hero-title"),
                        html.P(
                            "Auto-refresh every 10 seconds. Use Refresh Now to reload immediately.",
                            className="hero-subtitle",
                        ),
                    ]
                ),
                html.Div(
                    className="controls",
                    children=[
                        html.Button("Refresh now", id="refresh-btn", n_clicks=0, className="btn"),
                    ],
                ),
            ],
        ),
        dcc.Interval(id="refresh-interval", interval=10_000, n_intervals=0),
        dcc.Tabs(
            className="tabs",
            children=[
                dcc.Tab(
                    label="News",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.Div(
                                    className="filter-row",
                                    children=[
                                        dcc.Dropdown(id="news-year", placeholder="Year", style={"width": "160px"}),
                                        dcc.Input(id="news-company", type="text", placeholder="Company"),
                                        dcc.Input(id="news-keyword", type="text", placeholder="Keyword"),
                                    ],
                                ),
                                html.Div(id="news-filter-note", className="note"),
                                html.Div(id="news-count", className="count"),
                                dash_table.DataTable(
                                    id="news-table",
                                    page_size=20,
                                    filter_action="native",
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "minWidth": "120px", "maxWidth": "300px"},
                                    style_header={
                                        "backgroundColor": "#fff4e6",
                                        "fontWeight": "600",
                                        "border": "1px solid #eadfce",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Predictions",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.Div(id="pred-insight", className="insight-banner"),
                                html.Div(
                                    className="chart-card",
                                    children=[
                                        html.Div(
                                            className="chart-header",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.H3("Share snapshot", className="chart-title"),
                                                        html.P(
                                                            "Scores, mentions, and sentiment mix for a selected share.",
                                                            className="chart-subtitle",
                                                        ),
                                                    ]
                                                ),
                                                dcc.Dropdown(
                                                    id="pred-company-graph",
                                                    placeholder="Select share",
                                                    style={"width": "260px"},
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(
                                            id="pred-graph",
                                            config={"displayModeBar": False},
                                            style={"height": "320px"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="filter-row",
                                    children=[
                                        dcc.Input(id="pred-name", type="text", placeholder="Company"),
                                        dcc.Dropdown(
                                            id="pred-risk",
                                            placeholder="Risk",
                                            options=[
                                                {"label": "Low", "value": "low"},
                                                {"label": "Medium", "value": "medium"},
                                                {"label": "High", "value": "high"},
                                            ],
                                            style={"width": "160px"},
                                        ),
                                        dcc.Input(id="pred-score-min", type="number", placeholder="Min score"),
                                        dcc.Input(id="pred-score-max", type="number", placeholder="Max score"),
                                        dcc.Input(id="pred-price-min", type="number", placeholder="Min price"),
                                        dcc.Input(id="pred-price-max", type="number", placeholder="Max price"),
                                    ],
                                ),
                                html.Div(id="pred-filter-note", className="note"),
                                html.Div(id="pred-count", className="count"),
                                dash_table.DataTable(
                                    id="pred-table",
                                    page_size=20,
                                    filter_action="native",
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "minWidth": "120px", "maxWidth": "300px"},
                                    style_header={
                                        "backgroundColor": "#fff4e6",
                                        "fontWeight": "600",
                                        "border": "1px solid #eadfce",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Files",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.Div(
                                    className="filter-row",
                                    children=[
                                        dcc.Dropdown(id="file-select", placeholder="Select file", style={"width": "320px"}),
                                        dcc.Input(id="file-company", type="text", placeholder="Company"),
                                    ],
                                ),
                                html.Div(id="file-filter-note", className="note"),
                                html.Div(id="file-count", className="count"),
                                dash_table.DataTable(
                                    id="file-table",
                                    page_size=20,
                                    filter_action="native",
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "minWidth": "120px", "maxWidth": "300px"},
                                    style_header={
                                        "backgroundColor": "#fff4e6",
                                        "fontWeight": "600",
                                        "border": "1px solid #eadfce",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        ),
    ],
)


@app.callback(
    Output("news-year", "options"),
    Output("news-table", "data"),
    Output("news-table", "columns"),
    Output("news-count", "children"),
    Output("news-filter-note", "children"),
    Output("pred-table", "data"),
    Output("pred-table", "columns"),
    Output("pred-count", "children"),
    Output("pred-filter-note", "children"),
    Output("pred-insight", "children"),
    Output("pred-company-graph", "options"),
    Output("pred-company-graph", "value"),
    Output("pred-graph", "figure"),
    Output("file-select", "options"),
    Output("file-select", "value"),
    Output("file-table", "data"),
    Output("file-table", "columns"),
    Output("file-filter-note", "children"),
    Output("file-count", "children"),
    Input("refresh-btn", "n_clicks"),
    Input("refresh-interval", "n_intervals"),
    Input("pred-company-graph", "value"),
    State("news-year", "value"),
    State("news-company", "value"),
    State("news-keyword", "value"),
    State("pred-name", "value"),
    State("pred-risk", "value"),
    State("pred-score-min", "value"),
    State("pred-score-max", "value"),
    State("pred-price-min", "value"),
    State("pred-price-max", "value"),
    State("file-select", "value"),
    State("file-company", "value"),
)
def refresh_all(
    _refresh_clicks: int,
    _refresh_ticks: int,
    pred_company_graph: str | None,
    news_year: int | None,
    news_company: str | None,
    news_keyword: str | None,
    pred_name: str | None,
    pred_risk: str | None,
    pred_score_min: float | None,
    pred_score_max: float | None,
    pred_price_min: float | None,
    pred_price_max: float | None,
    file_value: str | None,
    file_company: str | None,
):
    data_map = load_all_data()

    news_df = build_news_df(data_map)
    news_df, news_notes = filter_news(news_df, news_year, news_keyword, news_company)
    news_year_options = []
    if not news_df.empty and "_year" in news_df.columns:
        years = sorted({int(y) for y in news_df["_year"].dropna().unique()})
        news_year_options = [{"label": str(y), "value": y} for y in years]

    pred_df = build_predictions_df(data_map)
    pred_df, pred_notes = filter_predictions(
        pred_df,
        pred_name,
        pred_risk,
        pred_score_min,
        pred_score_max,
        pred_price_min,
        pred_price_max,
    )

    pred_insight = build_insight_message(pred_df)

    pred_figure, pred_company_opts, pred_company_value = build_company_figure(
        pred_df,
        pred_company_graph,
    )

    file_opts = file_options(data_map)
    if not file_value and file_opts:
        file_value = file_opts[0]["value"]
    file_df = data_map.get(file_value, pd.DataFrame()) if file_value else pd.DataFrame()
    file_df, file_notes = filter_file_df(file_df, file_company)
    file_df = prepare_file_table(file_df)

    news_count = f"Rows: {len(news_df):,}" if not news_df.empty else "No news rows"
    pred_count = f"Rows: {len(pred_df):,}" if not pred_df.empty else "No prediction rows"
    file_count = f"Rows: {len(file_df):,}" if not file_df.empty else "No file rows"

    return (
        news_year_options,
        news_df.to_dict("records"),
        to_table_columns(news_df),
        news_count,
        " ".join(news_notes),
        pred_df.to_dict("records"),
        to_table_columns(pred_df),
        pred_count,
        " ".join(pred_notes),
        pred_insight,
        pred_company_opts,
        pred_company_value,
        pred_figure,
        file_opts,
        file_value,
        file_df.to_dict("records"),
        to_table_columns(file_df),
        " ".join(file_notes),
        file_count,
    )


if __name__ == "__main__":
    app.run(debug=True)
