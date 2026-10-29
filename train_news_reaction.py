"""Train how the five CSE prices moved after each old headline.

The headline archive covers roughly the last five years. CSE announcements
from the same years are included. Each headline is paired with the percent
change from the previous close to the next close that could react to it,
and with the change five sessions later.

The later 20 percent of headline dates are held out. They are not used to
fit the model or the historical averages.

    python fetch_news_history.py
    python train_news_reaction.py
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from news_reactions import (
    FEATURES,
    FORECAST_PATH,
    MODEL_PATH,
    PATTERNS_PATH,
    REACTIONS_PATH,
    REGION_LABELS,
    REPORT_PATH,
    build_reactions,
    latest_closes,
    load_headlines,
    load_prices,
    pattern_table,
    score_headlines,
)

TEST_FRACTION = 0.20


def _fit(train: pd.DataFrame):
    weight = 1.0 / train.groupby(["symbol", "react_date"])["return_1d"].transform("size")
    features = train[FEATURES]
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    regressor.fit(features, train["return_1d"], sample_weight=weight)
    classifier.fit(features, (train["return_1d"] > 0).astype(int), sample_weight=weight)
    five = train.dropna(subset=["return_5d"])
    five = five[five["return_5d"].abs() <= 0.50]
    five_weight = 1.0 / five.groupby(["symbol", "react_date"])["return_5d"].transform("size")
    regressor_5d = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    regressor_5d.fit(five[FEATURES], five["return_5d"], sample_weight=five_weight)
    return regressor, regressor_5d, classifier


def _session_frame(frame: pd.DataFrame, predicted: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    work = frame[["symbol", "react_date", "return_1d"]].copy()
    work["predicted"] = predicted
    work["probability"] = probability
    grouped = work.groupby(["symbol", "react_date"], as_index=False).agg(
        return_1d=("return_1d", "first"),
        predicted=("predicted", "mean"),
        probability=("probability", "mean"),
    )
    return grouped


def _lookup_prediction(test: pd.DataFrame, patterns: pd.DataFrame) -> np.ndarray:
    general = patterns[patterns["scope"] == "region_tone"]
    averages = {(row.region, row.tone): row.mean_1d for row in general.itertuples(index=False)}
    fallback = float(test["return_1d"].mean()) if len(test) else 0.0
    return np.array([averages.get((region, tone), fallback) for region, tone in zip(test["region"], test["tone"])])


def evaluate(train: pd.DataFrame, test: pd.DataFrame, regressor, classifier, patterns: pd.DataFrame) -> dict:
    predicted = regressor.predict(test[FEATURES])
    probability = classifier.predict_proba(test[FEATURES])
    up_index = list(classifier.classes_).index(1)
    proba = probability[:, up_index]
    sessions = _session_frame(test, predicted, proba)
    actual_direction = (sessions["return_1d"] > 0).astype(int)
    model_direction = (sessions["probability"] >= 0.5).astype(int)
    majority = int(actual_direction.mean() >= 0.5)
    lookup = _lookup_prediction(test, patterns)
    lookup_sessions = _session_frame(test, lookup, np.full(len(test), 0.5))
    train_sessions = train.drop_duplicates(["symbol", "react_date"])
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_headlines": int(train.groupby(["region", "title", "seen_at"]).ngroups),
        "test_sessions": int(len(sessions)),
        "train_start": pd.Timestamp(train["react_date"].min()).date().isoformat(),
        "train_end": pd.Timestamp(train["react_date"].max()).date().isoformat(),
        "test_start": pd.Timestamp(test["react_date"].min()).date().isoformat(),
        "test_end": pd.Timestamp(test["react_date"].max()).date().isoformat(),
        "typical_session": float(train_sessions["return_1d"].mean()),
        "session_mae": float(mean_absolute_error(sessions["return_1d"], sessions["predicted"]) * 100),
        "zero_mae": float(mean_absolute_error(sessions["return_1d"], np.zeros(len(sessions))) * 100),
        "lookup_mae": float(mean_absolute_error(lookup_sessions["return_1d"], lookup_sessions["predicted"]) * 100),
        "session_direction": float(accuracy_score(actual_direction, model_direction)),
        "majority_direction": float(accuracy_score(actual_direction, np.full(len(sessions), majority))),
    }


def _percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def write_report(reactions: pd.DataFrame, patterns: pd.DataFrame, metrics: dict, forecast: pd.DataFrame) -> None:
    counts = reactions.groupby("region").size()
    lines = [
        "How CSE prices moved after each headline",
        "",
        "World headlines are a sample of up to about 100 stories a week. Sri Lanka and stock-market headlines are collected by month.",
        "Each headline is matched to the next Colombo session that could still react to it.",
        "Stories seen before 14:30 Colombo use that session. Later stories use the next session.",
        "The change is from the previous close to that session's close. The five-session change uses the same starting close.",
        "A busy news day is not allowed to count more than a quiet day when the model is fit.",
        "The historical averages below use the training years only.",
        "",
        f"Headline rows used: {len(reactions)}",
        f"World rows: {int(counts.get('world', 0))}",
        f"Sri Lanka rows: {int(counts.get('sri_lanka', 0))}",
        f"Stock-market rows: {int(counts.get('market', 0))}",
        f"Train: {metrics['train_rows']} headline-stock rows, {metrics['train_start']} to {metrics['train_end']}",
        f"Test:  {metrics['test_rows']} headline-stock rows, {metrics['test_sessions']} stock-sessions, {metrics['test_start']} to {metrics['test_end']}",
        "",
        f"Average miss on the next session: {metrics['session_mae']:.3f} percentage points",
        f"Average miss if the move is always called 0: {metrics['zero_mae']:.3f} percentage points",
        f"Average miss if each headline uses the old average for its region and tone: {metrics['lookup_mae']:.3f} percentage points",
        f"Direction accuracy on the test sessions: {metrics['session_direction']:.1%}",
        f"Direction accuracy of always guessing the more common side: {metrics['majority_direction']:.1%}",
        "",
        (
            f"On an ordinary training session, with the headline ignored, the average move was "
            f"{_percent(metrics['typical_session'])}."
        ),
        "The averages below are the actual price change after that kind of headline. Compare them with the ordinary session.",
        "The direction call did not beat always guessing the more common side.",
        "The size of the predicted move was less accurate than assuming the price does not change.",
        "",
        "Average next-session move after each kind of training headline with at least 30 rows:",
    ]
    general = patterns[patterns["scope"] == "region_tone"].sort_values(["region", "tone"])
    shown = general[general["n"] >= 30]
    for row in shown.itertuples(index=False):
        five = "n/a" if pd.isna(row.mean_5d) else _percent(row.mean_5d)
        lines.append(
            f"  {REGION_LABELS.get(row.region, row.region)} {row.tone}: "
            f"next session {_percent(row.mean_1d)}, five sessions {five}, {int(row.n)} rows"
        )
    topic_rows = patterns[(patterns["scope"] == "topic") & (patterns["n"] >= 40)].copy()
    if not topic_rows.empty:
        topic_rows["abs_move"] = topic_rows["mean_1d"].abs()
        lines.extend(["", "Topic groups with at least 40 training rows and the largest average move:"])
        for row in topic_rows.sort_values("abs_move", ascending=False).head(12).itertuples(index=False):
            lines.append(
                f"  {REGION_LABELS.get(row.region, row.region)} {row.tone} {row.topic}: "
                f"next session {_percent(row.mean_1d)}, {int(row.n)} rows"
            )
    company_rows = patterns[(patterns["scope"] == "company") & (patterns["n"] >= 30)]
    if not company_rows.empty:
        lines.extend(["", "When the headline names the company:"])
        for row in company_rows.sort_values(["symbol", "tone"]).itertuples(index=False):
            lines.append(
                f"  {row.symbol} {row.tone}: next session {_percent(row.mean_1d)}, {int(row.n)} rows"
            )
    lines.extend(
        [
            "",
            "Latest saved headlines, scored from the average move after similar training headlines:",
        ]
    )
    if forecast.empty:
        lines.append("  No recent headlines were available to score.")
    else:
        for row in forecast.itertuples(index=False):
            lines.append(
                f"  {row.company}: next session {row.pred_1d * 100:+.2f}% "
                f"(up chance {row.up_probability:.0%}), five sessions {row.pred_5d * 100:+.2f}%"
            )
    lines.extend(
        [
            "",
            "A one-day stock move is mostly noise. Read the test miss above before using a forecast.",
            "The full headline-by-headline price changes are in data/news_reactions.csv.",
            "The averages are in cse_output/news_reaction_patterns.csv.",
            "This is a research model, not a trading instruction.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def recent_headlines(headlines: pd.DataFrame) -> pd.DataFrame:
    if headlines.empty:
        return headlines
    dated = headlines.copy()
    dated["seen_dt"] = pd.to_datetime(dated["seen_at"], utc=True, errors="coerce")
    dated = dated.dropna(subset=["seen_dt"]).sort_values("seen_dt")
    if dated.empty:
        return headlines.iloc[0:0]
    cutoff = dated["seen_dt"].max() - pd.Timedelta(days=3)
    recent = dated[dated["seen_dt"] >= cutoff].tail(40)
    return recent.drop(columns=["seen_dt"])


def save_reactions(reactions: pd.DataFrame) -> None:
    columns = [
        "seen_at",
        "date",
        "region",
        "title",
        "source",
        "url",
        "symbol",
        "company",
        "tone",
        "sentiment",
        "topics",
        "mentions_company",
        "ref_date",
        "react_date",
        "close_ref",
        "close_1d",
        "close_5d",
        "return_1d",
        "return_5d",
    ]
    saved = reactions[columns].copy()
    saved["return_1d_pct"] = (saved["return_1d"] * 100).round(3)
    saved["return_5d_pct"] = (saved["return_5d"] * 100).round(3)
    for column in ("close_ref", "close_1d", "close_5d"):
        saved[column] = saved[column].round(2)
    REACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved.to_csv(REACTIONS_PATH, index=False)


def main() -> None:
    headlines = load_headlines()
    if headlines.empty:
        raise SystemExit("No headlines found. Run python fetch_news_history.py first.")
    print(f"Headlines: {len(headlines)}", flush=True)
    prices = load_prices()
    print(f"Prices: {prices['date'].min().date()} to {prices['date'].max().date()}", flush=True)
    reactions = build_reactions(headlines, prices)
    print(f"Headline-price rows: {len(reactions)}", flush=True)
    if len(reactions) < 500:
        raise SystemExit(
            f"Only {len(reactions)} headline-price rows. "
            "Wait until fetch_news_history.py finishes, then train again."
        )
    reactions["react_date"] = pd.to_datetime(reactions["react_date"])
    save_reactions(reactions)
    dates = np.sort(reactions["react_date"].unique())
    cutoff = dates[int(len(dates) * (1 - TEST_FRACTION))]
    train = reactions[reactions["react_date"] < cutoff].copy()
    test = reactions[reactions["react_date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise SystemExit("The date split left an empty train or test set.")
    patterns = pattern_table(train)
    print("Fitting the reaction model.", flush=True)
    PATTERNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    patterns.to_csv(PATTERNS_PATH, index=False)
    regressor, regressor_5d, classifier = _fit(train)
    metrics = evaluate(train, test, regressor, classifier, patterns)
    bundle = {
        "regressor_1d": regressor,
        "regressor_5d": regressor_5d,
        "classifier_1d": classifier,
        "features": FEATURES,
        "patterns": patterns,
        "metrics": metrics,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    _headlines, forecast = score_headlines(recent_headlines(headlines), bundle)
    closes = latest_closes(prices)
    if not forecast.empty:
        forecast = forecast.merge(closes, on=["symbol", "company"], how="left")
        forecast["predicted_close"] = (forecast["close"] * (1 + forecast["pred_1d"])).round(2)
        forecast.to_csv(FORECAST_PATH, index=False)
    else:
        forecast = pd.DataFrame()
    write_report(reactions, patterns, metrics, forecast)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Headline reactions: {REACTIONS_PATH}")
    print(f"Test direction accuracy: {metrics['session_direction']:.1%}")
    print(f"Test average miss: {metrics['session_mae']:.3f} percentage points")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
