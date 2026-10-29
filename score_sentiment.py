"""
Score news and announcements with the finance-word scorer used by the price model.

The old TF-IDF file (cse_sentiment_model.pkl) is no longer used.
"""

import argparse
import os

import pandas as pd

from news_sentiment import label_from_score, score_text


DEFAULT_INPUT = "cse_output/cse_combined_sources.csv"
DEFAULT_OUTPUT = "cse_output/cse_scored_sources.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CSE news and announcements")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV with text data")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV with scores")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    frame = pd.read_csv(args.input)
    if "source_type" in frame.columns:
        frame = frame[frame["source_type"].isin(["news", "announcement"])].copy()
    text = frame["text"].fillna("").astype(str) if "text" in frame.columns else ""
    title = frame["title"].fillna("").astype(str) if "title" in frame.columns else ""
    if isinstance(text, str):
        frame["text"] = title
    else:
        frame["text"] = text.where(text.str.len() > 0, title)

    frame["sentiment_score"] = frame["text"].map(score_text)
    frame["predicted_label"] = frame["sentiment_score"].map(label_from_score)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[OK] Scored {len(frame)} rows -> {args.output}")


if __name__ == "__main__":
    main()
