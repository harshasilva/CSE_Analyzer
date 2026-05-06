"""
Score news and announcements using the trained sentiment model.
"""

import argparse
import os

import joblib
import pandas as pd


DEFAULT_INPUT = "cse_output/cse_combined_sources.csv"
DEFAULT_OUTPUT = "cse_output/cse_scored_sources.csv"
DEFAULT_MODEL = "cse_sentiment_model.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CSE news/announcements with sentiment model")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV with text data")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV with predictions")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Trained model path")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    df = df[df["source_type"].isin(["news", "announcement"])].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].where(df["text"].str.len() > 0, df["title"])

    model = joblib.load(args.model)
    df["predicted_label"] = model.predict(df["text"])

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[OK] Scored {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
