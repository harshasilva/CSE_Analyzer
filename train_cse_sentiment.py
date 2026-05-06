"""
Train a CSE-specific sentiment model using labeled news data.

Expected input CSV:
  data/cse_sentiment_labeled.csv
  columns: text,label
  labels: positive, negative, neutral

Outputs:
  models/cse-finbert
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset  # type: ignore
# Avoid sklearn import resolution issues by using a local stratified split.
# sklearn.model_selection.train_test_split was previously used but may not
# be available in some environments (import error reported). The project
# includes a custom stratified_split function below which mirrors the
# behavior needed for splitting by label groups.
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

DATA_PATH = Path("data/cse_sentiment_labeled.csv")
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = Path("models/cse-finbert")
MAX_LEN = 256
SEED = 42

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing training data at {DATA_PATH}. Create it with columns text,label."
        )

    df = pd.read_csv(DATA_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: text,label")

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(LABEL_MAP.keys())]
    if df.empty:
        raise ValueError("No valid labeled rows found.")
    return df


def stratified_split(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []

    for _, group in df.groupby("label_id", sort=False):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        n_test = min(n_test, len(idx) - 1) if len(idx) > 1 else 1
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        test_parts.append(df.loc[test_idx])
        train_parts.append(df.loc[train_idx])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, test_df


def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(labels)
    accuracy = float(np.mean(preds == labels))
    f1_scores = []
    for cls in sorted(LABEL_MAP.values()):
        tp = np.sum((preds == cls) & (labels == cls))
        fp = np.sum((preds == cls) & (labels != cls))
        fn = np.sum((preds != cls) & (labels == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def main() -> None:
    df = load_dataset()
    df["label_id"] = df["label"].map(LABEL_MAP).astype(int)

    train_df, temp_df = stratified_split(df, test_size=0.2, random_state=SEED)
    val_df, test_df = stratified_split(temp_df, test_size=0.5, random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_MAP)
    )

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]]).rename_column(
        "label_id", "labels"
    )
    val_ds = Dataset.from_pandas(val_df[["text", "label_id"]]).rename_column(
        "label_id", "labels"
    )
    test_ds = Dataset.from_pandas(test_df[["text", "label_id"]]).rename_column(
        "label_id", "labels"
    )

    train_ds = train_ds.map(lambda x: tokenize_function(tokenizer, x), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(tokenizer, x), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nEvaluating on test split...")
    metrics = trainer.evaluate(test_ds)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
