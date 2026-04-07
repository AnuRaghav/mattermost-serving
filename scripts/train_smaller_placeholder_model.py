#!/usr/bin/env python3
"""Train a smaller TF-IDF + LogisticRegression artifact for model-level optimization comparison."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Same data as train_placeholder_model.py — label 0 = non-toxic, 1 = toxic
TEXTS = [
    "Good morning team, ready for standup?",
    "Thanks for the help with the deploy!",
    "Can someone review PR 42 when they have a minute?",
    "The release looks stable on staging.",
    "I'll update the docs this afternoon.",
    "Nice work everyone on the milestone.",
    "Does anyone have the link to the runbook?",
    "Let's sync tomorrow on the integration plan.",
    "Appreciate the quick turnaround.",
    "This is frustrating but we can fix it calmly.",
    "You're an idiot and nobody wants you here.",
    "Go kill yourself, seriously.",
    "I hate you and hope you get fired today.",
    "Shut up you worthless piece of garbage.",
    "That's the stupidest idea I've ever heard, moron.",
    "Die in a fire, nobody likes you.",
]

LABELS = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    dtype=np.int64,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "models" / "toxicity_pipeline_small.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=400,
                    ngram_range=(1, 1),
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=200, random_state=42),
            ),
        ]
    )
    pipeline.fit(TEXTS, LABELS)
    joblib.dump(pipeline, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
