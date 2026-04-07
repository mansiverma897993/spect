import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline


DATASET_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
)
MODEL_PATH = Path("spam_model.pkl")
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def train_and_select_model(x_train, y_train) -> GridSearchCV:
    pipeline = build_pipeline()

    param_grid = {
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__min_df": [1, 2],
        "vectorizer__stop_words": [None, "english"],
        "classifier__C": [0.5, 1.0, 2.0, 4.0],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)
    return search


def evaluate(model, x_train, y_train, x_test, y_test):
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    train_f1 = f1_score(y_train, train_pred)
    test_f1 = f1_score(y_test, test_pred)
    gap = abs(train_accuracy - test_accuracy)

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy:  {test_accuracy:.4f}")
    print(f"Training F1:       {train_f1:.4f}")
    print(f"Testing F1:        {test_f1:.4f}")
    print(f"Accuracy Gap:      {gap:.4f}")

    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, test_pred, target_names=["ham", "spam"]))

    return {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "accuracy_gap": float(gap),
    }


def main():
    print("Downloading dataset...")
    df = pd.read_csv(DATASET_URL, sep="\t", header=None, names=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    x = df["message"]
    y = df["label"]

    print("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Training and tuning model...")
    search = train_and_select_model(x_train, y_train)
    best_model = search.best_estimator_

    print(f"Best CV score (F1): {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    metrics = evaluate(best_model, x_train, y_train, x_test, y_test)

    artifact = {
        "model": best_model,
        "metrics": metrics,
        "best_params": search.best_params_,
        "dataset_size": int(len(df)),
    }

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(artifact, model_file)

    print(f"\nSaved trained artifact to `{MODEL_PATH}`.")
    if metrics["accuracy_gap"] <= 0.03:
        print("Generalization check passed: train/test accuracy gap is small.")
    else:
        print("Warning: gap is higher than ideal. Consider adding more data.")


if __name__ == "__main__":
    main()
