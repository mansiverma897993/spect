import pickle
from pathlib import Path
from typing import Any, Dict


MODEL_PATH = Path("spam_model.pkl")


def load_artifact(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Run `python train_model.py` first."
        )

    with model_path.open("rb") as model_file:
        artifact = pickle.load(model_file)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(
            "Invalid model artifact format. Re-train using the latest `train_model.py`."
        )

    return artifact


def predict_message(message: str, artifact: Dict[str, Any]) -> Dict[str, Any]:
    model = artifact["model"]
    prediction = int(model.predict([message])[0])
    probabilities = model.predict_proba([message])[0]

    spam_score = float(probabilities[1])
    ham_score = float(probabilities[0])

    return {
        "label": "spam" if prediction == 1 else "ham",
        "prediction": prediction,
        "spam_probability": spam_score,
        "ham_probability": ham_score,
        "confidence": max(spam_score, ham_score),
    }

