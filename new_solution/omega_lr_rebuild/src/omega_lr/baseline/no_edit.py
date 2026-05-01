"""No-edit baseline."""


def predict(example: dict) -> dict:
    return {
        "prediction": example["target_seq"],
        "predicted_labels": ["COPY" for _ in example["target_seq"]],
        "trust": [0.0 for _ in example["target_seq"]],
    }

