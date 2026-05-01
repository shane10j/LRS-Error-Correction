"""Conservative consensus baseline."""

from __future__ import annotations


def predict(example: dict, agreement_threshold: float = 0.8, deletion_threshold: float = 0.75) -> dict:
    prediction = []
    labels = []
    features = example["features"]
    pos = 0
    while pos < len(example["target_seq"]):
        counts = features["support_base_counts"][pos]
        depth = max(1, features["support_depth"][pos])
        agreement = features["support_agreement"][pos]
        del_fraction = features["support_del_count"][pos] / depth
        if del_fraction >= deletion_threshold:
            labels.append("DEL")
            pos += 1
            continue
        if agreement >= agreement_threshold and sum(counts) > 0:
            base = "ACGT"[max(range(4), key=lambda idx: counts[idx])]
            if base != example["target_seq"][pos]:
                labels.append(f"SUB_{base}")
                prediction.append(base)
            else:
                labels.append("COPY")
                prediction.append(example["target_seq"][pos])
        else:
            labels.append("COPY")
            prediction.append(example["target_seq"][pos])
        pos += 1
    return {
        "prediction": "".join(prediction),
        "predicted_labels": labels,
        "trust": features["support_agreement"],
    }

