"""Early stopping."""

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    best_score: float = float("-inf")
    bad_epochs: int = 0

    def update(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs > self.patience

