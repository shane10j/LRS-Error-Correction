from __future__ import annotations

from typing import Iterable, List

from .vocab import BASE_TO_ID, ID_TO_BASE, PAD_BASE_ID


class DNATokenizer:
    def __init__(self, add_unknown: bool = True) -> None:
        self.add_unknown = add_unknown

    def encode(self, seq: str) -> List[int]:
        ids: List[int] = []
        for ch in seq.upper():
            ids.append(BASE_TO_ID.get(ch, PAD_BASE_ID if self.add_unknown else KeyError(ch)))
        return ids

    def batch_encode(self, seqs: Iterable[str]) -> List[List[int]]:
        return [self.encode(s) for s in seqs]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(ID_TO_BASE.get(int(token), "N") for token in ids)
