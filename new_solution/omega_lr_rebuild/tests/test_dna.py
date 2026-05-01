import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omega_lr.dna import reverse_complement
from omega_lr.homopolymer import run_lengths
from omega_lr.repeats import tandem_repeat_mask


class DnaTests(unittest.TestCase):
    def test_reverse_complement(self):
        self.assertEqual(reverse_complement("ACGTN"), "NACGT")

    def test_homopolymer_run_lengths(self):
        self.assertEqual(run_lengths("AAATCC"), [3, 3, 3, 1, 2, 2])

    def test_tandem_repeat_mask(self):
        self.assertEqual(tandem_repeat_mask("ATATG"), [1, 1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()

