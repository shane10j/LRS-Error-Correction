from omega_safe_seqedit.labels import apply_edit_labels, make_edit_labels


def test_substitution_label_round_trip():
    labels = make_edit_labels("ACGT", "ATGT")
    assert apply_edit_labels("ACGT", labels) == "ATGT"


def test_insertion_before_label_round_trip():
    labels = make_edit_labels("ACGT", "ACGTT")
    assert apply_edit_labels("ACGT", labels) == "ACGTT"


def test_deletion_label_round_trip():
    labels = make_edit_labels("ACGTT", "ACGT")
    assert apply_edit_labels("ACGTT", labels) == "ACGT"
