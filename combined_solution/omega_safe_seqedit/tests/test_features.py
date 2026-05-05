from omega_safe_seqedit.features import pileup_features


def test_support_pileup_sees_substitution():
    f = pileup_features("ACGT", ["ATGT", "ATGT", "ACGT"])
    assert f["support_base_counts"][1][3] == 2
    assert f["support_rule_type"][1] != 0


def test_support_pileup_sees_insertion():
    f = pileup_features("ACGT", ["ACGTT", "ACGTT", "ACGT"])
    assert max(f["support_ins_count"]) >= 2


def test_support_pileup_sees_deletion():
    f = pileup_features("ACGTT", ["ACGT", "ACGT", "ACGTT"])
    assert max(f["support_del_count"]) >= 2
