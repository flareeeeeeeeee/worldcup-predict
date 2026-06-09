import pytest
from worldcup.teams import all_teams, validate_groups


def _cfg(groups):
    return {"tournament": {"groups": groups}}


def test_all_teams_flattens():
    cfg = _cfg({"A": ["x", "y"], "B": ["z"]})
    assert all_teams(cfg) == ["x", "y", "z"]


def test_validate_rejects_wrong_size():
    with pytest.raises(ValueError):
        validate_groups(_cfg({"A": ["x", "y", "z", "w"]}))  # only 1 group


def test_validate_accepts_48():
    groups = {chr(65 + i): [f"t{i}{j}" for j in range(4)] for i in range(12)}
    validate_groups(_cfg(groups))  # no raise
