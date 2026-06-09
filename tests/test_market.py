from worldcup.market import devig


def test_devig_normalizes_and_orders():
    p = devig({"A": 2.0, "B": 4.0, "C": 4.0})
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["A"] > p["B"]
