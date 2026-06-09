from worldcup.config import load_config


def test_load_config_reads_yaml(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("simulation:\n  n_sims: 10\n", encoding="utf-8")
    cfg = load_config(str(f))
    assert cfg["simulation"]["n_sims"] == 10
