def all_teams(cfg: dict) -> list:
    out = []
    for g in cfg["tournament"]["groups"].values():
        out.extend(g)
    return out


def validate_groups(cfg: dict) -> None:
    groups = cfg["tournament"]["groups"]
    if len(groups) != 12:
        raise ValueError(f"expected 12 groups, got {len(groups)}")
    teams = all_teams(cfg)
    for name, g in groups.items():
        if len(g) != 4:
            raise ValueError(f"group {name} must have 4 teams, got {len(g)}")
    if len(set(teams)) != 48:
        raise ValueError(f"expected 48 unique teams, got {len(set(teams))}")
