def save_config(cfg_yaml: str) -> None:
    with open("config.yaml", "w") as f:
        f.write(cfg_yaml)
