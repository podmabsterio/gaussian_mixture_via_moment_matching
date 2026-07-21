from pathlib import Path
import shutil
import pandas as pd
import numpy as np

from omegaconf import OmegaConf


def get_save_path(directory: str | Path, run_name: str, override: bool = False) -> Path:
    dir_path = Path(directory)
    run_path = dir_path / run_name

    if not dir_path.exists():
        print(f"Directory not found. Creating directory {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=True)

    if run_path.exists():
        if not override:
            raise FileExistsError(f"Run '{run_name}' already exists and override=False")
        else:
            print("Override is enabled. Removing existing file.")
            shutil.rmtree(run_path)
            run_path.mkdir(parents=True, exist_ok=True)
    else:
        run_path.mkdir(parents=True, exist_ok=True)

    return run_path


def save_dict_to_path(path, data):
    path = Path(path)

    def _save(p, d):
        p.mkdir(parents=True, exist_ok=True)
        for k, v in d.items():
            name = str(k)
            if isinstance(v, dict):
                _save(p / name, v)
            elif isinstance(v, pd.DataFrame):
                v.to_csv(p / f"{name}.csv", index=False)
            elif isinstance(v, np.ndarray):
                np.save(p / f"{name}.npy", v)
            else:
                raise TypeError(f"Unsupported type for key '{name}': {type(v)}")

    _save(path, data)
    print(f"All data saved to '{path}'")


def init_saving(config, config_file_name="config"):
    save_path = get_save_path(config.save_dir, config.run_name, config.override)
    OmegaConf.save(config, save_path / f"{config_file_name}.yaml")
    return save_path
