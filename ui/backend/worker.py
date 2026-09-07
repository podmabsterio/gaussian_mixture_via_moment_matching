from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one compiled UI experiment")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    import run as project_run

    task = getattr(project_run.main, "__wrapped__", None)
    if task is None:
        raise RuntimeError("Hydra-decorated run.main does not expose its original task")
    task(config)


if __name__ == "__main__":
    main()
