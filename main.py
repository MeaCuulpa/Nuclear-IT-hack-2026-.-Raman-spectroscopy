from __future__ import annotations

from solver import Solver
from utils import load_config


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    solver = Solver(cfg)
    solver.run()
