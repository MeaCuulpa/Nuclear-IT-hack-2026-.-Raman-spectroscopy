from solver import Solver
from omegaconf import OmegaConf
import pandas as pd


if __name__ == '__main__':
    cfg = OmegaConf.load('config.yaml')
    if cfg.task.titanic and cfg.task.house_pricing:
        raise Exception('Only one task at a time')

    solver = Solver(cfg)
    if cfg.mode.is_train:
        solver.fit()
    if cfg.mode.is_inference:
        solver.predict()
