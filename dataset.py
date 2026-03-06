import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

class Dataset:
    def __init__(self, config=None):
        self.config = config

    def preprocessing(self, df: pd.DataFrame):

        return None