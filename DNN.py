import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import ast
import os
torch.set_num_threads(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class _PyTorchNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super().__init__()
        layers = []

        in_features = int(input_dim)
        if isinstance(hidden_layers, str):
            hidden_layers = ast.literal_eval(hidden_layers)
        hidden_layers = list(hidden_layers)

        for out_features in hidden_layers:
            out_features = int(out_features)

            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())

            layers.append(nn.Dropout(float(dropout_rate)))

            in_features = out_features

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.output_layer(x).squeeze(-1)

class DNN(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=(64, 32), dropout_rate=0.2,
                 epochs=50, batch_size=32, learning_rate=0.001, is_regression=False):

        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_regression = is_regression

        self.model_ = None

    def fit(self, X, y):
        input_dim = X.shape[1]

        self.model_ = _PyTorchNet(input_dim, self.hidden_layers, self.dropout_rate)

        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y

        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        criterion = nn.MSELoss() if self.is_regression else nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model_(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        if self.is_regression:
            raise ValueError("No regression for predict proba")

        self.model_.eval()
        X_arr = X.values if hasattr(X, 'values') else X
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model_(X_tensor)
            probas = torch.sigmoid(logits).numpy()

        return np.vstack((1 - probas, probas)).T

    def predict(self, X):
        if self.is_regression:
            self.model_.eval()
            X_arr = X.values if hasattr(X, 'values') else X
            X_tensor = torch.tensor(X_arr, dtype=torch.float32)
            with torch.no_grad():
                return self.model_(X_tensor).numpy()
        else:
            probas = self.predict_proba(X)[:, 1]
            return (probas >= 0.5).astype(int)