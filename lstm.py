# lstm_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)


def train_lstm(x_train_scaled, y_train, x_test_scaled, y_test):
    import numpy as np

    X_train = torch.tensor(x_train_scaled, dtype=torch.float32).unsqueeze(2)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)

    X_test = torch.tensor(x_test_scaled, dtype=torch.float32).unsqueeze(2)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

    model = LSTMModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = model(X_test).squeeze()
        preds = (preds > 0.5).float()
        acc = (preds == y_test).float().mean()

    return acc.item()