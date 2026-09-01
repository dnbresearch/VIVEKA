#!/usr/bin/env python3
"""
Time Series Hypothesis: Instance Normalization → Forecasting
==============================================================
Cross-paper hypothesis:
  - Instance Normalization (Ulyanov et al., 2016): Proposed for neural
    style transfer in CV — normalizes each sample independently
  - RevIN (Kim et al., ICLR 2022): Rediscovered for time series as
    "Reversible Instance Normalization" to handle distribution shift
  - LSTM forecasting: Standard approach suffers from non-stationarity

Hypothesis: "Applying reversible instance normalization (normalize input,
forecast, then denormalize output) to LSTM and simple MLP forecasters
improves accuracy on non-stationary time series. This CV technique
transfers to time series because both domains face distribution shift."

Datasets: ETTh1 (electricity transformer temperature) + synthetic
Runtime: ~5 minutes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json, time, os, math
import numpy as np
from torch.utils.data import Dataset, DataLoader


RESULTS_FILE = "ts_hypothesis_results.json"
EPOCHS = 50
BATCH_SIZE = 32
SEQ_LEN = 96        # input window
PRED_LEN = 24       # prediction horizon


# ---- Reversible Instance Normalization (from CV → Time Series) ----

class RevIN(nn.Module):
    """Reversible Instance Normalization.
    Originally Instance Norm from CV style transfer (Ulyanov 2016).
    Adapted for time series by Kim et al. (ICLR 2022).
    """
    def __init__(self, n_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(n_features))
        self.affine_bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x, mode='norm'):
        """x: (batch, seq_len, features)"""
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self.mean) / self.std
            x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            x = (x - self.affine_bias) / self.affine_weight
            x = x * self.std + self.mean
            return x


# ---- Models ----

class LSTM_Plain(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, pred_len=24):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, pred_len * n_features)
        self.pred_len = pred_len
        self.n_features = n_features

    def forward(self, x):
        # x: (B, seq_len, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last hidden state
        return out.view(-1, self.pred_len, self.n_features)


class LSTM_RevIN(nn.Module):
    """LSTM + Reversible Instance Normalization (the hypothesis)."""
    def __init__(self, n_features, hidden=64, n_layers=2, pred_len=24):
        super().__init__()
        self.revin = RevIN(n_features)
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, pred_len * n_features)
        self.pred_len = pred_len
        self.n_features = n_features

    def forward(self, x):
        x = self.revin(x, mode='norm')
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = out.view(-1, self.pred_len, self.n_features)
        out = self.revin(out, mode='denorm')
        return out


class MLP_Plain(nn.Module):
    def __init__(self, n_features, seq_len=96, hidden=256, pred_len=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, pred_len * n_features),
        )
        self.pred_len = pred_len
        self.n_features = n_features

    def forward(self, x):
        B = x.size(0)
        out = self.net(x.view(B, -1))
        return out.view(B, self.pred_len, self.n_features)


class MLP_RevIN(nn.Module):
    """MLP + RevIN."""
    def __init__(self, n_features, seq_len=96, hidden=256, pred_len=24):
        super().__init__()
        self.revin = RevIN(n_features)
        self.net = nn.Sequential(
            nn.Linear(seq_len * n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, pred_len * n_features),
        )
        self.pred_len = pred_len
        self.n_features = n_features

    def forward(self, x):
        B = x.size(0)
        x = self.revin(x, mode='norm')
        out = self.net(x.view(B, -1))
        out = out.view(B, self.pred_len, self.n_features)
        out = self.revin(out, mode='denorm')
        return out


class LinearModel(nn.Module):
    """Simple linear baseline."""
    def __init__(self, n_features, seq_len=96, pred_len=24):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)
        self.n_features = n_features

    def forward(self, x):
        # x: (B, seq_len, features) → per-feature linear mapping
        x = x.permute(0, 2, 1)  # (B, features, seq_len)
        out = self.linear(x)     # (B, features, pred_len)
        return out.permute(0, 2, 1)  # (B, pred_len, features)


class LinearModel_RevIN(nn.Module):
    """Linear + RevIN."""
    def __init__(self, n_features, seq_len=96, pred_len=24):
        super().__init__()
        self.revin = RevIN(n_features)
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = self.revin(x, mode='norm')
        x = x.permute(0, 2, 1)
        out = self.linear(x)
        out = out.permute(0, 2, 1)
        out = self.revin(out, mode='denorm')
        return out


# ---- Data ----

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=96, pred_len=24):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def generate_nonstationary_data(n_points=10000, n_features=7, seed=42):
    """Generate non-stationary time series with trend + seasonality + noise.
    This is where RevIN should help most."""
    np.random.seed(seed)
    t = np.arange(n_points)
    data = np.zeros((n_points, n_features))

    for f in range(n_features):
        # Trend (changes over time — non-stationary)
        trend = 0.002 * t * (f + 1)
        # Seasonality (different period per feature)
        season = 5 * np.sin(2 * np.pi * t / (50 + f * 10))
        # Level shifts (sudden distribution changes)
        shifts = np.zeros(n_points)
        for s in range(0, n_points, n_points // 5):
            shifts[s:] += np.random.randn() * 3
        # Noise
        noise = np.random.randn(n_points) * (1 + 0.3 * f)
        data[:, f] = trend + season + shifts + noise

    return data


def download_etth1():
    """Try to download ETTh1 dataset."""
    try:
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
        import urllib.request
        import csv
        from io import StringIO

        print("    Downloading ETTh1...")
        response = urllib.request.urlopen(url, timeout=15)
        content = response.read().decode('utf-8')
        reader = csv.reader(StringIO(content))
        header = next(reader)
        data = []
        for row in reader:
            data.append([float(x) for x in row[1:]])  # skip date column
        return np.array(data)
    except Exception as e:
        print(f"    ETTh1 download failed: {e}")
        return None


def train_eval(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    mse_total = mae_total = n = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_total += F.mse_loss(pred, y, reduction='sum').item()
            mae_total += F.l1_loss(pred, y, reduction='sum').item()
            n += y.numel()

    return mse_total / n, mae_total / n


def run_experiment(data, dataset_name, n_features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split: 70% train, 30% val
    split = int(len(data) * 0.7)
    train_data = data[:split]
    val_data = data[split:]

    train_ds = TimeSeriesDataset(train_data, SEQ_LEN, PRED_LEN)
    val_ds = TimeSeriesDataset(val_data, SEQ_LEN, PRED_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    configs = [
        ("Linear",            lambda: LinearModel(n_features, SEQ_LEN, PRED_LEN)),
        ("Linear + RevIN",    lambda: LinearModel_RevIN(n_features, SEQ_LEN, PRED_LEN)),
        ("MLP",               lambda: MLP_Plain(n_features, SEQ_LEN, 256, PRED_LEN)),
        ("MLP + RevIN",       lambda: MLP_RevIN(n_features, SEQ_LEN, 256, PRED_LEN)),
        ("LSTM",              lambda: LSTM_Plain(n_features, 64, 2, PRED_LEN)),
        ("LSTM + RevIN",      lambda: LSTM_RevIN(n_features, 64, 2, PRED_LEN)),
    ]

    results = {}
    print(f"\n  {'Config':<20} {'MSE':>10} {'MAE':>10} {'Δ MSE':>10} {'Time':>7}")
    print(f"  {'-'*60}")

    baseline_mses = {}
    for name, model_fn in configs:
        mses, maes = [], []
        t0 = time.time()

        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_fn()
            mse, mae = train_eval(model, train_loader, val_loader, device, EPOCHS)
            mses.append(mse)
            maes.append(mae)

        elapsed = time.time() - t0
        mean_mse = np.mean(mses)
        mean_mae = np.mean(maes)

        # Compare with/without RevIN
        base_name = name.replace(" + RevIN", "")
        if "+ RevIN" not in name:
            baseline_mses[name] = mean_mse
            delta_str = "—"
        else:
            base_mse = baseline_mses.get(base_name, mean_mse)
            pct = (mean_mse - base_mse) / base_mse * 100
            delta_str = f"{pct:+.1f}%"

        marker = ""
        if "+ RevIN" in name:
            base_mse = baseline_mses.get(base_name, mean_mse)
            if mean_mse < base_mse * 0.95:
                marker = " ★ improved"
            elif mean_mse < base_mse:
                marker = " ✓"

        print(f"  {name:<20} {mean_mse:>10.4f} {mean_mae:>10.4f} {delta_str:>10} {elapsed:>5.1f}s{marker}")

        results[name] = {
            "mse_mean": round(mean_mse, 6),
            "mae_mean": round(mean_mae, 6),
            "mse_std": round(np.std(mses), 6),
            "mses": [round(m, 6) for m in mses],
            "time": round(elapsed, 1),
        }

    return results


def main():
    print("=" * 65)
    print("TIME SERIES HYPOTHESIS: Instance Norm (CV) → Forecasting")
    print("  Cross-domain: Style Transfer (CV, 2016) → RevIN (TS, 2022)")
    print("=" * 65)

    all_results = {}

    # Dataset 1: Synthetic non-stationary (RevIN should shine here)
    print(f"\n{'='*65}")
    print("  DATASET: Synthetic Non-Stationary (trend + shifts + seasonality)")
    print(f"{'='*65}")
    synth_data = generate_nonstationary_data(10000, 7)
    print(f"  Shape: {synth_data.shape}")
    all_results["synthetic"] = run_experiment(synth_data, "synthetic", 7)

    # Dataset 2: ETTh1 (real electricity data)
    etth1 = download_etth1()
    if etth1 is not None:
        print(f"\n{'='*65}")
        print("  DATASET: ETTh1 (Electricity Transformer Temperature)")
        print(f"{'='*65}")
        print(f"  Shape: {etth1.shape}")
        all_results["ETTh1"] = run_experiment(etth1, "ETTh1", etth1.shape[1])

    # Summary
    print(f"\n{'='*65}")
    print("HYPOTHESIS VALIDATION")
    print(f"{'='*65}")

    for ds_name, results in all_results.items():
        print(f"\n  {ds_name}:")
        for base in ["Linear", "MLP", "LSTM"]:
            revin = f"{base} + RevIN"
            if base in results and revin in results:
                base_mse = results[base]["mse_mean"]
                revin_mse = results[revin]["mse_mean"]
                pct = (revin_mse - base_mse) / base_mse * 100
                verdict = "HELPS" if pct < -1 else "NEUTRAL" if abs(pct) < 1 else "HURTS"
                print(f"    {base}: {base_mse:.4f} → {revin}: {revin_mse:.4f} ({pct:+.1f}%) [{verdict}]")

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
