#!/usr/bin/env python3
"""
DM Hypothesis: Residual Connections Enable Deeper GNNs
=======================================================
Cross-paper hypothesis:
  - ResNet (He et al., CVPR 2016): Skip connections enable very deep CNNs
  - GCN (Kipf & Welling, ICLR 2017): Standard 2-layer GCN for node classification
  - JKNet (Xu et al., ICML 2018): Jumping knowledge for graph networks

Hypothesis: "Applying residual connections from ResNet to GCN enables
training 4-8 layer GCN models that match or exceed standard 2-layer GCN
performance on Cora and CiteSeer, preventing the oversmoothing that
normally degrades deeper GNNs."

This combines a CV technique (residual connections) with a DM model (GCN)
— exactly the kind of cross-domain transfer our system generates.
"""
import torch
import torch.nn.functional as F
import json, time, numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import dropout_edge


class GCN_Plain(torch.nn.Module):
    """Standard GCN without skip connections."""
    def __init__(self, in_ch, hid, out_ch, n_layers, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_ch, hid))
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hid, hid))
        self.convs.append(GCNConv(hid, out_ch))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GCN_Residual(torch.nn.Module):
    """GCN with residual (skip) connections — the hypothesis."""
    def __init__(self, in_ch, hid, out_ch, n_layers, dropout=0.5):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_ch, hid)
        self.convs = torch.nn.ModuleList()
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hid, hid))
        self.output = torch.nn.Linear(hid, out_ch)
        self.dropout = dropout
        self.norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hid) for _ in range(n_layers - 1)
        ])

    def forward(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = x + residual  # SKIP CONNECTION (from ResNet)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output(x)


class GCN_JumpingKnowledge(torch.nn.Module):
    """GCN with Jumping Knowledge — concatenates all layer outputs."""
    def __init__(self, in_ch, hid, out_ch, n_layers, dropout=0.5):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_ch, hid)
        self.convs = torch.nn.ModuleList()
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hid, hid))
        # JK: concatenate all layers, then project
        self.output = torch.nn.Linear(hid * n_layers, out_ch)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        layer_outputs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        # Jumping knowledge: concat all layers
        x = torch.cat(layer_outputs, dim=-1)
        return self.output(x)


class GCN_DenseResidual(torch.nn.Module):
    """GCN with DenseNet-style connections — each layer sees all previous."""
    def __init__(self, in_ch, hid, out_ch, n_layers, dropout=0.5):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_ch, hid)
        self.convs = torch.nn.ModuleList()
        self.projs = torch.nn.ModuleList()
        for i in range(n_layers - 1):
            input_dim = hid * (i + 1)
            self.convs.append(GCNConv(input_dim, hid))
            self.projs.append(torch.nn.Identity())  # placeholder
        self.output = torch.nn.Linear(hid * n_layers, out_ch)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        features = [x]
        for conv in self.convs:
            combined = torch.cat(features, dim=-1)
            x = F.relu(conv(combined, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            features.append(x)
        x = torch.cat(features, dim=-1)
        return self.output(x)


def train_eval(model, data, epochs=200, lr=0.01, wd=5e-4, seed=0):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val = best_test = 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(1)
            val = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if val > best_val:
            best_val = val
            best_test = test
    return best_test


def main():
    print("=" * 70)
    print("DM HYPOTHESIS: Residual Connections Enable Deeper GNNs")
    print("  Cross-paper: ResNet (CVPR'16) → GCN (ICLR'17)")
    print("=" * 70)

    datasets = ["Cora", "CiteSeer", "PubMed"]
    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*70}")

        dataset = Planetoid(root=f'/tmp/{ds_name}', name=ds_name)
        data = dataset[0]
        nf, nc = dataset.num_node_features, dataset.num_classes
        hid = 64

        configs = [
            # (name, model_factory)
            ("GCN-2L (baseline)",      lambda: GCN_Plain(nf, hid, nc, 2)),
            ("GCN-4L (plain)",         lambda: GCN_Plain(nf, hid, nc, 4)),
            ("GCN-8L (plain)",         lambda: GCN_Plain(nf, hid, nc, 8)),
            ("GCN-16L (plain)",        lambda: GCN_Plain(nf, hid, nc, 16)),
            ("GCN-4L + Residual",      lambda: GCN_Residual(nf, hid, nc, 4)),
            ("GCN-8L + Residual",      lambda: GCN_Residual(nf, hid, nc, 8)),
            ("GCN-16L + Residual",     lambda: GCN_Residual(nf, hid, nc, 16)),
            ("GCN-32L + Residual",     lambda: GCN_Residual(nf, hid, nc, 32)),
            ("GCN-4L + JK",           lambda: GCN_JumpingKnowledge(nf, hid, nc, 4)),
            ("GCN-8L + JK",           lambda: GCN_JumpingKnowledge(nf, hid, nc, 8)),
        ]

        ds_results = {}
        print(f"\n  {'Config':<25} {'Accuracy':>12} {'vs 2L':>8} {'Time':>7}")
        print(f"  {'-'*55}")

        baseline_acc = None
        for name, model_fn in configs:
            accs = []
            t0 = time.time()
            for seed in range(10):
                acc = train_eval(model_fn(), data, seed=seed)
                accs.append(acc)
            elapsed = time.time() - t0

            mean = np.mean(accs) * 100
            std = np.std(accs) * 100

            if baseline_acc is None:
                baseline_acc = mean
                delta_str = "—"
            else:
                delta = mean - baseline_acc
                delta_str = f"{delta:+.1f}%"

            marker = ""
            if "plain" in name and "2L" not in name and mean < baseline_acc - 2:
                marker = " ← degraded"
            if "Residual" in name and mean > baseline_acc:
                marker = " ★ improved"
            if "JK" in name and mean > baseline_acc:
                marker = " ★ improved"

            print(f"  {name:<25} {mean:>6.1f}% ±{std:.1f}  {delta_str:>8} {elapsed:>5.1f}s{marker}")

            ds_results[name] = {
                "acc_mean": round(mean, 2),
                "acc_std": round(std, 2),
                "accs": [round(a*100, 2) for a in accs],
                "time": round(elapsed, 1),
            }

        all_results[ds_name] = ds_results

    # Summary
    print(f"\n{'='*70}")
    print("HYPOTHESIS VALIDATION")
    print(f"{'='*70}")

    for ds_name in datasets:
        r = all_results[ds_name]
        base = r["GCN-2L (baseline)"]["acc_mean"]
        plain4 = r["GCN-4L (plain)"]["acc_mean"]
        plain8 = r["GCN-8L (plain)"]["acc_mean"]
        plain16 = r["GCN-16L (plain)"]["acc_mean"]
        res4 = r["GCN-4L + Residual"]["acc_mean"]
        res8 = r["GCN-8L + Residual"]["acc_mean"]
        res16 = r["GCN-16L + Residual"]["acc_mean"]
        res32 = r["GCN-32L + Residual"]["acc_mean"]

        print(f"\n  {ds_name}:")
        print(f"    Oversmoothing without residual:")
        print(f"      2L: {base:.1f}% → 4L: {plain4:.1f}% → 8L: {plain8:.1f}% → 16L: {plain16:.1f}%")
        print(f"    With residual connections:")
        print(f"      4L: {res4:.1f}% → 8L: {res8:.1f}% → 16L: {res16:.1f}% → 32L: {res32:.1f}%")
        print(f"    Residual improvement at 8L: {res8-plain8:+.1f}%")
        print(f"    Residual improvement at 16L: {res16-plain16:+.1f}%")
        print(f"    Best residual vs baseline 2L: {max(res4,res8,res16,res32)-base:+.1f}%")

    with open("dm_residual_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to dm_residual_results.json")


if __name__ == "__main__":
    main()
