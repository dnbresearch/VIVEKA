#!/usr/bin/env python3
"""
DM Hypothesis Validation: GNN Depth + DropEdge on Cora/Citeseer
================================================================
Cross-paper hypothesis combining findings from:
  - GAT (Velickovic et al.): 2-layer attention GNN on Cora
  - DropEdge (Rong et al.): Regularization enabling deeper GCNs
  - GIN (Xu et al.): Expressive GNN architecture

Hypothesis: "DropEdge regularization enables training deeper GAT models
(4-8 layers) on Cora and Citeseer, improving over the standard 2-layer
GAT. The DropEdge paper shows this works for GCN; transferring to GAT
is an untested cross-paper combination."

Expected: 1-2% accuracy improvement from deeper GAT + DropEdge

This is a genuine cross-paper hypothesis that requires knowledge from
multiple papers — exactly what our system generates.

Usage:
  pip install torch-geometric
  python3 run_dm_hypothesis.py
"""
import torch
import torch.nn.functional as F
import json, time, os
import numpy as np

# Check for torch_geometric
try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv, GATConv, GINConv
    from torch_geometric.utils import dropout_edge
except ImportError:
    print("Installing torch-geometric...")
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "torch-geometric",
                    "--break-system-packages", "-q"])
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv, GATConv, GINConv
    from torch_geometric.utils import dropout_edge


# ---- Models ----

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, n_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden))
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, n_layers=2, heads=8, dropout=0.6):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden, heads=heads, dropout=dropout))
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, n_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden))))
        for _ in range(n_layers - 2):
            self.convs.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden))))
        self.lin = torch.nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x


def train_and_eval(model, data, epochs=200, lr=0.005, weight_decay=5e-4,
                   drop_edge_rate=0.0, seed=42):
    """Train a GNN and return test accuracy."""
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # DropEdge: randomly remove edges during training
        if drop_edge_rate > 0:
            edge_index, _ = dropout_edge(data.edge_index, p=drop_edge_rate, training=True)
        else:
            edge_index = data.edge_index

        out = model(data.x, edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    return best_test_acc, best_val_acc


def main():
    print("=" * 65)
    print("DM HYPOTHESIS: GNN Depth + DropEdge on Node Classification")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    datasets = ["Cora", "CiteSeer"]
    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*65}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*65}")

        dataset = Planetoid(root=f'/tmp/{ds_name}', name=ds_name)
        data = dataset[0]
        n_features = dataset.num_node_features
        n_classes = dataset.num_classes
        print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")
        print(f"  Features: {n_features}, Classes: {n_classes}")

        # Configurations to test
        configs = [
            # (name, model_fn, drop_edge_rate)
            ("GCN-2L", lambda: GCN(n_features, 64, n_classes, n_layers=2), 0.0),
            ("GCN-4L", lambda: GCN(n_features, 64, n_classes, n_layers=4), 0.0),
            ("GCN-4L+DropEdge", lambda: GCN(n_features, 64, n_classes, n_layers=4), 0.3),
            ("GCN-8L+DropEdge", lambda: GCN(n_features, 64, n_classes, n_layers=8), 0.3),
            ("GAT-2L", lambda: GAT(n_features, 8, n_classes, n_layers=2), 0.0),
            ("GAT-4L", lambda: GAT(n_features, 8, n_classes, n_layers=4), 0.0),
            ("GAT-4L+DropEdge", lambda: GAT(n_features, 8, n_classes, n_layers=4), 0.3),
            ("GAT-8L+DropEdge", lambda: GAT(n_features, 8, n_classes, n_layers=8), 0.3),
            ("GIN-2L", lambda: GIN(n_features, 64, n_classes, n_layers=2), 0.0),
            ("GIN-4L+DropEdge", lambda: GIN(n_features, 64, n_classes, n_layers=4), 0.3),
        ]

        ds_results = {}
        print(f"\n  {'Config':<22} {'Test Acc':>10} {'Val Acc':>10} {'Time':>8}")
        print(f"  {'-'*52}")

        for name, model_fn, de_rate in configs:
            # Run 5 seeds for stability
            test_accs = []
            t0 = time.time()
            for seed in range(5):
                model = model_fn()
                test_acc, val_acc = train_and_eval(
                    model, data, epochs=200, drop_edge_rate=de_rate, seed=seed
                )
                test_accs.append(test_acc)
            elapsed = time.time() - t0

            mean_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)
            print(f"  {name:<22} {mean_acc:>9.2%} ±{std_acc:.2%} {elapsed:>6.1f}s")

            ds_results[name] = {
                "test_acc_mean": round(mean_acc * 100, 2),
                "test_acc_std": round(std_acc * 100, 2),
                "test_accs": [round(a * 100, 2) for a in test_accs],
                "drop_edge": de_rate,
                "time": round(elapsed, 1),
            }

        all_results[ds_name] = ds_results

    # Summary
    print(f"\n{'='*65}")
    print("HYPOTHESIS VALIDATION SUMMARY")
    print(f"{'='*65}")

    for ds_name in datasets:
        ds_r = all_results[ds_name]
        print(f"\n  {ds_name}:")

        # Key comparisons
        gat2 = ds_r.get("GAT-2L", {}).get("test_acc_mean", 0)
        gat4 = ds_r.get("GAT-4L", {}).get("test_acc_mean", 0)
        gat4de = ds_r.get("GAT-4L+DropEdge", {}).get("test_acc_mean", 0)
        gat8de = ds_r.get("GAT-8L+DropEdge", {}).get("test_acc_mean", 0)
        gcn2 = ds_r.get("GCN-2L", {}).get("test_acc_mean", 0)
        gcn4de = ds_r.get("GCN-4L+DropEdge", {}).get("test_acc_mean", 0)

        print(f"    GAT-2L (baseline):     {gat2:.1f}%")
        print(f"    GAT-4L (deeper, no DE):{gat4:.1f}% (Δ={gat4-gat2:+.1f}%)")
        print(f"    GAT-4L+DropEdge:       {gat4de:.1f}% (Δ={gat4de-gat2:+.1f}%)")
        print(f"    GAT-8L+DropEdge:       {gat8de:.1f}% (Δ={gat8de-gat2:+.1f}%)")
        print()
        print(f"    Does DropEdge help deeper GAT?  {'YES' if gat4de > gat4 else 'NO'} ({gat4de-gat4:+.1f}%)")
        print(f"    Does deeper GAT beat 2-layer?   {'YES' if gat4de > gat2 else 'NO'} ({gat4de-gat2:+.1f}%)")
        print(f"    Does DropEdge help deeper GCN?  {'YES' if gcn4de > gcn2 else 'MIXED'} ({gcn4de-gcn2:+.1f}%)")

    # Overall verdict
    cora_r = all_results.get("Cora", {})
    gat2_cora = cora_r.get("GAT-2L", {}).get("test_acc_mean", 0)
    gat4de_cora = cora_r.get("GAT-4L+DropEdge", {}).get("test_acc_mean", 0)

    print(f"\n  VERDICT: ", end="")
    if gat4de_cora > gat2_cora + 0.5:
        print(f"HYPOTHESIS VALIDATED — GAT+DropEdge improves by {gat4de_cora-gat2_cora:+.1f}%")
    elif gat4de_cora > gat2_cora:
        print(f"PARTIALLY VALIDATED — small improvement ({gat4de_cora-gat2_cora:+.1f}%)")
    else:
        print(f"REFUTED — deeper GAT+DropEdge does not help ({gat4de_cora-gat2_cora:+.1f}%)")

    with open("dm_hypothesis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to dm_hypothesis_results.json")


if __name__ == "__main__":
    main()
