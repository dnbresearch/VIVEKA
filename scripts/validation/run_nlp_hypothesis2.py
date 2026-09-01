#!/usr/bin/env python3
"""
NLP Hypothesis 2: R-Drop Consistency Regularization for BERT
=============================================================
Cross-paper hypothesis:
  - Dropout (Srivastava et al., JMLR 2014): Random neuron masking
  - Consistency Regularization (from semi-supervised CV, e.g. MeanTeacher):
    Two augmented views of same input should give same prediction
  - BERT fine-tuning: Standard approach doesn't exploit dropout as augmentation

Hypothesis: "Running each training sample through the model twice with
different dropout masks, then adding a KL-divergence penalty between the
two output distributions, improves BERT fine-tuning. This treats dropout
as a data augmentation and enforces prediction consistency — a technique
from semi-supervised CV that transfers to NLP."

This is known as R-Drop (Liang et al., NeurIPS 2021) and reliably improves
fine-tuning. It's a genuine cross-domain transfer from CV consistency
training to NLP.

Also tests: Focal Loss from object detection (RetinaNet, ICCV 2017)

Dataset: AG News (4-class), IMDB (2-class sentiment)
Runtime: ~10-15 minutes
"""
import torch
import torch.nn.functional as F
import json, time, random, os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RESULTS_FILE = "nlp_hypothesis2_results.json"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128, text_key="text", label_key="label"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_key = text_key
        self.label_key = label_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item[self.text_key], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": item[self.label_key],
        }


def focal_loss(logits, labels, gamma=2.0, alpha=None):
    """Focal Loss from RetinaNet (Lin et al., ICCV 2017, object detection)."""
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = ((1 - pt) ** gamma) * ce_loss
    return focal.mean()


def rdrop_loss(logits1, logits2, labels, alpha=5.0):
    """R-Drop: CE + KL divergence between two forward passes."""
    ce1 = F.cross_entropy(logits1, labels)
    ce2 = F.cross_entropy(logits2, labels)
    ce = (ce1 + ce2) / 2

    # KL divergence (symmetric)
    p = F.log_softmax(logits1, dim=-1)
    q = F.log_softmax(logits2, dim=-1)
    kl_pq = F.kl_div(p, q.exp(), reduction='batchmean')
    kl_qp = F.kl_div(q, p.exp(), reduction='batchmean')
    kl = (kl_pq + kl_qp) / 2

    return ce + alpha * kl


def train_and_eval(train_loader, val_loader, device, config, n_labels, epochs=3, lr=2e-5):
    """Train with specified config and evaluate."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_labels
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    loss_type = config.get("loss", "ce")
    rdrop_alpha = config.get("rdrop_alpha", 5.0)
    focal_gamma = config.get("focal_gamma", 2.0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if loss_type == "rdrop":
                # Two forward passes with different dropout
                out1 = model(input_ids=input_ids, attention_mask=attention_mask)
                out2 = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = rdrop_loss(out1.logits, out2.logits, labels, rdrop_alpha)
            elif loss_type == "focal":
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = focal_loss(out.logits, labels, focal_gamma)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(out.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        print(f"      Epoch {epoch+1}: loss={total_loss/n:.4f}")

    # Evaluate
    model.eval()
    correct = total = 0
    val_loss = 0
    n_val = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            val_loss += F.cross_entropy(out.logits, labels).item()
            n_val += 1
            preds = out.logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    del model
    torch.cuda.empty_cache()
    return correct / total, val_loss / n_val


def run_dataset(ds_name, train_data, val_data, n_labels, text_key, label_key):
    """Run all configs on one dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = TextDataset(train_data, tokenizer, MAX_LEN, text_key, label_key)
    val_ds = TextDataset(val_data, tokenizer, MAX_LEN, text_key, label_key)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    configs = [
        ("Baseline (CE)", {"loss": "ce"}),
        ("Focal Loss γ=2 (from CV)", {"loss": "focal", "focal_gamma": 2.0}),
        ("Focal Loss γ=3", {"loss": "focal", "focal_gamma": 3.0}),
        ("R-Drop α=1", {"loss": "rdrop", "rdrop_alpha": 1.0}),
        ("R-Drop α=5", {"loss": "rdrop", "rdrop_alpha": 5.0}),
        ("R-Drop α=10", {"loss": "rdrop", "rdrop_alpha": 10.0}),
    ]

    results = {}
    print(f"\n  {'Config':<28} {'Accuracy':>10} {'Val Loss':>10} {'Time':>7}")
    print(f"  {'-'*58}")

    for name, config in configs:
        accs, losses = [], []
        t0 = time.time()

        for seed in range(3):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            print(f"    [{name}] Seed {seed}:")
            acc, vloss = train_and_eval(
                train_loader, val_loader, device, config,
                n_labels, epochs=EPOCHS
            )
            accs.append(acc)
            losses.append(vloss)
            print(f"      → Acc={acc:.4f}")

        elapsed = time.time() - t0
        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        mean_loss = np.mean(losses)

        print(f"  {name:<28} {mean_acc:>7.2f}% ±{std_acc:.1f} {mean_loss:>10.4f} {elapsed:>5.0f}s")

        results[name] = {
            "acc_mean": round(mean_acc, 2),
            "acc_std": round(std_acc, 2),
            "val_loss": round(mean_loss, 4),
            "accs": [round(a*100, 2) for a in accs],
        }

    return results


def main():
    print("=" * 65)
    print("NLP HYPOTHESIS 2: R-Drop + Focal Loss for BERT Fine-tuning")
    print("  Cross-domain: Consistency Reg (CV) + Focal Loss (Detection)")
    print("=" * 65)

    from datasets import load_dataset
    all_results = {}

    # Dataset 1: AG News
    print(f"\n{'='*65}")
    print("  DATASET: AG News (4-class news)")
    print(f"{'='*65}")
    ds = load_dataset("ag_news")
    train_data = list(ds["train"].shuffle(seed=42).select(range(2000)))
    val_data = list(ds["test"].shuffle(seed=42).select(range(500)))
    all_results["ag_news"] = run_dataset("ag_news", train_data, val_data, 4, "text", "label")

    # Dataset 2: IMDB
    print(f"\n{'='*65}")
    print("  DATASET: IMDB (2-class sentiment)")
    print(f"{'='*65}")
    ds = load_dataset("imdb")
    train_data = list(ds["train"].shuffle(seed=42).select(range(2000)))
    val_data = list(ds["test"].shuffle(seed=42).select(range(500)))
    all_results["imdb"] = run_dataset("imdb", train_data, val_data, 2, "text", "label")

    # Summary
    print(f"\n{'='*65}")
    print("FINAL RESULTS")
    print(f"{'='*65}")

    for ds_name, results in all_results.items():
        baseline = results["Baseline (CE)"]["acc_mean"]
        print(f"\n  {ds_name}:")
        print(f"    {'Config':<28} {'Acc':>8} {'Δ':>8}")
        print(f"    {'-'*46}")
        for name, r in results.items():
            delta = r["acc_mean"] - baseline
            marker = " ★" if delta > 0.5 else ""
            print(f"    {name:<28} {r['acc_mean']:>7.2f}% {delta:>+7.2f}%{marker}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
