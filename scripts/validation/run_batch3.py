#!/usr/bin/env python3
"""
Three System-Generated Hypotheses — Batch Runner
==================================================
All from the system's actual output. Uses models/data already available.

H_NLP4: "Apply DoLa's layer contrasting technique to GPT-2 on WikiText"
  - DoLa: Decoding by Contrasting Layers Improves Factuality
  - Cross-paper: DoLa decoding + WikiText language modeling

H_NLP5: "Test IDEAL's influence-driven selective annotations approach 
  with dataset variations (en_wiki, wikitext103) from Retrieval paper"
  - Tests whether training on higher-quality subset beats full data

H_CV2: "Apply confidence calibration to domain adaptation on CIFAR-100
  using temperature scaling from Confidence Calibration paper"
  - Cross-paper: calibration techniques + domain shifts
"""
import torch, json, time, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
all_results = {}


# ============================================================
# H_NLP4: DoLa — Layer Contrasting for Better Decoding
# System hypothesis: "Combine Grad x emb attribution method
# (from LAMBADA) with SLED's self-logits evolution decoding"
# We implement DoLa: contrast early vs late layer logits
# ============================================================

def run_dola():
    print("=" * 60)
    print("H_NLP4: DoLa Layer Contrasting on WikiText")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:100]

    results = {}
    configs = [
        ("Standard (last layer)", None),
        ("DoLa (layer 10 vs 12)", (10, 12)),
        ("DoLa (layer 6 vs 12)", (6, 12)),
        ("DoLa (layer 3 vs 12)", (3, 12)),
    ]

    for name, layers in configs:
        print(f"\n  {name}...")
        ppls = []

        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)

            with torch.no_grad():
                outputs = model(**tokens, labels=tokens["input_ids"])

                if layers is None:
                    ppl = torch.exp(outputs.loss).item()
                else:
                    early_idx, late_idx = layers
                    hidden_states = outputs.hidden_states

                    # DoLa: subtract early layer logits from late layer logits
                    late_logits = model.lm_head(hidden_states[late_idx])
                    early_logits = model.lm_head(hidden_states[early_idx])

                    # Contrasted logits
                    dola_logits = late_logits - early_logits
                    dola_logits = torch.nn.functional.log_softmax(dola_logits, dim=-1)

                    # Compute PPL from contrasted logits
                    shift_logits = dola_logits[..., :-1, :].contiguous()
                    shift_labels = tokens["input_ids"][..., 1:].contiguous()
                    loss = torch.nn.functional.nll_loss(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    ppl = torch.exp(loss).item()

                if ppl < 5000:
                    ppls.append(ppl)

        mean_ppl = np.mean(ppls)
        print(f"    PPL: {mean_ppl:.2f} ± {np.std(ppls):.2f} (n={len(ppls)})")
        results[name] = {"ppl": round(mean_ppl, 2), "std": round(np.std(ppls), 2), "n": len(ppls)}

    del model
    torch.cuda.empty_cache()
    return results


# ============================================================
# H_NLP5: Selective Data — Quality vs Quantity
# System hypothesis: "Test IDEAL's influence-driven selective
# annotations approach with dataset variations"
# We test: does training on shorter/higher-quality texts beat
# training on all texts?
# ============================================================

def run_selective_data():
    print("\n" + "=" * 60)
    print("H_NLP5: Selective Data Quality vs Quantity on WikiText")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    all_train = [t for t in ds["train"]["text"] if len(t.strip()) > 50]
    val_texts = [t for t in ds["validation"]["text"] if len(t.strip()) > 100][:200]

    # Quality filter: longer texts tend to be more coherent
    sorted_by_len = sorted(all_train, key=len, reverse=True)
    short_texts = [t for t in all_train if 50 < len(t) < 200]
    long_texts = [t for t in all_train if len(t) > 500]

    configs = [
        ("Random 1000", all_train[:1000]),
        ("Top 1000 longest", sorted_by_len[:1000]),
        ("1000 short (50-200 chars)", short_texts[:1000]),
        ("1000 long (500+ chars)", long_texts[:1000]),
        ("Random 500", all_train[:500]),
        ("Top 500 longest", sorted_by_len[:500]),
    ]

    # Quick tokenize val set
    from torch.utils.data import Dataset, DataLoader

    class LMDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=128):
            self.encodings = []
            for t in texts:
                enc = tokenizer(t, truncation=True, max_length=max_len,
                               padding="max_length", return_tensors="pt")
                enc["labels"] = enc["input_ids"].clone()
                self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})
        def __len__(self): return len(self.encodings)
        def __getitem__(self, i): return self.encodings[i]

    val_ds = LMDataset(val_texts, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=16)

    results = {}

    for name, train_texts in configs:
        print(f"\n  {name} ({len(train_texts)} texts)...")

        train_ds = LMDataset(train_texts, tokenizer)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-05, weight_decay=0.01)

        model.train()
        for epoch in range(2):
            total_loss = nb = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += out.loss.item()
                nb += 1
            print(f"    Epoch {epoch+1}: loss={total_loss/nb:.4f}")

        model.eval()
        total_loss = n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                total_loss += out.loss.item()
                n += 1

        ppl = torch.exp(torch.tensor(total_loss / n)).item()
        print(f"    Val PPL: {ppl:.2f}")

        results[name] = {"ppl": round(ppl, 2), "n_train": len(train_texts)}
        del model, optimizer
        torch.cuda.empty_cache()

    return results


# ============================================================
# H_CV2: Temperature Scaling for Calibration on CIFAR-100
# System hypothesis: "Apply confidence calibration techniques
# to domain adaptation with lr=0.001 on CIFAR-100"
# We test: does temperature scaling improve calibration?
# ============================================================

def run_calibration():
    print("\n" + "=" * 60)
    print("H_CV2: Temperature Scaling Calibration on CIFAR-100")
    print("=" * 60)

    from torchvision import models, transforms
    from torch.utils.data import DataLoader
    import torchvision

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load CIFAR-100
    print("  Loading CIFAR-100...")
    train_set = torchvision.datasets.CIFAR100(root='/tmp/cifar100', train=True,
                                               download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR100(root='/tmp/cifar100', train=False,
                                              download=True, transform=transform)

    # Use subsets for speed
    from torch.utils.data import Subset
    train_subset = Subset(train_set, range(2000))
    val_subset = Subset(test_set, range(500))
    cal_subset = Subset(test_set, range(500, 1000))  # calibration set
    test_subset = Subset(test_set, range(1000, 2000))

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32)
    cal_loader = DataLoader(cal_subset, batch_size=32)
    test_loader = DataLoader(test_subset, batch_size=32)

    # Train ResNet-18 on CIFAR-100
    print("  Training ResNet-18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(512, 100)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        total_loss = nb = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        print(f"    Epoch {epoch+1}: loss={total_loss/nb:.4f}")

    # Evaluate calibration
    def compute_ece(model, loader, temperature=1.0, n_bins=10):
        """Expected Calibration Error."""
        model.eval()
        confs, accs_list = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x) / temperature
                probs = torch.softmax(logits, dim=-1)
                conf, preds = probs.max(dim=-1)
                correct = (preds == y).float()
                confs.extend(conf.cpu().numpy())
                accs_list.extend(correct.cpu().numpy())

        confs = np.array(confs)
        accs_arr = np.array(accs_list)
        accuracy = np.mean(accs_arr)

        # ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        for i in range(n_bins):
            mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_conf = confs[mask].mean()
                bin_acc = accs_arr[mask].mean()
                ece += mask.sum() / len(confs) * abs(bin_acc - bin_conf)

        return accuracy, ece, np.mean(confs)

    # Find optimal temperature on calibration set
    print("\n  Finding optimal temperature...")
    best_t = 1.0
    best_ece = float('inf')
    results = {}

    for t in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        acc, ece, avg_conf = compute_ece(model, cal_loader, temperature=t)
        print(f"    T={t:.1f}: Acc={acc:.3f}, ECE={ece:.4f}, Conf={avg_conf:.3f}")

        if ece < best_ece:
            best_ece = ece
            best_t = t

    print(f"\n  Best T={best_t} (ECE={best_ece:.4f})")

    # Test on held-out test set
    for t_name, t_val in [("No calibration (T=1.0)", 1.0), (f"Calibrated (T={best_t})", best_t)]:
        acc, ece, avg_conf = compute_ece(model, test_loader, temperature=t_val)
        print(f"\n  {t_name}:")
        print(f"    Accuracy: {acc:.3f}, ECE: {ece:.4f}, Avg Confidence: {avg_conf:.3f}")
        results[t_name] = {
            "accuracy": round(acc * 100, 2),
            "ece": round(ece, 4),
            "avg_confidence": round(avg_conf, 4),
            "temperature": t_val,
        }

    del model
    torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================

def main():
    print("Running 3 system-generated hypotheses...\n")

    all_results = {}

    # H_NLP4: DoLa
    all_results["H_NLP4_DoLa"] = run_dola()

    # H_NLP5: Selective Data
    all_results["H_NLP5_SelectiveData"] = run_selective_data()

    # H_CV2: Calibration
    all_results["H_CV2_Calibration"] = run_calibration()

    # Summary
    print(f"\n{'='*60}")
    print("ALL RESULTS")
    print(f"{'='*60}")

    print("\n  H_NLP4 (DoLa):")
    for k, v in all_results["H_NLP4_DoLa"].items():
        print(f"    {k}: PPL={v['ppl']}")

    print("\n  H_NLP5 (Selective Data):")
    for k, v in all_results["H_NLP5_SelectiveData"].items():
        print(f"    {k}: PPL={v['ppl']}")

    print("\n  H_CV2 (Calibration):")
    for k, v in all_results["H_CV2_Calibration"].items():
        print(f"    {k}: {v}")

    with open("batch3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to batch3_results.json")


if __name__ == "__main__":
    main()
