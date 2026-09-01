#!/usr/bin/env python3
"""
H1: Learning Rate Ablation for BLIP on COCO Captioning
========================================================
Tests whether different learning rates improve captioning performance.
Uses pure HuggingFace Transformers — no LAVIS needed.

Usage:
  python3 run_h1_hf.py
"""
import torch, json, time, os, random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

LEARNING_RATES = [1e-05, 2e-05, 3e-05, 5e-05, 1e-04]
TRAIN_SAMPLES = 500
VAL_SAMPLES = 100
BATCH_SIZE = 4
MAX_EPOCHS = 1
RESULTS_FILE = "h1_results.json"

# COCO val2017 image URLs and dummy captions for quick testing
COCO_API = "http://images.cocodataset.org/val2017"


class COCOCaptionDataset(Dataset):
    """Loads COCO images + captions from HuggingFace or local."""
    def __init__(self, processor, split="train", max_samples=500):
        self.processor = processor
        self.max_samples = max_samples
        self.data = []

        print(f"    Loading COCO captions ({split}, max={max_samples})...")
        try:
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceM4/COCO", split="train" if split == "train" else "validation",
                              trust_remote_code=True)
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                self.data.append({
                    "image": item["image"],
                    "caption": item["sentences"]["raw"][0] if item.get("sentences") else "a photo"
                })
        except Exception as e1:
            print(f"    HuggingFaceM4/COCO failed: {e1}")
            try:
                from datasets import load_dataset
                ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)
                for i, item in enumerate(ds):
                    if i >= max_samples:
                        break
                    self.data.append({
                        "image": item["image"],
                        "caption": item["caption"][0] if isinstance(item["caption"], list) else item["caption"]
                    })
            except Exception as e2:
                print(f"    Flickr30k failed: {e2}")
                print("    Using COCO val2017 URLs directly...")
                # Fallback: download COCO images directly
                import urllib.request, json as jjson
                ann_url = "http://images.cocodataset.org/annotations/captions_val2017.zip"
                # Use a list of known COCO val image IDs
                coco_ids = [
                    39769, 40083, 40570, 42276, 42563, 43338, 43511, 43950,
                    44260, 44699, 45057, 45472, 45500, 45550, 46252, 46431,
                    47029, 47112, 47740, 47828, 48153, 48291, 48355, 48378,
                    48396, 48555, 49091, 49151, 49215, 49375, 49550, 49667,
                    49759, 49785, 49986, 50058, 50165, 50326, 50679, 50710,
                ]
                for img_id in coco_ids[:max_samples]:
                    url = f"{COCO_API}/{str(img_id).zfill(12)}.jpg"
                    try:
                        resp = requests.get(url, timeout=10)
                        img = Image.open(BytesIO(resp.content)).convert("RGB")
                        self.data.append({"image": img, "caption": "a photo of a scene"})
                    except:
                        pass

        print(f"    Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        encoding = self.processor(
            images=image, text=item["caption"],
            padding="max_length", truncation=True, max_length=64,
            return_tensors="pt"
        )
        # Squeeze batch dim
        return {k: v.squeeze(0) for k, v in encoding.items()}


def evaluate(model, processor, val_loader, device):
    """Compute val loss and generate sample captions."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    # Generate a few captions
    captions = []
    try:
        # Use a sample image
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        img = Image.open(requests.get(url, timeout=10).raw).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        captions.append(processor.decode(out[0], skip_special_tokens=True))

        url2 = "http://images.cocodataset.org/val2017/000000040083.jpg"
        img2 = Image.open(requests.get(url2, timeout=10).raw).convert("RGB")
        inputs2 = processor(images=img2, return_tensors="pt").to(device)
        with torch.no_grad():
            out2 = model.generate(**inputs2, max_new_tokens=50)
        captions.append(processor.decode(out2[0], skip_special_tokens=True))
    except:
        pass

    return avg_loss, captions


def train_one_epoch(model, train_loader, optimizer, device):
    """Train for one epoch, return avg loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % 25 == 0:
            print(f"      Batch {n_batches}, Loss: {loss.item():.4f}")

    return total_loss / max(n_batches, 1)


def main():
    print("=" * 60)
    print("H1: BLIP Learning Rate Ablation on COCO")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  LRs: {LEARNING_RATES}")
    print(f"  Train samples: {TRAIN_SAMPLES}, Val samples: {VAL_SAMPLES}")

    # Load processor (shared across all runs)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load datasets once
    print("\n  Loading datasets...")
    train_dataset = COCOCaptionDataset(processor, "train", TRAIN_SAMPLES)
    val_dataset = COCOCaptionDataset(processor, "val", VAL_SAMPLES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Baseline: eval pretrained model (no fine-tuning)
    print("\n  --- Baseline (pretrained, no fine-tuning) ---")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    baseline_loss, baseline_caps = evaluate(model, processor, val_loader, device)
    print(f"  Baseline val loss: {baseline_loss:.4f}")
    print(f"  Baseline captions: {baseline_caps}")
    del model
    torch.cuda.empty_cache()

    results = {
        "baseline": {
            "val_loss": round(baseline_loss, 4),
            "captions": baseline_caps
        }
    }

    # Test each learning rate
    for lr in LEARNING_RATES:
        print(f"\n  {'='*50}")
        print(f"  LR = {lr}")
        print(f"  {'='*50}")

        # Fresh model for each LR
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.05
        )

        t0 = time.time()

        for epoch in range(MAX_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            print(f"    Epoch {epoch+1}: train_loss={train_loss:.4f}")

        train_time = time.time() - t0

        # Evaluate
        val_loss, captions = evaluate(model, processor, val_loader, device)
        improvement = baseline_loss - val_loss

        print(f"    Val loss: {val_loss:.4f} (baseline: {baseline_loss:.4f}, Δ={improvement:+.4f})")
        print(f"    Captions: {captions}")
        print(f"    Time: {train_time:.0f}s")

        results[str(lr)] = {
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "improvement": round(improvement, 4),
            "captions": captions,
            "train_time": round(train_time, 1),
        }

        del model, optimizer
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Config':<15} {'Val Loss':>10} {'Δ vs Base':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Baseline':<15} {results['baseline']['val_loss']:>10.4f} {'—':>10}")
    for lr in LEARNING_RATES:
        r = results[str(lr)]
        print(f"  {'LR='+str(lr):<15} {r['val_loss']:>10.4f} {r['improvement']:>+10.4f}")

    best_lr = min(
        [str(lr) for lr in LEARNING_RATES],
        key=lambda k: results[k]["val_loss"]
    )
    print(f"\n  Best LR: {best_lr} (val_loss={results[best_lr]['val_loss']:.4f})")
    print(f"  Improvement over baseline: {results[best_lr]['improvement']:+.4f}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
