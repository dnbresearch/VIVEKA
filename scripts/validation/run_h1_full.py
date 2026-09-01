#!/usr/bin/env python3
"""H1: Full-scale BLIP LR Ablation on COCO (local data)"""
import torch, json, time, os, random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration

LEARNING_RATES = [1e-05, 2e-05, 3e-05, 5e-05, 1e-04]
COCO_IMG_DIR = "/content/coco_data/val2017"
COCO_ANN_FILE = "/content/coco_data/annotations/captions_val2017.json"
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500
BATCH_SIZE = 8
MAX_EPOCHS = 3
RESULTS_FILE = "h1_results_full.json"


class LocalCocoDataset(Dataset):
    def __init__(self, processor, img_dir, annotations, max_length=32):
        self.processor = processor
        self.img_dir = img_dir
        self.data = annotations
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["file_name"])
        image = Image.open(img_path).convert("RGB")
        encoding = self.processor(
            images=image, text=item["caption"],
            padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        encoding["labels"] = encoding["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in encoding.items()}


def load_coco_annotations(ann_file, img_dir, max_samples):
    """Load COCO captions and pair with image filenames."""
    with open(ann_file) as f:
        coco = json.load(f)

    # Build image_id -> filename map
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    # Group captions by image (take first caption per image)
    seen_images = set()
    data = []
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id in seen_images:
            continue
        fname = id_to_file.get(img_id)
        if fname and os.path.exists(os.path.join(img_dir, fname)):
            data.append({"file_name": fname, "caption": ann["caption"], "image_id": img_id})
            seen_images.add(img_id)
        if len(data) >= max_samples:
            break

    return data


def evaluate(model, processor, val_loader, device, img_dir, val_data):
    """Compute val loss + generate sample captions."""
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            n += 1
    avg_loss = total_loss / max(n, 1)

    # Generate 5 sample captions
    captions = []
    for item in val_data[:5]:
        img = Image.open(os.path.join(img_dir, item["file_name"])).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=30)
        cap = processor.decode(gen[0], skip_special_tokens=True)
        captions.append({"file": item["file_name"], "generated": cap, "reference": item["caption"][:80]})

    return avg_loss, captions


def main():
    print("=" * 60)
    print("H1: Full-Scale BLIP LR Ablation on COCO")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load annotations
    print(f"\n  Loading COCO annotations...")
    all_data = load_coco_annotations(COCO_ANN_FILE, COCO_IMG_DIR, TRAIN_SAMPLES + VAL_SAMPLES)
    random.seed(42)
    random.shuffle(all_data)

    train_data = all_data[:TRAIN_SAMPLES]
    val_data = all_data[TRAIN_SAMPLES:TRAIN_SAMPLES + VAL_SAMPLES]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    train_ds = LocalCocoDataset(processor, COCO_IMG_DIR, train_data)
    val_ds = LocalCocoDataset(processor, COCO_IMG_DIR, val_data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)

    # Baseline
    print(f"\n  --- Baseline (pretrained, no fine-tuning) ---")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    base_loss, base_caps = evaluate(model, processor, val_loader, device, COCO_IMG_DIR, val_data)
    print(f"  Val loss: {base_loss:.4f}")
    for c in base_caps[:3]:
        print(f"    {c['file']}: {c['generated']}")
        print(f"      ref: {c['reference']}")
    del model; torch.cuda.empty_cache()

    results = {"baseline": {"val_loss": round(base_loss, 4), "captions": base_caps}}

    # Each LR
    for lr in LEARNING_RATES:
        print(f"\n  {'='*50}")
        print(f"  LR = {lr}")
        print(f"  {'='*50}")

        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

        t0 = time.time()
        for epoch in range(MAX_EPOCHS):
            model.train()
            epoch_loss = 0
            n_batch = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                out.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += out.loss.item()
                n_batch += 1
                if n_batch % 50 == 0:
                    print(f"    Epoch {epoch+1} batch {n_batch}/{len(train_loader)} loss={out.loss.item():.4f}")
            epoch_loss /= max(n_batch, 1)
            print(f"    Epoch {epoch+1}/{MAX_EPOCHS}: avg_loss={epoch_loss:.4f}")

        train_time = time.time() - t0
        val_loss, captions = evaluate(model, processor, val_loader, device, COCO_IMG_DIR, val_data)
        delta = base_loss - val_loss

        print(f"    Val loss: {val_loss:.4f} (Δ={delta:+.4f})")
        print(f"    Time: {train_time:.0f}s")
        for c in captions[:3]:
            print(f"    {c['file']}: {c['generated']}")
            print(f"      ref: {c['reference']}")

        results[str(lr)] = {
            "train_loss": round(epoch_loss, 4),
            "val_loss": round(val_loss, 4),
            "delta": round(delta, 4),
            "captions": captions,
            "time": round(train_time, 1),
        }
        del model, optimizer; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  {'Config':<15} {'Train Loss':>12} {'Val Loss':>10} {'Δ':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Baseline':<15} {'—':>12} {results['baseline']['val_loss']:>10.4f}")
    for lr in LEARNING_RATES:
        r = results[str(lr)]
        print(f"  {'LR='+str(lr):<15} {r['train_loss']:>12.4f} {r['val_loss']:>10.4f} {r['delta']:>+10.4f}")

    best = min([str(lr) for lr in LEARNING_RATES], key=lambda k: results[k]["val_loss"])
    print(f"\n  Best: LR={best}, val_loss={results[best]['val_loss']:.4f}, Δ={results[best]['delta']:+.4f}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
