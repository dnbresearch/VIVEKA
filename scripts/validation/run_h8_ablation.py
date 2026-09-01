#!/usr/bin/env python3
"""
H8: RealNet Resolution Ablation on MVTec-AD
"""
import subprocess, json, os, sys, re, time, shutil, yaml
from pathlib import Path
import numpy as np

RESOLUTIONS = [224, 256, 320]
TEST_CLASSES = ["bottle", "cable", "capsule", "carpet", "hazelnut"]
CONFIG_PATH = "experiments/MVTec-AD/realnet.yaml"
RESULTS_FILE = "h8_results.json"


def patch_resolution(config_path, resolution):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "dataset" in cfg and "input_size" in cfg["dataset"]:
        old = cfg["dataset"]["input_size"]
        cfg["dataset"]["input_size"] = [resolution, resolution]
        print(f"    Patched input_size: {old} → [{resolution}, {resolution}]")
    else:
        print(f"    WARNING: dataset.input_size not found!")
        return False
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return True


def train_class(dataset, class_name):
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "train_realnet.py",
        "--dataset", dataset,
        "--class_name", class_name
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        return proc
    except subprocess.TimeoutExpired:
        return None


def eval_class(dataset, class_name):
    cmd = ["python3", "evaluation_realnet.py", "--dataset", dataset, "--class_name", class_name]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return proc
    except subprocess.TimeoutExpired:
        return None


def parse_metrics(output):
    img_auroc = pix_auroc = pro = None
    # Find ALL table lines with class names and extract the LAST (best) one
    for cls in TEST_CLASSES:
        matches = []
        for line in output.split("\n"):
            if cls in line and "|" in line:
                nums = re.findall(r'(\d+\.\d+)', line)
                if len(nums) >= 2:
                    matches.append((float(nums[0]), float(nums[1])))
        if matches:
            img_auroc, pix_auroc = matches[-1]  # last eval = best
    # Also check for "best" lines
    for line in output.split("\n"):
        if "best" in line.lower() and "|" in line:
            nums = re.findall(r'(\d+\.\d+)', line)
            if len(nums) >= 2:
                img_auroc = float(nums[0])
                pix_auroc = float(nums[1])
    return img_auroc, pix_auroc, pro


def main():
    print("=" * 60)
    print("H8: RealNet Resolution Ablation on MVTec-AD")
    print("=" * 60)

    if not os.path.exists(CONFIG_PATH):
        print(f"  ERROR: Config not found at {CONFIG_PATH}")
        sys.exit(1)

    backup = CONFIG_PATH + ".backup"
    if not os.path.exists(backup):
        shutil.copy2(CONFIG_PATH, backup)
        print(f"  Backed up config → {backup}")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    print(f"  Default input_size: {cfg.get('dataset', {}).get('input_size', '?')}")
    print(f"  Resolutions to test: {RESOLUTIONS}")
    print(f"  Classes: {TEST_CLASSES}")
    print(f"  Total runs: {len(RESOLUTIONS) * len(TEST_CLASSES)}")

    all_results = {}

    for res in RESOLUTIONS:
        print(f"\n{'='*60}")
        print(f"  RESOLUTION: {res}x{res}")
        print(f"{'='*60}")

        shutil.copy2(backup, CONFIG_PATH)
        if not patch_resolution(CONFIG_PATH, res):
            continue

        res_results = {}

        for cls_name in TEST_CLASSES:
            print(f"\n  [{cls_name}] Training...", end=" ", flush=True)
            t0 = time.time()

            proc = train_class("MVTec-AD", cls_name)
            if proc is None:
                print("TIMEOUT")
                res_results[cls_name] = {"status": "timeout"}
                continue

            train_time = time.time() - t0

            if proc.returncode != 0:
                err_lines = (proc.stderr or "").strip().split("\n")[-3:]
                print(f"FAILED ({train_time:.0f}s)")
                for l in err_lines:
                    print(f"    {l[:120]}")
                res_results[cls_name] = {"status": "train_failed"}
                continue

            output = (proc.stdout or "") + (proc.stderr or "")
            img_auroc, pix_auroc, pro = parse_metrics(output)

            if img_auroc is None:
                print(f"OK ({train_time:.0f}s) → Evaluating...", end=" ", flush=True)
                eval_proc = eval_class("MVTec-AD", cls_name)
                if eval_proc:
                    eval_output = (eval_proc.stdout or "") + (eval_proc.stderr or "")
                    img_auroc, pix_auroc, pro = parse_metrics(eval_output)

            res_results[cls_name] = {
                "status": "ok",
                "image_auroc": img_auroc,
                "pixel_auroc": pix_auroc,
                "pro": pro,
                "train_time": round(train_time, 1),
            }
            print(f"OK ({train_time:.0f}s) ImgAUROC={img_auroc}, PixAUROC={pix_auroc}")

        all_results[str(res)] = res_results

    shutil.copy2(backup, CONFIG_PATH)
    print(f"\n  Restored original config")

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    header = f"  {'Class':<12}"
    for res in RESOLUTIONS:
        header += f"  {res}px_img  {res}px_pix"
    print(header)
    print(f"  {'-'*12}" + f"  {'-'*14}" * len(RESOLUTIONS))

    for cls in TEST_CLASSES:
        row = f"  {cls:<12}"
        for res in RESOLUTIONS:
            r = all_results.get(str(res), {}).get(cls, {})
            img = r.get("image_auroc")
            pix = r.get("pixel_auroc")
            img_s = f"{img:.4f}" if img is not None else "—"
            pix_s = f"{pix:.4f}" if pix is not None else "—"
            row += f"  {img_s:>7}  {pix_s:>7}"
        print(row)

    print(f"\n  {'MEAN':<12}", end="")
    for res in RESOLUTIONS:
        imgs = [all_results[str(res)][c]["image_auroc"]
                for c in TEST_CLASSES
                if all_results.get(str(res), {}).get(c, {}).get("image_auroc") is not None]
        pixs = [all_results[str(res)][c]["pixel_auroc"]
                for c in TEST_CLASSES
                if all_results.get(str(res), {}).get(c, {}).get("pixel_auroc") is not None]
        img_m = f"{np.mean(imgs):.4f}" if imgs else "—"
        pix_m = f"{np.mean(pixs):.4f}" if pixs else "—"
        print(f"  {img_m:>7}  {pix_m:>7}", end="")
    print()

    output = {
        "hypothesis": "H8: RealNet resolution ablation on MVTec-AD",
        "resolutions": RESOLUTIONS,
        "classes": TEST_CLASSES,
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
