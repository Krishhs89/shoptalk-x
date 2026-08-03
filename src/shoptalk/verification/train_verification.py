"""
Trains the verification MLP head on the Siamese pairs from build_pairs.py:
embed both images with CLIP -> pairwise features -> small MLP -> match
probability. Reports ROC-AUC and FAR/FRR at a threshold chosen by Youden's J
statistic on the validation split (design doc §3.3b, §9).

Usage:
  python -m shoptalk.verification.train_verification
"""
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn

from shoptalk.config import load_config
from shoptalk.verification.model import VerificationMLP, pair_features


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_images(paths: list, clip_model, preprocess, device: str, batch_size: int = 32) -> dict:
    unique_paths = sorted(set(paths))
    embeddings = {}
    for start in range(0, len(unique_paths), batch_size):
        batch_paths = unique_paths[start : start + batch_size]
        images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        for path, vec in zip(batch_paths, emb.cpu().numpy()):
            embeddings[path] = vec
        print(f"  embedded {min(start + batch_size, len(unique_paths))}/{len(unique_paths)} images", end="\r")
    print()
    return embeddings


def far_frr_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple:
    preds = (scores >= threshold).astype(int)
    positives, negatives = labels == 1, labels == 0
    frr = float(np.mean(preds[positives] == 0)) if positives.any() else 0.0  # false reject: true match called mismatch
    far = float(np.mean(preds[negatives] == 1)) if negatives.any() else 0.0  # false accept: true mismatch called match
    return far, frr


def main():
    cfg = load_config()
    vcfg = cfg["verification"]

    pairs_path = Path(vcfg["pairs_path"])
    if not pairs_path.exists():
        print(f"error: {pairs_path} not found -- run build_pairs.py first", file=sys.stderr)
        return 1

    pairs = [json.loads(line) for line in open(pairs_path)]
    print(f"loaded {len(pairs)} pairs")

    device = resolve_device(vcfg["device"])
    print(f"device: {device}")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        vcfg["clip_model_name"], pretrained=vcfg["clip_pretrained"], device=device
    )
    clip_model.eval()

    all_paths = [p["image_a_path"] for p in pairs] + [p["image_b_path"] for p in pairs]
    print("embedding all images with CLIP...")
    embeddings = embed_images(all_paths, clip_model, preprocess, device)

    X, y = [], []
    for p in pairs:
        if p["image_a_path"] not in embeddings or p["image_b_path"] not in embeddings:
            continue
        X.append(pair_features(embeddings[p["image_a_path"]], embeddings[p["image_b_path"]]))
        y.append(p["label"])
    X, y = np.stack(X), np.array(y, dtype=np.float32)
    print(f"{len(X)} usable pairs, {y.sum():.0f} positive / {len(y) - y.sum():.0f} negative")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=vcfg["seed"], stratify=y if len(set(y)) > 1 else None
    )

    model = VerificationMLP(input_dim=X.shape[1], hidden_dim=vcfg["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=vcfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_auc, best_state = -1.0, None
    for epoch in range(vcfg["epochs"]):
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        for start in range(0, len(perm), vcfg["batch_size"]):
            idx = perm[start : start + vcfg["batch_size"]]
            optimizer.zero_grad()
            logits = model(X_train_t[idx])
            loss = loss_fn(logits, y_train_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(X_val_t)).cpu().numpy()
        auc = roc_auc_score(y_val, val_scores) if len(set(y_val)) > 1 else float("nan")
        print(f"epoch {epoch + 1}/{vcfg['epochs']}  loss={epoch_loss / len(X_train_t):.4f}  val_auc={auc:.4f}")
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(X_val_t)).cpu().numpy()

    if len(set(y_val)) > 1:
        fpr, tpr, thresholds = roc_curve(y_val, val_scores)
        youden_j = tpr - fpr
        threshold = float(thresholds[np.argmax(youden_j)])
    else:
        threshold = vcfg["threshold"]

    far, frr = far_frr_at_threshold(y_val, val_scores, threshold)
    final_auc = roc_auc_score(y_val, val_scores) if len(set(y_val)) > 1 else float("nan")

    print(f"\nfinal: val_auc={final_auc:.4f}  threshold={threshold:.4f}  FAR={far:.4f}  FRR={frr:.4f}")

    model_path = Path(vcfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X.shape[1],
            "hidden_dim": vcfg["hidden_dim"],
            "threshold": threshold,
            "clip_model_name": vcfg["clip_model_name"],
            "clip_pretrained": vcfg["clip_pretrained"],
        },
        model_path,
    )
    print(f"saved -> {model_path}")

    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "day5_verification_eval.md", "w") as f:
        f.write("# Day 5 — Verification head evaluation\n\n")
        f.write(f"- Pairs: {len(X)} ({int(y.sum())} positive / {int(len(y) - y.sum())} negative)\n")
        f.write(f"- Validation ROC-AUC: {final_auc:.4f}\n")
        f.write(f"- Threshold (Youden's J): {threshold:.4f}\n")
        f.write(f"- FAR (false accept rate): {far:.4f}\n")
        f.write(f"- FRR (false reject rate): {frr:.4f}\n")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="verification_head"):
        mlflow.log_params({"hidden_dim": vcfg["hidden_dim"], "epochs": vcfg["epochs"], "lr": vcfg["lr"]})
        mlflow.log_metrics({"roc_auc": final_auc, "far": far, "frr": frr, "threshold": threshold})
    return 0


if __name__ == "__main__":
    sys.exit(main())
