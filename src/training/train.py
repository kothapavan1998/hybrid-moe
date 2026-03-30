"""
Train extraction MLPs + router signal on cached hidden states.
Focuses on 4 viable experts: DSCR, LTV, Cap Rate, Debt Yield.
Uses mean-pooled hidden states (proven better in extraction tests).

Usage: python -m src.training.train --cache_dir cache/hidden_states/ --layer 12
"""
import torch
import torch.nn as nn
import json, os, argparse, time
import numpy as np
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader

ACTIVE_EXPERTS = ["dscr", "ltv", "cap_rate", "debt_yield"]

INPUT_PARAM_MAP = {
    "dscr": ["noi", "ads"],
    "ltv": ["loan", "value"],
    "cap_rate": ["noi", "price"],
    "debt_yield": ["noi", "loan"],
}


def load_cached_data(cache_dir, layer, method="mean_pool"):
    prefix = method if method != "mean_pool" else "mean_pool"
    hs = torch.load(os.path.join(cache_dir, f"{prefix}_layer_{layer}.pt"),
                     weights_only=True)
    with open(os.path.join(cache_dir, "metadata.json")) as f:
        meta = json.load(f)
    print(f"  Loaded {hs.shape[0]} hidden states, layer {layer}, dim={hs.shape[1]}")
    return hs, meta


def build_extraction_targets(meta, active_experts):
    """Build per-expert training data: scenario_idx → input parameter values."""
    targets_by_expert = defaultdict(list)

    for i, m in enumerate(meta):
        for marker in m["markers"]:
            expert_name = marker["expert"]
            if expert_name not in active_experts:
                continue
            for input_name, input_val in marker["inputs"].items():
                targets_by_expert[(expert_name, input_name)].append({
                    "scenario_idx": i,
                    "value": input_val,
                })
            targets_by_expert[(expert_name, "_output")].append({
                "scenario_idx": i,
                "value": marker["output"],
            })

    return targets_by_expert


def train_single_probe(X, y, hidden_dim, device, epochs=200, lr=1e-3, label=""):
    """Train one extraction probe on GPU."""
    X = X.to(device)
    y_log = (torch.log1p(torch.abs(y)) * torch.sign(y)).to(device)

    n = len(X)
    perm = torch.randperm(n)
    n_train = int(0.85 * n)
    X_tr, X_val = X[perm[:n_train]], X[perm[n_train:]]
    y_tr, y_val = y_log[perm[:n_train]], y_log[perm[n_train:]]
    y_raw_val = y[perm[n_train:]].to(device)

    probe = nn.Sequential(
        nn.Linear(hidden_dim, 512), nn.SiLU(), nn.Dropout(0.1),
        nn.Linear(512, 128), nn.SiLU(), nn.Dropout(0.05),
        nn.Linear(128, 1),
    ).to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.HuberLoss()

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        probe.train()
        for xb, yb in loader:
            pred = probe(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0 and len(X_val) > 0:
            probe.eval()
            with torch.no_grad():
                vp = probe(X_val).squeeze(-1)
                vl = nn.MSELoss()(vp, y_val).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    if best_state:
        probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    probe.eval()
    r2_log, r2_raw = -999, -999
    if len(X_val) > 2:
        with torch.no_grad():
            pred_log = probe(X_val).squeeze(-1)
        ss_res = ((pred_log - y_val) ** 2).sum().item()
        ss_tot = ((y_val - y_val.mean()) ** 2).sum().item()
        r2_log = 1 - ss_res / (ss_tot + 1e-8)

        pred_raw = torch.sign(pred_log) * (torch.exp(torch.abs(pred_log)) - 1)
        ss_res_raw = ((pred_raw - y_raw_val) ** 2).sum().item()
        ss_tot_raw = ((y_raw_val - y_raw_val.mean()) ** 2).sum().item()
        r2_raw = 1 - ss_res_raw / (ss_tot_raw + 1e-8)

    return probe.cpu(), r2_log, r2_raw


def train_router_signal(hidden_states, meta, hidden_dim, device, epochs=30):
    """Binary classifier: does this scenario need computation?"""
    print("\n  Training router signal classifier...")
    X = hidden_states.to(device)
    y = torch.tensor([1.0 if len(m["markers"]) > 0 else 0.0 for m in meta]).to(device)

    n = len(X)
    perm = torch.randperm(n)
    n_tr = int(0.85 * n)

    probe = nn.Sequential(
        nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Dropout(0.1),
        nn.Linear(128, 32), nn.SiLU(),
        nn.Linear(32, 1), nn.Sigmoid(),
    ).to(device)

    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    loader = DataLoader(TensorDataset(X[perm[:n_tr]], y[perm[:n_tr]]),
                        batch_size=64, shuffle=True)

    for epoch in range(epochs):
        probe.train()
        for xb, yb in loader:
            pred = probe(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        val_pred = (probe(X[perm[n_tr:]]).squeeze(-1) > 0.5).float()
        val_true = y[perm[n_tr:]]
        acc = (val_pred == val_true).float().mean().item()
        tp = ((val_pred == 1) & (val_true == 1)).sum().item()
        fp = ((val_pred == 1) & (val_true == 0)).sum().item()
        fn = ((val_pred == 0) & (val_true == 1)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

    print(f"    Accuracy: {acc:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}")
    return probe.cpu(), acc


def main(cache_dir, layer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    hs, meta = load_cached_data(cache_dir, layer, method="mean_pool")
    hidden_dim = hs.shape[1]

    targets = build_extraction_targets(meta, ACTIVE_EXPERTS)

    print(f"\n  Extraction targets by (expert, param):")
    for key, items in sorted(targets.items()):
        print(f"    {key[0]:15s} / {key[1]:10s}: {len(items)} samples")

    # Train extraction probes
    print(f"\n{'='*60}")
    print(f"  TRAINING EXTRACTION PROBES (layer {layer})")
    print(f"{'='*60}")

    all_probes = {}
    all_r2 = {}
    t0 = time.time()

    for (expert_name, param_name), items in sorted(targets.items()):
        if len(items) < 30:
            continue

        X = torch.stack([hs[t["scenario_idx"]] for t in items]).float()
        y = torch.tensor([t["value"] for t in items], dtype=torch.float32)

        label = f"{expert_name}/{param_name}"
        probe, r2_log, r2_raw = train_single_probe(
            X, y, hidden_dim, device, epochs=200, lr=1e-3, label=label)

        all_probes[label] = probe
        all_r2[label] = {"r2_log": r2_log, "r2_raw": r2_raw, "n": len(items)}
        print(f"  {label:30s}: R²(log)={r2_log:.4f}  R²(raw)={r2_raw:.4f}  n={len(items)}")

    print(f"\n  Extraction training took {time.time()-t0:.1f}s")

    # Train router
    router_probe, router_acc = train_router_signal(hs, meta, hidden_dim, device)

    # Save everything
    print(f"\n{'='*60}")
    print(f"  SAVING CHECKPOINTS")
    print(f"{'='*60}")

    for label, probe in all_probes.items():
        safe_name = label.replace("/", "__")
        path = os.path.join(save_dir, f"probe_{safe_name}.pt")
        torch.save(probe.state_dict(), path)

    torch.save(router_probe.state_dict(), os.path.join(save_dir, "router_signal.pt"))

    results_summary = {"layer": layer, "r2_scores": all_r2, "router_accuracy": router_acc}
    with open(os.path.join(save_dir, "training_results.json"), "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"  Saved {len(all_probes)} extraction probes + router to {save_dir}/")

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    input_r2s = [v["r2_log"] for k, v in all_r2.items() if "/_output" not in k]
    output_r2s = [v["r2_log"] for k, v in all_r2.items() if "/_output" in k]
    if input_r2s:
        print(f"  Avg input extraction R²:  {np.mean(input_r2s):.4f}")
    if output_r2s:
        print(f"  Avg output prediction R²: {np.mean(output_r2s):.4f}")
    print(f"  Router accuracy:           {router_acc:.2%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="cache/hidden_states/")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--save_dir", default="checkpoints/")
    args = p.parse_args()
    main(args.cache_dir, args.layer, args.save_dir)
