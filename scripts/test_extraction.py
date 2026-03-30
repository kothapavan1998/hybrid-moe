"""
CRITICAL EXTRACTION TEST — Run this FIRST Saturday morning.

Tests whether a small MLP can extract numerical values from hidden states.
This is the make-or-break experiment for the entire project.

V2: Per-expert probes + more data + deeper MLP + multiple layers

If R² > 0.90: FULL SPEED AHEAD
If R² 0.70-0.90: Use deeper MLP or attention-based extraction  
If R² < 0.70: Pivot the approach

Usage: python scripts/test_extraction.py --model_path openai/gpt-oss-20b
"""
import torch
import torch.nn as nn
import json, argparse, time
import numpy as np
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader


def extract_hidden_states(model, tokenizer, scenarios, target_layers, max_scenarios=200):
    """Extract hidden states at the last token position for each scenario."""
    print(f"  Extracting hidden states from {min(len(scenarios), max_scenarios)} scenarios...")

    hidden_states_data = []
    model.eval()

    for i, s in enumerate(scenarios[:max_scenarios]):
        if i % 50 == 0:
            print(f"    Processing {i}/{min(len(scenarios), max_scenarios)}...")

        text = s["input_prompt"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for marker in s["computation_markers"]:
            target_val = marker["output"]
            expert_name = marker["expert"]
            for layer_idx in target_layers:
                if layer_idx < len(outputs.hidden_states):
                    hs = outputs.hidden_states[layer_idx][0, -1, :].cpu()
                    hidden_states_data.append({
                        "hidden_state": hs,
                        "target_value": target_val,
                        "expert": expert_name,
                        "layer": layer_idx,
                        "scenario_id": s["id"],
                    })

    return hidden_states_data


def train_per_expert_probe(data_by_expert, hidden_dim, epochs=100, lr=5e-4):
    """Train separate probes for each expert type."""
    results = {}

    for expert_name, expert_data in data_by_expert.items():
        if len(expert_data) < 10:
            print(f"\n  Skipping {expert_name}: only {len(expert_data)} samples")
            continue

        X = torch.stack([d["hidden_state"] for d in expert_data]).float()
        y = torch.tensor([d["target_value"] for d in expert_data], dtype=torch.float32)
        y_log = torch.log1p(torch.abs(y)) * torch.sign(y)

        n = len(X)
        perm = torch.randperm(n)
        n_train = max(int(0.8 * n), 1)
        X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]
        y_train, y_test = y_log[perm[:n_train]], y_log[perm[n_train:]]
        y_raw_test = y[perm[n_train:]]

        if len(X_test) == 0:
            print(f"\n  Skipping {expert_name}: not enough data for test split")
            continue

        probe = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        best_loss = float('inf')
        for epoch in range(epochs):
            probe.train()
            epoch_loss = 0
            for xb, yb in loader:
                pred = probe(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss

        probe.eval()
        with torch.no_grad():
            pred_log = probe(X_test).squeeze(-1)

        ss_res = ((pred_log - y_test) ** 2).sum().item()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
        r2_log = 1 - ss_res / (ss_tot + 1e-8)

        pred_raw = torch.sign(pred_log) * (torch.exp(torch.abs(pred_log)) - 1)
        ss_res_raw = ((pred_raw - y_raw_test) ** 2).sum().item()
        ss_tot_raw = ((y_raw_test - y_raw_test.mean()) ** 2).sum().item()
        r2_raw = 1 - ss_res_raw / (ss_tot_raw + 1e-8)

        print(f"\n  [{expert_name}] n={n}, train={n_train}, test={len(X_test)}")
        print(f"    Target range: {y.min():.2f} to {y.max():.2f}")
        print(f"    R² (log): {r2_log:.4f}  |  R² (raw): {r2_raw:.4f}")

        n_show = min(5, len(pred_raw))
        for j in range(n_show):
            print(f"    Expected: {y_raw_test[j]:>14,.2f}  "
                  f"Predicted: {pred_raw[j]:>14,.2f}  "
                  f"Error: {abs(pred_raw[j]-y_raw_test[j]):>12,.2f}")

        results[expert_name] = {
            "r2_log": r2_log, "r2_raw": r2_raw,
            "n_train": n_train, "n_test": len(X_test), "probe": probe
        }

    return results


def train_global_probe(data, hidden_dim, epochs=100, lr=5e-4):
    """Train one probe across all expert types (baseline comparison)."""
    X = torch.stack([d["hidden_state"] for d in data]).float()
    y = torch.tensor([d["target_value"] for d in data], dtype=torch.float32)
    y_log = torch.log1p(torch.abs(y)) * torch.sign(y)

    n = len(X)
    perm = torch.randperm(n)
    n_train = int(0.8 * n)
    X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]
    y_train, y_test = y_log[perm[:n_train]], y_log[perm[n_train:]]

    probe = nn.Sequential(
        nn.Linear(hidden_dim, 512), nn.SiLU(), nn.Dropout(0.1),
        nn.Linear(512, 256), nn.SiLU(),
        nn.Linear(256, 64), nn.SiLU(),
        nn.Linear(64, 1),
    )
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        probe.train()
        for xb, yb in loader:
            pred = probe(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        pred_log = probe(X_test).squeeze(-1)
    ss_res = ((pred_log - y_test) ** 2).sum().item()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    return 1 - ss_res / (ss_tot + 1e-8)


def main(model_path):
    print("=" * 60)
    print("  CRITICAL EXTRACTION TEST v2")
    print("  Per-expert probes + 200 scenarios + deeper MLP")
    print("=" * 60)

    with open("data/train.json") as f:
        scenarios = json.load(f)[:200]
    print(f"\n  Loaded {len(scenarios)} training scenarios")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n  Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, output_hidden_states=True)

    target_layers = [8, 12, 16, 20]
    t0 = time.time()
    data = extract_hidden_states(model, tokenizer, scenarios, target_layers=target_layers)
    extract_time = time.time() - t0
    print(f"\n  Extracted {len(data)} pairs in {extract_time:.1f}s")

    hidden_dim = data[0]["hidden_state"].shape[0]
    print(f"  Hidden dim: {hidden_dim}")

    # Group by expert
    data_by_expert = defaultdict(list)
    for d in data:
        data_by_expert[d["expert"]].append(d)

    print(f"\n  Expert distribution:")
    for expert, items in sorted(data_by_expert.items()):
        vals = [d["target_value"] for d in items]
        print(f"    {expert}: {len(items)} samples, range [{min(vals):.2f}, {max(vals):.2f}]")

    # Per-expert probes (THE REAL TEST)
    print(f"\n{'='*60}")
    print("  PER-EXPERT PROBE RESULTS")
    print(f"{'='*60}")
    t0 = time.time()
    expert_results = train_per_expert_probe(data_by_expert, hidden_dim)
    print(f"\n  Per-expert training took {time.time()-t0:.1f}s")

    # Global probe (baseline comparison)
    print(f"\n{'='*60}")
    print("  GLOBAL PROBE (baseline)")
    print(f"{'='*60}")
    global_r2 = train_global_probe(data, hidden_dim)
    print(f"  Global R² (log): {global_r2:.4f}")

    # Per-layer analysis (which layers encode numbers best?)
    print(f"\n{'='*60}")
    print("  PER-LAYER ANALYSIS")
    print(f"{'='*60}")
    for layer in target_layers:
        layer_data = [d for d in data if d["layer"] == layer]
        if len(layer_data) > 20:
            r2 = train_global_probe(layer_data, hidden_dim, epochs=50)
            print(f"  Layer {layer}: R² = {r2:.4f} ({len(layer_data)} samples)")

    # VERDICT
    print(f"\n{'='*60}")
    print("  FINAL VERDICT")
    print(f"{'='*60}")

    if expert_results:
        r2_values = [r["r2_log"] for r in expert_results.values()]
        avg_r2 = np.mean(r2_values)
        best_expert = max(expert_results.items(), key=lambda x: x[1]["r2_log"])
        worst_expert = min(expert_results.items(), key=lambda x: x[1]["r2_log"])

        print(f"  Average per-expert R²: {avg_r2:.4f}")
        print(f"  Best:  {best_expert[0]} (R² = {best_expert[1]['r2_log']:.4f})")
        print(f"  Worst: {worst_expert[0]} (R² = {worst_expert[1]['r2_log']:.4f})")
        print(f"  Global baseline R²: {global_r2:.4f}")

        if avg_r2 > 0.90:
            print("\n  ✅ Avg R² > 0.90 — EXTRACTION WORKS! Full speed ahead.")
        elif avg_r2 > 0.70:
            print("\n  ⚠️  Avg R² 0.70-0.90 — Promising. Proceed with caution.")
            print("     Consider: attention-based extraction for weaker experts.")
        elif avg_r2 > 0.40:
            print("\n  ⚠️  Avg R² 0.40-0.70 — Moderate signal. Needs improvement.")
            print("     Try: attention over input tokens, larger MLP, or input-level extraction.")
        else:
            print("\n  ❌ Avg R² < 0.40 — Extraction is weak.")
            print("     Consider: attention over recent tokens, or pivoting the approach.")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="openai/gpt-oss-20b")
    main(p.parse_args().model_path)
