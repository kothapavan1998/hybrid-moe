"""
Master pipeline: precompute → train → evaluate.

Runs the full Hybrid MoE experiment:
1. Pre-compute hidden states from frozen gpt-oss-20b
2. Train extraction MLPs + router signal on cached states
3. Evaluate Hybrid MoE vs baselines (zero-shot, CoT) on eval set

Usage: python scripts/run_pipeline.py --model_path ./models/gpt-oss-20b
"""
import torch, json, os, time, argparse, sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ACTIVE_EXPERTS = ["dscr", "ltv", "cap_rate", "debt_yield"]
TARGET_LAYERS = [12, 16, 20]
EPSILON = {"dscr": 0.05, "ltv": 1.0, "cap_rate": 0.5, "debt_yield": 1.0}

from src.model.cre_experts import (
    compute_dscr, compute_ltv, compute_cap_rate, compute_debt_yield
)

EXPERT_FUNCTIONS = {
    "dscr": {"fn": compute_dscr, "inputs": ["noi", "ads"]},
    "ltv": {"fn": compute_ltv, "inputs": ["loan", "value"]},
    "cap_rate": {"fn": compute_cap_rate, "inputs": ["noi", "price"]},
    "debt_yield": {"fn": compute_debt_yield, "inputs": ["noi", "loan"]},
}


def step1_precompute(model_path, data_path, cache_dir, layers, batch_size=8):
    """Pre-compute mean-pooled hidden states for all scenarios."""
    print(f"\n{'#'*60}")
    print(f"  STEP 1: PRE-COMPUTE HIDDEN STATES")
    print(f"{'#'*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    os.makedirs(cache_dir, exist_ok=True)

    print(f"  Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(data_path) as f:
        scenarios = json.load(f)
    print(f"  Loaded {len(scenarios)} scenarios")

    mean_pool_hs = {l: [] for l in layers}
    last_token_hs = {l: [] for l in layers}
    all_metadata = []
    t0 = time.time()

    for i in tqdm(range(0, len(scenarios), batch_size), desc="Precompute"):
        batch = scenarios[i:i + batch_size]
        texts = [s["input_prompt"] for s in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for j, s in enumerate(batch):
            seq_len = inputs["attention_mask"][j].sum().item()
            for layer_idx in layers:
                if layer_idx < len(outputs.hidden_states):
                    hs = outputs.hidden_states[layer_idx][j]
                    mean_pool_hs[layer_idx].append(hs[:seq_len].mean(0).cpu().float())
                    last_token_hs[layer_idx].append(hs[seq_len - 1].cpu().float())

            all_metadata.append({
                "scenario_id": s["id"], "complexity": s["complexity"],
                "markers": s["computation_markers"],
                "ground_truth": s["ground_truth"],
            })

    for layer_idx in layers:
        mp = torch.stack(mean_pool_hs[layer_idx])
        lt = torch.stack(last_token_hs[layer_idx])
        torch.save(mp, os.path.join(cache_dir, f"mean_pool_layer_{layer_idx}.pt"))
        torch.save(lt, os.path.join(cache_dir, f"last_token_layer_{layer_idx}.pt"))
        print(f"  Layer {layer_idx}: {mp.shape}")

    with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
        json.dump(all_metadata, f)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed / len(scenarios) * 1000:.0f}ms/scenario)")

    del model
    torch.cuda.empty_cache()
    return tokenizer


def step2_train(cache_dir, layers, save_dir):
    """Train extraction MLPs + router for each layer."""
    print(f"\n{'#'*60}")
    print(f"  STEP 2: TRAIN EXTRACTION MLPS + ROUTER")
    print(f"{'#'*60}")
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(cache_dir, "metadata.json")) as f:
        meta = json.load(f)

    best_layer_results = {}

    for layer in layers:
        print(f"\n  --- Layer {layer} ---")
        hs = torch.load(os.path.join(cache_dir, f"mean_pool_layer_{layer}.pt"),
                         weights_only=True)
        hidden_dim = hs.shape[1]

        layer_probes = {}
        layer_r2 = {}

        for expert_name in ACTIVE_EXPERTS:
            for param_type in ["inputs", "output"]:
                if param_type == "inputs":
                    for input_name in EXPERT_FUNCTIONS[expert_name]["inputs"]:
                        items = []
                        for i, m in enumerate(meta):
                            for marker in m["markers"]:
                                if marker["expert"] == expert_name and input_name in marker["inputs"]:
                                    items.append((i, marker["inputs"][input_name]))
                        if len(items) < 30:
                            continue
                        X = torch.stack([hs[idx] for idx, _ in items]).float()
                        y = torch.tensor([v for _, v in items], dtype=torch.float32)
                        label = f"{expert_name}/{input_name}"
                        probe, r2 = _train_probe(X, y, hidden_dim, device)
                        layer_probes[label] = probe
                        layer_r2[label] = r2
                        print(f"    {label:25s}: R²={r2:.4f} (n={len(items)})")
                else:
                    items = []
                    for i, m in enumerate(meta):
                        for marker in m["markers"]:
                            if marker["expert"] == expert_name:
                                items.append((i, marker["output"]))
                    if len(items) < 30:
                        continue
                    X = torch.stack([hs[idx] for idx, _ in items]).float()
                    y = torch.tensor([v for _, v in items], dtype=torch.float32)
                    label = f"{expert_name}/_output"
                    probe, r2 = _train_probe(X, y, hidden_dim, device)
                    layer_probes[label] = probe
                    layer_r2[label] = r2
                    print(f"    {label:25s}: R²={r2:.4f} (n={len(items)})")

        # Router signal
        y_router = torch.tensor([
            1.0 if any(m["expert"] in ACTIVE_EXPERTS for m in meta_item["markers"]) else 0.0
            for meta_item in meta
        ])
        router_probe, router_acc = _train_router(hs, y_router, hidden_dim, device)

        avg_r2 = np.mean([v for k, v in layer_r2.items() if "/_output" not in k]) if layer_r2 else -999
        best_layer_results[layer] = {
            "avg_input_r2": avg_r2, "router_acc": router_acc,
            "probes": layer_probes, "router": router_probe, "all_r2": layer_r2,
        }
        print(f"  Layer {layer} avg input R²: {avg_r2:.4f}, router acc: {router_acc:.2%}")

    best_layer = max(best_layer_results.keys(),
                     key=lambda l: best_layer_results[l]["avg_input_r2"])
    print(f"\n  Best layer: {best_layer} (R²={best_layer_results[best_layer]['avg_input_r2']:.4f})")

    best = best_layer_results[best_layer]
    for label, probe in best["probes"].items():
        safe = label.replace("/", "__")
        torch.save(probe.state_dict(), os.path.join(save_dir, f"probe_{safe}.pt"))
    torch.save(best["router"].state_dict(), os.path.join(save_dir, "router_signal.pt"))

    summary = {
        "best_layer": best_layer,
        "r2_scores": best["all_r2"],
        "router_accuracy": best["router_acc"],
    }
    with open(os.path.join(save_dir, "training_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved checkpoints to {save_dir}/")

    return best_layer, best


def _train_probe(X, y, hidden_dim, device, epochs=200, lr=1e-3):
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    X_d = X.to(device)
    y_log = (torch.log1p(torch.abs(y)) * torch.sign(y)).to(device)

    n = len(X_d)
    perm = torch.randperm(n)
    n_tr = int(0.85 * n)

    probe = nn.Sequential(
        nn.Linear(hidden_dim, 512), nn.SiLU(), nn.Dropout(0.1),
        nn.Linear(512, 128), nn.SiLU(), nn.Dropout(0.05),
        nn.Linear(128, 1),
    ).to(device)

    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss()

    loader = DataLoader(TensorDataset(X_d[perm[:n_tr]], y_log[perm[:n_tr]]),
                        batch_size=64, shuffle=True)

    for epoch in range(epochs):
        probe.train()
        for xb, yb in loader:
            pred = probe(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(X_d[perm[n_tr:]]).squeeze(-1)
        y_val = y_log[perm[n_tr:]]
    ss_res = ((pred - y_val) ** 2).sum().item()
    ss_tot = ((y_val - y_val.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return probe.cpu(), r2


def _train_router(hs, y, hidden_dim, device, epochs=30):
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    X = hs.to(device)
    y = y.to(device)
    n = len(X)
    perm = torch.randperm(n)
    n_tr = int(0.85 * n)

    probe = nn.Sequential(
        nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Dropout(0.1),
        nn.Linear(128, 32), nn.SiLU(),
        nn.Linear(32, 1), nn.Sigmoid(),
    ).to(device)

    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X[perm[:n_tr]], y[perm[:n_tr]]),
                        batch_size=64, shuffle=True)

    for epoch in range(epochs):
        probe.train()
        for xb, yb in loader:
            pred = probe(xb).squeeze(-1)
            loss = nn.BCELoss()(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        val_pred = (probe(X[perm[n_tr:]]).squeeze(-1) > 0.5).float()
        acc = (val_pred == y[perm[n_tr:]]).float().mean().item()
    return probe.cpu(), acc


def step3_evaluate(model_path, eval_data_path, cache_dir, checkpoints_dir,
                   best_layer, results_dir):
    """Evaluate hybrid MoE vs baselines on eval set."""
    print(f"\n{'#'*60}")
    print(f"  STEP 3: EVALUATION")
    print(f"{'#'*60}")
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.evaluation.baselines.prompts import format_prompt
    from src.evaluation.metrics import extract_numeric

    os.makedirs(results_dir, exist_ok=True)

    with open(eval_data_path) as f:
        eval_scenarios = json.load(f)
    print(f"  Loaded {len(eval_scenarios)} eval scenarios")

    # === Hybrid MoE evaluation ===
    print(f"\n  --- Hybrid MoE (extraction + deterministic computation) ---")
    print(f"  Loading model for hidden state extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load probes
    probes = {}
    for fname in os.listdir(checkpoints_dir):
        if fname.startswith("probe_") and fname.endswith(".pt"):
            label = fname[6:-3].replace("__", "/")
            probe = nn.Sequential(
                nn.Linear(2880, 512), nn.SiLU(), nn.Dropout(0.1),
                nn.Linear(512, 128), nn.SiLU(), nn.Dropout(0.05),
                nn.Linear(128, 1),
            )
            probe.load_state_dict(torch.load(
                os.path.join(checkpoints_dir, fname), weights_only=True))
            probe.eval()
            probes[label] = probe
    print(f"  Loaded {len(probes)} extraction probes")

    hybrid_results = []
    from tqdm import tqdm

    for s in tqdm(eval_scenarios[:200], desc="Hybrid MoE"):
        t0 = time.time()
        inputs = tokenizer(s["input_prompt"], return_tensors="pt",
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        seq_len = inputs["attention_mask"][0].sum().item()
        hs = outputs.hidden_states[best_layer][0, :seq_len].mean(0).cpu().float()

        predictions = {}
        for expert_name, cfg in EXPERT_FUNCTIONS.items():
            extracted_inputs = {}
            for input_name in cfg["inputs"]:
                probe_key = f"{expert_name}/{input_name}"
                if probe_key in probes:
                    with torch.no_grad():
                        pred_log = probes[probe_key](hs.unsqueeze(0)).item()
                    pred_val = np.sign(pred_log) * (np.exp(abs(pred_log)) - 1)
                    extracted_inputs[input_name] = pred_val

            if len(extracted_inputs) == len(cfg["inputs"]):
                params = torch.tensor([[extracted_inputs[n] for n in cfg["inputs"]]],
                                      dtype=torch.float32)
                result = cfg["fn"](params).item()
                predictions[expert_name] = {
                    "extracted_inputs": extracted_inputs,
                    "computed_result": result,
                }

        latency = (time.time() - t0) * 1000

        gt = s["ground_truth"]
        result = {"scenario_id": s["id"], "method": "hybrid_moe",
                  "latency_ms": latency, "predictions": predictions, "matches": {}}

        for expert_name, pred in predictions.items():
            if expert_name in gt:
                gt_val = gt[expert_name]
                pred_val = pred["computed_result"]
                eps = EPSILON.get(expert_name, 1.0)
                result["matches"][expert_name] = {
                    "predicted": pred_val, "ground_truth": gt_val,
                    "error": abs(pred_val - gt_val),
                    "correct": abs(pred_val - gt_val) <= eps,
                    "pct_error": abs(pred_val - gt_val) / max(abs(gt_val), 1e-8) * 100,
                }
        hybrid_results.append(result)

    # === Baseline evaluations ===
    print(f"\n  --- Running baselines ---")
    baseline_results = {}

    for method in ["zeroshot", "cot"]:
        print(f"\n  Running {method}...")
        method_results = []
        for s in tqdm(eval_scenarios[:200], desc=method):
            prompt = format_prompt(method, s)
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=1024).to(model.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            latency = (time.time() - t0) * 1000
            text = tokenizer.decode(out[0], skip_special_tokens=True)

            gt = s["ground_truth"]
            result = {"scenario_id": s["id"], "method": method,
                      "latency_ms": latency, "matches": {}}

            for expert_name in ACTIVE_EXPERTS:
                if expert_name in gt:
                    pred_val = extract_numeric(text, expert_name)
                    gt_val = gt[expert_name]
                    if pred_val is not None:
                        eps = EPSILON.get(expert_name, 1.0)
                        result["matches"][expert_name] = {
                            "predicted": pred_val, "ground_truth": gt_val,
                            "error": abs(pred_val - gt_val),
                            "correct": abs(pred_val - gt_val) <= eps,
                            "pct_error": abs(pred_val - gt_val) / max(abs(gt_val), 1e-8) * 100,
                        }
            method_results.append(result)
        baseline_results[method] = method_results

    del model
    torch.cuda.empty_cache()

    # === Aggregate and compare ===
    print(f"\n{'#'*60}")
    print(f"  RESULTS COMPARISON")
    print(f"{'#'*60}")

    all_methods = {"hybrid_moe": hybrid_results, **baseline_results}
    summary = {}

    for method, results in all_methods.items():
        correct_by_expert = defaultdict(list)
        pct_errors = defaultdict(list)
        latencies = []

        for r in results:
            latencies.append(r["latency_ms"])
            for expert, match in r.get("matches", {}).items():
                correct_by_expert[expert].append(match["correct"])
                pct_errors[expert].append(match["pct_error"])

        summary[method] = {
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "per_expert_accuracy": {
                e: np.mean(v) * 100 for e, v in correct_by_expert.items()
            },
            "per_expert_pct_error": {
                e: np.median(v) for e, v in pct_errors.items()
            },
            "overall_accuracy": np.mean([
                np.mean(v) for v in correct_by_expert.values()
            ]) * 100 if correct_by_expert else 0,
        }

    print(f"\n  {'Method':<15} {'Overall Acc':>12} {'Avg Latency':>12}")
    print(f"  {'-'*40}")
    for method, s in summary.items():
        print(f"  {method:<15} {s['overall_accuracy']:>10.1f}% {s['avg_latency_ms']:>10.0f}ms")

    print(f"\n  Per-expert accuracy:")
    print(f"  {'Expert':<15}", end="")
    for method in summary:
        print(f" {method:>12}", end="")
    print()
    for expert in ACTIVE_EXPERTS:
        print(f"  {expert:<15}", end="")
        for method, s in summary.items():
            acc = s["per_expert_accuracy"].get(expert, 0)
            print(f" {acc:>10.1f}%", end="")
        print()

    print(f"\n  Median % error:")
    print(f"  {'Expert':<15}", end="")
    for method in summary:
        print(f" {method:>12}", end="")
    print()
    for expert in ACTIVE_EXPERTS:
        print(f"  {expert:<15}", end="")
        for method, s in summary.items():
            err = s["per_expert_pct_error"].get(expert, 999)
            print(f" {err:>10.1f}%", end="")
        print()

    # Save all results
    with open(os.path.join(results_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    for method, results in all_methods.items():
        with open(os.path.join(results_dir, f"results_{method}.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to {results_dir}/")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./models/gpt-oss-20b")
    p.add_argument("--train_data", default="data/train.json")
    p.add_argument("--eval_data", default="data/eval.json")
    p.add_argument("--cache_dir", default="cache/hidden_states/")
    p.add_argument("--checkpoints", default="checkpoints/")
    p.add_argument("--results", default="results/")
    p.add_argument("--skip_precompute", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    args = p.parse_args()

    total_t0 = time.time()

    # Step 1
    if not args.skip_precompute:
        step1_precompute(args.model_path, args.train_data, args.cache_dir, TARGET_LAYERS)
    else:
        print("  Skipping precompute (--skip_precompute)")

    # Step 2
    if not args.skip_train:
        best_layer, best_results = step2_train(
            args.cache_dir, TARGET_LAYERS, args.checkpoints)
    else:
        print("  Skipping training (--skip_train)")
        with open(os.path.join(args.checkpoints, "training_results.json")) as f:
            tr = json.load(f)
        best_layer = tr["best_layer"]

    # Step 3
    step3_evaluate(args.model_path, args.eval_data, args.cache_dir,
                   args.checkpoints, best_layer, args.results)

    total_elapsed = time.time() - total_t0
    print(f"\n{'#'*60}")
    print(f"  TOTAL PIPELINE TIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
