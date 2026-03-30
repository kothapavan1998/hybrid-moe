"""
v5: Use ORIGINAL cached hidden states (proven good) + 5-fold ensemble + structured baseline.
Does NOT expand training data (batched caching corrupts representations).

Phase A: 5-fold CV ensemble on original 3000 samples (uses existing cache)
Phase B: Evaluate Hybrid MoE (ensemble input + direct output)
Phase C: Structured output baseline (new prompt format)
Phase D: Aggregate final results
"""
import torch, json, os, time, sys, re
import numpy as np
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import transformers.integrations.moe as _moe_module
    _moe_module._can_use_grouped_mm = lambda *args, **kwargs: False
    print("[patch] B200 grouped_mm fallback")
except Exception:
    pass

ACTIVE_EXPERTS = ["dscr", "ltv", "cap_rate", "debt_yield"]
SANE_RANGES = {"dscr": (0.2, 8.0), "ltv": (10, 100), "cap_rate": (1, 20), "debt_yield": (2, 30)}

from src.model.cre_experts import (
    compute_dscr, compute_ltv, compute_cap_rate, compute_debt_yield
)
EXPERT_FUNCTIONS = {
    "dscr": {"fn": compute_dscr, "inputs": ["noi", "ads"]},
    "ltv": {"fn": compute_ltv, "inputs": ["loan", "value"]},
    "cap_rate": {"fn": compute_cap_rate, "inputs": ["noi", "price"]},
    "debt_yield": {"fn": compute_debt_yield, "inputs": ["noi", "loan"]},
}

STRUCTURED_PROMPT = """You are a CRE underwriting analyst. Calculate ALL metrics from the data below.
Output ONLY the final numerical answers in this EXACT format:
DSCR: [number]
LTV: [number]%
Cap Rate: [number]%
Debt Yield: [number]%

If a metric cannot be calculated, write N/A.

{prompt}

DSCR:"""


def train_probe(X, y, hidden_dim, device, epochs=300):
    X_d = X.to(device)
    y_log = (torch.log1p(torch.abs(y)) * torch.sign(y)).to(device)
    probe = nn.Sequential(
        nn.Linear(hidden_dim, 512), nn.SiLU(), nn.Dropout(0.1),
        nn.Linear(512, 128), nn.SiLU(), nn.Dropout(0.05),
        nn.Linear(128, 1),
    ).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(X_d, y_log), batch_size=64, shuffle=True)
    for _ in range(epochs):
        probe.train()
        for xb, yb in loader:
            loss = nn.HuberLoss()(probe(xb).squeeze(-1), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    probe.eval()
    return probe


def eval_r2(probe, X_val, y_val, device):
    X_v = X_val.to(device)
    y_log = (torch.log1p(torch.abs(y_val)) * torch.sign(y_val)).to(device)
    with torch.no_grad():
        pred = probe(X_v).squeeze(-1)
    ss_res = ((pred - y_log) ** 2).sum().item()
    ss_tot = ((y_log - y_log.mean()) ** 2).sum().item()
    return 1 - ss_res / (ss_tot + 1e-8)


def ensemble_predict(probes, hs):
    preds = []
    for p in probes:
        p.eval()
        with torch.no_grad():
            v = p(hs.unsqueeze(0)).item()
        preds.append(np.sign(v) * (np.exp(abs(v)) - 1))
    return float(np.median(preds))


def smart_extract(text, metric):
    lo, hi = SANE_RANGES[metric]
    pats = {
        "dscr": [r"DSCR\s*[:=]\s*[\d,.$]+\s*/\s*[\d,.$]+\s*=\s*([\d.]+)",
                 r"DSCR\s*[:=]\s*([\d.]+)", r"DSCR\s*(?:is|of)\s*([\d.]+)"],
        "ltv": [r"LTV\s*[:=]\s*[\d,.$]+\s*/\s*[\d,.$]+\s*=?\s*([\d.]+)",
                r"LTV\s*[:=]\s*([\d.]+)", r"([\d.]+)\s*%\s*LTV"],
        "cap_rate": [r"[Cc]ap\s*[Rr]ate\s*[:=]\s*[\d,.$]+\s*/\s*[\d,.$]+\s*=?\s*([\d.]+)",
                     r"[Cc]ap\s*[Rr]ate\s*[:=]\s*([\d.]+)"],
        "debt_yield": [r"[Dd]ebt\s*[Yy]ield\s*[:=]\s*[\d,.$]+\s*/\s*[\d,.$]+\s*=?\s*([\d.]+)",
                       r"[Dd]ebt\s*[Yy]ield\s*[:=]\s*([\d.]+)"],
    }
    for pat in pats.get(metric, []):
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                v = float(m.group(1))
                if lo <= v <= hi:
                    return v
            except (ValueError, IndexError):
                continue
    if metric == "dscr":
        for m in re.finditer(r"DSCR[^.]{0,30}?([\d]+\.[\d]+)", text, re.IGNORECASE):
            try:
                v = float(m.group(1))
                if lo <= v <= hi:
                    return v
            except (ValueError, IndexError):
                continue
    return None


def parse_structured(text):
    results = {}
    for label, key in [("DSCR", "dscr"), ("LTV", "ltv"),
                        ("Cap Rate", "cap_rate"), ("Debt Yield", "debt_yield")]:
        for m in re.finditer(rf"{label}\s*[:=]\s*([\d.]+)", text, re.IGNORECASE):
            try:
                v = float(m.group(1))
                lo, hi = SANE_RANGES[key]
                if lo <= v <= hi:
                    results[key] = v
                    break
            except ValueError:
                continue
    return results


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.evaluation.baselines.prompts import format_prompt
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "checkpoints/"
    results_dir = "results/"
    os.makedirs(results_dir, exist_ok=True)
    total_t0 = time.time()

    # ===== PHASE A: 5-fold CV on ORIGINAL cached data =====
    print(f"\n{'='*60}")
    print(f"  PHASE A: 5-FOLD CV ENSEMBLE (original cache)")
    print(f"{'='*60}")

    cache_dir = "cache/hidden_states/"
    hs = torch.load(os.path.join(cache_dir, "mean_pool_layer_12.pt"), weights_only=True)
    with open(os.path.join(cache_dir, "metadata.json")) as f:
        meta = json.load(f)
    hidden_dim = hs.shape[1]
    print(f"  Loaded {hs.shape[0]} hidden states (dim={hidden_dim})")

    targets = defaultdict(list)
    for i, m in enumerate(meta):
        for marker in m["markers"]:
            exp = marker["expert"]
            if exp not in ACTIVE_EXPERTS:
                continue
            for inp_name, inp_val in marker["inputs"].items():
                targets[(exp, inp_name)].append({"idx": i, "value": inp_val})
            targets[(exp, "_output")].append({"idx": i, "value": marker["output"]})

    K = 5
    N_SEEDS = 3  # 3 seeds per fold = 15 probes per label
    ensemble_probes = {}
    ensemble_r2 = {}

    for (exp, param), items in sorted(targets.items()):
        if len(items) < 50:
            continue
        label = f"{exp}/{param}"
        X = torch.stack([hs[t["idx"]] for t in items]).float()
        y = torch.tensor([t["value"] for t in items], dtype=torch.float32)
        n = len(X)

        gen = torch.Generator().manual_seed(42)
        perm = torch.randperm(n, generator=gen)
        fold_sz = n // K

        all_probes = []
        all_r2s = []

        for fold in range(K):
            vs = fold * fold_sz
            ve = vs + fold_sz if fold < K-1 else n
            val_idx = perm[vs:ve]
            tr_idx = torch.cat([perm[:vs], perm[ve:]])

            best_probe = None
            best_r2 = -999
            for seed in range(N_SEEDS):
                torch.manual_seed(fold * 100 + seed)
                probe = train_probe(X[tr_idx], y[tr_idx], hidden_dim, device)
                r2 = eval_r2(probe, X[val_idx], y[val_idx], device)
                if r2 > best_r2:
                    best_r2 = r2
                    best_probe = probe.cpu()

            all_probes.append(best_probe)
            all_r2s.append(best_r2)

        mean_r2 = np.mean(all_r2s)
        std_r2 = np.std(all_r2s)
        ensemble_probes[label] = all_probes
        ensemble_r2[label] = {"mean": mean_r2, "std": std_r2, "folds": all_r2s}
        print(f"  {label:30s}: R²={mean_r2:.4f}±{std_r2:.4f} "
              f"[{' '.join(f'{r:.3f}' for r in all_r2s)}]")

    # ===== Load model for eval =====
    print(f"\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("./models/gpt-oss-20b",
                                               trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "./models/gpt-oss-20b", dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    with open("data/eval.json") as f:
        eval_scenarios = json.load(f)[:200]

    # ===== PHASE B: Hybrid MoE eval =====
    print(f"\n{'='*60}")
    print(f"  PHASE B: HYBRID MOE (ensemble input + direct)")
    print(f"{'='*60}")

    res_ens = []
    res_dir = []

    for s in tqdm(eval_scenarios, desc="Hybrid MoE"):
        t0 = time.time()
        inp = tokenizer(s["input_prompt"], return_tensors="pt",
                        truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        seq_len = inp["attention_mask"][0].sum().item()
        h = out.hidden_states[12][0, :seq_len].mean(0).cpu().float()
        lat = (time.time() - t0) * 1000
        gt = s["ground_truth"]

        # Ensemble input extraction → deterministic compute
        preds_a = {}
        for exp, cfg in EXPERT_FUNCTIONS.items():
            ext = {}
            for inp_n in cfg["inputs"]:
                key = f"{exp}/{inp_n}"
                if key in ensemble_probes:
                    ext[inp_n] = ensemble_predict(ensemble_probes[key], h)
            if len(ext) == len(cfg["inputs"]):
                params = torch.tensor([[ext[n] for n in cfg["inputs"]]],
                                      dtype=torch.float32)
                preds_a[exp] = cfg["fn"](params).item()

        ra = {"scenario_id": s["id"], "method": "ensemble_input",
              "latency_ms": lat, "matches": {}}
        for exp, val in preds_a.items():
            if exp in gt:
                gv = gt[exp]
                ra["matches"][exp] = {"predicted": val, "ground_truth": gv,
                    "pct_error": abs(val-gv)/max(abs(gv),1e-8)*100}
        res_ens.append(ra)

        # Direct output prediction
        preds_b = {}
        for exp in ACTIVE_EXPERTS:
            key = f"{exp}/_output"
            if key in ensemble_probes:
                preds_b[exp] = ensemble_predict(ensemble_probes[key], h)

        rb = {"scenario_id": s["id"], "method": "ensemble_direct",
              "latency_ms": lat, "matches": {}}
        for exp, val in preds_b.items():
            if exp in gt:
                gv = gt[exp]
                rb["matches"][exp] = {"predicted": val, "ground_truth": gv,
                    "pct_error": abs(val-gv)/max(abs(gv),1e-8)*100}
        res_dir.append(rb)

    # ===== PHASE C: All baselines (batch=32) =====
    print(f"\n{'='*60}")
    print(f"  PHASE C: BASELINES (zeroshot + cot + structured, batch=32)")
    print(f"{'='*60}")

    BS = 32
    bl_results = {}
    bl_raw = {}

    for mname, pfn in [
        ("zeroshot", lambda s: format_prompt("zeroshot", s)),
        ("cot", lambda s: format_prompt("cot", s)),
        ("structured", lambda s: STRUCTURED_PROMPT.format(prompt=s["input_prompt"])),
    ]:
        results = []; raw = []
        for i in tqdm(range(0, len(eval_scenarios), BS), desc=mname):
            batch = eval_scenarios[i:i+BS]
            prompts = [pfn(s) for s in batch]
            inp = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=1024).to(model.device)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=256, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            per_lat = (time.time() - t0) * 1000 / len(batch)

            for j, s in enumerate(batch):
                text = tokenizer.decode(gen[j], skip_special_tokens=True)
                raw.append({"scenario_id": s["id"], "text": text[:2000]})
                gt = s["ground_truth"]
                r = {"scenario_id": s["id"], "method": mname,
                     "latency_ms": per_lat, "matches": {}}

                if mname == "structured":
                    parsed = parse_structured(text)
                    for exp in ACTIVE_EXPERTS:
                        if exp in gt and exp in parsed:
                            gv = gt[exp]
                            r["matches"][exp] = {"predicted": parsed[exp],
                                "ground_truth": gv,
                                "pct_error": abs(parsed[exp]-gv)/max(abs(gv),1e-8)*100}
                else:
                    for exp in ACTIVE_EXPERTS:
                        if exp in gt:
                            val = smart_extract(text, exp)
                            if val is not None:
                                gv = gt[exp]
                                r["matches"][exp] = {"predicted": val,
                                    "ground_truth": gv,
                                    "pct_error": abs(val-gv)/max(abs(gv),1e-8)*100}
                results.append(r)
        bl_results[mname] = results
        bl_raw[mname] = raw

    # ===== PHASE D: Aggregate =====
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS (N=200)")
    print(f"{'='*60}")

    all_m = {"ensemble_input": res_ens, "ensemble_direct": res_dir, **bl_results}
    summary = {}
    for method, res in all_m.items():
        errs = defaultdict(list); ns = defaultdict(int); lats = []
        for r in res:
            lats.append(r["latency_ms"])
            for exp, m in r.get("matches", {}).items():
                errs[exp].append(m["pct_error"]); ns[exp] += 1
        summary[method] = {
            "N": dict(ns), "lat": float(np.mean(lats)) if lats else 0,
            "w10": {e: sum(1 for x in v if x<10)/len(v)*100 for e,v in errs.items()},
            "w20": {e: sum(1 for x in v if x<20)/len(v)*100 for e,v in errs.items()},
            "med": {e: float(np.median(v)) for e,v in errs.items()},
        }

    show = ["ensemble_input", "ensemble_direct", "zeroshot", "cot", "structured"]
    for exp in ACTIVE_EXPERTS:
        print(f"\n  {exp}:")
        for thresh, key in [("<10%", "w10"), ("<20%", "w20")]:
            print(f"    {thresh:<5}", end="")
            for m in show:
                if m in summary:
                    n = summary[m]['N'].get(exp, 0)
                    acc = summary[m][key].get(exp, 0)
                    print(f"  {acc:>5.1f}%(N={n:<3d})", end="")
            print()
        print(f"    med  ", end="")
        for m in show:
            if m in summary:
                print(f"  {summary[m]['med'].get(exp, 999):>13.1f}%", end="")
        print()

    print(f"\n  Latency:")
    for m in show:
        if m in summary:
            print(f"    {m:<20}: {summary[m]['lat']:>8.0f} ms")

    print(f"\n  Probe R² (5-fold CV, best-of-{N_SEEDS} per fold):")
    for label in sorted(ensemble_r2):
        r = ensemble_r2[label]
        print(f"    {label:30s}: {r['mean']:.4f}±{r['std']:.4f}")

    final = {"summary": summary,
             "ensemble_r2": {k: {"mean": v["mean"], "std": v["std"]}
                             for k, v in ensemble_r2.items()}}
    with open(os.path.join(results_dir, "final_v5.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)
    for method, res in all_m.items():
        with open(os.path.join(results_dir, f"results_{method}.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
    for mn, raw in bl_raw.items():
        with open(os.path.join(results_dir, f"raw_{mn}_v5.json"), "w") as f:
            json.dump(raw, f, indent=2)

    del model; torch.cuda.empty_cache()
    print(f"\n  DONE: {time.time()-total_t0:.0f}s ({(time.time()-total_t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
