"""
Two final measurements:
1. Single-scenario (unbatched) latency for fair paper comparison
2. Expanded router test with 500+ negatives for statistical significance
"""
import torch, json, os, time, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import transformers.integrations.moe as _moe_module
    _moe_module._can_use_grouped_mm = lambda *args, **kwargs: False
    print("[patch] B200 grouped_mm fallback enabled")
except Exception:
    pass


def measure_single_latency(model, tokenizer, eval_scenarios, n=20):
    """Measure per-scenario latency WITHOUT batching (real-world single-query)."""
    from src.evaluation.baselines.prompts import format_prompt

    print(f"\n{'='*60}")
    print(f"  SINGLE-SCENARIO LATENCY (n={n}, no batching)")
    print(f"{'='*60}")

    # Warm up
    prompt = format_prompt("zeroshot", eval_scenarios[0])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print("  Warmed up.")

    # Hybrid MoE latency (single forward pass)
    hybrid_lats = []
    for s in eval_scenarios[:n]:
        inputs = tokenizer(s["input_prompt"], return_tensors="pt",
                           truncation=True, max_length=512).to(model.device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        hybrid_lats.append((time.time() - t0) * 1000)

    print(f"\n  Hybrid MoE (single forward pass, no generation):")
    print(f"    Mean: {np.mean(hybrid_lats):.1f} ms")
    print(f"    Median: {np.median(hybrid_lats):.1f} ms")
    print(f"    Min: {min(hybrid_lats):.1f} ms, Max: {max(hybrid_lats):.1f} ms")
    print(f"    Stdev: {np.std(hybrid_lats):.1f} ms")

    # Baseline latency (single autoregressive generation, 256 tokens)
    for method in ["zeroshot", "cot"]:
        method_lats = []
        for s in eval_scenarios[:n]:
            prompt = format_prompt(method, s)
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=1024).to(model.device)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            torch.cuda.synchronize()
            method_lats.append((time.time() - t0) * 1000)

        print(f"\n  {method} (single scenario, max 256 tokens):")
        print(f"    Mean: {np.mean(method_lats):.1f} ms")
        print(f"    Median: {np.median(method_lats):.1f} ms")
        print(f"    Min: {min(method_lats):.1f} ms, Max: {max(method_lats):.1f} ms")
        print(f"    Tokens generated: ~256")

    # Also with 512 tokens (original paper setting)
    method_lats_512 = []
    for s in eval_scenarios[:n]:
        prompt = format_prompt("zeroshot", s)
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=1024).to(model.device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        torch.cuda.synchronize()
        method_lats_512.append((time.time() - t0) * 1000)

    print(f"\n  zeroshot (single scenario, max 512 tokens — original setting):")
    print(f"    Mean: {np.mean(method_lats_512):.1f} ms")
    print(f"    Median: {np.median(method_lats_512):.1f} ms")

    speedup_256 = np.mean(method_lats) / np.mean(hybrid_lats)
    speedup_512 = np.mean(method_lats_512) / np.mean(hybrid_lats)
    print(f"\n  Speedup (256 tok): {speedup_256:.0f}x")
    print(f"  Speedup (512 tok): {speedup_512:.0f}x")

    return {
        "hybrid_moe_mean_ms": float(np.mean(hybrid_lats)),
        "hybrid_moe_median_ms": float(np.median(hybrid_lats)),
        "zeroshot_256_mean_ms": float(np.mean(method_lats)),
        "zeroshot_512_mean_ms": float(np.mean(method_lats_512)),
        "speedup_256": float(speedup_256),
        "speedup_512": float(speedup_512),
        "n_measured": n,
    }


def expanded_router_test(model, tokenizer, cache_dir):
    """Train and test router with 500+ negatives."""
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import random

    print(f"\n{'='*60}")
    print(f"  EXPANDED ROUTER TEST (500+ negatives)")
    print(f"{'='*60}")

    random.seed(42)

    general_questions = [
        "What is commercial real estate?",
        "Explain the difference between Class A and Class B office buildings.",
        "What factors affect property values in urban areas?",
        "Describe the typical lease structure for a retail property.",
        "What is a triple net lease?",
        "How does inflation affect commercial real estate investments?",
        "What are the main risks in multifamily investing?",
        "Explain the concept of cap rate in simple terms.",
        "What is the difference between gross and net operating income?",
        "How do interest rate changes impact CRE valuations?",
        "What is a 1031 exchange and how does it work?",
        "Describe the due diligence process for acquiring a commercial property.",
        "What are common exit strategies for CRE investors?",
        "How do you evaluate a property management company?",
        "What is the difference between a REIT and direct property ownership?",
        "Explain the concept of internal rate of return for real estate.",
        "What makes a good location for a multifamily property?",
        "How do environmental regulations affect commercial development?",
        "What is a ground lease?",
        "Describe the typical commercial mortgage application process.",
        "What are the advantages of investing in industrial real estate?",
        "How do you assess tenant creditworthiness?",
        "What is the difference between stabilized and value-add properties?",
        "Explain the concept of absorption rate in commercial real estate.",
        "What role do brokers play in commercial real estate transactions?",
        "Describe the typical timeline for a CRE acquisition.",
        "What are the key factors in underwriting a multifamily deal?",
        "How do you calculate effective gross income?",
        "What is a preferred return in CRE syndications?",
        "How does depreciation work for commercial real estate?",
    ]

    non_cre = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "Write a Python function to sort a list.",
        "What are the main causes of climate change?",
        "Describe the history of the Roman Empire.",
        "How does a neural network learn?",
        "What is the theory of relativity?",
        "Explain supply and demand in economics.",
        "What are the symptoms of type 2 diabetes?",
        "How does blockchain technology work?",
        "What is the Pythagorean theorem?",
        "Describe the water cycle.",
        "What are the benefits of regular exercise?",
        "How do vaccines work?",
        "What is quantum computing?",
        "Explain the process of mitosis.",
        "What causes earthquakes?",
        "How does the stock market work?",
        "What is machine learning?",
        "Describe the structure of DNA.",
        "What are renewable energy sources?",
        "How does GPS navigation work?",
        "What is the greenhouse effect?",
        "Explain the difference between TCP and UDP.",
        "What is artificial intelligence?",
        "How do electric cars work?",
        "What is the Higgs boson?",
        "Explain how a compiler works.",
        "What is the difference between HTTP and HTTPS?",
        "How does solar energy work?",
    ]

    qualitative_cre = [
        "What should I look for when touring a potential office building acquisition?",
        "How do I negotiate a commercial lease renewal?",
        "What are the warning signs of a bad CRE deal?",
        "Describe the pros and cons of investing in a hotel property.",
        "What market conditions favor buying vs leasing commercial space?",
        "How do you build relationships with commercial real estate lenders?",
        "What are the most common mistakes new CRE investors make?",
        "How do you evaluate a property's neighborhood?",
        "What is the role of a property inspector in a CRE transaction?",
        "How do zoning laws affect commercial property development?",
        "What insurance do I need for a commercial property?",
        "How do you handle tenant disputes in a commercial building?",
        "What are the tax implications of owning commercial real estate?",
        "How do you market a commercial property for sale?",
        "What are the key terms in a commercial purchase agreement?",
        "How do you assess the structural condition of a building?",
        "What is the role of an appraiser in CRE?",
        "How do co-working spaces impact the office real estate market?",
        "What are the environmental assessment requirements for CRE?",
        "How does technology change commercial real estate management?",
        "What are the benefits of green building certifications?",
        "How do you plan a capital improvement budget for a commercial property?",
        "What are the different types of commercial property financing?",
        "How do demographic trends affect multifamily demand?",
        "What is the future outlook for retail real estate?",
        "How do you handle a tenant bankruptcy?",
        "What are the risks of investing in secondary markets?",
        "How do you evaluate a property management agreement?",
        "What are the key metrics in hotel real estate investing?",
        "How do you perform a competitive market analysis for CRE?",
    ]

    ambiguous = [
        "The property was purchased for $15 million three years ago. How has the market changed?",
        "Our portfolio includes 12 properties across 5 states. How should we diversify?",
        "The building is 50,000 square feet. What tenant mix would you recommend?",
        "We have a $2 million renovation budget. How should we prioritize improvements?",
        "The lease expires in 18 months. What should we consider for renewal?",
        "Property taxes increased 15% last year. Is this typical?",
        "The occupancy rate is 85%. How does this compare to the market?",
        "We received an offer of $22 million. Should we counter or accept?",
        "The building was constructed in 1985. What upgrades are likely needed?",
        "Operating expenses are $8 per square foot. Is this reasonable?",
        "The previous owner invested $500,000 in HVAC upgrades. How does this affect value?",
        "We have 3 years remaining on our loan. What refinancing options exist?",
        "The property generates $1.2 million in gross revenue. How can we improve?",
        "There are 200 units in the complex. What amenities attract tenants?",
        "The parking ratio is 4 per 1,000 SF. Is this adequate?",
        "Annual maintenance costs are $150,000. How can we reduce them?",
        "The property sits on 5 acres. What development potential exists?",
        "We have 40 tenants with varying lease terms. How do we manage rollover risk?",
        "The building has a Walk Score of 75. How important is this?",
        "Market rents increased 8% this year. Should we adjust our rents?",
        "We spent $3 million on tenant improvements. What is the expected ROI?",
        "The property is 20 years old. When should we plan major capital expenditures?",
        "Insurance premiums are $200,000 annually. Can we reduce this?",
        "The property has 95% occupancy. Should we raise rents aggressively?",
        "Comparable properties sold for $180-220 per square foot. Where do we fall?",
        "Our weighted average lease term is 4.2 years. Is this healthy?",
        "The submarket has 12% vacancy. How does this impact our strategy?",
        "We need to replace the roof at a cost of $1.5M. When should we do this?",
        "The anchor tenant occupies 40% of the building. What is the concentration risk?",
        "Our property management fee is 5% of gross revenue. Is this competitive?",
    ]

    # 120 negatives per category = 480 total
    negatives = []
    for cat, prompts in [("general_cre", general_questions), ("non_cre", non_cre),
                         ("qualitative_cre", qualitative_cre), ("ambiguous", ambiguous)]:
        for i, prompt in enumerate(prompts):
            negatives.append({"input_prompt": prompt, "category": cat})
            # Augment with variations
            for suffix in [" Explain briefly.", " Give a detailed answer.", " What do you think?",
                          " Summarize the key points."]:
                if len(negatives) < 500:
                    negatives.append({"input_prompt": prompt + suffix, "category": cat})

    random.shuffle(negatives)
    negatives = negatives[:500]
    print(f"  Generated {len(negatives)} negative prompts")

    # Pre-compute hidden states for negatives
    from tqdm import tqdm
    neg_hs = []
    model.eval()
    for i in tqdm(range(0, len(negatives), 8), desc="Precompute negatives"):
        batch = negatives[i:i+8]
        texts = [n["input_prompt"] for n in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for j in range(len(batch)):
            seq_len = inputs["attention_mask"][j].sum().item()
            hs = outputs.hidden_states[12][j, :seq_len].mean(0).cpu().float()
            neg_hs.append(hs)

    neg_hs = torch.stack(neg_hs)
    print(f"  Negative hidden states: {neg_hs.shape}")

    # Load positive hidden states
    pos_hs = torch.load(os.path.join(cache_dir, "mean_pool_layer_12.pt"), weights_only=True)
    print(f"  Positive hidden states: {pos_hs.shape}")
    hidden_dim = pos_hs.shape[1]

    # Balance and train
    n_pos = len(pos_hs)
    n_neg = len(neg_hs)
    n_use = min(n_pos, n_neg)

    perm_pos = torch.randperm(n_pos)[:n_use]
    perm_neg = torch.randperm(n_neg)[:n_use]

    X = torch.cat([pos_hs[perm_pos], neg_hs[perm_neg]], dim=0)
    y = torch.cat([torch.ones(n_use), torch.zeros(n_use)])

    n = len(X)
    perm = torch.randperm(n)
    n_tr = int(0.80 * n)
    n_val = n - n_tr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    y = y.to(device)

    print(f"  Training: {n_tr} samples, Validation: {n_val} samples")
    print(f"  Balanced: {n_use} pos + {n_use} neg")

    router = nn.Sequential(
        nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Dropout(0.1),
        nn.Linear(128, 32), nn.SiLU(),
        nn.Linear(32, 1), nn.Sigmoid(),
    ).to(device)

    opt = torch.optim.AdamW(router.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X[perm[:n_tr]], y[perm[:n_tr]]),
                        batch_size=64, shuffle=True)

    for epoch in range(50):
        router.train()
        for xb, yb in loader:
            pred = router(xb).squeeze(-1)
            loss = nn.BCELoss()(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    router.eval()
    with torch.no_grad():
        val_X = X[perm[n_tr:]]
        val_y = y[perm[n_tr:]]
        val_pred = router(val_X).squeeze(-1)
        val_binary = (val_pred > 0.5).float()
        acc = (val_binary == val_y).float().mean().item()

        tp = ((val_binary == 1) & (val_y == 1)).sum().item()
        fp = ((val_binary == 1) & (val_y == 0)).sum().item()
        fn = ((val_binary == 0) & (val_y == 1)).sum().item()
        tn = ((val_binary == 0) & (val_y == 0)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n  Router Results (n_val={n_val}):")
    print(f"    Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Per-category breakdown
    router.eval()
    print(f"\n  Per-category false positive rate:")
    cat_offset = 0
    for cat in ["general_cre", "non_cre", "qualitative_cre", "ambiguous"]:
        cat_indices = [i for i, n in enumerate(negatives) if n["category"] == cat]
        if cat_indices:
            cat_hs = neg_hs[cat_indices].to(device)
            with torch.no_grad():
                cat_pred = (router(cat_hs).squeeze(-1) > 0.5).float()
            fp_rate = cat_pred.mean().item()
            print(f"    {cat:25s}: {fp_rate:.2%} FP rate (n={len(cat_indices)})")

    metrics = {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_val": n_val, "n_pos_total": n_use, "n_neg_total": n_use,
    }

    torch.save(router.cpu().state_dict(), "checkpoints/router_signal_expanded.pt")
    with open("checkpoints/router_expanded_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn as nn

    print(f"\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("./models/gpt-oss-20b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "./models/gpt-oss-20b", dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open("data/eval.json") as f:
        eval_scenarios = json.load(f)[:200]

    # 1. Single-scenario latency
    latency = measure_single_latency(model, tokenizer, eval_scenarios, n=20)
    with open("results/single_latency.json", "w") as f:
        json.dump(latency, f, indent=2)

    # 2. Expanded router test
    router_metrics = expanded_router_test(model, tokenizer, "cache/hidden_states/")

    del model
    torch.cuda.empty_cache()
    print(f"\n  Done! All measurements saved.")


if __name__ == "__main__":
    main()
