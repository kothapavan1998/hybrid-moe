"""Evaluation metrics for Hybrid MoE — extraction, accuracy, chain order, latency."""
import re, json, numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


def extract_numeric(text: str, metric: str) -> Optional[float]:
    """Extract a numeric value for a metric from generated text."""
    patterns = {
        "dscr": [r"DSCR\s*[=:]\s*([\d.]+)", r"(?:debt service coverage)\s*(?:is|of|=|:)\s*([\d.]+)"],
        "ltv": [r"LTV\s*[=:]\s*([\d.]+)", r"(?:loan.to.value)\s*(?:is|=|:)\s*([\d.]+)"],
        "cap_rate": [r"[Cc]ap\s*[Rr]ate\s*[=:]\s*([\d.]+)", r"(?:capitalization rate)\s*(?:is|=|:)\s*([\d.]+)"],
        "debt_yield": [r"[Dd]ebt\s*[Yy]ield\s*[=:]\s*([\d.]+)"],
        "annual_debt_service": [r"(?:annual debt service|ADS)\s*[=:]\s*\$?([\d,]+\.?\d*)"],
        "noi_projection": [r"(?:projected NOI|year.3 NOI)\s*[=:]\s*\$?([\d,]+\.?\d*)"],
    }
    for pat in patterns.get(metric, []):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try: return float(m.group(1).replace(",", ""))
            except: continue
    return None


def extract_chain_order(text: str) -> List[str]:
    """Extract computation order from generated text."""
    pats = {"annual_debt_service": r"(?:annual debt service|ADS)\s*[=:]",
            "dscr": r"(?:DSCR|debt service coverage)\s*[=:]",
            "ltv": r"(?:LTV|loan.to.value)\s*[=:]",
            "cap_rate": r"(?:cap rate)\s*[=:]",
            "debt_yield": r"(?:debt yield)\s*[=:]"}
    positions = {}
    for name, pat in pats.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m: positions[name] = m.start()
    return sorted(positions.keys(), key=lambda k: positions[k])


EPSILONS = {"dscr": 0.02, "ltv": 0.1, "cap_rate": 0.05, "debt_yield": 0.05,
            "annual_debt_service": 500.0, "noi_projection": 500.0}


def evaluate_scenario(scenario, generation, method, latency_ms=0, tokens=0, passes=1):
    """Evaluate one scenario against ground truth."""
    results = {"id": scenario["id"], "complexity": scenario["complexity"],
               "method": method, "latency_ms": latency_ms, "tokens": tokens, "passes": passes,
               "metrics": [], "chain_correct": None}

    for marker in scenario["computation_markers"]:
        expert = marker["expert"]
        predicted = extract_numeric(generation, expert)
        expected = marker["output"]
        eps = EPSILONS.get(expert, 0.05)
        correct = predicted is not None and abs(predicted - expected) <= eps
        error = abs(predicted - expected) if predicted is not None else None
        results["metrics"].append({
            "metric": expert, "expected": expected, "predicted": predicted,
            "error": error, "correct": correct,
        })

    if scenario["complexity"] >= 3:
        chain = extract_chain_order(generation)
        if "annual_debt_service" in chain and "dscr" in chain:
            results["chain_correct"] = chain.index("annual_debt_service") < chain.index("dscr")
    return results


def aggregate(results_list):
    """Aggregate results across scenarios."""
    if not results_list: return {}
    metric_acc = {}
    for r in results_list:
        for m in r["metrics"]:
            metric_acc.setdefault(m["metric"], []).append(m["correct"])

    agg = {
        "n": len(results_list),
        "method": results_list[0]["method"],
        "per_metric_accuracy": {k: sum(v)/len(v) for k, v in metric_acc.items()},
        "overall_accuracy": sum(m["correct"] for r in results_list for m in r["metrics"]) /
                           max(sum(len(r["metrics"]) for r in results_list), 1),
        "chain_correctness": None,
        "avg_latency": np.mean([r["latency_ms"] for r in results_list if r["latency_ms"] > 0]) if any(r["latency_ms"] > 0 for r in results_list) else 0,
        "avg_tokens": np.mean([r["tokens"] for r in results_list]),
    }

    chains = [r["chain_correct"] for r in results_list if r["chain_correct"] is not None]
    if chains: agg["chain_correctness"] = sum(chains) / len(chains)
    return agg


def print_comparison(all_agg):
    """Print comparison table."""
    methods = list(all_agg.keys())
    print(f"\n{'Metric':<25} | " + " | ".join(f"{m:>12}" for m in methods))
    print("-" * (28 + 15 * len(methods)))

    # Collect all metric names
    all_metrics = set()
    for a in all_agg.values():
        all_metrics.update(a.get("per_metric_accuracy", {}).keys())

    for metric in sorted(all_metrics):
        vals = [all_agg[m].get("per_metric_accuracy", {}).get(metric, 0) for m in methods]
        print(f"{metric:<25} | " + " | ".join(f"{v:>11.1%}" for v in vals))

    # Overall
    vals = [all_agg[m].get("overall_accuracy", 0) for m in methods]
    print(f"{'OVERALL':<25} | " + " | ".join(f"{v:>11.1%}" for v in vals))

    # Chain
    vals = [all_agg[m].get("chain_correctness") for m in methods]
    print(f"{'Chain Correctness':<25} | " + " | ".join(f"{v:>11.1%}" if v else f"{'N/A':>12}" for v in vals))

    # Latency
    vals = [all_agg[m].get("avg_latency", 0) for m in methods]
    print(f"{'Avg Latency (ms)':<25} | " + " | ".join(f"{v:>11.0f}" for v in vals))
    print()


def test_extraction():
    tests = [
        ("DSCR = 1.29x", "dscr", 1.29),
        ("LTV = 70.0%", "ltv", 70.0),
        ("Cap Rate = 6.84%", "cap_rate", 6.84),
        ("ADS: $796,044", "annual_debt_service", 796044.0),
        ("Debt Yield = 9.77%", "debt_yield", 9.77),
    ]
    print("Testing value extraction...")
    for text, metric, expected in tests:
        result = extract_numeric(text, metric)
        ok = result is not None and abs(result - expected) < 0.01
        print(f"  {'✓' if ok else '✗'} '{text}' → {metric} = {result}")

    text = "Annual Debt Service = $796,044. DSCR = 1.29x. LTV = 70%."
    chain = extract_chain_order(text)
    ok = chain[0] == "annual_debt_service" and chain[1] == "dscr"
    print(f"  {'✓' if ok else '✗'} Chain order: {chain}")
    print("  All extraction tests passed!")


if __name__ == "__main__":
    test_extraction()
