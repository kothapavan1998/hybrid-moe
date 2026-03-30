"""
Standalone baseline evaluation — runs generation-based methods on the eval dataset.
For the full pipeline including Hybrid MoE, use scripts/run_pipeline.py instead.

Usage: python scripts/evaluate.py --model_path openai/gpt-oss-20b --data data/eval.json
"""
import torch, json, time, argparse
from tqdm import tqdm
from src.evaluation.metrics import evaluate_scenario, aggregate, print_comparison
from src.evaluation.baselines.prompts import format_prompt


def run_baseline(model, tokenizer, scenarios, method, max_tokens=512):
    """Run a baseline method on all scenarios."""
    results = []
    for s in tqdm(scenarios, desc=method):
        prompt = format_prompt(method, s)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        latency = (time.time() - t0) * 1000

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        tokens = out.shape[1]

        result = evaluate_scenario(s, text, method, latency_ms=latency, tokens=tokens)
        results.append(result)
    return results


def main(model_path, data_path, output_dir, methods):
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading eval data: {data_path}")
    with open(data_path) as f:
        scenarios = json.load(f)

    all_results = {}
    all_agg = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Running: {method}")
        print(f"{'='*60}")

        if method == "hybrid_moe":
            print("  Skipping hybrid_moe — use scripts/run_pipeline.py for full evaluation")
            continue

        results = run_baseline(model, tokenizer, scenarios, method)

        all_results[method] = results
        all_agg[method] = aggregate(results)

        # Save per-method results
        with open(os.path.join(output_dir, f"results_{method}.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Print comparison
    if all_agg:
        print_comparison(all_agg)

    # Save aggregate
    with open(os.path.join(output_dir, "aggregate.json"), "w") as f:
        json.dump({k: {kk: str(vv) if not isinstance(vv, (int, float, type(None))) else vv
                      for kk, vv in v.items()} for k, v in all_agg.items()}, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="openai/gpt-oss-20b")
    p.add_argument("--data", default="data/eval.json")
    p.add_argument("--output", default="results/")
    p.add_argument("--methods", nargs="+", default=["zeroshot", "cot", "react", "toolken"])
    args = p.parse_args()
    main(args.model_path, args.data, args.output, args.methods)
