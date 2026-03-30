"""
Pre-compute hidden states from frozen gpt-oss-20b.
Saves mean-pooled and last-token representations per layer.

Usage: python -m src.training.precompute --model_path ./models/gpt-oss-20b
"""
import torch, json, os, time, argparse
from tqdm import tqdm


def precompute(model_path, data_path, output_dir, layers, batch_size=8, max_seq_len=512):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading data: {data_path}")
    with open(data_path) as f:
        scenarios = json.load(f)

    last_token_hs = {l: [] for l in layers}
    mean_pool_hs = {l: [] for l in layers}
    all_metadata = []

    print(f"Processing {len(scenarios)} scenarios, batch_size={batch_size}...")
    t0 = time.time()

    for i in tqdm(range(0, len(scenarios), batch_size)):
        batch = scenarios[i:i + batch_size]
        texts = [s["input_prompt"] for s in batch]

        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_seq_len).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for j, s in enumerate(batch):
            attn_mask = inputs["attention_mask"][j]
            seq_len = attn_mask.sum().item()

            for layer_idx in layers:
                if layer_idx < len(outputs.hidden_states):
                    hs = outputs.hidden_states[layer_idx][j]
                    last_tok = hs[seq_len - 1, :].cpu().float()
                    mean_p = hs[:seq_len, :].mean(dim=0).cpu().float()
                    last_token_hs[layer_idx].append(last_tok)
                    mean_pool_hs[layer_idx].append(mean_p)

            all_metadata.append({
                "scenario_id": s["id"],
                "complexity": s["complexity"],
                "markers": s["computation_markers"],
                "ground_truth": s["ground_truth"],
            })

    elapsed = time.time() - t0
    print(f"\nExtracted {len(all_metadata)} scenarios in {elapsed:.1f}s "
          f"({elapsed / len(scenarios) * 1000:.0f}ms/scenario)")

    for layer_idx in layers:
        lt = torch.stack(last_token_hs[layer_idx])
        mp = torch.stack(mean_pool_hs[layer_idx])
        torch.save(lt, os.path.join(output_dir, f"last_token_layer_{layer_idx}.pt"))
        torch.save(mp, os.path.join(output_dir, f"mean_pool_layer_{layer_idx}.pt"))
        print(f"  Layer {layer_idx}: last_token {lt.shape}, mean_pool {mp.shape}")

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f)
    print(f"  Saved metadata ({len(all_metadata)} scenarios)")

    del model
    torch.cuda.empty_cache()
    print(f"\nDone! Cached in {output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./models/gpt-oss-20b")
    p.add_argument("--data", default="data/train.json")
    p.add_argument("--output", default="cache/hidden_states/")
    p.add_argument("--layers", nargs="+", type=int, default=[12, 16, 20])
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()
    precompute(args.model_path, args.data, args.output, args.layers, args.batch_size)
