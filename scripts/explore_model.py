"""
Explore gpt-oss-20b internals. Run first thing Saturday morning.
Prints: layer structure, MoE module paths, router shapes, expert shapes.
Usage: python scripts/explore_model.py [--model_path PATH]
"""
import sys, argparse, torch

def main(model_path):
    print("=" * 60)
    print("  gpt-oss-20b Architecture Explorer")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\nLoading model (1-2 min)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"  Class: {model.__class__.__name__}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Top level
    print("\n--- Top-level modules ---")
    for name, mod in model.named_children():
        print(f"  {name}: {mod.__class__.__name__} ({sum(p.numel() for p in mod.parameters()):,} params)")

    # Find MoE / router / expert modules
    print("\n--- MoE-related modules ---")
    router_info = {}
    for name, mod in model.named_modules():
        cn = mod.__class__.__name__.lower()
        if any(k in cn for k in ['moe', 'expert', 'mixture', 'sparse']):
            print(f"  {name}: {mod.__class__.__name__}")
        if any(k in name.lower() for k in ['router', 'gate']):
            print(f"  ROUTER: {name} ({mod.__class__.__name__})")
            if hasattr(mod, 'weight'):
                print(f"    weight: {mod.weight.shape}")
                print(f"    bias: {mod.bias.shape if mod.bias is not None else 'None'}")
                router_info = {"path": name, "shape": tuple(mod.weight.shape),
                              "bias": mod.bias is not None}

    # Inspect layer 0 in detail
    print("\n--- Layer 0 deep inspection ---")
    for name, mod in model.named_modules():
        if '.0.' in name and name.count('.') <= 4:
            n_params = sum(p.numel() for p in mod.parameters())
            if n_params > 0:
                print(f"  {name}: {mod.__class__.__name__} ({n_params:,})")

    # Expert parameter shapes
    print("\n--- Expert parameter shapes (first 10) ---")
    count = 0
    for name, p in model.named_parameters():
        if 'expert' in name.lower():
            if count < 10: print(f"  {name}: {p.shape}")
            count += 1
    if count > 10: print(f"  ... and {count - 10} more")
    print(f"  Total expert param tensors: {count}")

    # Quick generation test
    print("\n--- Generation test ---")
    prompt = "Calculate the DSCR for NOI of $500,000 and annual debt service of $400,000."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    print(f"  Prompt: {prompt}")
    print(f"  Output: {tokenizer.decode(out[0], skip_special_tokens=True)[:300]}")

    # Summary
    print(f"\n{'='*60}")
    print("  PATCHING GUIDE")
    print(f"{'='*60}")
    if router_info:
        n_exp = router_info["shape"][0]
        h_dim = router_info["shape"][1]
        print(f"  Router at: {router_info['path']}")
        print(f"  Shape: ({n_exp}, {h_dim}) → extend to ({n_exp + 6}, {h_dim})")
        print(f"  Has bias: {router_info['bias']}")
    else:
        print("  ⚠ Router not auto-detected. Look for Linear with output_dim=32")
    print(f"\n  GPU memory used: {torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="openai/gpt-oss-20b")
    main(p.parse_args().model_path)
