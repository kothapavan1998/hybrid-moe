#!/bin/bash
set -e
echo "============================================"
echo "  Hybrid MoE — Environment Setup"
echo "============================================"

pip install torch transformers accelerate safetensors \
    pandas numpy matplotlib seaborn tqdm scikit-learn pyyaml \
    --quiet --break-system-packages 2>&1 | tail -3

echo ""
echo "Downloading gpt-oss-20b..."
pip install huggingface_hub --quiet --break-system-packages
huggingface-cli download openai/gpt-oss-20b \
    --include "original/*" \
    --local-dir ./models/gpt-oss-20b/ \
    --quiet || echo "Download failed — check HF access"

echo ""
echo "Verifying..."
python -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_mem/1e9:.0f}GB)')
"

echo ""
echo "Running unit tests..."
python -m src.model.cre_experts
python -m src.evaluation.metrics

echo ""
echo "============================================"
echo "  READY! Run the full pipeline:"
echo ""
echo "  python scripts/run_pipeline.py \\"
echo "      --model_path ./models/gpt-oss-20b"
echo ""
echo "  Or step by step:"
echo "  1. python scripts/explore_model.py"
echo "  2. python scripts/test_extraction.py"
echo "  3. python scripts/run_pipeline.py"
echo "============================================"
