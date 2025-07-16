#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from torchinfo import summary

def inspect_model(model_id: str, cache_dir: str, device: str, max_seq_len: int):
    device = torch.device(device)
    print(f"\n=== Loading config for {model_id} ===")
    cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    
    # 1) Print config hyper-parameters
    print("\n>>> Model Config:")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    
    # 2) Instantiate the model (no weights) to inspect architecture only
    #    If the model uses a custom class you may instead load pretrained:
    #    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir,
    #                                                 device_map="auto",
    #                                                 torch_dtype="auto",
    #                                                 trust_remote_code=True)
    print("\n=== Instantiating model from config (random init) ===")
    model = AutoModelForCausalLM.from_config(cfg)
    model.to(device)
    
    # parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n>>> Total parameters:     {total_params:,}")
    print(f">>> Trainable parameters: {trainable_params:,}")
    
    # 3) Print the top-level architecture
    print("\n=== Model Architecture ===")
    print(model)
    
    # 4) Named modules
    print("\n=== Named Modules ===")
    for name, module in model.named_modules():
        print(name, module)
    
    # 5) Dive into a GPT-style block if it exists
    #    (many causal LM’s follow .transformer.h[i].attn)
    try:
        blocks = model.transformer.h
        print(f"\n>>> Detected {len(blocks)} transformer blocks")
        blk3 = blocks[3]
        attn = blk3.attn
        print(f"Block 3 attention heads: {attn.num_heads}")
        print(f"Block 3 head_dim:         {attn.head_dim}")
    except Exception:
        print("\n>>> Model does not expose transformer.h[].attn in this way.")
    
    # 6) torchinfo summary
    print("\n=== torchinfo Summary ===")
    dummy_input = torch.zeros((1, max_seq_len), dtype=torch.long, device=device)
    summary(model, input_data=dummy_input)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Inspect and dump a Hugging Face model's full architecture"
    )
    p.add_argument(
        "model_id",
        help="Hugging Face model identifier, e.g. microsoft/Phi-4-reasoning-plus",
    )
    p.add_argument(
        "--cache_dir",
        default=None,
        help="HF cache directory (defaults to ~/.cache/huggingface)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="torch device to move the model to (e.g. cpu or cuda)",
    )
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=10,
        help="sequence length for the torchinfo summary",
    )
    args = p.parse_args()
    inspect_model(args.model_id, args.cache_dir, args.device, args.max_seq_len)