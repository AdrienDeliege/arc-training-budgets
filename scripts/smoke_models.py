#!/usr/bin/env python3
"""Reduced CPU forward/backward check; does not reproduce paper scores."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=["varc", "trm"])
    a = p.parse_args()
    sys.path.insert(0, str(ROOT / "vendor" / a.model))
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(2)
    if a.model == "varc":
        from src.ARC_ViT import ARCViT
        model = ARCViT(num_tasks=2, image_size=8, patch_size=2, num_colors=12, embed_dim=32, depth=1, num_heads=4, mlp_dim=64)
        logits = model(torch.randint(0, 10, (2, 8, 8)), torch.tensor([0, 1]))
    else:
        from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
        model = TinyRecursiveReasoningModel_ACTV1(dict(batch_size=2, seq_len=16, puzzle_emb_ndim=32, num_puzzle_identifiers=2, vocab_size=12, H_cycles=1, L_cycles=1, H_layers=0, L_layers=1, hidden_size=32, expansion=2, num_heads=4, pos_encodings="rope", halt_max_steps=2, halt_exploration_prob=0., forward_dtype="float32"))
        batch = {"inputs": torch.randint(0, 12, (2, 16)), "labels": torch.randint(0, 12, (2, 16)), "puzzle_identifiers": torch.tensor([0, 1])}
        _, outputs = model(model.initial_carry(batch), batch)
        logits = outputs["logits"]
    loss = logits.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    print(f"PASS: {a.model} CPU forward/backward, output={tuple(logits.shape)}, loss={loss.item():.6f}, torch={torch.__version__}")


if __name__ == "__main__":
    main()
