"""
GPU training entry-point for MAQA / AmbigQA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from loaders.maqa_preprocess import load_maqa_direct
from models.maqa_credal_model import (
    MAQADataset,
    CredalMAQA,
    MAQACredalLoss,
    MAQACredalTrainer,
    maqa_collate_fn,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="maqa")
    ap.add_argument("--hf_dataset", default="ttomov/ambigqa_star")
    ap.add_argument("--encoder", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--eval_dump_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--use_paired", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    eval_dump_dir = Path(args.eval_dump_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    eval_dump_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_maqa_direct(args.hf_dataset, split="train")
    val_raw = load_maqa_direct(args.hf_dataset, split="validation")
    test_raw = load_maqa_direct(args.hf_dataset, split="test")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = MAQADataset(train_raw, tokenizer, args.max_length, use_paired=args.use_paired)
    val_ds = MAQADataset(val_raw, tokenizer, args.max_length, use_paired=args.use_paired)
    test_ds = MAQADataset(test_raw, tokenizer, args.max_length, use_paired=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=maqa_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=maqa_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=maqa_collate_fn,
    )

    encoder = AutoModel.from_pretrained(args.encoder)
    model = CredalMAQA(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
        num_answers=10,
    )
    trainer = MAQACredalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
    )
    trainer.criterion = MAQACredalLoss(
        alpha_kl=1.0,
        alpha_reg=0.0,
        alpha_cal=1.0,
        alpha_cont=0.0,
    )

    history = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch()
        record = {"epoch": epoch, "split": "train", **train_metrics}
        history.append(record)
        print(f"[epoch {epoch}] train: {record}")

        if epoch % args.eval_every == 0:
            val_metrics = trainer.evaluate(val_loader)
            history.append({"epoch": epoch, "split": "val", **val_metrics})
            print(f"[epoch {epoch}] val:   {val_metrics}")
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), save_dir / "best.pt")

    test_metrics = trainer.evaluate(test_loader)
    print(f"[test] {test_metrics}")

    with (save_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    with (eval_dump_dir / "test_metrics.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)

    trainer.model.eval()
    sig_epi, sig_ale, ent_gt = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            b = {k: v.to(args.device) for k, v in batch.items()}
            if "amb" in b:
                b = b["amb"]
            params = trainer.model(b["input_ids"], b["attention_mask"])
            sig_epi.append(params.sigma_epi.cpu().numpy())
            sig_ale.append(params.sigma_ale.cpu().numpy())
            ent_gt.append(b["entropy"].cpu().numpy())

    np.savez(
        eval_dump_dir / "test_arrays.npz",
        sigma_epi=np.concatenate(sig_epi),
        sigma_ale=np.concatenate(sig_ale),
        entropy_gt=np.concatenate(ent_gt),
    )

    print(f"[done] saved to {save_dir} and {eval_dump_dir}")


if __name__ == "__main__":
    main()
