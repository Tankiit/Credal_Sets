"""
Train a credal QA model on MAQA/AmbigQA-style datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from load_ambigqa_dataset import load_combined_maqa_ambigqa
from maqa_credal_model import CredalMAQA, MAQADataset, MAQACredalTrainer, maqa_collate_fn


DEFAULTS = {
    "ambigqa": {"batch_size": 16, "max_length": 256, "epochs": 10, "lr": 1e-4},
    "maqa": {"batch_size": 16, "max_length": 256, "epochs": 10, "lr": 1e-4},
}


def _build_loaders(raw_splits, tokenizer, batch_size: int, max_length: int, num_workers: int):
    from torch.utils.data import DataLoader

    train_dataset = MAQADataset(raw_splits["train"], tokenizer, max_length=max_length, use_paired=True)
    val_key = "validation" if "validation" in raw_splits else "val" if "val" in raw_splits else "train"
    test_key = "test" if "test" in raw_splits else val_key
    val_dataset = MAQADataset(raw_splits[val_key], tokenizer, max_length=max_length, use_paired=False)
    test_dataset = MAQADataset(raw_splits[test_key], tokenizer, max_length=max_length, use_paired=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=maqa_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=maqa_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=maqa_collate_fn)
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train a credal QA model.")
    parser.add_argument("--dataset", choices=["ambigqa", "maqa"], default="ambigqa")
    parser.add_argument("--hf_dataset", default="ttomov/ambigqa_star")
    parser.add_argument("--encoder", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--eval_dump_dir", type=Path, default=None)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    defaults = DEFAULTS[args.dataset]
    batch_size = args.batch_size or defaults["batch_size"]
    max_length = args.max_length or defaults["max_length"]
    epochs = args.epochs or defaults["epochs"]
    lr = args.lr or defaults["lr"]

    torch.manual_seed(args.seed)
    raw_splits = load_combined_maqa_ambigqa(dataset_name=args.hf_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader, test_loader = _build_loaders(raw_splits, tokenizer, batch_size, max_length, args.num_workers)

    encoder = AutoModel.from_pretrained(args.encoder)
    hidden_size = encoder.config.hidden_size
    max_answers = max(
        max((len(item["p_star"]) for item in raw_splits["train"]), default=0),
        max((len(item["p_star"]) for item in raw_splits.get("validation", raw_splits["train"])), default=0),
        max((len(item["p_star"]) for item in raw_splits.get("test", raw_splits["train"])), default=0),
    )
    model = CredalMAQA(encoder=encoder, hidden_size=hidden_size, num_answers=max(1, max_answers))
    save_dir = args.save_dir or Path(f"checkpoints/{args.dataset}_credal")
    trainer = MAQACredalTrainer(model, train_loader, val_loader, device=args.device, learning_rate=lr)
    trainer.fit(num_epochs=epochs, save_dir=save_dir, eval_every=args.eval_every)

    ckpt = torch.load(save_dir / "best_model.pt", map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = trainer.evaluate(test_loader)
    dump_dir = args.eval_dump_dir or (save_dir / "eval_dumps")
    trainer.dump_eval_arrays(test_loader, dump_dir)
    with (save_dir / "test_metrics.json").open("w") as f:
        json.dump(test_metrics, f, indent=2, default=float)
    print(test_metrics)


if __name__ == "__main__":
    main()
