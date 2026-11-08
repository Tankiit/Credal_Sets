"""Train the concept-only reference head: nn.Linear fitted on ground-truth
concepts (g) from the reference-train split, predicting task labels (y)
directly -- no encoder, no readout, no z involved.

This is the baseline audit_consequence-style comparisons will eventually
be checked against: "how well could you predict y if you had perfect
concept scores?" Not yet consumed by any audit function -- this script
only trains and saves the head.
"""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from experiments.arguments import parse_arguments
from concept_audit.data import load_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON config; CLI flags override it")
    parser.add_argument("--task", choices=["model", "reference", "certification"], required=True,
                         help="Must be 'reference' -- this script trains the reference head")
    parser.add_argument("--reference-features-dir", type=Path,
                         help="reference-train cache (from split_cache_4way) with concepts.npy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda / cpu; auto-detected if omitted")
    parser.add_argument("--out", type=Path, default=Path("results/real/reference_head.pt"))
    args = parse_arguments(parser, argv)
    if args.task != "reference":
        parser.error("reference.py trains the reference head; pass --task reference")
    if args.reference_features_dir is None:
        parser.error("--reference-features-dir is required")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    _, g, y = load_cache(args.reference_features_dir)
    g, y = g.to(device), y.to(device)
    k, classes = g.shape[1], int(y.max()) + 1

    head = nn.Linear(k, classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.02)

    for epoch in range(args.epochs):
        for rows in torch.arange(len(g)).split(128):
            logits = head(g[rows])
            loss = F.cross_entropy(logits, y[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        accuracy = (head(g).argmax(1) == y).float().mean().item()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": {k_: v.cpu() for k_, v in head.state_dict().items()},
        "num_concepts": k,
        "num_classes": classes,
        "seed": args.seed,
        "reference_cache": str(args.reference_features_dir),
        "train_accuracy": accuracy,
    }, args.out)
    print(f"Saved {args.out}; train accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()