"""Concept-level read-out of a trained PyC model (pyc-cbm, pyc-cem, pyc-hyper) with PyC tools.

    python semantics.py runs/cebab-pyc-cem-lr0.001-p8-s0
    python semantics.py runs/goemotions-pyc-hyper-lr0.001-p8-s0 --split val

Writes runs/<run>/semantics.json and prints a summary.  Every concept is addressed by name
(pyc Annotations / AnnotatedTensor), and every intervention goes through a pyc
InterventionModule wrapped around the model's concept encoder:

* concepts   per concept: AUC, base rate, mean predicted probability.
* cace       causal effect of each concept on the answer (pyc cace_score):
             mean P(answer) under do(concept = 1) minus under do(concept = 0).
* weights    pyc-cbm / pyc-hyper: (mean) concept -> answer weights of the linear predictor.
* ncc        pyc number_of_contributing_concepts: how many concepts are needed to explain
             95% of each decision (pyc-cbm, pyc-hyper; per text for pyc-hyper).
* interventions  task accuracy when the n least certain concepts (pyc
             UncertaintyInterventionPolicy) or n random ones (pyc RandomPolicy) are set to
             their human labels (pyc GroundTruthIntervention), for several n.
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_concepts.nn import (DoIntervention, GroundTruthIntervention, InterventionModule,
                               RandomPolicy, UncertaintyInterventionPolicy, UniformPolicy)
from torch_concepts.nn import functional as pycF

from concept_datasets import load_meta
from concept_models.features import get_features
from concept_models.models import load_model

BIG = 30.0  # logit used for do(C = 1) / do(C = 0): sigmoid(+-30) is 0/1 in float32


@contextmanager
def wrapped_encoder(model, **intervention):
    """Temporarily replace model.concept_encoder by a pyc InterventionModule around it."""
    original = model.concept_encoder
    model.concept_encoder = InterventionModule(original_module=original, **intervention)
    try:
        yield
    finally:
        model.concept_encoder = original


def answer_probs(model, X):
    return torch.softmax(model(X)["logits"], 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run", help="run directory of a pyc-* model")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=0, help="for the random intervention policy")
    args = p.parse_args()

    run = Path(args.run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(run / "model.pt", device)
    kind = model.config["kind"]
    if not kind.startswith("pyc-"):
        raise SystemExit(f"{run} is a {kind!r} model; semantics.py reads pyc-* models")
    meta = load_meta(model.config["dataset"])
    concepts, classes = meta["concept_names"], meta["class_names"]
    feats = get_features(model.config["dataset"], args.split, model.config["encoder"])
    X, y, C = feats["X"].to(device), feats["y"].to(device), feats["C"].to(device).float()
    k = len(concepts)
    report = {"run": run.name, "kind": kind, "split": args.split, "concept_names": concepts,
              "class_names": classes}

    with torch.no_grad():
        out = model(X)
        probs = model.annotate(out["concept_probs"])  # AnnotatedTensor: probs["food_pos"] etc.
        acc = float((out["logits"].argmax(1) == y).float().mean())
        report["task_acc"] = acc

        # 1. Concepts
        Cn = C.cpu().numpy()
        rows = {}
        for j, name in enumerate(concepts):
            pj = probs[name].squeeze(1).cpu().numpy()
            rows[name] = {"base_rate": float(Cn[:, j].mean()), "mean_prob": float(pj.mean()),
                          "auc": float(roc_auc_score(Cn[:, j], pj)) if 0 < Cn[:, j].sum() < len(Cn)
                          else None}
        report["concepts"] = rows

        # 2. CaCE: do(c_j = 1) vs do(c_j = 0), through an InterventionModule on concept j only
        cace = {}
        for j, name in enumerate(concepts):
            preds = []
            for value in (-BIG, BIG):
                with wrapped_encoder(model, intervention_strategy=DoIntervention(constants=value),
                                     intervention_policy=UniformPolicy(),
                                     out_concepts_to_intervene_on=[j]):
                    preds.append(answer_probs(model, X))
            cace[name] = dict(zip(classes, pycF.cace_score(*preds).tolist()))
        report["cace"] = cace

        # 3. Predictor weights and 4. NCC
        if kind == "pyc-cbm":
            W = model.predictor.predictor.weight  # (classes, k)
            report["weights"] = {c: dict(zip(concepts, W[i].tolist())) for i, c in enumerate(classes)}
            report["ncc"] = {"all_classes": pycF.number_of_contributing_concepts(W, out["concept_probs"]),
                             "predicted_class": pycF.number_of_contributing_concepts(
                                 W, out["concept_probs"], predicted_class_only=True)}
        elif kind == "pyc-hyper":
            Wx = out["concept_weights"]  # (n, classes, k), one weight matrix per text
            report["weights"] = {c: dict(zip(concepts, Wx[:, i].mean(0).tolist()))
                                 for i, c in enumerate(classes)}
            report["weights_std_over_texts"] = {c: dict(zip(concepts, Wx[:, i].std(0).tolist()))
                                                for i, c in enumerate(classes)}
            ncc = [(pycF.number_of_contributing_concepts(Wx[i], out["concept_probs"][i:i + 1]),
                    pycF.number_of_contributing_concepts(Wx[i], out["concept_probs"][i:i + 1],
                                                         predicted_class_only=True))
                   for i in range(len(X))]
            report["ncc"] = {"all_classes": float(np.mean([a for a, _ in ncc])),
                             "predicted_class": float(np.mean([b for _, b in ncc]))}
        else:  # pyc-cem: the predictor reads embeddings, not concept scalars
            report["ncc"] = None

        # 5. Interventions: replace the n lowest-scoring concepts (policy) by the human labels
        truth = torch.where(C > 0.5, BIG, -BIG)
        budgets = sorted({0, 1, round(k / 4), round(k / 2), round(3 * k / 4), k})
        curves = {}
        for pname, make_policy in [("uncertainty", lambda: UncertaintyInterventionPolicy(0.0)),
                                   ("random", lambda: RandomPolicy())]:
            torch.manual_seed(args.seed)
            curve = {}
            for n in budgets:
                if n == 0:
                    curve[0] = acc
                    continue
                with wrapped_encoder(model, intervention_strategy=GroundTruthIntervention(truth),
                                     intervention_policy=make_policy(),
                                     quantile=(n - 1) / max(k - 1, 1)):
                    pred = model(X)["logits"].argmax(1)
                curve[n] = float((pred == y).float().mean())
            curves[pname] = curve
        report["interventions"] = {"n_concepts_replaced": budgets, "task_acc": curves}

    (run / f"semantics-{args.split}.json").write_text(json.dumps(report, indent=2))

    # Summary
    print(f"\nModel: {run.name}  ({kind})  split: {args.split}  task acc {acc:.3f}")
    print(f"\n{'concept':<24}{'AUC':>6}  " + "  ".join(f"CaCE {c[:10]:>10}" for c in classes))
    for name in concepts:
        auc = rows[name]["auc"]
        print(f"{name:<24}{auc if auc is not None else float('nan'):6.3f}  "
              + "  ".join(f"{cace[name][c]:+15.3f}" for c in classes))
    if report["ncc"]:
        print(f"\nNCC (95%): {report['ncc']['all_classes']:.2f} concepts per class, "
              f"{report['ncc']['predicted_class']:.2f} for the predicted class (out of {k})")
    print("\nTask accuracy after setting n concepts to their human labels:")
    print(f"  {'n':>4}  " + "  ".join(f"{pn:>11}" for pn in curves))
    for n in budgets:
        print(f"  {n:>4}  " + "  ".join(f"{curves[pn][n]:11.3f}" for pn in curves))
    print(f"\nSaved {run}/semantics-{args.split}.json")


if __name__ == "__main__":
    main()
