"""Build the twin of a trained model and check what is preserved and what moved.

    python twin.py runs/cebab-cem-s0
    python twin.py runs/cebab-cbm-r16-s0 --rotate 2 --mix 1 --shift 1 --seed 3

The twin has the same architecture; its weights are the original's with the concept layer
moved along readout-invisible directions and the head compensated (see
concept_models/reparam.py). It is saved to runs/<run>/twins/<tag>.pt with a report.json.

Reported on --split:
  must be ~0 : max |d task logits|, max |d concept probs|
  moved      : relative change of z, mean cosine(z, z_twin)
  first diagnostic: task logits under concept interventions (random 50% of concepts set
             to their true value).  Whether this notices depends on the model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from concept_models.features import get_features
from concept_models.models import load_model, save_model
from concept_models.reparam import make_twin
from concept_models.training import predict

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("run", type=Path, help="run directory containing model.pt")
p.add_argument("--rotate", type=float, default=1.0, help="rotation strength inside the null space")
p.add_argument("--mix", type=float, help="how much visible coords leak into invisible ones "
               "(default 1, or 0 when the head is not linear in z: CBM with head_input=probs)")
p.add_argument("--shift", type=float, default=0.0, help="constant shift inside the null space")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--split", default="test")
args = p.parse_args()

model = load_model(args.run / "model.pt")
if args.mix is None:
    args.mix = 1.0 if model.head_linear_in_z else 0.0
twin, info = make_twin(model, args.rotate, args.mix, args.shift, args.seed)
feats = get_features(model.config["dataset"], args.split, model.config["encoder"])

a, b = predict(model, feats["X"]), predict(twin, feats["X"])
sig = lambda v: 1 / (1 + np.exp(-v))
dz = a["z"] - b["z"]
cos = (a["z"] * b["z"]).sum(1) / (np.linalg.norm(a["z"], axis=1) * np.linalg.norm(b["z"], axis=1))
report = dict(info, split=args.split,
              max_abs_d_logits=float(np.abs(a["logits"] - b["logits"]).max()),
              max_abs_d_concept_probs=float(np.abs(sig(a["concept_logits"]) - sig(b["concept_logits"])).max()),
              rel_change_z=float(np.linalg.norm(dz) / np.linalg.norm(a["z"])),
              mean_cos_z=float(cos.mean()))
if "C" in feats:
    mask = torch.rand(feats["C"].shape, generator=torch.Generator().manual_seed(args.seed)) < 0.5
    ai, bi = predict(model, feats["X"], feats["C"], mask), predict(twin, feats["X"], feats["C"], mask)
    y = feats["y"].numpy()
    report.update(intervention_max_abs_d_logits=float(np.abs(ai["logits"] - bi["logits"]).max()),
                  intervention_pred_agreement=float((ai["logits"].argmax(1) == bi["logits"].argmax(1)).mean()),
                  intervention_acc_original=float((ai["logits"].argmax(1) == y).mean()),
                  intervention_acc_twin=float((bi["logits"].argmax(1) == y).mean()))

tag = f"rot{args.rotate}-mix{args.mix}-shift{args.shift}-seed{args.seed}"
out = args.run / "twins"
out.mkdir(exist_ok=True)
save_model(twin, out / f"{tag}.pt")
(out / f"{tag}.report.json").write_text(json.dumps(report, indent=2))
print(f"Twin of {args.run} ({tag}) -> {out}/{tag}.pt")
for k, v in report.items():
    print(f"  {k:<32} {v:.3g}" if isinstance(v, float) else f"  {k:<32} {v}")
