"""Re-run the ICML 2026 text experiments on Modal for additional seeds."""

from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("icml-2026-credal-multiseed")
results_volume = modal.Volume.from_name("icml-2026-credal-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "datasets==3.5.0",
        "huggingface_hub==0.30.2",
        "numpy==2.1.3",
        "scipy==1.15.2",
        "scikit-learn==1.6.1",
        "tqdm==4.67.1",
        "pandas==2.2.3",
        "sentencepiece==0.2.0",
        "accelerate==1.6.0",
    )
    .run_commands(
        "git clone --depth 1 --branch ICML_2026 "
        "https://github.com/Tankiit/Credal_Sets.git /root/icml_2026"
        " && git -C /root/icml_2026 checkout "
        "532fd05f67238f4459c371e2dbb14bca3bdecd59"
    )
    # The ICML branch intentionally omitted these data artifacts from Git, but
    # its HateXplain loader requires the label encoder and official split file.
    .add_local_dir(
        "data/hatexplain",
        remote_path="/root/icml_2026/data/hatexplain",
        copy=True,
    )
    # MAQA support files exist in the local ICML_2026 worktree but were omitted
    # from the remote branch used to construct the base image.
    .add_local_file(
        "/Users/tanmoy/research/Credal_Sets/Variational_CBM/load_maqa_real.py",
        remote_path="/root/icml_2026/load_maqa_real.py",
        copy=True,
    )
    .add_local_file(
        "/Users/tanmoy/research/Credal_Sets/Variational_CBM/maqa_credal_model.py",
        remote_path="/root/icml_2026/maqa_credal_model.py",
        copy=True,
    )
    .add_local_file(
        "/Users/tanmoy/research/Credal_Sets/Variational_CBM/maqa_credal_loss_v3.py",
        remote_path="/root/icml_2026/maqa_credal_loss_v3.py",
        copy=True,
    )
    .add_local_file(
        "/Users/tanmoy/research/Credal_Sets/Variational_CBM/maqa_credal_loss_v4.py",
        remote_path="/root/icml_2026/maqa_credal_loss_v4.py",
        copy=True,
    )
    .add_local_file(
        "/Users/tanmoy/research/Credal_Sets/Variational_CBM/maqa_credal_loss_v5.py",
        remote_path="/root/icml_2026/maqa_credal_loss_v5.py",
        copy=True,
    )
    .add_local_file(
        "/Users/tanmoy/research/Credal_Sets/Variational_CBM/maqa_credal_loss_v6.py",
        remote_path="/root/icml_2026/maqa_credal_loss_v6.py",
        copy=True,
    )
    .add_local_file(
        "/Users/tanmoy/research/Credal_Sets/Variational_CBM/maqa_credal_loss_v7b_fixed.py",
        remote_path="/root/icml_2026/maqa_credal_loss_v7b_fixed.py",
        copy=True,
    )
    .add_local_file(
        "/Users/tanmoy/research/Credal_Sets/Variational_CBM/v7b_complete_integration.py",
        remote_path="/root/icml_2026/v7b_complete_integration.py",
        copy=True,
    )
    .add_local_file(
        "/Users/tanmoy/research/Credal_Sets/Variational_CBM/maqa_fixed_config.py",
        remote_path="/root/icml_2026/maqa_fixed_config.py",
        copy=True,
    )
)


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={"/results": results_volume},
)
def train_one(
    dataset: str,
    seed: int,
    epochs: int,
    encoder: str = "distilbert",
    cebab_three_class: bool = False,
    cebab_binary: bool = False,
    run_tag: str = "",
    phased_schedule: bool = False,
    worker_review_entropy: bool = False,
    binary_concept_entropy: bool = False,
    mask_unannotated: bool = False,
    aleatoric_weight: float = -1.0,
    decorr_weight: float = 0.0,
    qa_subset: str = "",
) -> dict:
    import os
    import random
    import sys

    import numpy as np
    import torch

    os.chdir("/root/icml_2026")
    sys.path.insert(0, "/root/icml_2026")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import main_train_hybrid_multi_dataset as experiment

    if cebab_three_class and cebab_binary:
        raise ValueError("Choose only one CEBaB label mode")
    encoder_tag = encoder.replace("/", "_").replace("-", "_")
    if dataset == "cebab" and cebab_binary:
        run_id = f"cebab_2class_{encoder_tag}_seed{seed}"
    elif dataset == "cebab" and cebab_three_class:
        run_id = f"cebab_3class_{encoder_tag}_seed{seed}"
    else:
        run_id = f"{qa_subset or dataset}_{encoder_tag}_seed{seed}"
    if run_tag:
        run_id = f"{run_id}_{run_tag}"
    if phased_schedule:
        run_id = f"{run_id}_phased"
    output_dir = Path("/results") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment.DATASET_CONFIGS[dataset]["save_dir"] = str(output_dir)

    if dataset == "cebab" and (cebab_three_class or cebab_binary):
        import load_cebab_direct as cebab
        from datasets import load_dataset

        # Use the established train_inclusive/validation/test split convention
        # with a reduced binary or ternary review-sentiment target.
        def load_cebab_paper(split_ids_path=None, include_edits=False, seed=42):
            del split_ids_path, include_edits, seed
            ds = load_dataset("CEBaB/CEBaB")
            return {
                "train": list(ds["train_inclusive"]),
                "validation": list(ds["validation"]),
                "test": list(ds["test"]),
            }

        original_process = cebab.process_cebab_raw

        def process_cebab_reduced(raw_data):
            # Reduced-label loaders exclude reviews without a majority rating.
            raw_data = [
                item for item in raw_data
                if str(item.get("review_majority", "")).strip() in {"1", "2", "3", "4", "5"}
            ]
            if mask_unannotated:
                # Keep only rows the original processor keeps, so raw and
                # processed items stay aligned one-to-one.
                raw_data = [r for r in raw_data if (r.get("description") or "").strip()]
            processed = original_process(raw_data)
            if mask_unannotated:
                assert len(processed) == len(raw_data)
                aspects = ("food", "service", "ambiance", "noise")
                for item, raw in zip(processed, raw_data):
                    assert item["text"] == raw["description"]
                    for j, aspect in enumerate(aspects):
                        if raw.get(f"{aspect}_aspect_majority") not in ("Negative", "Positive", "unknown"):
                            # No annotators ('') or no majority: no point label
                            # and no entropy. Label 1 removes the aspect from the
                            # concept, error, and entropy losses; H = -1 marks it
                            # for the unknown-AU term patched below.
                            item["concepts"][j] = 1
                            item["_concept_entropies"][j] = -1.0
            for item in processed:
                rating = item["label"] + 1
                if cebab_binary:
                    item["label"] = 0 if rating <= 2 else 1
                else:
                    item["label"] = 0 if rating <= 2 else (1 if rating == 3 else 2)
            return processed

        cebab.load_cebab = load_cebab_paper
        cebab.process_cebab_raw = process_cebab_reduced
        if worker_review_entropy:
            original_getitem = cebab.CEBaBDataset.__getitem__

            def getitem_with_worker_review_entropy(self, idx):
                result = original_getitem(self, idx)
                # The paper's H is worker uncertainty over the review/task
                # label distribution. Repeat the scalar target across the four
                # AU outputs so their mean is a task-level AU estimate.
                rating_entropy = result["_rating_entropy"].float()
                result["annotator_entropy"] = rating_entropy.repeat(4)
                result["entropy_weights"] = torch.ones(4, dtype=torch.float)
                return result

            cebab.CEBaBDataset.__getitem__ = getitem_with_worker_review_entropy
        if binary_concept_entropy:
            original_getitem_binary = cebab.CEBaBDataset.__getitem__

            def getitem_with_binary_concept_entropy(self, idx):
                result = original_getitem_binary(self, idx)
                distributions = self.data[idx]["_concept_distributions"]
                targets = []
                weights = []
                for negative, _unknown, positive in distributions:
                    binary_total = float(negative + positive)
                    if binary_total <= 0:
                        targets.append(0.0)
                        weights.append(0.0)
                        continue
                    p_positive = float(positive / binary_total)
                    entropy = -(
                        p_positive * np.log(p_positive + 1e-10)
                        + (1.0 - p_positive) * np.log(1.0 - p_positive + 1e-10)
                    ) / np.log(2.0)
                    targets.append(float(entropy))
                    weights.append(1.0)
                result["annotator_entropy"] = torch.tensor(targets, dtype=torch.float)
                # Persist validity for evaluation/provenance. The pinned trainer
                # does not consume weights, so all-unknown distributions receive
                # a zero target rather than contributing an invented entropy.
                result["entropy_weights"] = torch.tensor(weights, dtype=torch.float)
                return result

            cebab.CEBaBDataset.__getitem__ = getitem_with_binary_concept_entropy
        experiment.DATASET_CONFIGS["cebab"]["num_classes"] = 2 if cebab_binary else 3
        experiment.DATASET_CONFIGS["cebab"]["loader_kwargs"]["include_edits"] = False

    # The branch's CLI does not persist final test metrics. Mark the evaluation
    # following load_best_model() as test evaluation and save its complete dict.
    import torch.nn.functional as F
    import VCBM

    model_cls = VCBM.HybridCredalCBM
    if aleatoric_weight >= 0 or mask_unannotated or decorr_weight > 0:
        original_init = model_cls.__init__
        original_losses = model_cls._compute_losses

        def init_with_weights(self, config, *args, **kwargs):
            original_init(self, config, *args, **kwargs)
            if aleatoric_weight >= 0:
                # aleatoric_weight=0 also disables the unknown-AU term (0.1 x weight).
                self.config.aleatoric_weight = aleatoric_weight

        def losses_with_patches(self, result, labels, concept_labels, annotator_entropy, batch=None):
            losses = original_losses(self, result, labels, concept_labels, annotator_entropy, batch)
            if mask_unannotated and "aleatoric_unknown" in losses and annotator_entropy is not None:
                # Only aspects whose annotators chose "unknown" are pushed towards
                # high AU; unannotated aspects (H = -1) are left unsupervised.
                keep = (concept_labels == 1) & (annotator_entropy >= 0)
                new = (F.mse_loss(result["aleatoric"][keep], torch.ones_like(result["aleatoric"][keep]))
                       if keep.any() else result["aleatoric"].sum() * 0.0)
                w = 0.1 * self.config.aleatoric_weight
                losses["loss"] = losses["loss"] + w * (new - losses["aleatoric_unknown"])
                losses["aleatoric_unknown"] = new
            if decorr_weight > 0:
                # Squared Pearson correlation of the per-example scores, as in the
                # MAQA v7b loss; this couples the two heads' gradients.
                eu = result["epistemic"].mean(dim=-1)
                au = result["aleatoric"].mean(dim=-1)
                eu_c, au_c = eu - eu.mean(), au - au.mean()
                corr = (eu_c * au_c).mean() / (eu.std() * au.std() + 1e-8)
                losses["decorr"] = corr ** 2
                losses["loss"] = losses["loss"] + decorr_weight * losses["decorr"]
            return losses

        model_cls.__init__ = init_with_weights
        model_cls._compute_losses = losses_with_patches

    if dataset == "maqa" and qa_subset:
        original_qa_loader = experiment.load_combined_maqa_ambigqa

        def qa_subset_loader(*args, **kwargs):
            # Same 80/10/10 split of the combined pool, restricted to one source,
            # so the subset's test questions are a subset of the MAQA* test set.
            splits = original_qa_loader(*args, **kwargs)
            return {k: [x for x in v if x.get("dataset") == qa_subset] for k, v in splits.items()}

        experiment.load_combined_maqa_ambigqa = qa_subset_loader

    trainer_cls = experiment.HybridCredalCBMTrainer
    original_load = trainer_cls.load_best_model
    original_evaluate = trainer_cls.evaluate
    original_train_epoch = trainer_cls.train_epoch

    def train_epoch_phased(self, *args, **kwargs):
        if self.current_epoch <= 20:
            phase = "task_concept_warmup"
            weights = (2.0, 0.0, 0.0, 0.01, 0.001)
        elif self.current_epoch <= 40:
            phase = "uncertainty_ramp"
            weights = (1.0, 1.0, 2.0, 0.01, 0.001)
        else:
            phase = "joint_finetune"
            weights = (2.0, 1.0, 2.0, 0.01, 0.001)

        (
            self.config.concept_weight,
            self.config.error_supervision_weight,
            self.config.aleatoric_weight,
            self.config.kl_weight,
            self.config.orth_weight,
        ) = weights
        print(
            f"[phased] epoch={self.current_epoch} phase={phase} "
            f"concept={weights[0]} epi={weights[1]} ale={weights[2]}"
        )
        return original_train_epoch(self, *args, **kwargs)

    def load_and_mark(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        self._icml_save_next_eval = True
        return result

    def evaluate_and_save(self, *args, **kwargs):
        metrics = original_evaluate(self, *args, **kwargs)
        if getattr(self, "_icml_save_next_eval", False):
            payload = metrics.to_dict()
            payload["dataset"] = dataset
            payload["seed"] = seed
            payload["epochs"] = epochs
            payload["cebab_three_class"] = cebab_three_class
            payload["cebab_binary"] = cebab_binary
            payload["phased_schedule"] = phased_schedule
            payload["worker_review_entropy"] = worker_review_entropy
            payload["binary_concept_entropy"] = binary_concept_entropy
            (output_dir / "test_metrics.json").write_text(
                json.dumps(payload, indent=2, default=float)
            )
            self._icml_save_next_eval = False
        return metrics

    trainer_cls.load_best_model = load_and_mark
    trainer_cls.evaluate = evaluate_and_save
    if phased_schedule:
        trainer_cls.train_epoch = train_epoch_phased

    sys.argv = [
        "main_train_hybrid_multi_dataset.py",
        "--dataset",
        dataset,
        "--encoder",
        encoder,
        "--num_epochs",
        str(epochs),
    ]
    experiment.main()

    # MAQA writes its final test result as part of final_results.json. Mirror
    # that payload to test_metrics.json so the replication matrix is uniform.
    if dataset == "maqa":
        final_results_path = output_dir / "final_results.json"
        final_results = json.loads(final_results_path.read_text())
        payload = final_results["test_metrics"]
        payload.update({
            "dataset": "maqa_ambigqa",
            "seed": seed,
            "epochs": epochs,
            "encoder": "distilbert-base-uncased",
            "loss_version": "v7b",
        })
        (output_dir / "test_metrics.json").write_text(
            json.dumps(payload, indent=2, default=float)
        )

    metadata = {
        "run_id": run_id,
        "dataset": dataset,
        "encoder": encoder,
        "seed": seed,
        "epochs": epochs,
        "cebab_three_class": cebab_three_class,
        "cebab_binary": cebab_binary,
        "run_tag": run_tag,
        "phased_schedule": phased_schedule,
        "worker_review_entropy": worker_review_entropy,
        "binary_concept_entropy": binary_concept_entropy,
        "mask_unannotated": mask_unannotated,
        "aleatoric_weight": aleatoric_weight if aleatoric_weight >= 0 else "default (2.0)",
        "decorr_weight": decorr_weight,
        "qa_subset": qa_subset or None,
        "phase_epochs": {
            "task_concept_warmup": [1, 20],
            "uncertainty_ramp": [21, 40],
            "joint_finetune": [41, epochs],
        } if phased_schedule else None,
        "cebab_label_mapping": (
            "1-2=negative,3-5=positive"
            if cebab_binary
            else "1-2=negative,3=neutral,4-5=positive"
            if cebab_three_class
            else None
        ),
        "cebab_splits": "train_inclusive/validation/test"
        if cebab_three_class or cebab_binary
        else None,
        "git_branch": "ICML_2026",
        "git_commit": "532fd05f67238f4459c371e2dbb14bca3bdecd59",
        "git_repo": "https://github.com/Tankiit/Credal_Sets.git",
        "torch_version": torch.__version__,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    results_volume.commit()
    return metadata


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={"/results": results_volume},
)
def evaluate_cebab_concepts(run_id: str, encoder: str, seed: int) -> dict:
    """Evaluate task and binary known-concept accuracy from an existing checkpoint."""
    import os
    import sys
    import importlib

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.chdir("/root/icml_2026")
    sys.path.insert(0, "/root/icml_2026")

    import load_cebab_direct as cebab
    import main_train_hybrid_multi_dataset as experiment
    from VCBM import HybridCredalCBM, HybridCredalConfig

    # Modal workers may be reused for multiple evaluations. Restore the module's
    # original processing functions before installing the temporary ternary map.
    cebab = importlib.reload(cebab)

    def load_cebab_paper(split_ids_path=None, include_edits=False, seed=42):
        del split_ids_path, include_edits, seed
        ds = load_dataset("CEBaB/CEBaB")
        return {
            "train": list(ds["train_inclusive"]),
            "validation": list(ds["validation"]),
            "test": list(ds["test"]),
        }

    original_process = cebab.process_cebab_raw

    def process_cebab_ternary(raw_data):
        raw_data = [
            item for item in raw_data
            if str(item.get("review_majority", "")).strip() in {"1", "2", "3", "4", "5"}
        ]
        processed = original_process(raw_data)
        for item in processed:
            rating = item["label"] + 1
            item["label"] = 0 if rating <= 2 else (1 if rating == 3 else 2)
        return processed

    cebab.load_cebab = load_cebab_paper
    cebab.process_cebab_raw = process_cebab_ternary

    tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)
    _, _, test_loader, _, _ = cebab.get_cebab_dataloaders(
        tokenizer=tokenizer,
        batch_size=8,
        max_length=256,
        num_workers=0,
        include_edits=False,
    )

    config = HybridCredalConfig(
        encoder_name=encoder,
        freeze_encoder=True,
        num_concepts=4,
        concept_names=["food", "service", "ambiance", "noise"],
        num_classes=3,
        prior_sigma=0.5,
        error_scale=2.0,
    )
    model = HybridCredalCBM(config).cuda()
    checkpoint_path = Path("/results") / run_id / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    concept_names = ["food", "service", "ambiance", "noise"]
    task_correct = task_total = 0
    correct = np.zeros(4, dtype=np.int64)
    total = np.zeros(4, dtype=np.int64)
    class_correct = np.zeros((4, 2), dtype=np.int64)
    class_total = np.zeros((4, 2), dtype=np.int64)
    unknown_total = np.zeros(4, dtype=np.int64)
    all_aleatoric = []
    all_entropies = []
    all_rating_entropies = []
    all_concept_labels = []

    with torch.inference_mode():
        for batch in test_loader:
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)
            concept_labels = batch["concept_labels"].cuda(non_blocking=True)
            all_concept_labels.append(concept_labels.cpu().numpy())
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_aleatoric.append(outputs["aleatoric"].cpu().numpy())
            if "annotator_entropy" in batch:
                all_entropies.append(batch["annotator_entropy"].cpu().numpy())
            if "_rating_entropy" in batch:
                all_rating_entropies.append(batch["_rating_entropy"].cpu().numpy())

            task_correct += int((outputs["predictions"] == labels).sum().item())
            task_total += int(labels.numel())

            concept_pred = (outputs["concept_probs"] >= 0.5).long()
            known = concept_labels != 1
            target = (concept_labels == 2).long()
            for aspect in range(4):
                mask = known[:, aspect]
                correct[aspect] += int(
                    (concept_pred[mask, aspect] == target[mask, aspect]).sum().item()
                )
                total[aspect] += int(mask.sum().item())
                unknown_total[aspect] += int((concept_labels[:, aspect] == 1).sum().item())
                for binary_class in (0, 1):
                    class_mask = mask & (target[:, aspect] == binary_class)
                    class_correct[aspect, binary_class] += int(
                        (concept_pred[class_mask, aspect] == binary_class).sum().item()
                    )
                    class_total[aspect, binary_class] += int(class_mask.sum().item())

    per_aspect = {}
    aleatoric = np.concatenate(all_aleatoric, axis=0)
    entropies = np.concatenate(all_entropies, axis=0)
    rating_entropies = np.concatenate(all_rating_entropies, axis=0)
    concept_labels_all = np.concatenate(all_concept_labels, axis=0)
    concept_vote_probs = np.asarray(
        [item["_concept_distributions"] for item in test_loader.dataset.data],
        dtype=np.float64,
    )
    from scipy import stats

    def spearman(x, y):
        if x.std() == 0 or y.std() == 0:
            return 0.0
        return float(stats.spearmanr(x, y).statistic)

    def masked_spearman(x, y, mask):
        mask = np.asarray(mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return None
        return spearman(x[mask], y[mask])

    def aggregation_correlations(au, target):
        aggregators = {
            "mean": lambda x: np.mean(x, axis=-1),
            "max": lambda x: np.max(x, axis=-1),
            "rms": lambda x: np.sqrt(np.mean(np.square(x), axis=-1)),
            "median": lambda x: np.median(x, axis=-1),
            "std": lambda x: np.std(x, axis=-1),
        }
        return {
            name: spearman(fn(au), fn(target))
            for name, fn in aggregators.items()
        }

    # Alternative CEBaB worker-disagreement definitions that remove unknown.
    known_majority = concept_labels_all != 1
    binary_mass = concept_vote_probs[:, :, 0] + concept_vote_probs[:, :, 2]
    binary_valid = binary_mass > 0
    binary_positive = np.divide(
        concept_vote_probs[:, :, 2],
        binary_mass,
        out=np.zeros_like(binary_mass),
        where=binary_valid,
    )
    binary_entropy = -(
        binary_positive * np.log(binary_positive + 1e-10)
        + (1.0 - binary_positive) * np.log(1.0 - binary_positive + 1e-10)
    ) / np.log(2.0)

    for i, name in enumerate(concept_names):
        per_aspect[name] = {
            "accuracy": float(correct[i] / total[i]) if total[i] else 0.0,
            "negative_accuracy": (
                float(class_correct[i, 0] / class_total[i, 0]) if class_total[i, 0] else 0.0
            ),
            "positive_accuracy": (
                float(class_correct[i, 1] / class_total[i, 1]) if class_total[i, 1] else 0.0
            ),
            "known_count": int(total[i]),
            "unknown_count": int(unknown_total[i]),
            "rho_au_entropy": spearman(aleatoric[:, i], entropies[:, i]),
            "rho_au_entropy_known_majority": masked_spearman(
                aleatoric[:, i], entropies[:, i], known_majority[:, i]
            ),
            "rho_au_binary_entropy_no_unknown_votes": masked_spearman(
                aleatoric[:, i], binary_entropy[:, i], binary_valid[:, i]
            ),
        }

    payload = {
        "run_id": run_id,
        "encoder": encoder,
        "seed": seed,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "task_accuracy": float(task_correct / task_total),
        "mean_concept_accuracy": float(correct.sum() / total.sum()),
        "concept_coverage": float(total.sum() / (task_total * 4)),
        "negative_accuracy": float(class_correct[:, 0].sum() / class_total[:, 0].sum()),
        "positive_accuracy": float(class_correct[:, 1].sum() / class_total[:, 1].sum()),
        "rho_au_entropy_sample_mean": spearman(
            aleatoric.mean(axis=-1), entropies.mean(axis=-1)
        ),
        "rho_au_entropy_flattened": spearman(
            aleatoric.reshape(-1), entropies.reshape(-1)
        ),
        "rho_au_worker_rating_entropy": spearman(
            aleatoric.mean(axis=-1), rating_entropies
        ),
        "rho_au_entropy_known_majority_pooled": masked_spearman(
            aleatoric.reshape(-1),
            entropies.reshape(-1),
            known_majority.reshape(-1),
        ),
        "rho_au_binary_entropy_no_unknown_votes_pooled": masked_spearman(
            aleatoric.reshape(-1),
            binary_entropy.reshape(-1),
            binary_valid.reshape(-1),
        ),
        "rho_au_entropy_by_aggregation": aggregation_correlations(
            aleatoric, entropies
        ),
        "rho_au_binary_entropy_by_aggregation": aggregation_correlations(
            aleatoric, binary_entropy
        ),
        "per_aspect": per_aspect,
        "counts": {
            "test_examples": task_total,
            "known_concepts": int(total.sum()),
            "unknown_concepts": int(unknown_total.sum()),
            "negative_concepts": int(class_total[:, 0].sum()),
            "positive_concepts": int(class_total[:, 1].sum()),
        },
    }
    output_path = Path("/results") / run_id / "task_concept_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2))
    results_volume.commit()
    return payload


@app.local_entrypoint()
def main(
    datasets: str = "cebab,hatexplain,goemotions",
    encoder: str = "distilbert",
    cebab_three_class: bool = False,
    cebab_binary: bool = False,
    seeds: str = "123,2024",
    epochs_override: int = 0,
    run_tag: str = "",
    phased_schedule: bool = False,
    worker_review_entropy: bool = False,
    binary_concept_entropy: bool = False,
    mask_unannotated: bool = False,
    aleatoric_weight: float = -1.0,
    decorr_weight: float = 0.0,
    qa_subset: str = "",
    eval_concepts: bool = False,
    eval_binary_runs: bool = False,
    eval_encoder_filter: str = "",
):
    if eval_binary_runs:
        jobs = []
        for seed in (123, 2024):
            for phased in (False, True):
                suffix = "_phased" if phased else ""
                run_id = f"cebab_3class_roberta_base_seed{seed}_binaryH_100ep{suffix}"
                jobs.append(evaluate_cebab_concepts.spawn(run_id, "roberta-base", seed))
        for job in jobs:
            print(f"spawned binary-H concept evaluation {job.object_id}")
        return

    if eval_concepts:
        encoders = {
            "distilbert": "distilbert-base-uncased",
            "answerdotai_ModernBERT_base": "answerdotai/ModernBERT-base",
            "roberta_base": "roberta-base",
            "microsoft_deberta_v3_base": "microsoft/deberta-v3-base",
        }
        jobs = []
        for encoder_tag, full_encoder in encoders.items():
            if eval_encoder_filter and encoder_tag != eval_encoder_filter:
                continue
            for seed in (123, 2024):
                for phased in (False, True):
                    suffix = "_phased" if phased else ""
                    run_id = (
                        f"cebab_3class_{encoder_tag}_seed{seed}_100ep{suffix}"
                        if encoder_tag != "distilbert"
                        else f"cebab_3class_seed{seed}_100ep{suffix}"
                    )
                    jobs.append(
                        evaluate_cebab_concepts.spawn(run_id, full_encoder, seed)
                    )
        for job in jobs:
            print(f"spawned concept evaluation {job.object_id}")
        return

    requested = {name.strip() for name in datasets.split(",") if name.strip()}
    requested_seeds = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    epoch_config = (
        ("cebab", 50),
        ("hatexplain", 30),
        ("goemotions", 30),
        ("maqa", 100),
    )
    jobs = [
        train_one.spawn(
            dataset,
            seed,
            epochs_override or epochs,
            encoder,
            cebab_three_class,
            cebab_binary,
            run_tag,
            phased_schedule,
            worker_review_entropy,
            binary_concept_entropy,
            mask_unannotated,
            aleatoric_weight,
            decorr_weight,
            qa_subset,
        )
        for dataset, epochs in epoch_config
        if dataset in requested
        for seed in requested_seeds
    ]
    for job in jobs:
        print(f"spawned {job.object_id}")
