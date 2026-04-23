from __future__ import annotations

from types import SimpleNamespace

import torch

import VCBM


class DummyEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        hidden = torch.randn(batch_size, seq_len, self.config.hidden_size, device=input_ids.device)
        return SimpleNamespace(last_hidden_state=hidden)


def test_variational_config_accepts_legacy_fields():
    config = VCBM.VariationalCredalConfig.from_dict(
        {
            "encoder_name": "distilbert-base-uncased",
            "covariance_family": "mean_field",
            "prior_sigma": 0.25,
            "min_sigma_epi": 0.1,
            "max_sigma_ale": 2.5,
            "unknown_field": "ignored",
        }
    )

    assert config.prior_sigma == 0.25
    assert config.min_sigma_epi == 0.1
    assert config.max_sigma_ale == 2.5
    assert not hasattr(config, "unknown_field")


def test_forward_exposes_sigma_and_legacy_aliases(monkeypatch):
    monkeypatch.setattr(VCBM.AutoModel, "from_pretrained", lambda *args, **kwargs: DummyEncoder())

    model = VCBM.VariationalCredalCBM(VCBM.VariationalCredalConfig())
    input_ids = torch.ones(2, 4, dtype=torch.long)
    attention_mask = torch.ones(2, 4, dtype=torch.long)
    labels = torch.zeros(2, dtype=torch.long)
    concept_labels = torch.randint(0, 3, (2, model.config.num_concepts), dtype=torch.long)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        concept_labels=concept_labels,
    )

    assert outputs["concept_probs"].shape == (2, model.config.num_concepts)
    assert outputs["epistemic"].shape == (2, model.config.num_concepts)
    assert outputs["aleatoric"].shape == (2, model.config.num_concepts)
    assert outputs["sigma_epi"].shape == (2,)
    assert outputs["sigma_ale"].shape == (2,)
    assert outputs["mu"].shape == (2, model.config.num_concepts)
    assert outputs["label_probs"].shape == (2, model.config.num_classes)
    assert "kl_loss" in outputs
    assert "concept_loss" in outputs

