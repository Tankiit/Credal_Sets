"""Concept models assembled from the PyC (pytorch-concepts) low-level API.

https://pytorch-concepts.readthedocs.io/en/latest/guides/using_low_level.html

They share the trunk and the ``forward`` interface of ``models.ConceptModel``, so ``fit``,
``evaluate`` and ``explain.py`` work on them unchanged, but every concept-specific layer
is a PyC layer, wired the way PyC's own models wire them:

pyc-cbm    x -> trunk -> LinearEmbeddingToConcept -> concept logits
                          sigmoid -> LinearConceptToConcept -> task logits
           Same function class as our vanilla CBM with a linear head; a cross-check.

pyc-cem    x -> trunk -> LinearEmbeddingEncoder -> one embedding e_i per concept (k, m)
                       e_i -> LinearEmbeddingToConcept (shared across concepts) -> logit_i
           MixConceptEmbeddingToConcept: e_i -> Linear + LeakyReLU -> (c_i+, c_i-),
           z_i = p_i c_i+ + (1 - p_i) c_i-, task logits = Linear(concat_i z_i).
           Differs from our CEM: the concept is read from one embedding (not from
           [c_i+; c_i-]) and c_i+/- come out of a nonlinearity, so no exact twin folding.

pyc-hyper  x -> trunk -> LinearEmbeddingToConcept -> concept logits
                      -> LinearEmbeddingEncoder -> one embedding per class (n_classes, m)
           HyperlinearConceptEmbeddingToConcept: an MLP turns each class embedding into
           per-example concept weights W(x) (n_classes, k); task logits = W(x) p + b.
           (C2BM-style predictor, De Felice et al. 2025.)  Each answer is exactly a sum of
           one term per concept, with weights that change from one text to the next.
           b is a learned per-class bias added here: without it, a text whose concepts are
           all 0 (e.g. after intervening on a CEBaB review with every aspect unknown) gets
           all-zero logits and its prediction is decided by rounding noise.

None of them implements ``readout_blocks`` / ``fold_concept_layer``: no twins (twin.py).
"""

import torch
from torch import nn

import torch_concepts as pyc
from torch_concepts.nn import (HyperlinearConceptEmbeddingToConcept, LinearConceptToConcept,
                               LinearEmbeddingEncoder, LinearEmbeddingToConcept,
                               MixConceptEmbeddingToConcept)

from .models import ConceptModel


def concept_annotations(config: dict) -> pyc.Annotations:
    names = config.get("concept_names") or [f"c{i}" for i in range(config["n_concepts"])]
    return pyc.Annotations(labels=list(names), cardinalities=[1] * len(names),
                           types=["binary"] * len(names))


class PycModel(ConceptModel):
    """Common part: concept annotations, interventions on probabilities, CEM-style RandInt."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.annotations = concept_annotations(config)
        self.p_int = config.get("p_int", 0.0)

    def concept_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Concept logits from the trunk output. Wrapped by pyc InterventionModules in semantics.py."""
        return self.concept_encoder(h)

    def probs(self, concept_logits, concepts, intervene):
        p = torch.sigmoid(concept_logits)
        if intervene is None and self.training and self.p_int > 0 and concepts is not None:
            intervene = torch.rand_like(p) < self.p_int
        if intervene is not None:
            p = torch.where(intervene, concepts.float(), p)
        return p

    def annotate(self, t: torch.Tensor) -> pyc.AnnotatedTensor:
        """Name the concept axis, e.g. ``model.annotate(out["concept_probs"])["food_pos"]``."""
        return pyc.AnnotatedTensor(t, self.annotations)


class PycCBM(PycModel):
    def __init__(self, config: dict):
        super().__init__(config)
        h = config["hidden"]
        self.concept_encoder = LinearEmbeddingToConcept(in_embeddings=h, out_concepts=self.annotations)
        self.predictor = LinearConceptToConcept(in_concepts=self.k, out_concepts=config["n_classes"])

    def forward(self, x, concepts=None, intervene=None):
        concept_logits = self.concept_logits(self.trunk(x))
        p = self.probs(concept_logits, concepts, intervene)
        return {"z": concept_logits, "concept_logits": concept_logits, "concept_probs": p,
                "logits": self.predictor(p)}


class PycCEM(PycModel):
    def __init__(self, config: dict):
        super().__init__(config)
        h, self.m = config["hidden"], config.get("emb_dim", 16)
        self.embedding_encoder = LinearEmbeddingEncoder(in_features=h, out_features=self.m,
                                                        n_embeddings=self.k)
        self.concept_encoder = pyc.nn.Sequential(
            LinearEmbeddingToConcept(in_embeddings=self.m, out_concepts=1), nn.Flatten(start_dim=1))
        self.predictor = MixConceptEmbeddingToConcept(in_concepts=self.annotations, in_embeddings=self.m,
                                                      out_concepts=config["n_classes"])

    def concept_logits(self, emb: torch.Tensor) -> torch.Tensor:
        return self.concept_encoder(emb)

    def forward(self, x, concepts=None, intervene=None):
        emb = self.embedding_encoder(self.trunk(x))                           # (b, k, m)
        concept_logits = self.concept_logits(emb)
        p = self.probs(concept_logits, concepts, intervene)
        z = self.predictor._mix(p, emb).flatten(1)                            # (b, k * m)
        c = self.predictor.bernoulli_to_categorical_embedding_splitter(emb)  # (b, k, 2, m)
        return {"z": z, "concept_logits": concept_logits, "concept_probs": p,
                "logits": self.predictor.predictor(z), "embeddings": emb,
                "c_plus": c[:, :, 0], "c_minus": c[:, :, 1]}


class PycHyper(PycModel):
    def __init__(self, config: dict):
        super().__init__(config)
        h, self.m = config["hidden"], config.get("emb_dim", 16)
        n_classes = config["n_classes"]
        self.concept_encoder = LinearEmbeddingToConcept(in_embeddings=h, out_concepts=self.annotations)
        self.task_encoder = LinearEmbeddingEncoder(in_features=h, out_features=self.m,
                                                   n_embeddings=n_classes)
        # use_bias=False: PyC's bias is one scalar for all classes, sampled at random on every
        # call (even at test time); a deterministic per-class bias is added instead.
        self.predictor = HyperlinearConceptEmbeddingToConcept(
            in_concepts=self.k, in_embeddings=self.m, out_concepts=n_classes,
            hidden_size=config.get("hyper_hidden", 64), use_bias=False)
        self.task_bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x, concepts=None, intervene=None):
        h = self.trunk(x)
        concept_logits = self.concept_logits(h)
        p = self.probs(concept_logits, concepts, intervene)
        task_emb = self.task_encoder(h)                                       # (b, classes, m)
        weights = self.predictor.hypernet(task_emb)                           # (b, classes, k)
        return {"z": concept_logits, "concept_logits": concept_logits, "concept_probs": p,
                "logits": torch.einsum("bc,bnc->bn", p, weights) + self.task_bias,
                "concept_weights": weights}


PYC_MODELS = {"pyc-cbm": PycCBM, "pyc-cem": PycCEM, "pyc-hyper": PycHyper}
