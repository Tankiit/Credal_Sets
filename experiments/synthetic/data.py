"""Controlled synthetic features; no image dataset or backbone dependencies.

The baseline is a *complete* concept bottleneck:

    c* ~ Bernoulli,   z = B c* + eps,   y = g(c*)

so the label is a function of the supervised concepts alone. The previous
generator drew ``y`` partly from ``z[:, 3]`` -- an unsupervised input coordinate
-- which makes the label unpredictable from c* by construction and silently
turns every "concept incompleteness" result into a property of the generator.

``incomplete_concepts`` keeps that case, deliberately and separately, as
``y = g(c*, u)`` with u an explicit unobserved cause.

DATA_SEED is frozen. Cross-seed comparison of learned representations requires
the data and split to be identical across runs, so the model seed must move
init and batch order only -- never the dataset.
"""
import torch

DATA_SEED = 0


def make_data(seed=None, condition="complete", n=512, latent_dim=12, n_concepts=3,
              noise=0.1, n_train=384):
    """Return ``(z, concepts, labels, n_train)``.

    ``seed`` is accepted for backward compatibility but ignored: the dataset is
    fixed at DATA_SEED so that varying a model seed cannot move it. Pass
    ``data_seed`` explicitly to generate a genuinely different dataset.
    """
    return make_dataset(DATA_SEED, condition=condition, n=n, latent_dim=latent_dim,
                        n_concepts=n_concepts, noise=noise, n_train=n_train)


def make_dataset(data_seed=DATA_SEED, condition="complete", n=512, latent_dim=12,
                 n_concepts=3, noise=0.1, n_train=384):
    if condition not in {"complete", "incomplete_concepts"}:
        raise ValueError(f"condition must be 'complete' or 'incomplete_concepts', got {condition!r}")
    generator = torch.Generator().manual_seed(data_seed)

    # c* ~ Bernoulli(0.5), independent coordinates.
    concepts = (torch.rand(n, n_concepts, generator=generator) > 0.5).float()
    # x = B c* + eps: the concepts are fully recoverable from the input.
    mixing = torch.randn(n_concepts, latent_dim, generator=generator)
    z = concepts @ mixing + noise * torch.randn(n, latent_dim, generator=generator)

    # y = g(c*): a function of the supervised concepts only.
    labels = (concepts.sum(dim=1) >= 2).long()
    if condition == "incomplete_concepts":
        # y = g(c*, u) with u an explicit unobserved cause, not a stray input coordinate.
        u = (torch.rand(n, generator=generator) > 0.5).long()
        labels = (labels + u) % 2

    return z, concepts, labels, n_train
