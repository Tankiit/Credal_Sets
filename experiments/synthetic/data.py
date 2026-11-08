"""Controlled synthetic features; no image dataset or backbone dependencies.

The baseline is a *complete* concept bottleneck:

    c* ~ Bernoulli,   z = B c* + eps,   y = g(c*)

so the label is a function of the supervised concepts alone. The previous
generator drew ``y`` partly from ``z[:, 3]`` -- an unsupervised input coordinate
-- which makes the label unpredictable from c* by construction and silently
turns every "concept incompleteness" result into a property of the generator.

``incomplete_concepts`` keeps that case, deliberately and separately, as
``y = g(c*, u)`` with u an explicit unobserved cause.

Incompleteness and leakage are two different failure modes and are kept on
two separate axes, controlled independently:

  * ``condition`` (incompleteness) governs *label generation*: whether c* is
    sufficient to determine y, or whether an unobserved cause u also
    contributes. This is about what determines the label.

  * ``leak_strength`` lambda (leakage) governs the *representation*: whether z
    carries a decodable shortcut to y that bypasses the concepts entirely.
    Concretely, z is perturbed along a fixed random direction (independent of
    the concept mixing matrix B) by an amount proportional to lambda and
    signed by y:

        z = B c* + eps + lambda * sign(y) * v,   v a fixed unit vector

    At lambda = 0 (the default) this term vanishes and z is exactly the
    baseline ``B c* + eps`` -- unchanged from prior behavior. At lambda > 0,
    a linear probe on z can recover y directly through v, regardless of
    whether the concepts are complete. This lets leakage be studied on its
    own, or composed with incompleteness (e.g. concepts that are both
    incomplete *and* leaky, which is the harder case for concept-based
    interpretability evaluations).

DATA_SEED is frozen. Cross-seed comparison of learned representations requires
the data and split to be identical across runs, so the model seed must move
init and batch order only -- never the dataset.
"""
import torch

DATA_SEED = 0


def make_data(seed=None, condition="complete", n=512, latent_dim=12, n_concepts=3,
              noise=0.1, n_train=384, leak_strength=0.0):
    """Return ``(z, concepts, labels, n_train)``.

    ``seed`` is accepted for backward compatibility but ignored: the dataset is
    fixed at DATA_SEED so that varying a model seed cannot move it. Pass
    ``data_seed`` explicitly to generate a genuinely different dataset.

    ``leak_strength`` (lambda) is independent of ``condition``: it controls how
    much extra, concept-bypassing signal about ``y`` is injected into ``z``.
    See the module docstring for details.
    """
    return make_dataset(DATA_SEED, condition=condition, n=n, latent_dim=latent_dim,
                         n_concepts=n_concepts, noise=noise, n_train=n_train,
                         leak_strength=leak_strength)


def make_dataset(data_seed=DATA_SEED, condition="complete", n=512, latent_dim=12,
                  n_concepts=3, noise=0.1, n_train=384, leak_strength=0.0):
    if condition not in {"complete", "incomplete_concepts"}:
        raise ValueError(f"condition must be 'complete' or 'incomplete_concepts', got {condition!r}")
    if leak_strength < 0:
        raise ValueError(f"leak_strength must be non-negative, got {leak_strength!r}")

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

    # Leakage: an extra, concept-bypassing channel through which y is
    # decodable from z. Drawn unconditionally (not gated on leak_strength) so
    # that the fixed direction v -- and everything drawn before it -- is
    # identical across calls that differ only in lambda, keeping lambda
    # sweeps comparable at fixed data_seed. At leak_strength == 0 this is a
    # no-op: z is exactly the baseline B c* + eps.
    leak_direction = torch.randn(latent_dim, generator=generator)
    leak_direction = leak_direction / leak_direction.norm()
    signed_label = (2.0 * labels.float() - 1.0).unsqueeze(1)  # -1 / +1
    z = z + leak_strength * signed_label * leak_direction.unsqueeze(0)

    return z, concepts, labels, n_train
