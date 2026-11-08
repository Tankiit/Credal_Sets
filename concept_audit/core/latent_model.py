"""The one abstraction above every model family.

    model family -> admissible equivalence class -> diagnostic invariance

A ``LatentModel`` says three things: how to get a latent, what must stay fixed
(observables plus family constraints), and how to rebuild itself under
``c' = A c`` so that those things do stay fixed.

PCA, NMF and SAE should never have to pretend to be concept models to be
audited. They are different points in this interface, not special cases of a
CBM; what differs is the *size and shape of the admissible class*, which is
exactly what the paper compares.

Column convention ``c' = A c``; batches store rows, hence ``c' = c @ A.T``.
"""
from abc import ABC, abstractmethod

import torch
from torch import nn

from concept_audit.core.constraints import evaluate
from concept_audit.core.equivalence import decide
from concept_audit.core.observables import LinearObservable


def apply_transform(c, a):
    """``C -> C A^T``, the row-batch form of ``c -> A c``."""
    c = torch.as_tensor(c)
    return c @ torch.as_tensor(a, dtype=c.dtype).T


class LatentModel(nn.Module, ABC):
    """A model whose latent parameterisation may not be identified.

    Subclasses that only implement ``encode`` keep working; the equivalence
    machinery is opt-in through ``observable_maps`` and ``compensate``.
    """

    family = "latent"
    variant = "base"

    @abstractmethod
    def encode(self, z):
        """(n, latent_dim) latents."""

    def observables(self, c):
        """Everything training/evaluation treats as fixed, evaluated at c."""
        return {o.name: o(c) for o in self.observable_maps()}

    def observable_maps(self):
        """The observables as objects. Linear ones enable exact matrix checks."""
        return []

    def constraints(self):
        """Family membership conditions. Empty means unconstrained."""
        return []

    def constraint_parts(self, a, c=None):
        """Named pieces the constraints inspect after applying A."""
        return {}

    def predictions(self, c):
        """Task outputs when the model has a head; None otherwise."""
        return None

    def compensate(self, a):
        """The model that *is* ``c' = A c``, or None if the family cannot express it.

        Returning None is a real answer: it says A moved the model outside what
        this family can represent, which is itself a fact about the family.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define compensation")

    def latent_dim(self):
        for observable in self.observable_maps():
            if isinstance(observable, LinearObservable):
                return observable.latent_dim
        raise NotImplementedError("latent_dim is unknown; override it")

    def admissible(self, a, c=None, tol=1e-8):
        """Check observable invariance, family constraints and predictions."""
        a = torch.as_tensor(a)
        cond = float(torch.linalg.cond(a.double()))
        try:
            moved = self.compensate(a)
        except (ValueError, TypeError, RuntimeError):
            moved = None
        if moved is None:
            return decide(observable_error=float("inf"), condition_number=cond, tol=tol,
                          detail={"family": self.family, "variant": self.variant,
                                  "compensation": "not representable in this family"})

        observable_error = 0.0
        checked_exactly = True

        # Maps that are themselves specified (a supervision readout) must not move.
        for observable in self.observable_maps():
            if getattr(observable, "fixed_map", False) and isinstance(observable, LinearObservable):
                observable_error = max(observable_error, observable.matrix_error(a))

        # Every observable's *value* must survive, evaluated through the
        # compensated model. Using this model's maps on a moved latent compares
        # a transformed code against an untransformed decoder, which is wrong.
        if c is not None:
            before, after = self.observables(c), moved.observables(apply_transform(c, a))
            if set(before) != set(after):
                raise ValueError(f"observable keys changed: {set(before)} -> {set(after)}")
            for key, value in before.items():
                b, x = torch.as_tensor(value), torch.as_tensor(after[key])
                if b.shape != x.shape:
                    raise ValueError(f"observable {key!r} changed shape")
                if b.numel():
                    observable_error = max(observable_error, float((b - x).abs().max()))
        elif any(not getattr(o, "fixed_map", False) for o in self.observable_maps()):
            checked_exactly = False

        prediction_error = None
        if c is not None and self.predictions(c) is not None:
            with torch.no_grad():
                before_pred = torch.as_tensor(self.predictions(c)).detach()
                after_pred = moved.predictions(apply_transform(c, a))
                prediction_error = (
                    float("inf") if after_pred is None
                    else float((before_pred - torch.as_tensor(after_pred).detach()).abs().max())
                )

        return decide(
            observable_error=observable_error,
            prediction_error=prediction_error,
            constraint_error=evaluate(self.constraints(), **self.constraint_parts(a, c)),
            condition_number=cond,
            tol=tol,
            detail={"family": self.family, "variant": self.variant,
                    "observables_checked_exactly": checked_exactly},
        )
