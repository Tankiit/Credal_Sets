"""
GACS Model: Text VAE with Interpretable Latent Factors
=======================================================

Architecture:
    Input text → [BERT encoder] → h (CLS token)
                                   ↓
                   [EncoderHead] → μ_z, log σ²_z
                                   ↓
                   [Reparameterize] → z ~ N(μ, σ²)
                                   ↓
                   [FactorDecoder] → s ∈ [0,1]^K   (interpretable latent factors)
                                   ↓           ↓
                   [Classifier] → ŷ    [ReconstructionHead] → ĥ_norm

Notation maps to paper as:
    h          → BERT representation
    (μ, log σ²) → q_φ(z|x) encoder posterior parameters
    z          → stochastic latent
    s          → interpretable latent factors  (paper: s ∈ R^K)
    ŷ          → task prediction
    ĥ_norm     → L2-normalised reconstruction target (D1: scale invariance)

Probe scope for geometric probes (Framing A):
    Use get_probeable_parameters("decoder") which returns:
        factor_decoder + classifier + recon_head parameters only.
    The BERT encoder and encoder_head are excluded from probing —
    epistemic uncertainty over the factor-to-prediction mapping
    is what GACS measures, not BERT parameter uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Tuple


class EncoderHead(nn.Module):
    """Maps BERT CLS hidden state → latent posterior parameters (μ, log σ²)."""

    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        return mu, logvar


class FactorDecoder(nn.Module):
    """
    Maps latent z → interpretable factor activations s ∈ [0,1]^K.

    FIX 1: Added Sigmoid as final activation.
    - s ∈ [0,1]^K is required for interpretability (each factor is a
      normalised activation, analogous to a probability).
    - Also required for D3 diversity loss stability: unbounded outputs
      cause the [K×K] factor covariance to explode during training.

    Renamed from ConceptDecoder → FactorDecoder to match paper terminology.
    ("interpretable latent factors", never "concepts")
    """

    def __init__(self, z_dim: int, num_factors: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_factors),
            nn.Sigmoid(),    # FIX 1: s ∈ [0,1]^K — do not remove
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Classifier(nn.Module):
    """
    Maps factor activations s → class logits.

    FIX 2: Dropout moved before importance weighting.
    Original order: importance_weight → dropout → linear
    This randomly zeroed the most important factors during training,
    destabilising importance learning.
    Correct order: dropout → importance_weight → linear
    """

    def __init__(self, num_factors: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout   = nn.Dropout(dropout)
        self.linear    = nn.Linear(num_factors, num_classes)
        # Learnable per-factor importance weights
        # Initialised to 1 → uniform importance at start
        self.factor_importance = nn.Parameter(torch.ones(num_factors))

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits [B, C], importance [K]) where importance ∈ (0,1)^K.
        importance[k] reflects the learned contribution of factor k to the prediction.
        """
        # FIX 2: dropout before importance weighting
        s_dropped  = self.dropout(s)
        importance = torch.sigmoid(self.factor_importance)   # [K] ∈ (0,1)
        weighted   = s_dropped * importance                  # [B, K]
        logits     = self.linear(weighted)                   # [B, C]
        return logits, importance


class ReconstructionHead(nn.Module):
    """
    Reconstructs the L2-normalised BERT CLS embedding from latent z.

    FIX 3 (partial — see GACSModel.forward): The reconstruction TARGET
    must be L2-normalised h, not raw h. This is enforced in forward()
    by normalising h before storing it as the reconstruction target.
    The head itself is unchanged — it outputs in R^D and the loss
    (NLPContrastiveReconLoss in losses.py) applies cosine similarity.

    D1 (scale invariance): raw BERT CLS embeddings have magnitude ~10–15
    and vary with sequence length. MSE on raw h would make the Hessian
    eigenspectrum reflect embedding magnitude, not representational structure.
    Normalising h and using cosine/contrastive loss removes this artifact.
    """

    def __init__(self, z_dim: int, output_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            # No final activation — cosine loss handles normalisation
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GACSModel(nn.Module):
    """
    Text VAE with Interpretable Latent Factors for GACS.

    Forward returns a dict with all intermediate representations
    needed for loss computation and geometric probing.

    Probe scope (Framing A):
        call get_probeable_parameters("decoder") to get the parameter
        subset used by both geometric probes. This covers:
            factor_decoder + classifier + recon_head
        The BERT encoder and encoder_head are excluded.
        See get_probeable_parameters() docstring for full scope options.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        mc = config.model

        # 1. Pre-trained text encoder (partially frozen)
        encoder_config   = AutoConfig.from_pretrained(mc.encoder_name)
        self.encoder     = AutoModel.from_pretrained(mc.encoder_name)
        self.encoder_dim = encoder_config.hidden_size

        if mc.freeze_encoder_layers > 0:
            self._freeze_encoder_layers(mc.freeze_encoder_layers)

        # 2. Latent posterior: h → (μ, log σ²)
        self.encoder_head = EncoderHead(
            self.encoder_dim, mc.z_dim, mc.hidden_dim, mc.dropout
        )

        # 3. Interpretable latent factors: z → s ∈ [0,1]^K
        # FIX 1 applied here — FactorDecoder has Sigmoid output
        self.factor_decoder = FactorDecoder(
            mc.z_dim, mc.num_factors, mc.hidden_dim, mc.dropout
        )

        # 4. Classifier: s → ŷ
        # FIX 2 applied here — dropout before importance weighting
        self.classifier = Classifier(
            mc.num_factors, config.data.num_classes, mc.dropout
        )

        # 5. Reconstruction: z → ĥ_norm  (D1: target is normalised h)
        self.recon_head = ReconstructionHead(
            mc.z_dim, self.encoder_dim, mc.hidden_dim, mc.dropout
        )

    def _freeze_encoder_layers(self, num_layers: int):
        """Freeze BERT embedding layer + first num_layers transformer blocks."""
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick.
        At eval time returns mu (deterministic) — required for stable probing.
        The perturbation probe evaluates loss at a fixed point; stochastic z
        would add noise to the loss surface and corrupt rho estimates.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run BERT encoder and latent head.

        Returns
        -------
        h_norm : [B, D]  L2-normalised CLS embedding  (D1: reconstruction target)
        mu     : [B, z_dim]
        logvar : [B, z_dim]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h       = outputs.last_hidden_state[:, 0, :]   # CLS token  [B, D]

        # FIX 3: normalise h here — this is the D1-compliant reconstruction target.
        # The raw h is discarded; h_norm is returned and used as the recon target
        # in the training loop. Do NOT use raw h as the reconstruction target.
        h_norm = F.normalize(h, dim=-1)                # [B, D]

        mu, logvar = self.encoder_head(h_norm)         # encode normalised representation
        return h_norm, mu, logvar

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns dict with keys:
            h_norm   : [B, D]     L2-normalised BERT CLS embedding (D1 reconstruction target)
            z_mu     : [B, z_dim] posterior mean
            z_logvar : [B, z_dim] posterior log variance
            z        : [B, z_dim] sampled latent (mu at eval)
            s        : [B, K]     interpretable factor activations ∈ [0,1]^K
            logits   : [B, C]     class logits
            importance:[K]        per-factor importance weights ∈ (0,1)
            recon    : [B, D]     reconstruction of h_norm (use cosine loss, not MSE)

        Training loop usage:
            cls_loss   = cross_entropy(out["logits"], labels)
            recon_loss = NLPContrastiveReconLoss(out["recon"], out["h_norm"])
            kl_loss    = kl_divergence(out["z_mu"], out["z_logvar"])
            d3_loss    = FactorHeadLoss(out["s"])
        """
        h_norm, z_mu, z_logvar = self.encode(input_ids, attention_mask)
        z                      = self.reparameterize(z_mu, z_logvar)
        s                      = self.factor_decoder(z)
        logits, importance     = self.classifier(s)
        recon                  = self.recon_head(z)

        return {
            "h_norm":    h_norm,
            "z_mu":      z_mu,
            "z_logvar":  z_logvar,
            "z":         z,
            "s":         s,
            "logits":    logits,
            "importance": importance,
            "recon":     recon,
        }

    # ── Geometric probe interface ─────────────────────────────────────────────

    def get_probeable_parameters(self, scope: str = "decoder") -> list:
        """
        Return parameter list for geometric probing.

        Scopes
        ------
        "decoder"      (default, Framing A) — factor_decoder + classifier + recon_head.
                        This is the FactorHead-equivalent space: maps z → s → ŷ.
                        Use this for all GACS geometric probes.

        "classifier"   — classifier only. Narrower scope; use for ablation only.

        "concept_only" — factor_decoder only. Ablates whether the z→s mapping
                         or the s→ŷ mapping is the source of degeneracy.

        "all"          — all trainable parameters including BERT layers that are
                         not frozen. Much larger space; probing is slow and noisy.
                         Do not use for paper results unless you have a specific reason.

        FIX 4: Default changed from "all" to "decoder" and scopes renamed
               to match paper terminology (no "concept" in scope names).
        """
        if scope == "decoder":
            # Primary GACS probe scope (Framing A)
            return (
                list(self.factor_decoder.parameters())
                + list(self.classifier.parameters())
                + list(self.recon_head.parameters())
            )
        elif scope == "classifier":
            return list(self.classifier.parameters())
        elif scope == "concept_only":
            # TODO: rename key in any caller code — was "concept_layer" before
            return list(self.factor_decoder.parameters())
        elif scope == "all":
            return [p for p in self.parameters() if p.requires_grad]
        else:
            raise ValueError(
                f"Unknown probe scope '{scope}'. "
                f"Use 'decoder' (default), 'classifier', 'concept_only', or 'all'."
            )

    def num_probeable_parameters(self, scope: str = "decoder") -> int:
        """Total number of parameters in the probe scope."""
        return sum(p.numel() for p in self.get_probeable_parameters(scope))
