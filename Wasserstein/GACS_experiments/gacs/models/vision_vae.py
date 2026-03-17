"""
Vision VAE with Concept Bottleneck for MedMNIST

Lightweight CNN architecture for 28×28 images.
Trains in ~2-5 minutes per epoch on RTX 3090 with batch_size=128.

Architecture:
    x (28×28 image)
    → CNN encoder → h (flat feature vector)
    → μ_z, log σ²_z
    → z ~ N(μ, σ²) via reparameterization
    → concept decoder → c ∈ R^K
    → classifier → ŷ
    → image decoder → x̂ (pixel reconstruction)

Key difference from NLP VAE: we DO reconstruct pixels here because
(a) 28×28 is cheap, and (b) it gives us a proper generative model
whose loss landscape geometry is richer to probe.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ConvEncoder(nn.Module):
    """CNN encoder for 28×28 images → flat features."""
    
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        # 28×28 → 14×14 → 7×7 → 3×3
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # → 14×14
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # → 7×7
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),          # → 4×4
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        # 128 * 4 * 4 = 2048
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.GELU(),
        )
    
    def forward(self, x):
        h = self.conv(x)
        return self.fc(h)


class ConvDecoder(nn.Module):
    """CNN decoder: flat features → 28×28 image reconstruction."""
    
    def __init__(self, z_dim: int, out_channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128 * 4 * 4),
            nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0),  # → 7×7
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # → 14×14
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, out_channels, 3, stride=2, padding=1, output_padding=1),  # → 28×28
            nn.Sigmoid(),  # pixel values in [0, 1]
        )
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 128, 4, 4)
        return self.deconv(h)


class VisionConceptVAE(nn.Module):
    """
    Vision VAE with interpretable concept bottleneck.
    
    Same logical structure as the NLP model:
        x → z → c → y (and x → z → x̂ for reconstruction)
    
    But uses CNN encoder/decoder for images.
    
    Args:
        in_channels: input image channels (1 for grayscale, 3 for RGB)
        z_dim: latent dimension
        num_concepts: number of concept activations
        num_classes: number of output classes
        hidden_dim: intermediate feature dimension
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        z_dim: int = 64,
        num_concepts: int = 8,
        num_classes: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = ConvEncoder(in_channels, hidden_dim)
        
        # Latent space
        self.z_mu = nn.Linear(hidden_dim, z_dim)
        self.z_logvar = nn.Linear(hidden_dim, z_dim)
        
        # Concept decoder: z → concepts
        self.concept_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
        )
        
        # Classifier: concepts → class logits
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # Image decoder: z → reconstructed image
        self.image_decoder = ConvDecoder(z_dim, in_channels)
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode image to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.z_mu(h)
        logvar = self.z_logvar(h)
        return {"h": h, "mu": mu, "logvar": logvar}
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu
    
    def forward(self, image: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Accepts 'image' key to match DataLoader output format.
        """
        # Encode
        enc = self.encode(image)
        z = self.reparameterize(enc["mu"], enc["logvar"])
        
        # Concepts
        concept_logits = self.concept_decoder(z)
        concept_probs = torch.sigmoid(concept_logits)
        
        # Classify from concepts
        logits = self.classifier(concept_probs)
        
        # Reconstruct image
        x_recon = self.image_decoder(z)
        
        return {
            "z": z,
            "mu": enc["mu"],
            "logvar": enc["logvar"],
            "concept_logits": concept_logits,
            "concept_probs": concept_probs,
            "logits": logits,
            "x_recon": x_recon,
            "x_original": image,
        }
    
    def get_concept_parameters(self):
        """Parameters for concept decoder + classifier (for geometric probes)."""
        params = []
        for name, param in self.named_parameters():
            if "concept_decoder" in name or "classifier" in name:
                if param.requires_grad:
                    params.append(param)
        return params
    
    def get_all_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class PythaeVisionConceptVAE(nn.Module):
    """
    Vision model that uses `pythae` as the image VAE backbone,
    while keeping the same concept bottleneck/classifier heads used by GACS.
    """

    def __init__(
        self,
        in_channels: int = 3,
        z_dim: int = 64,
        num_concepts: int = 8,
        num_classes: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        input_size: int = 28,
        pythae_model_name: str = "vae",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        try:
            from pythae.models import VAE, VAEConfig, BetaVAE, BetaVAEConfig
        except Exception as e:
            raise ImportError(
                "pythae is required for encoder_name='pythae'. Install with: pip install pythae"
            ) from e

        model_name = pythae_model_name.lower()
        if model_name == "vae":
            vae_config = VAEConfig(
                input_dim=(in_channels, input_size, input_size),
                latent_dim=z_dim,
                reconstruction_loss="bce",
            )
            self.image_vae = VAE(model_config=vae_config)
        elif model_name == "betavae":
            vae_config = BetaVAEConfig(
                input_dim=(in_channels, input_size, input_size),
                latent_dim=z_dim,
                reconstruction_loss="bce",
            )
            self.image_vae = BetaVAE(model_config=vae_config)
        else:
            raise ValueError(
                f"Unsupported pythae model '{pythae_model_name}'. Supported: 'vae', 'betavae'."
            )

        self.concept_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        pythae_out = self.image_vae({"data": image})

        encoder_out = self.image_vae.encoder(image)
        mu = encoder_out.embedding
        logvar = encoder_out.log_covariance

        z = pythae_out.z
        x_recon = pythae_out.recon_x

        concept_logits = self.concept_decoder(z)
        concept_probs = torch.sigmoid(concept_logits)
        logits = self.classifier(concept_probs)

        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "concept_logits": concept_logits,
            "concept_probs": concept_probs,
            "logits": logits,
            "x_recon": x_recon,
            "x_original": image,
        }

    def get_concept_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "concept_decoder" in name or "classifier" in name:
                if param.requires_grad:
                    params.append(param)
        return params

    def get_all_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_vision_model(dataset_name: str, config=None) -> nn.Module:
    """
    Factory: build model matched to a MedMNIST dataset.
    """
    from gacs.data.medmnist import MEDMNIST_INFO
    
    info = MEDMNIST_INFO[dataset_name]
    
    z_dim = config.model.z_dim if config else 64
    num_concepts = config.model.num_concepts if config else 8
    hidden_dim = config.model.hidden_dim if config else 256
    dropout = config.model.dropout if config else 0.1

    backend = (config.model.encoder_name if config else "cnn").lower()
    if backend == "pythae":
        pythae_model_name = getattr(config.model, "pythae_model_name", "vae") if config else "vae"
        model = PythaeVisionConceptVAE(
            in_channels=info["n_channels"],
            z_dim=z_dim,
            num_concepts=num_concepts,
            num_classes=info["n_classes"],
            hidden_dim=hidden_dim,
            dropout=dropout,
            input_size=28,
            pythae_model_name=pythae_model_name,
        )
    else:
        model = VisionConceptVAE(
            in_channels=info["n_channels"],
            z_dim=z_dim,
            num_concepts=num_concepts,
            num_classes=info["n_classes"],
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Vision model ({backend}) for {dataset_name}: {n_params:,} params")
    
    return model
