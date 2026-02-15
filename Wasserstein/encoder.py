

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple, List
import numpy as np
from sklearn.decomposition import PCA


class FrozenDistilBERTEncoder(nn.Module):


    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 768,
        freeze: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.hidden_dim = hidden_dim

        # Load pre-trained DistilBERT
        self.backbone = AutoModel.from_pretrained(model_name)

        # Freeze all parameters if specified
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.frozen = freeze



    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_cls_only: bool = True,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        # Set to eval mode if frozen to disable dropout
        if self.frozen:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )

        if output_hidden_states:
            # Return all hidden states for low-rank analysis
            return {
                'last_hidden_state': outputs.last_hidden_state,
                'hidden_states': outputs.hidden_states,  # Tuple of hidden states from all layers
                'pooler_output': getattr(outputs, 'pooler_output', None),
            }
        elif return_cls_only:
            # Extract [CLS] token representation (first token)
            # outputs.last_hidden_state: [batch_size, seq_len, hidden_dim]
            cls_representation = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
            return cls_representation
        else:
            return outputs.last_hidden_state

    def encode_batch(self, batch: Dict[str, torch.Tensor], output_all_hidden: bool = False) -> torch.Tensor:
        return self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            return_cls_only=not output_all_hidden,
            output_hidden_states=output_all_hidden,
        )

    def encode_texts(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 128,
        device: str = "cpu",
        output_all_hidden: bool = False,
    ) -> torch.Tensor:
        # Tokenize
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors='pt',
        )

        # Move to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Encode
        with torch.no_grad():
            representations = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_cls_only=not output_all_hidden,
                output_hidden_states=output_all_hidden,
            )

        return representations

    def extract_low_rank_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rank: int = 64,
        method: str = "svd",
        layer: str = "last",
        aggregate: str = "mean",
    ) -> torch.Tensor:
        # Get all hidden states
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_cls_only=False,
            output_hidden_states=True,
        )

        # Select layer(s)
        if layer == "last":
            hidden_states = outputs['hidden_states'][-1]  # [batch, seq_len, hidden_dim]
        elif layer == "first":
            hidden_states = outputs['hidden_states'][0]  # [batch, seq_len, hidden_dim]
        elif layer == "mean":
            # Average across all layers
            all_states = torch.stack(outputs['hidden_states'], dim=0)  # [num_layers, batch, seq_len, hidden_dim]
            hidden_states = all_states.mean(dim=0)  # [batch, seq_len, hidden_dim]
        else:
            raise ValueError(f"Unknown layer selection: {layer}")

        # Aggregate sequence
        if aggregate == "cls":
            # Use CLS token only
            pooled = hidden_states[:, 0, :]  # [batch, hidden_dim]
        elif aggregate == "mean":
            # Mean pooling (masked)
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            masked_hidden = hidden_states * mask  # [batch, seq_len, hidden_dim]
            pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [batch, hidden_dim]
        elif aggregate == "max":
            # Max pooling
            pooled = hidden_states.max(dim=1)[0]  # [batch, hidden_dim]
        else:
            raise ValueError(f"Unknown aggregation: {aggregate}")

        # Convert to numpy for SVD/PCA
        pooled_np = pooled.cpu().numpy()  # [batch, hidden_dim]

        # Apply dimensionality reduction
        if method == "svd":
            # Truncated SVD (equivalent to PCA for centered data)
            from sklearn.decomposition import TruncatedSVD

            # Center the data
            mean = pooled_np.mean(axis=0, keepdims=True)
            centered = pooled_np - mean

            # Apply SVD
            svd = TruncatedSVD(n_components=rank, random_state=42)
            low_rank = svd.fit_transform(centered)  # [batch, rank]

            # Store for inverse transform if needed
            if not hasattr(self, '_svd_components'):
                self._svd_components = {
                    'mean': mean,
                    'components': svd.components_,
                    'explained_variance': svd.explained_variance_ratio_,
                }

        elif method == "pca":
            # PCA (requires n_samples >= n_components)
            if pooled_np.shape[0] < rank:
                # Fall back to SVD if batch is too small
                from sklearn.decomposition import TruncatedSVD
                mean = pooled_np.mean(axis=0, keepdims=True)
                centered = pooled_np - mean
                svd = TruncatedSVD(n_components=min(rank, pooled_np.shape[0]-1), random_state=42)
                low_rank = svd.fit_transform(centered)

                if not hasattr(self, '_pca_components'):
                    self._pca_components = {
                        'mean': mean,
                        'components': svd.components_,
                        'explained_variance': svd.explained_variance_ratio_,
                    }
            else:
                # Standard PCA
                pca = PCA(n_components=rank, random_state=42)
                low_rank = pca.fit_transform(pooled_np)  # [batch, rank]

                # Store components
                if not hasattr(self, '_pca_components'):
                    self._pca_components = {
                        'mean': pca.mean_,
                        'components': pca.components_,
                        'explained_variance': pca.explained_variance_ratio_,
                    }
        else:
            raise ValueError(f"Unknown method: {method}")

        # Convert back to tensor
        low_rank_tensor = torch.from_numpy(low_rank).to(pooled.device).float()  # [batch, rank]

        return low_rank_tensor

    def extract_multi_layer_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layers: List[int] = None,
        aggregation: str = "concat",
    ) -> torch.Tensor:
        # Get all hidden states
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_cls_only=False,
            output_hidden_states=True,
        )

        all_hidden = outputs['hidden_states']  # Tuple of [batch, seq_len, hidden_dim]

        # Select layers
        if layers is None:
            layers = list(range(len(all_hidden)))

        # Extract CLS token from each selected layer
        layer_latents = []
        for layer_idx in layers:
            layer_hidden = all_hidden[layer_idx]  # [batch, seq_len, hidden_dim]
            cls_token = layer_hidden[:, 0, :]  # [batch, hidden_dim]
            layer_latents.append(cls_token)

        # Aggregate
        if aggregation == "concat":
            # Concatenate all layers
            multi_layer = torch.cat(layer_latents, dim=-1)  # [batch, num_layers * hidden_dim]
        elif aggregation == "mean":
            # Average across layers
            multi_layer = torch.stack(layer_latents, dim=0).mean(dim=0)  # [batch, hidden_dim]
        elif aggregation == "sum":
            # Sum across layers
            multi_layer = torch.stack(layer_latents, dim=0).sum(dim=0)  # [batch, hidden_dim]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return multi_layer

    def extract_token_level_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        # Get hidden states
        if layer == -1:
            # Just last hidden state
            hidden_states = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_cls_only=False,
                output_hidden_states=False,
            )
        else:
            # Get specific layer
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_cls_only=False,
                output_hidden_states=True,
            )
            hidden_states = outputs['hidden_states'][layer]

        return hidden_states

    def get_embedding_dim(self) -> int:
        return self.hidden_dim

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()
        self.frozen = False

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.frozen = True


class EncoderWithClassifier(nn.Module):


    def __init__(
        self,
        encoder: FrozenDistilBERTEncoder,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = encoder
        self.num_classes = num_classes

        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.hidden_dim, num_classes)



    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Get frozen representations
        with torch.no_grad():
            representations = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_cls_only=True,
            )

        # Apply dropout and classifier
        outputs = self.dropout(representations)
        logits = self.classifier(outputs)

        return logits

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids, attention_mask, return_cls_only=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_encoder(
    model_name: str = "distilbert-base-uncased",
    freeze: bool = True,
    device: str = "cpu",
) -> Tuple[FrozenDistilBERTEncoder, AutoTokenizer]:

    encoder = FrozenDistilBERTEncoder(
        model_name=model_name,
        freeze=freeze,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Move to device
    encoder = encoder.to(device)

    return encoder, tokenizer


def extract_latents_from_loader(
    encoder: FrozenDistilBERTEncoder,
    dataloader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
    extraction_method: str = "cls",
    **extraction_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    
    encoder.eval()

    all_latents = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Encode based on method
            if extraction_method == "cls":
                # Standard CLS token extraction
                latents = encoder(input_ids, attention_mask, return_cls_only=True)

            elif extraction_method == "low_rank":
                # Low-rank approximation
                rank = extraction_kwargs.get('rank', 128)
                method = extraction_kwargs.get('method', 'svd')
                layer = extraction_kwargs.get('layer', 'last')
                aggregate = extraction_kwargs.get('aggregate', 'mean')

                latents = encoder.extract_low_rank_latents(
                    input_ids,
                    attention_mask,
                    rank=rank,
                    method=method,
                    layer=layer,
                    aggregate=aggregate,
                )

                # Ensure consistent dimensions across batches
                # (Pad/truncate if needed due to variable batch sizes)
                target_dim = extraction_kwargs.get('rank', 128)
                if latents.shape[1] < target_dim:
                    # Pad with zeros if dimension is too small
                    padding = torch.zeros(latents.shape[0], target_dim - latents.shape[1], device=latents.device, dtype=latents.dtype)
                    latents = torch.cat([latents, padding], dim=1)
                elif latents.shape[1] > target_dim:
                    # Truncate if dimension is too large
                    latents = latents[:, :target_dim]

            elif extraction_method == "multi_layer":
                # Multi-layer aggregation
                latents = encoder.extract_multi_layer_latents(
                    input_ids,
                    attention_mask,
                    **extraction_kwargs
                )

            elif extraction_method == "token_level":
                # Token-level (will be larger)
                latents = encoder.extract_token_level_latents(
                    input_ids,
                    attention_mask,
                    **extraction_kwargs
                )
                # Flatten: [batch, seq_len, hidden_dim] -> [batch, seq_len * hidden_dim]
                batch_size, seq_len, hidden_dim = latents.shape
                latents = latents.reshape(batch_size, -1)

            else:
                raise ValueError(f"Unknown extraction method: {extraction_method}")

            # Collect
            all_latents.append(latents.cpu().numpy())
            all_labels.append(batch['labels'].cpu().numpy())

    # Concatenate
    latents = np.vstack(all_latents)
    labels = np.concatenate(all_labels)



    return latents, labels


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("FROZEN DISTILBERT ENCODER TEST")
    print("="*80)

    # Create encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    encoder, tokenizer = create_encoder(
        model_name="distilbert-base-uncased",
        freeze=True,
        device=device,
    )

    # Test with dummy text
    print("\n" + "-"*80)
    print("Testing with sample texts:")
    print("-"*80)

    sample_texts = [
        "I loved this movie! It was absolutely fantastic.",
        "The food was terrible and the service was slow.",
        "It was okay, nothing special but not bad either.",
    ]

    # Encode
    latents = encoder.encode_texts(
        texts=sample_texts,
        tokenizer=tokenizer,
        max_length=128,
        device=device,
    )

    print(f"\nEncoded {len(sample_texts)} texts")
    print(f"Latent shape: {latents.shape}")
    print(f"Latent dim: {encoder.get_embedding_dim()}")
    print(f"\nSample latent values (first text, first 10 dims):")
    print(latents[0, :10])

    # Test with dummy batch
    print("\n" + "-"*80)
    print("Testing with dummy batch:")
    print("-"*80)

    batch_size = 4
    seq_len = 128

    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_len)).to(device)
    dummy_attention_mask = torch.ones(batch_size, seq_len).to(device)

    dummy_batch = {
        'input_ids': dummy_input_ids,
        'attention_mask': dummy_attention_mask,
    }

    latents_batch = encoder.encode_batch(dummy_batch)

    print(f"\nBatch latent shape: {latents_batch.shape}")
    print(f"Expected: [{batch_size}, {encoder.get_embedding_dim()}]")

    # Test LOW-RANK extraction
    print("\n" + "-"*80)
    print("Testing LOW-RANK extraction (SVD):")
    print("-"*80)

    low_rank_latents = encoder.extract_low_rank_latents(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        rank=64,
        method="svd",
        layer="last",
        aggregate="mean",
    )

    print(f"\nLow-rank latent shape: {low_rank_latents.shape}")
    print(f"Expected: [{batch_size}, 64]")
    print(f"Compression ratio: {encoder.get_embedding_dim() / 64:.2f}x")
    print(f"\nSample low-rank values (first sample, first 10 dims):")
    print(low_rank_latents[0, :10])

    # Test PCA low-rank
    print("\n" + "-"*80)
    print("Testing LOW-RANK extraction (PCA):")
    print("-"*80)

    pca_latents = encoder.extract_low_rank_latents(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        rank=32,
        method="pca",
        layer="last",
        aggregate="cls",
    )

    print(f"\nPCA latent shape: {pca_latents.shape}")
    print(f"Expected: [{batch_size}, 32]")
    print(f"Compression ratio: {encoder.get_embedding_dim() / 32:.2f}x")

    # Test multi-layer extraction
    print("\n" + "-"*80)
    print("Testing MULTI-LAYER extraction:")
    print("-"*80)

    multi_layer_latents = encoder.extract_multi_layer_latents(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        layers=[0, 3, 5, -1],  # First, middle, last layers
        aggregation="concat",
    )

    print(f"\nMulti-layer latent shape: {multi_layer_latents.shape}")
    print(f"Expected: [{batch_size}, {4 * encoder.get_embedding_dim()}] (4 layers concatenated)")
    print(f"Effective dim: {multi_layer_latents.shape[1]}")

    # Test token-level extraction
    print("\n" + "-"*80)
    print("Testing TOKEN-LEVEL extraction:")
    print("-"*80)

    token_latents = encoder.extract_token_level_latents(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        layer=-1,
    )

    print(f"\nToken-level latent shape: {token_latents.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {encoder.get_embedding_dim()}]")

    # Test classifier head
    print("\n" + "-"*80)
    print("Testing classifier head:")
    print("-"*80)

    num_classes = 3
    model = EncoderWithClassifier(
        encoder=encoder,
        num_classes=num_classes,
        dropout=0.1,
    ).to(device)

    logits = model(dummy_input_ids, dummy_attention_mask)

    print(f"\nLogits shape: {logits.shape}")
    print(f"Expected: [{batch_size}, {num_classes}]")
    print(f"\nSample logits:")
    print(logits[0])

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Encoder: {encoder.model_name}")
    print(f"✓ Hidden dim: {encoder.get_embedding_dim()}")
    print(f"✓ Frozen: {encoder.frozen}")
    print(f"✓ Device: {device}")
    print(f"✓ Classifier output dim: {num_classes}")
    print(f"\n✓ Low-rank extraction: SVD, PCA")
    print(f"✓ Multi-layer extraction: concat, mean, sum")
    print(f"✓ Token-level extraction: full sequence")
    print("="*80)

