#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class CredalCBMConfig:
    base_model_name: str = "distilbert-base-uncased"
    num_concepts: int = 20
    num_classes: int = 2
    n_ensemble_heads: int = 5
    use_credal: bool = True
    use_rationales: bool = False
    hidden_dim: int = 256
    dropout: float = 0.3
    concept_loss_weight: float = 1.0
    classification_loss_weight: float = 1.0


class CredalSet:
    def __init__(self, ensemble_predictions: np.ndarray, aleatoric_variance: np.ndarray):
        self.ensemble_predictions = ensemble_predictions
        self.aleatoric_variance = aleatoric_variance
        self.n_ensemble, self.n_concepts = ensemble_predictions.shape

    def epistemic_uncertainty(self) -> float:
        variance_per_concept = np.var(self.ensemble_predictions, axis=0)
        return float(np.mean(variance_per_concept))

    def aleatoric_uncertainty(self) -> float:
        return float(np.mean(self.aleatoric_variance))

    def total_uncertainty(self) -> float:
        return self.epistemic_uncertainty() + self.aleatoric_uncertainty()

    def get_concept_mean(self) -> np.ndarray:
        return np.mean(self.ensemble_predictions, axis=0)


class SimpleRationaleExtractor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(sequence_output).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        rationale_features = torch.sum(sequence_output * attn_weights.unsqueeze(-1), dim=1)
        return rationale_features, attn_weights


class CredalCBM(nn.Module):
    def __init__(self, config: CredalCBMConfig):
        super().__init__()
        self.config = config

        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        self.hidden_dim = self.encoder.config.hidden_size

        if config.use_rationales:
            self.rationale_extractor = SimpleRationaleExtractor(self.hidden_dim)
        else:
            self.rationale_extractor = None

        if config.use_credal:
            self.concept_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.num_concepts)
                )
                for _ in range(config.n_ensemble_heads)
            ])

            self.aleatoric_head = nn.Sequential(
                nn.Linear(self.hidden_dim, config.hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(config.hidden_dim // 4, config.num_concepts),
                nn.Softplus()
            )
        else:
            self.concept_head = nn.Sequential(
                nn.Linear(self.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.num_concepts)
            )

        self.classifier = nn.Sequential(
            nn.Linear(config.num_concepts, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                concept_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state

        if self.rationale_extractor is not None:
            features, rationale_mask = self.rationale_extractor(sequence_output, attention_mask)
        else:
            features = sequence_output.mean(dim=1)
            rationale_mask = None

        if self.config.use_credal:
            concept_preds_list = []
            for head in self.concept_heads:
                logits = head(features)
                preds = torch.sigmoid(logits)
                concept_preds_list.append(preds)

            concept_preds = torch.stack(concept_preds_list, dim=1)
            aleatoric_var = self.aleatoric_head(features)
            concept_mean = concept_preds.mean(dim=1)
        else:
            concept_logits = self.concept_head(features)
            concept_mean = torch.sigmoid(concept_logits)
            concept_preds = concept_mean.unsqueeze(1)
            aleatoric_var = None

        logits = self.classifier(concept_mean)

        loss = None
        if concept_labels is not None:
            concept_loss = F.binary_cross_entropy(concept_mean, concept_labels.float())
            loss = self.config.concept_loss_weight * concept_loss

        uncertainty_metrics = {}
        if self.config.use_credal:
            uncertainty_metrics['epistemic'] = concept_preds.var(dim=1)
            uncertainty_metrics['aleatoric'] = aleatoric_var
        else:
            uncertainty_metrics['epistemic'] = torch.zeros_like(concept_mean)
            uncertainty_metrics['aleatoric'] = torch.zeros_like(concept_mean)

        return {
            'logits': logits,
            'concept_preds': concept_preds,
            'concept_mean': concept_mean,
            'aleatoric_var': aleatoric_var,
            'rationale_mask': rationale_mask,
            'loss': loss,
            'uncertainty_metrics': uncertainty_metrics
        }

    def get_credal_sets(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[CredalSet]:
        if not self.config.use_credal:
            raise ValueError("Model not configured with credal sets (use_credal=False)")

        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

            concept_preds = outputs['concept_preds'].cpu().numpy()
            aleatoric_var = outputs['aleatoric_var'].cpu().numpy()

            credal_sets = []
            for i in range(concept_preds.shape[0]):
                cs = CredalSet(
                    ensemble_predictions=concept_preds[i],
                    aleatoric_variance=aleatoric_var[i]
                )
                credal_sets.append(cs)

            return credal_sets


def train_credal_cbm(model: CredalCBM,
                     train_loader,
                     val_loader,
                     optimizer,
                     device: str = 'cuda',
                     epochs: int = 50):
    model.to(device)
    classification_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            concept_labels = batch.get('concepts', None)
            if concept_labels is not None:
                concept_labels = concept_labels.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, concept_labels)
            class_loss = classification_loss_fn(outputs['logits'], labels)

            if outputs['loss'] is not None:
                total_loss = class_loss + outputs['loss']
            else:
                total_loss = class_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs['logits'], dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}")

    return model


def train_credal_cbm_with_bootstrap(model: CredalCBM,
                                     train_loader,
                                     val_loader,
                                     optimizer,
                                     device: str = 'cuda',
                                     epochs: int = 50,
                                     bootstrap_ratio: float = 1.0):
    """
    Train Credal CBM with bootstrap sampling for ensemble diversity.

    Each ensemble head is trained on a different bootstrap sample, which increases
    diversity and improves epistemic uncertainty estimation.

    Args:
        model: CredalCBM model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        device: Device to train on
        epochs: Number of training epochs
        bootstrap_ratio: Ratio of batch size for bootstrap sampling (default 1.0 = same size)
    """
    if not model.config.use_credal:
        raise ValueError("Bootstrap training requires use_credal=True in model config")

    model.to(device)
    classification_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_concept_loss = 0
        total_class_loss = 0
        total_aleatoric_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            batch_size = input_ids.size(0)

            concept_labels = batch.get('concepts', None)
            if concept_labels is not None:
                concept_labels = concept_labels.to(device)

            optimizer.zero_grad()

            # === STEP 1: Get shared encoder features ===
            encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = encoder_outputs.last_hidden_state

            if model.rationale_extractor is not None:
                features, _ = model.rationale_extractor(sequence_output, attention_mask)
            else:
                features = sequence_output.mean(dim=1)

            # === STEP 2: Train each concept head on bootstrap sample ===
            concept_preds_list = []
            concept_loss = 0
            bootstrap_size = int(batch_size * bootstrap_ratio)

            for head_idx, head in enumerate(model.concept_heads):
                # Bootstrap sampling: sample with replacement
                bootstrap_indices = torch.randint(0, batch_size, (bootstrap_size,), device=device)

                # Get bootstrap sample
                boot_features = features[bootstrap_indices]

                # Forward through this concept head
                head_logits = head(boot_features)
                head_preds = torch.sigmoid(head_logits)

                # Store predictions for full batch (use non-bootstrapped for consistency)
                with torch.no_grad():
                    full_logits = head(features)
                    full_preds = torch.sigmoid(full_logits)
                    concept_preds_list.append(full_preds)

                # Concept loss (if we have concept labels)
                if concept_labels is not None:
                    boot_concept_labels = concept_labels[bootstrap_indices]
                    head_concept_loss = F.binary_cross_entropy(head_preds, boot_concept_labels.float())
                    concept_loss += head_concept_loss

            # Average concept loss across heads
            if concept_labels is not None:
                concept_loss = concept_loss / len(model.concept_heads)

            # === STEP 3: Train aleatoric head (on full batch) ===
            aleatoric_var = model.aleatoric_head(features)

            # Aleatoric regularization: encourage reasonable uncertainty
            aleatoric_loss = 0.01 * torch.mean(torch.abs(aleatoric_var - 0.1))

            # === STEP 4: Get mean concepts and classify ===
            concept_preds = torch.stack(concept_preds_list, dim=1)
            concept_mean = concept_preds.mean(dim=1)

            logits = model.classifier(concept_mean)
            class_loss = classification_loss_fn(logits, labels)

            # === STEP 5: Combine losses ===
            total_loss = class_loss

            if concept_labels is not None:
                total_loss = total_loss + model.config.concept_loss_weight * concept_loss

            total_loss = total_loss + aleatoric_loss

            # === STEP 6: Backward and optimize ===
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track losses
            train_loss += total_loss.item()
            total_class_loss += class_loss.item()
            if concept_labels is not None:
                total_concept_loss += concept_loss.item()
            total_aleatoric_loss += aleatoric_loss.item()

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'class': f'{class_loss.item():.4f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        avg_concept_loss = total_concept_loss / len(train_loader) if concept_labels is not None else 0
        avg_aleatoric_loss = total_aleatoric_loss / len(train_loader)

        # === VALIDATION ===
        model.eval()
        val_correct = 0
        val_total = 0
        val_epistemic = []
        val_aleatoric = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs['logits'], dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # Track uncertainties
                if outputs['concept_preds'] is not None:
                    concept_preds = outputs['concept_preds']
                    epistemic = concept_preds.var(dim=1).mean(dim=1).cpu().numpy()
                    val_epistemic.extend(epistemic)

                if outputs['aleatoric_var'] is not None:
                    aleatoric = outputs['aleatoric_var'].mean(dim=1).cpu().numpy()
                    val_aleatoric.extend(aleatoric)

        val_acc = val_correct / val_total
        val_epistemic_mean = np.mean(val_epistemic) if val_epistemic else 0
        val_aleatoric_mean = np.mean(val_aleatoric) if val_aleatoric else 0

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss={avg_train_loss:.4f} "
              f"(Class={avg_class_loss:.4f}, Concept={avg_concept_loss:.4f}, Aleat={avg_aleatoric_loss:.4f}), "
              f"Val Acc={val_acc:.4f}, "
              f"Epistemic={val_epistemic_mean:.4f}, "
              f"Aleatoric={val_aleatoric_mean:.4f}")

    return model


def train_credal_cbm_with_diverse_dropout(model: CredalCBM,
                                     train_loader,
                                     val_loader,
                                     optimizer,
                                     device: str = 'cuda',
                                     epochs: int = 50,
                                     dropout_rates: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Train Credal CBM with diverse dropout for ensemble diversity.
    Each ensemble head sees a different dropout mask.
    """
    if not model.config.use_credal:
        raise ValueError("Diverse dropout training requires use_credal=True in model config")

    if len(dropout_rates) != len(model.concept_heads):
        raise ValueError(f"Number of dropout rates ({len(dropout_rates)}) must match number of concept heads ({len(model.concept_heads)})")

    model.to(device)
    classification_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_concept_loss = 0
        total_class_loss = 0
        total_aleatoric_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            batch_size = input_ids.size(0)

            concept_labels = batch.get('concepts', None)
            if concept_labels is not None:
                concept_labels = concept_labels.to(device)

            optimizer.zero_grad()

            # === STEP 1: Get shared encoder features ===
            encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = encoder_outputs.last_hidden_state

            if model.rationale_extractor is not None:
                features, _ = model.rationale_extractor(sequence_output, attention_mask)
            else:
                features = sequence_output.mean(dim=1)

            # === STEP 2: Train each concept head with different dropout ===
            concept_preds_list = []
            concept_loss = 0

            for head, dropout_rate in zip(model.concept_heads, dropout_rates):
                # Apply different dropout to each head
                dropped_features = F.dropout(features, p=dropout_rate, training=True)
                
                # Forward through this concept head
                head_logits = head(dropped_features)
                head_preds = torch.sigmoid(head_logits)
                concept_preds_list.append(head_preds)

                # Concept loss (if we have concept labels)
                if concept_labels is not None:
                    head_concept_loss = F.binary_cross_entropy(head_preds, concept_labels.float())
                    concept_loss += head_concept_loss

            # Average concept loss across heads
            if concept_labels is not None:
                concept_loss = concept_loss / len(model.concept_heads)

            # === STEP 3: Train aleatoric head (on full batch) ===
            aleatoric_var = model.aleatoric_head(features)

            # Aleatoric regularization: encourage reasonable uncertainty
            aleatoric_loss = 0.01 * torch.mean(torch.abs(aleatoric_var - 0.1))

            # === STEP 4: Get mean concepts and classify ===
            concept_preds = torch.stack(concept_preds_list, dim=1)
            concept_mean = concept_preds.mean(dim=1)

            logits = model.classifier(concept_mean)
            class_loss = classification_loss_fn(logits, labels)

            # === STEP 5: Combine losses ===
            total_loss = class_loss

            if concept_labels is not None:
                total_loss = total_loss + model.config.concept_loss_weight * concept_loss

            total_loss = total_loss + aleatoric_loss

            # === STEP 6: Backward and optimize ===
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track losses
            train_loss += total_loss.item()
            total_class_loss += class_loss.item()
            if concept_labels is not None:
                total_concept_loss += concept_loss.item()
            total_aleatoric_loss += aleatoric_loss.item()

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'class': f'{class_loss.item():.4f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        avg_concept_loss = total_concept_loss / len(train_loader) if concept_labels is not None else 0
        avg_aleatoric_loss = total_aleatoric_loss / len(train_loader)

        # === VALIDATION ===
        model.eval()
        val_correct = 0
        val_total = 0
        val_epistemic = []
        val_aleatoric = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs['logits'], dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # Track uncertainties
                if outputs['concept_preds'] is not None:
                    concept_preds = outputs['concept_preds']
                    epistemic = concept_preds.var(dim=1).mean(dim=1).cpu().numpy()
                    val_epistemic.extend(epistemic)

                if outputs['aleatoric_var'] is not None:
                    aleatoric = outputs['aleatoric_var'].mean(dim=1).cpu().numpy()
                    val_aleatoric.extend(aleatoric)

        val_acc = val_correct / val_total
        val_epistemic_mean = np.mean(val_epistemic) if val_epistemic else 0
        val_aleatoric_mean = np.mean(val_aleatoric) if val_aleatoric else 0

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss={avg_train_loss:.4f} "
              f"(Class={avg_class_loss:.4f}, Concept={avg_concept_loss:.4f}, Aleat={avg_aleatoric_loss:.4f}), "
              f"Val Acc={val_acc:.4f}, "
              f"Epistemic={val_epistemic_mean:.4f}, "
              f"Aleatoric={val_aleatoric_mean:.4f}")

    return model
