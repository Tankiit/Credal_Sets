"""
Fast Variational Credal CBM training using token embeddings from any model
Integration of VariationalCredalCBM with the existing token embeddings pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import argparse
import os
import json
from run_token_embeddings import (
    TokenEmbeddingProcessor, DATASETS, ENCODERS,
    encode_concept, convert_cebab_concepts_to_numerical,
    calculate_concept_accuracy, expected_calibration_error,
    calculate_concept_ece
)
from variational_credal_cbm import (
    VariationalCredalCBM, VariationalConfig, CredalSet,
    compute_calibration_metrics, epistemic_error_correlation
)


class VariationalCBMIntegrator:
    """
    Integrates VariationalCredalCBM with token embedding pipeline
    """

    def __init__(self,
                 encoder_name: str = "distilbert-base-uncased",
                 variational_family: str = "mean_field",
                 low_rank_dim: int = 5,
                 num_concepts: int = 20,
                 num_classes: int = 2,
                 kl_weight: float = 1e-5,
                 num_mc_samples: int = 10,
                 device: str = "auto"):
        """
        Initialize the integrator

        Args:
            encoder_name: HuggingFace model name
            variational_family: "mean_field" or "low_rank"
            low_rank_dim: Rank for low-rank approximation
            num_concepts: Number of concepts to model
            num_classes: Number of output classes
            kl_weight: KL divergence regularization weight
            num_mc_samples: Monte Carlo samples for uncertainty
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.config = VariationalConfig(
            encoder_name=encoder_name,
            num_concepts=num_concepts,
            num_classes=num_classes,
            variational_family=variational_family,
            low_rank_dim=low_rank_dim,
            kl_weight=kl_weight,
            num_mc_samples=num_mc_samples
        )

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize model
        self.model = VariationalCredalCBM(self.config).to(self.device)

        # Initialize tokenizer for preprocessing
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_data(self, texts: List[str], labels: List[int] = None):
        """
        Prepare data for training/inference

        Args:
            texts: List of input texts
            labels: Optional list of labels

        Returns:
            Dict with tokenized inputs
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        result = {
            'input_ids': inputs['input_ids'].to(self.device),
            'attention_mask': inputs['attention_mask'].to(self.device)
        }

        if labels is not None:
            result['labels'] = torch.tensor(labels, dtype=torch.long).to(self.device)

        return result

    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str] = None, val_labels: List[int] = None,
              num_epochs: int = 10, batch_size: int = 16,
              learning_rate: float = 2e-5, warmup_steps: int = 100,
              checkpoint_dir: str = "./checkpoints", save_every: int = 5):
        """
        Train the variational credal CBM

        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for learning rate schedule
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs

        Returns:
            Training history
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        from transformers import get_linear_schedule_with_warmup
        from torch.utils.data import DataLoader, TensorDataset

        # Prepare training data
        train_data = self.prepare_data(train_texts, train_labels)

        # Create DataLoader
        train_dataset = TensorDataset(
            train_data['input_ids'],
            train_data['attention_mask'],
            train_data['labels']
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Training history
        history = {
            'train_loss': [],
            'train_ce_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # Validation data preparation
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_data = self.prepare_data(val_texts, val_labels)
            val_dataset = TensorDataset(
                val_data['input_ids'],
                val_data['attention_mask'],
                val_data['labels']
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Starting training for {num_epochs} epochs...")

        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            total_ce_loss = 0
            total_kl_loss = 0
            num_batches = 0

            # Training batches
            for batch_input_ids, batch_attention_mask, batch_labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
            ):
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )

                # Backward pass
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Track losses
                total_loss += loss.item()
                total_ce_loss += outputs['ce_loss'].item()
                total_kl_loss += outputs['kl_loss'].item()
                num_batches += 1

            # Calculate average losses
            avg_train_loss = total_loss / num_batches
            avg_ce_loss = total_ce_loss / num_batches
            avg_kl_loss = total_kl_loss / num_batches

            history['train_loss'].append(avg_train_loss)
            history['train_ce_loss'].append(avg_ce_loss)
            history['train_kl_loss'].append(avg_kl_loss)

            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])

                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                      f"Val Loss={val_metrics['loss']:.4f}, "
                      f"Val Acc={val_metrics['accuracy']:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': self.config,
                    'history': history,
                    'val_metrics': val_metrics if val_loader is not None else None
                }, checkpoint_path)
                print(f"Checkpoint saved to: {checkpoint_path}")

        # Save final model
        final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': history
        }, final_checkpoint_path)
        print(f"Final model saved to: {final_checkpoint_path}")

        return history

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch number and history
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'history' in checkpoint:
            print(f"Training history loaded with {len(checkpoint['history']['train_loss'])} epochs")

        return checkpoint.get('epoch', 0), checkpoint.get('history', {})

    def evaluate(self, data_loader):
        """
        Evaluate the model on a DataLoader

        Args:
            data_loader: DataLoader with (input_ids, attention_mask, labels)

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_concept_probs = []
        all_epistemic_uncertainties = []

        with torch.no_grad():
            for batch_input_ids, batch_attention_mask, batch_labels in data_loader:
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )

                total_loss += outputs['loss'].item()

                # Collect predictions and labels
                preds = outputs['predictions'].cpu().numpy()
                labels = batch_labels.cpu().numpy()

                all_predictions.extend(preds)
                all_labels.extend(labels)
                all_concept_probs.append(outputs['concept_probs'].cpu().numpy())
                all_epistemic_uncertainties.append(outputs['epistemic_uncertainty'].cpu().numpy())

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_micro = f1_score(all_labels, all_predictions, average='micro')

        # Combine concept probabilities
        all_concept_probs = np.vstack(all_concept_probs)
        all_epistemic_uncertainties = np.vstack(all_epistemic_uncertainties)

        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'concept_probs': all_concept_probs,
            'epistemic_uncertainties': all_epistemic_uncertainties,
            'predictions': all_predictions,
            'labels': all_labels
        }

    def predict_with_uncertainty(self, texts: List[str]):
        """
        Make predictions with uncertainty quantification

        Args:
            texts: List of input texts

        Returns:
            Dict with predictions and uncertainties
        """
        self.model.eval()
        data = self.prepare_data(texts)

        with torch.no_grad():
            outputs = self.model(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask']
            )

        # Convert to numpy
        concept_probs = outputs['concept_probs'].cpu().numpy()
        epistemic_uncertainty = outputs['epistemic_uncertainty'].cpu().numpy()
        aleatoric_uncertainty = outputs['aleatoric_uncertainty'].cpu().numpy()
        predictions = outputs['predictions'].cpu().numpy()
        logits = outputs['logits'].cpu().numpy()

        # Create credal sets
        credal_sets = CredalSet.from_uncertainty(
            mean=torch.from_numpy(concept_probs),
            epistemic=torch.from_numpy(epistemic_uncertainty),
            aleatoric=torch.from_numpy(aleatoric_uncertainty),
            confidence_level=0.95
        )

        return {
            'predictions': predictions,
            'concept_probs': concept_probs,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'credal_lower': credal_sets.lower.cpu().numpy(),
            'credal_upper': credal_sets.upper.cpu().numpy(),
            'credal_width': credal_sets.width.cpu().numpy(),
            'logits': logits
        }


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Variational Credal CBM with token embeddings')

    # Model arguments
    parser.add_argument('--model', type=str,
                       default='distilbert-base-uncased',
                       help='Hugging Face model name (or use --encoder for preset)')
    parser.add_argument('--encoder', type=str, default='distilbert',
                       choices=list(ENCODERS.keys()),
                       help='Encoder preset name (overrides --model)')
    parser.add_argument('--variational-family', type=str, default='mean_field',
                       choices=['mean_field', 'low_rank'],
                       help='Variational family to use')
    parser.add_argument('--low-rank-dim', type=int, default=5,
                       help='Low-rank dimension (only for low_rank family)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='sst2',
                       choices=['sst2', 'imdb', 'cebab', 'hatexplain', 'goemotions', 'ag_news', 'civil_comments', 'snli'],
                       help='Dataset name')
    parser.add_argument('--train-size', type=int, default=None,
                       help='Number of training samples to use')
    parser.add_argument('--test-size', type=int, default=None,
                       help='Number of test samples to use')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Proportion of data to use for test set')

    # Training arguments
    parser.add_argument('--n-concepts', type=int, default=20,
                       help='Number of concepts to model')
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--kl-weight', type=float, default=1e-5,
                       help='KL divergence regularization weight')
    parser.add_argument('--num-mc-samples', type=int, default=10,
                       help='Monte Carlo samples for uncertainty')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./results_variational',
                       help='Output directory for results')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_variational',
                       help='Directory to save model checkpoints')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Resolve encoder if specified
    if args.encoder in ENCODERS:
        model_name = ENCODERS[args.encoder]
        print(f"Using encoder preset: {args.encoder} -> {model_name}")
    else:
        model_name = args.model
        print(f"Using custom model: {model_name}")

    # Get dataset config
    dataset_config = DATASETS[args.dataset]
    print(f"Dataset: {dataset_config.name}")
    print(f"Task: {dataset_config.task}")

    # Load dataset
    try:
        if dataset_config.name == 'sst2':
            ds = load_dataset("glue", "sst2")
            train_texts = ds["train"]["sentence"]
            train_labels = ds["train"]["label"]
            test_texts = ds["validation"]["sentence"]
            test_labels = ds["validation"]["label"]
            num_classes = 2

        elif dataset_config.name == 'imdb':
            ds = load_dataset("imdb")
            train_texts = ds["train"]["text"]
            train_labels = ds["train"]["label"]
            test_texts = ds["test"]["text"]
            test_labels = ds["test"]["label"]
            num_classes = 2

        elif dataset_config.name == 'cebab':
            ds = load_dataset("CEBaB/CEBaB")
            train_split = ds["train_inclusive"]
            test_split = ds["test"]

            train_texts = train_split["description"]
            test_texts = test_split["description"]

            # Convert CEBaB labels to numeric
            def convert_cebab_label(label):
                if isinstance(label, str):
                    if label.lower() in ['positive', '1']:
                        return 1
                    elif label.lower() in ['negative', '0']:
                        return 0
                    else:
                        return 1  # Default to positive
                return int(label)

            train_labels = [convert_cebab_label(l) for l in train_split["review_majority"]]
            test_labels = [convert_cebab_label(l) for l in test_split["review_majority"]]
            num_classes = 2

        elif dataset_config.name == 'ag_news':
            ds = load_dataset("ag_news")
            train_texts = ds["train"]["text"]
            train_labels = ds["train"]["label"]
            test_texts = ds["test"]["text"]
            test_labels = ds["test"]["label"]
            num_classes = 4

        else:
            # For other datasets, use a subset for demonstration
            print(f"Using generic loader for {dataset_config.name}")
            ds = load_dataset(dataset_config.hf_path)

            # Handle different dataset structures
            if "train" in ds and "test" in ds:
                train_split = ds["train"]
                test_split = ds["test"]
            else:
                # Use train/validation split
                train_split = ds["train"]
                test_split = ds["validation"]

            train_texts = train_split[dataset_config.text_col]
            test_texts = test_split[dataset_config.text_col]
            train_labels = train_split[dataset_config.label_col]
            test_labels = test_split[dataset_config.label_col]
            num_classes = len(set(train_labels + test_labels))

        # Limit samples if requested
        if args.train_size is not None:
            train_texts = train_texts[:args.train_size]
            train_labels = train_labels[:args.train_size]

        if args.test_size is not None:
            test_texts = test_texts[:args.test_size]
            test_labels = test_labels[:args.test_size]

        print(f"Train: {len(train_texts)} samples, Test: {len(test_texts)} samples")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize integrator
    integrator = VariationalCBMIntegrator(
        encoder_name=model_name,
        variational_family=args.variational_family,
        low_rank_dim=args.low_rank_dim,
        num_concepts=args.n_concepts,
        num_classes=num_classes,
        kl_weight=args.kl_weight,
        num_mc_samples=args.num_mc_samples
    )

    # Split train into train/validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_checkpoint:
        try:
            start_epoch, loaded_history = integrator.load_checkpoint(args.resume_checkpoint)
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
            start_epoch = 0

    # Train model
    history = integrator.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every
    )

    # Evaluate on test set
    test_data = integrator.prepare_data(test_texts, test_labels)
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(
        test_data['input_ids'],
        test_data['attention_mask'],
        test_data['labels']
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    test_results = integrator.evaluate(test_loader)

    # Get predictions with uncertainty
    predictions = integrator.predict_with_uncertainty(test_texts)

    # Calculate additional metrics
    mean_epistemic = predictions['epistemic_uncertainty'].mean()
    mean_aleatoric = predictions['aleatoric_uncertainty'].mean()
    mean_credal_width = predictions['credal_width'].mean()

    # Error correlation
    errors = (test_results['predictions'] != test_results['labels']).astype(float)
    error_correlation = epistemic_error_correlation(
        torch.from_numpy(predictions['epistemic_uncertainty']),
        torch.from_numpy(errors)
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    output = {
        'model_name': model_name,
        'dataset': args.dataset,
        'variational_family': args.variational_family,
        'low_rank_dim': args.low_rank_dim,
        'n_concepts': args.n_concepts,
        'n_classes': num_classes,
        'train_samples': len(train_texts),
        'val_samples': len(val_texts),
        'test_samples': len(test_texts),
        'num_epochs': args.num_epochs,
        'test_accuracy': test_results['accuracy'],
        'test_f1_macro': test_results['f1_macro'],
        'test_f1_micro': test_results['f1_micro'],
        'mean_epistemic_uncertainty': float(mean_epistemic),
        'mean_aleatoric_uncertainty': float(mean_aleatoric),
        'mean_credal_width': float(mean_credal_width),
        'error_correlation': float(error_correlation),
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'final_val_accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else None,
        'history': history
    }

    output_file = os.path.join(args.output_dir,
                              f"{args.dataset}_{args.variational_family}_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Final Results ===")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_results['f1_macro']:.4f}")
    print(f"Mean Epistemic Uncertainty: {mean_epistemic:.4f}")
    print(f"Mean Aleatoric Uncertainty: {mean_aleatoric:.4f}")
    print(f"Mean Credal Width: {mean_credal_width:.4f}")
    print(f"Error-Uncertainty Correlation: {error_correlation:.4f}")
    print(f"Results saved to: {output_file}")

    return output


if __name__ == "__main__":
    main()