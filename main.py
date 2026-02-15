import argparse
import torch
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict

from dataloader import load_dataset_splits, DatasetConfig
from model import (
    VariationalCredalCBM,
    ModelConfig,
    create_trainer,
    add_model_config_args
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class CheckpointLogger(Callback):
    """Callback to log checkpoint saves"""
    
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.logger = logger
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log at end of each training epoch"""
        metrics = trainer.callback_metrics
        if 'train/loss' in metrics:
            self.logger.info(
                f"Epoch {trainer.current_epoch}: "
                f"train_loss={metrics['train/loss'].item():.4f}, "
                f"train_acc={metrics.get('train/acc', torch.tensor(0.0)).item():.4f}, "
                f"val_loss={metrics.get('val/loss', torch.tensor(0.0)).item():.4f}, "
                f"val_acc={metrics.get('val/acc', torch.tensor(0.0)).item():.4f}"
            )
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Log when checkpoint is saved"""
        checkpoint_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer.checkpoint_callback, 'best_model_path') else None
        if checkpoint_path:
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to both file and console"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def save_training_summary(output_dir: str, args, model_config: ModelConfig, metadata: Dict, trainer=None):
    """Save training summary to JSON file"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'name': args.dataset,
            'label_type': args.label_type,
            'train_size': metadata['train_size'],
            'val_size': metadata['val_size'],
            'test_size': metadata['test_size'],
            'num_classes': metadata['num_classes'],
            'num_concepts': metadata['num_concepts'],
            'concept_names': metadata['concept_names']
        },
        'model': {
            'encoder_name': model_config.encoder_name,
            'freeze_encoder': model_config.freeze_encoder,
            'variational_family': model_config.variational_family,
            'num_mc_samples': model_config.num_mc_samples,
            'prior_std': model_config.prior_std,
            'kl_weight': model_config.kl_weight,
            'concept_weight': model_config.concept_weight,
            'aleatoric_weight': model_config.aleatoric_weight
        },
        'training': {
            'max_epochs': args.max_epochs,
            'batch_size': args.batch_size,
            'learning_rate': model_config.learning_rate,
            'weight_decay': model_config.weight_decay,
            'warmup_ratio': model_config.warmup_ratio,
            'gradient_clip_val': args.gradient_clip_val,
            'early_stopping_patience': args.early_stopping_patience,
            'accelerator': args.accelerator,
            'devices': args.devices,
            'precision': args.precision
        },
        'checkpoints': {
            'directory': str(Path(output_dir) / "checkpoints"),
            'save_every_n_epochs': args.save_every_n_epochs
        }
    }
    
    # Add final metrics if trainer is provided
    if trainer and hasattr(trainer, 'callback_metrics'):
        metrics = trainer.callback_metrics
        summary['final_metrics'] = {
            'train_loss': float(metrics.get('train/loss', torch.tensor(0.0)).item()),
            'train_acc': float(metrics.get('train/acc', torch.tensor(0.0)).item()),
            'val_loss': float(metrics.get('val/loss', torch.tensor(0.0)).item()),
            'val_acc': float(metrics.get('val/acc', torch.tensor(0.0)).item()),
            'test_acc': float(metrics.get('test/acc', torch.tensor(0.0)).item()),
            'test_f1': float(metrics.get('test/f1', torch.tensor(0.0)).item())
        }
    
    # Save summary
    summary_file = Path(output_dir) / "logs" / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_file


def log_config(logger: logging.Logger, args, model_config: ModelConfig, metadata: Dict):
    """Log configuration details"""
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    
    logger.info("\nDataset Configuration:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Label type: {args.label_type}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Tokenizer: {args.tokenizer_name}")
    logger.info(f"  Train size: {metadata['train_size']:,}")
    logger.info(f"  Val size: {metadata['val_size']:,}")
    logger.info(f"  Test size: {metadata['test_size']:,}")
    logger.info(f"  Num classes: {metadata['num_classes']}")
    logger.info(f"  Num concepts: {metadata['num_concepts']}")
    logger.info(f"  Concept names: {metadata['concept_names']}")
    
    logger.info("\nModel Configuration:")
    logger.info(f"  Encoder: {model_config.encoder_name}")
    logger.info(f"  Freeze encoder: {model_config.freeze_encoder}")
    logger.info(f"  Variational family: {model_config.variational_family}")
    logger.info(f"  MC samples: {model_config.num_mc_samples}")
    logger.info(f"  Prior std: {model_config.prior_std}")
    logger.info(f"  KL weight: {model_config.kl_weight}")
    logger.info(f"  Concept weight: {model_config.concept_weight}")
    logger.info(f"  Aleatoric weight: {model_config.aleatoric_weight}")
    
    logger.info("\nTraining Configuration:")
    logger.info(f"  Max epochs: {args.max_epochs}")
    logger.info(f"  Learning rate: {model_config.learning_rate}")
    logger.info(f"  Weight decay: {model_config.weight_decay}")
    logger.info(f"  Warmup ratio: {model_config.warmup_ratio}")
    logger.info(f"  Gradient clip: {args.gradient_clip_val}")
    logger.info(f"  Early stopping patience: {args.early_stopping_patience}")
    logger.info(f"  LR Scheduler: {'Enabled' if model_config.use_lr_scheduler else 'Disabled'}")
    if model_config.use_lr_scheduler:
        logger.info(f"    - Factor: {model_config.lr_scheduler_factor}")
        logger.info(f"    - Patience: {model_config.lr_scheduler_patience} epochs")
        logger.info(f"    - Min LR: {model_config.lr_scheduler_min_lr}")
        logger.info(f"    - Mode: {model_config.lr_scheduler_mode}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Accelerator: {args.accelerator}")
    logger.info(f"  Devices: {args.devices}")
    logger.info(f"  Precision: {args.precision}")
    
    logger.info("=" * 60)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Variational Credal CBM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cebab',
                       choices=['cebab', 'hatexplain', 'goemotions', 'civil_comments', 'sst2', 'ag_news'],
                       help='Dataset name')
    parser.add_argument('--label_type', type=str, default='default',
                       help='Label type (default uses dataset default)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for dataloaders')
    parser.add_argument('--tokenizer_name', type=str, default='distilbert-base-uncased',
                       help='Tokenizer name')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per split (for debugging)')
    
    # Model config arguments (will be added)
    parser = add_model_config_args(parser)
    
    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--accelerator', type=str, default='auto',
                       choices=['auto', 'gpu', 'cpu', 'mps'],
                       help='Accelerator type')
    parser.add_argument('--devices', type=int, default=1,
                       help='Number of devices')
    parser.add_argument('--precision', type=str, default='16-mixed',
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Early stopping patience (set to -1 to disable early stopping)')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--log_every_n_steps', type=int, default=10,
                       help='Logging frequency')
    parser.add_argument('--save_every_n_epochs', type=int, default=10,
                       help='Save checkpoint every N epochs (in addition to best checkpoints)')
    
    # Inference arguments
    parser.add_argument('--test_only', action='store_true',
                       help='Only run testing (requires checkpoint)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint for testing')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting Variational Credal CBM Training")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load dataset and get metadata
    data_config = DatasetConfig(
        label_type=args.label_type if args.label_type != 'default' else 'default',
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length
    )
    
    logger.info(f"Loading dataset: {args.dataset}")
    train_loader, val_loader, test_loader, tokenizer, metadata = load_dataset_splits(
        args.dataset,
        config=data_config,
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size
    )
    logger.info(f"Dataset loaded successfully")
    
    # 2. Create model config from args, then update with metadata
    model_config = ModelConfig.from_args(args)
    
    # Override with dataset metadata
    model_config.num_classes = metadata['num_classes']
    model_config.num_concepts = metadata['num_concepts']
    model_config.max_length = args.max_length
    
    # Log configuration
    log_config(logger, args, model_config, metadata)
    
    # 3. Create model
    logger.info("Creating model...")
    model = VariationalCredalCBM(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # 4. Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        early_stopping_patience=args.early_stopping_patience,
        log_every_n_steps=args.log_every_n_steps,
        save_every_n_epochs=args.save_every_n_epochs
    )
    
    # Add checkpoint logger callback
    checkpoint_logger = CheckpointLogger(logger)
    trainer.callbacks.append(checkpoint_logger)
    logger.info("Trainer created with checkpoint logging")
    
    # 5. Train or test
    if args.test_only:
        logger.info("Running test only mode...")
        if args.checkpoint_path is None:
            # Try to find best checkpoint
            checkpoint_dir = Path(args.output_dir) / "checkpoints"
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoints:
                args.checkpoint_path = str(checkpoints[0])
                logger.info(f"Using checkpoint: {args.checkpoint_path}")
            else:
                raise ValueError("No checkpoint found. Please specify --checkpoint_path")
        
        logger.info("Starting testing...")
        trainer.test(model, test_loader, ckpt_path=args.checkpoint_path)
        logger.info("Testing completed")
    else:
        logger.info("Starting training...")
        logger.info(f"Training will run for up to {args.max_epochs} epochs")
        if args.early_stopping_patience > 0:
            logger.info(f"Early stopping enabled with patience={args.early_stopping_patience} epochs")
        else:
            logger.info(f"Early stopping disabled - will train for full {args.max_epochs} epochs")
        
        trainer.fit(model, train_loader, val_loader)
        
        # Log training completion details
        if trainer.current_epoch < args.max_epochs - 1:
            logger.info(f"Training stopped early at epoch {trainer.current_epoch + 1} (out of {args.max_epochs} max)")
            logger.info("This is likely due to early stopping - validation accuracy did not improve")
            logger.info(f"Best validation accuracy: {trainer.callback_metrics.get('val/acc', torch.tensor(0.0)).item():.4f}")
        else:
            logger.info(f"Training completed all {args.max_epochs} epochs")
        
        # Log checkpoint information
        checkpoint_dir = Path(args.output_dir) / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        logger.info(f"Checkpoints saved: {len(checkpoints)}")
        for ckpt in checkpoints:
            logger.info(f"  - {ckpt.name}")
        
        logger.info("Starting testing...")
        trainer.test(model, test_loader)
        logger.info("Testing completed")
        
        # Save training summary
        summary_file = save_training_summary(args.output_dir, args, model_config, metadata, trainer)
        logger.info(f"Training summary saved to: {summary_file}")
    
    logger.info("=" * 60)
    logger.info("Training session completed")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

