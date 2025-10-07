"""
Training Script for Legal Document Role Classification

This script provides comprehensive training functionality for both InLegalBERT
and BiLSTM-CRF models using custom datasets.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from data_loader import create_data_loaders, LegalDocumentDataset
import sys
# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from role_classifier import InLegalBERTClassifier, BiLSTMCRFClassifier, RhetoricalRole

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoleClassifierTrainer:
    """
    Trainer class for legal document role classification models
    """
    
    def __init__(self, 
                 model_type: str = "inlegalbert",
                 model_name: str = "law-ai/InLegalBERT",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 output_dir: str = "./outputs",
                 num_labels: int = 7):
        """
        Initialize the trainer
        
        Args:
            model_type: Type of model ("inlegalbert" or "bilstm_crf")
            model_name: Name of the pre-trained model
            device: Device to train on
            output_dir: Directory to save outputs
            num_labels: Number of classification labels
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_labels = num_labels
        
        # Initialize model
        self.model = self._initialize_model()
        self.model.to(self.device)
        
        # Initialize tokenizer
        if model_type == "inlegalbert":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'epoch': []
        }
        
        logger.info(f"Initialized {model_type} trainer on {device}")
    
    def _initialize_model(self):
        """Initialize the model based on type"""
        if self.model_type == "inlegalbert":
            return InLegalBERTClassifier(
                model_name=self.model_name,
                num_labels=self.num_labels
            )
        elif self.model_type == "bilstm_crf":
            return BiLSTMCRFClassifier(
                num_labels=self.num_labels
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self,
              train_data_path: str,
              val_data_path: str = None,
              test_data_path: str = None,
              context_mode: str = "prev",
              batch_size: int = 16,
              num_epochs: int = 10,
              learning_rate: float = 2e-5,
              weight_decay: float = 0.01,
              warmup_steps: int = 500,
              max_length: int = 512,
              save_best_model: bool = True,
              evaluation_strategy: str = "epoch",
              dataset_sample_ratio: float = 1.0):
        """
        Train the model
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            test_data_path: Path to test data
            context_mode: Context mode for training
            batch_size: Batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps for scheduler
            max_length: Maximum sequence length
            save_best_model: Whether to save the best model
            evaluation_strategy: When to evaluate ("epoch" or "steps")
        """
        
        # Create data loaders
        logger.info("Creating data loaders...")
        data_loaders = create_data_loaders(
            train_path=train_data_path,
            val_path=val_data_path,
            test_path=test_data_path,
            tokenizer_name=self.model_name,
            context_mode=context_mode,
            batch_size=batch_size,
            max_length=max_length,
            dataset_sample_ratio=dataset_sample_ratio
        )
        
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        test_loader = data_loaders.get('test', None)
        
        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Initialize TensorBoard writer
        writer = SummaryWriter(self.output_dir / "tensorboard_logs")
        
        # Training loop
        best_val_f1 = 0.0
        global_step = 0
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_loss, train_f1 = self._train_epoch(
                train_loader, optimizer, scheduler, criterion, epoch, writer
            )
            
            # Validation phase
            val_loss, val_f1 = self._evaluate(val_loader, criterion, epoch, writer, "val")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['epoch'].append(epoch + 1)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if save_best_model and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self._save_model("best_model")
                logger.info(f"New best model saved with Val F1: {val_f1:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(epoch, optimizer, scheduler, val_f1)
        
        # Final evaluation on test set
        if test_loader:
            logger.info("Evaluating on test set...")
            test_loss, test_f1 = self._evaluate(test_loader, criterion, epoch, writer, "test")
            logger.info(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")
        
        # Save final model
        self._save_model("final_model")
        
        # Save training history
        self._save_training_history()
        
        # Plot training curves
        self._plot_training_curves()
        
        writer.close()
        logger.info("Training completed!")
    
    def _train_epoch(self, train_loader, optimizer, scheduler, criterion, epoch, writer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if self.model_type == "inlegalbert":
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                predictions = torch.argmax(logits, dim=-1)
            else:
                # For BiLSTM-CRF, implement specific forward pass
                # This would require sentence embeddings instead of token-level inputs
                pass
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            if batch_idx % 50 == 0:
                writer.add_scalar('Loss/Train_Step', loss.item(), 
                                epoch * len(train_loader) + batch_idx)
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, f1
    
    def _evaluate(self, data_loader, criterion, epoch, writer, split_name):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if self.model_type == "inlegalbert":
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                    predictions = torch.argmax(logits, dim=-1)
                
                # Update metrics
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Log to TensorBoard
        writer.add_scalar(f'Loss/{split_name.capitalize()}', avg_loss, epoch)
        writer.add_scalar(f'F1/{split_name.capitalize()}', f1, epoch)
        
        # Generate classification report for validation
        if split_name == "val" and epoch % 5 == 0:  # Every 5 epochs
            role_names = [role.value for role in RhetoricalRole]
            report = classification_report(
                all_labels, all_predictions, 
                target_names=role_names[:self.num_labels],
                output_dict=True
            )
            
            # Save classification report
            report_path = self.output_dir / f"classification_report_epoch_{epoch}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return avg_loss, f1
    
    def _save_model(self, model_name: str):
        """Save model checkpoint"""
        model_path = self.output_dir / f"{model_name}.pt"
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'history': self.history
        }
        
        torch.save(save_dict, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def _save_checkpoint(self, epoch, optimizer, scheduler, val_f1):
        """Save training checkpoint"""
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_f1': val_f1,
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def _save_training_history(self):
        """Save training history"""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _plot_training_curves(self):
        """Plot and save training curves"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # F1 Score plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history['epoch'], self.history['train_f1'], label='Train F1')
        plt.plot(self.history['epoch'], self.history['val_f1'], label='Val F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png")
        plt.close()
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        logger.info(f"Model loaded from {model_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Legal Document Role Classifier")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--val_data", type=str, default=None,
                       help="Path to validation data")
    parser.add_argument("--test_data", type=str, default=None,
                       help="Path to test data")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="inlegalbert",
                       choices=["inlegalbert", "bilstm_crf"],
                       help="Type of model to train")
    parser.add_argument("--model_name", type=str, default="law-ai/InLegalBERT",
                       help="Pre-trained model name")
    parser.add_argument("--context_mode", type=str, default="prev",
                       choices=["single", "prev", "prev_two", "surrounding"],
                       help="Context mode for training")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model_type}_{timestamp}"
    
    # Initialize trainer
    trainer = RoleClassifierTrainer(
        model_type=args.model_type,
        model_name=args.model_name,
        device=device,
        output_dir=str(output_dir)
    )
    
    # Start training
    trainer.train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        test_data_path=args.test_data,
        context_mode=args.context_mode,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main()