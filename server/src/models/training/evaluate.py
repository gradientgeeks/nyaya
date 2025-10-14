"""
Evaluation and Testing Script for Legal Document Role Classification

This script provides comprehensive evaluation functionality including
metrics calculation, confusion matrix generation, and error analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

# Import our custom modules
from data_loader import LegalDocumentDataset
from train import RoleClassifierTrainer
import sys
import os
# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from role_classifier import RhetoricalRole

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation class
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to the trained model
            device: Device to run evaluation on
        """
        self.device = device
        self.model_path = model_path
        
        # Load model
        self.trainer = self._load_model()
        
        # Role names for reporting
        self.role_names = [role.value for role in RhetoricalRole]
        
        logger.info(f"Model evaluator initialized with {model_path}")
    
    def _load_model(self):
        """Load the trained model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Initialize trainer with saved config
        trainer = RoleClassifierTrainer(
            model_type=checkpoint['model_type'],
            model_name=checkpoint['model_name'],
            device=self.device,
            num_labels=checkpoint['num_labels']
        )
        
        # Load model weights
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()
        
        return trainer
    
    def evaluate_dataset(self, 
                        test_data_path: str,
                        context_mode: str = "prev",
                        batch_size: int = 16,
                        max_length: int = 512,
                        output_dir: str = "./evaluation_results") -> dict:
        """
        Evaluate model on a test dataset
        
        Args:
            test_data_path: Path to test data
            context_mode: Context mode used during training
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
            output_dir: Directory to save results
        
        Returns:
            Dictionary containing evaluation metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test dataset
        test_dataset = LegalDocumentDataset(
            data_path=test_data_path,
            tokenizer_name=self.trainer.model_name,
            context_mode=context_mode,
            max_length=max_length,
            test_split=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        logger.info(f"Evaluating on {len(test_dataset)} samples...")
        
        # Get predictions
        all_predictions, all_labels, all_probs = self._get_predictions(test_loader)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probs)
        
        # Generate detailed reports
        self._generate_classification_report(
            all_labels, all_predictions, output_dir
        )
        self._generate_confusion_matrix(
            all_labels, all_predictions, output_dir
        )
        self._generate_per_role_analysis(
            all_labels, all_predictions, all_probs, output_dir
        )
        
        # Error analysis
        self._error_analysis(
            test_dataset, all_predictions, all_labels, output_dir
        )
        
        # Save metrics
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        
        return metrics
    
    def _get_predictions(self, data_loader):
        """Get model predictions on the data loader"""
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Getting predictions"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if self.trainer.model_type == "inlegalbert":
                    logits = self.trainer.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
                    probs = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return all_predictions, all_labels, all_probs
    
    def _calculate_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'macro_f1': f1_score(true_labels, predictions, average='macro'),
            'weighted_f1': f1_score(true_labels, predictions, average='weighted'),
            'macro_precision': precision_score(true_labels, predictions, average='macro'),
            'weighted_precision': precision_score(true_labels, predictions, average='weighted'),
            'macro_recall': recall_score(true_labels, predictions, average='macro'),
            'weighted_recall': recall_score(true_labels, predictions, average='weighted')
        }
        
        # Per-class metrics
        per_class_f1 = f1_score(true_labels, predictions, average=None)
        per_class_precision = precision_score(true_labels, predictions, average=None)
        per_class_recall = recall_score(true_labels, predictions, average=None)
        
        metrics['per_class'] = {}
        for i, role_name in enumerate(self.role_names[:len(per_class_f1)]):
            metrics['per_class'][role_name] = {
                'f1': float(per_class_f1[i]),
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i])
            }
        
        return metrics
    
    def _generate_classification_report(self, true_labels, predictions, output_dir):
        """Generate and save classification report"""
        report = classification_report(
            true_labels, 
            predictions,
            target_names=self.role_names[:max(max(true_labels), max(predictions)) + 1],
            output_dict=True
        )
        
        # Save as JSON
        report_path = output_dir / "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save as text
        report_text = classification_report(
            true_labels, 
            predictions,
            target_names=self.role_names[:max(max(true_labels), max(predictions)) + 1]
        )
        
        report_text_path = output_dir / "classification_report.txt"
        with open(report_text_path, 'w') as f:
            f.write(report_text)
        
        logger.info("Classification report saved")
    
    def _generate_confusion_matrix(self, true_labels, predictions, output_dir):
        """Generate and save confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        num_classes = len(np.unique(true_labels))
        role_labels = self.role_names[:num_classes]
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=role_labels,
            yticklabels=role_labels
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save confusion matrix as CSV
        cm_df = pd.DataFrame(cm, columns=role_labels, index=role_labels)
        cm_csv_path = output_dir / "confusion_matrix.csv"
        cm_df.to_csv(cm_csv_path)
        
        logger.info("Confusion matrix saved")
    
    def _generate_per_role_analysis(self, true_labels, predictions, probabilities, output_dir):
        """Generate per-role analysis"""
        num_classes = len(np.unique(true_labels))
        role_labels = self.role_names[:num_classes]
        
        analysis_data = []
        
        for i, role in enumerate(role_labels):
            # Get indices for this role
            role_indices = np.where(np.array(true_labels) == i)[0]
            
            if len(role_indices) > 0:
                role_predictions = np.array(predictions)[role_indices]
                role_probs = np.array(probabilities)[role_indices]
                
                # Calculate metrics for this role
                correct_predictions = np.sum(role_predictions == i)
                total_predictions = len(role_indices)
                accuracy = correct_predictions / total_predictions
                
                # Average confidence for correct predictions
                correct_mask = role_predictions == i
                if np.sum(correct_mask) > 0:
                    avg_confidence_correct = np.mean(role_probs[correct_mask, i])
                else:
                    avg_confidence_correct = 0.0
                
                # Average confidence for incorrect predictions
                incorrect_mask = role_predictions != i
                if np.sum(incorrect_mask) > 0:
                    avg_confidence_incorrect = np.mean(
                        np.max(role_probs[incorrect_mask], axis=1)
                    )
                else:
                    avg_confidence_incorrect = 0.0
                
                analysis_data.append({
                    'role': role,
                    'total_samples': total_predictions,
                    'correct_predictions': correct_predictions,
                    'accuracy': accuracy,
                    'avg_confidence_correct': avg_confidence_correct,
                    'avg_confidence_incorrect': avg_confidence_incorrect
                })
        
        # Save analysis
        analysis_df = pd.DataFrame(analysis_data)
        analysis_path = output_dir / "per_role_analysis.csv"
        analysis_df.to_csv(analysis_path, index=False)
        
        # Plot role-wise accuracy
        plt.figure(figsize=(12, 6))
        plt.bar(analysis_df['role'], analysis_df['accuracy'])
        plt.title('Per-Role Accuracy')
        plt.xlabel('Role')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        accuracy_plot_path = output_dir / "per_role_accuracy.png"
        plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Per-role analysis saved")
    
    def _error_analysis(self, dataset, predictions, true_labels, output_dir):
        """Perform detailed error analysis"""
        errors = []
        
        for i, (pred, true_label) in enumerate(zip(predictions, true_labels)):
            if pred != true_label:
                sample = dataset[i]
                errors.append({
                    'sample_index': i,
                    'text': sample['text'],
                    'original_sentence': sample['original_sentence'],
                    'true_label': self.role_names[true_label],
                    'predicted_label': self.role_names[pred],
                    'true_label_id': true_label,
                    'predicted_label_id': pred
                })
        
        # Save errors to CSV
        if errors:
            errors_df = pd.DataFrame(errors)
            errors_path = output_dir / "error_analysis.csv"
            errors_df.to_csv(errors_path, index=False)
            
            # Error statistics
            error_stats = {
                'total_errors': len(errors),
                'total_samples': len(predictions),
                'error_rate': len(errors) / len(predictions),
                'most_confused_pairs': {}
            }
            
            # Find most confused pairs
            confusion_pairs = {}
            for error in errors:
                pair = f"{error['true_label']} -> {error['predicted_label']}"
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
            
            # Sort by frequency
            sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
            error_stats['most_confused_pairs'] = dict(sorted_pairs[:10])  # Top 10
            
            # Save error statistics
            error_stats_path = output_dir / "error_statistics.json"
            with open(error_stats_path, 'w') as f:
                json.dump(error_stats, f, indent=2)
            
            logger.info(f"Error analysis completed. Found {len(errors)} errors out of {len(predictions)} samples")
        else:
            logger.info("No errors found in predictions!")
    
    def predict_single(self, text: str, context_mode: str = "prev") -> dict:
        """
        Predict role for a single text
        
        Args:
            text: Input text
            context_mode: Context mode
        
        Returns:
            Dictionary with prediction results
        """
        # Tokenize
        encoding = self.trainer.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.trainer.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
        
        # Format results
        predicted_label = self.role_names[prediction.item()]
        confidence = probs[0, prediction.item()].item()
        
        # Get top-k predictions
        top_k = 3
        top_probs, top_indices = torch.topk(probs[0], top_k)
        top_predictions = [
            {
                'role': self.role_names[idx.item()],
                'confidence': prob.item()
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return {
            'text': text,
            'predicted_role': predicted_label,
            'confidence': confidence,
            'top_predictions': top_predictions
        }

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate Legal Document Role Classifier")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test data")
    parser.add_argument("--context_mode", type=str, default="prev",
                       choices=["single", "prev", "prev_two", "surrounding"],
                       help="Context mode")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        device=device
    )
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        test_data_path=args.test_data,
        context_mode=args.context_mode,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()