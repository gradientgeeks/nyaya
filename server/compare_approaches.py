"""
Comparison Framework: Supervised vs Unsupervised Role Classification

This script compares:
1. Supervised InLegalBERT Classifier (fine-tuned)
2. Unsupervised Clustering Approach (K-Means, Hierarchical, etc.)

Metrics compared:
- Accuracy
- F1-Score (Macro/Weighted)
- Precision/Recall per role
- Confidence distribution
- Inference time
- Interpretability
"""

import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import json

from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from src.models.role_classifier import RoleClassifier, RhetoricalRole
from src.models.clustering_role_classifier import ClusteringRoleClassifier, ClusteringConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApproachComparison:
    """Compare supervised and unsupervised approaches"""
    
    ROLES = [
        "None",
        "Facts",
        "Issue",
        "Arguments of Petitioner",
        "Arguments of Respondent",
        "Reasoning",
        "Decision"
    ]
    
    def __init__(self, data_dir: str, output_dir: str = "comparison_results"):
        """
        Initialize comparison framework
        
        Args:
            data_dir: Directory containing test data
            output_dir: Directory to save comparison results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize classifiers
        logger.info("Initializing supervised classifier...")
        self.supervised_classifier = RoleClassifier(model_type="inlegalbert", device="cpu")
        
        logger.info("Initializing unsupervised classifier...")
        self.unsupervised_classifier = ClusteringRoleClassifier(
            config=ClusteringConfig(
                num_clusters=7,
                clustering_algorithm="kmeans",
                embedding_model="all-MiniLM-L6-v2"
            )
        )
        
        self.test_data = []
        self.results = {
            "supervised": {},
            "unsupervised": {},
            "comparison": {}
        }
    
    def load_test_data(self, max_files: int = None) -> List[Dict[str, Any]]:
        """
        Load test data from files
        
        Args:
            max_files: Maximum number of files to load
            
        Returns:
            List of documents with sentences and ground truth roles
        """
        logger.info(f"Loading test data from {self.data_dir}")
        
        txt_files = sorted(self.data_dir.glob("*.txt"))
        if max_files:
            txt_files = txt_files[:max_files]
        
        self.test_data = []
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                document = {
                    "file": file_path.name,
                    "sentences": [],
                    "roles": []
                }
                
                for line in lines:
                    if '\t' in line:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            sentence, role = parts
                            document["sentences"].append(sentence)
                            document["roles"].append(role)
                
                if document["sentences"]:
                    self.test_data.append(document)
                    
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        total_sentences = sum(len(doc["sentences"]) for doc in self.test_data)
        logger.info(f"Loaded {len(self.test_data)} documents with {total_sentences} sentences")
        
        return self.test_data
    
    def train_unsupervised(self, use_train_data: bool = True):
        """
        Train/fit unsupervised clustering model
        
        Args:
            use_train_data: Whether to use training data for fitting
        """
        if use_train_data:
            train_dir = self.data_dir.parent / "train"
            if train_dir.exists():
                logger.info(f"Fitting clustering model on training data from {train_dir}")
                train_files = sorted(train_dir.glob("*.txt"))[:100]  # Limit for speed
                
                all_sentences = []
                all_roles = []
                
                for file_path in train_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if '\t' in line:
                                parts = line.strip().split('\t')
                                if len(parts) == 2:
                                    all_sentences.append(parts[0])
                                    all_roles.append(parts[1])
                
                metrics = self.unsupervised_classifier.fit(all_sentences, all_roles)
                logger.info(f"Clustering metrics: {metrics}")
            else:
                logger.warning(f"Training directory not found: {train_dir}")
        else:
            # Fit on test data (not ideal but shows clustering capability)
            all_test_sentences = []
            for doc in self.test_data:
                all_test_sentences.extend(doc["sentences"])
            
            self.unsupervised_classifier.fit(all_test_sentences)
    
    def evaluate_supervised(self) -> Dict[str, Any]:
        """Evaluate supervised classifier"""
        logger.info("Evaluating supervised classifier...")
        
        all_predictions = []
        all_true_labels = []
        all_confidences = []
        inference_times = []
        
        for doc in self.test_data:
            start_time = time.time()
            
            # Predict with supervised model
            results = self.supervised_classifier.classify_document(
                " ".join(doc["sentences"]),
                context_mode="prev"
            )
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Collect predictions
            for result, true_role in zip(results, doc["roles"]):
                all_predictions.append(result["role"])
                all_true_labels.append(true_role)
                all_confidences.append(result["confidence"])
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_true_labels, all_predictions, all_confidences)
        metrics["avg_inference_time"] = np.mean(inference_times)
        metrics["total_inference_time"] = sum(inference_times)
        
        self.results["supervised"] = metrics
        
        logger.info(f"Supervised - Accuracy: {metrics['accuracy']:.4f}, F1 (Macro): {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def evaluate_unsupervised(self) -> Dict[str, Any]:
        """Evaluate unsupervised clustering classifier"""
        logger.info("Evaluating unsupervised classifier...")
        
        all_predictions = []
        all_true_labels = []
        all_confidences = []
        inference_times = []
        
        for doc in self.test_data:
            start_time = time.time()
            
            # Predict with unsupervised model
            results = self.unsupervised_classifier.classify_document(
                " ".join(doc["sentences"])
            )
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Collect predictions
            for result, true_role in zip(results, doc["roles"]):
                all_predictions.append(result.predicted_role)
                all_true_labels.append(true_role)
                all_confidences.append(result.confidence)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_true_labels, all_predictions, all_confidences)
        metrics["avg_inference_time"] = np.mean(inference_times)
        metrics["total_inference_time"] = sum(inference_times)
        
        self.results["unsupervised"] = metrics
        
        logger.info(f"Unsupervised - Accuracy: {metrics['accuracy']:.4f}, F1 (Macro): {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: List[str], y_pred: List[str], 
                          confidences: List[float]) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Overall metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average='macro', labels=self.ROLES, zero_division=0)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average='weighted', labels=self.ROLES, zero_division=0)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.ROLES, zero_division=0
        )
        
        metrics["per_class"] = {}
        for i, role in enumerate(self.ROLES):
            metrics["per_class"][role] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }
        
        # Confidence statistics
        metrics["confidence"] = {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences))
        }
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=self.ROLES).tolist()
        
        # Classification report
        metrics["classification_report"] = classification_report(
            y_true, y_pred, labels=self.ROLES, zero_division=0, output_dict=True
        )
        
        return metrics
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        logger.info("Generating comparison report...")
        
        comparison = {
            "summary": {
                "supervised": {
                    "accuracy": self.results["supervised"]["accuracy"],
                    "f1_macro": self.results["supervised"]["f1_macro"],
                    "f1_weighted": self.results["supervised"]["f1_weighted"],
                    "avg_confidence": self.results["supervised"]["confidence"]["mean"],
                    "inference_time": self.results["supervised"]["avg_inference_time"]
                },
                "unsupervised": {
                    "accuracy": self.results["unsupervised"]["accuracy"],
                    "f1_macro": self.results["unsupervised"]["f1_macro"],
                    "f1_weighted": self.results["unsupervised"]["f1_weighted"],
                    "avg_confidence": self.results["unsupervised"]["confidence"]["mean"],
                    "inference_time": self.results["unsupervised"]["avg_inference_time"]
                }
            },
            "winner": self._determine_winner(),
            "insights": self._generate_insights()
        }
        
        self.results["comparison"] = comparison
        
        # Save to file
        with open(self.output_dir / "comparison_report.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Comparison report saved to {self.output_dir / 'comparison_report.json'}")
        
        return comparison
    
    def _determine_winner(self) -> Dict[str, str]:
        """Determine which approach performs better"""
        sup = self.results["supervised"]
        unsup = self.results["unsupervised"]
        
        winner = {}
        winner["accuracy"] = "supervised" if sup["accuracy"] > unsup["accuracy"] else "unsupervised"
        winner["f1_macro"] = "supervised" if sup["f1_macro"] > unsup["f1_macro"] else "unsupervised"
        winner["confidence"] = "supervised" if sup["confidence"]["mean"] > unsup["confidence"]["mean"] else "unsupervised"
        winner["speed"] = "supervised" if sup["avg_inference_time"] < unsup["avg_inference_time"] else "unsupervised"
        
        # Overall winner (majority vote)
        vote_count = Counter(winner.values())
        winner["overall"] = vote_count.most_common(1)[0][0]
        
        return winner
    
    def _generate_insights(self) -> List[str]:
        """Generate insights from comparison"""
        insights = []
        
        sup = self.results["supervised"]
        unsup = self.results["unsupervised"]
        
        # Accuracy comparison
        acc_diff = abs(sup["accuracy"] - unsup["accuracy"])
        if acc_diff < 0.05:
            insights.append(f"Both approaches achieve similar accuracy (difference: {acc_diff:.2%})")
        else:
            better = "supervised" if sup["accuracy"] > unsup["accuracy"] else "unsupervised"
            insights.append(f"{better.capitalize()} approach has significantly better accuracy (+{acc_diff:.2%})")
        
        # F1 comparison
        f1_diff = abs(sup["f1_macro"] - unsup["f1_macro"])
        if sup["f1_macro"] > unsup["f1_macro"]:
            insights.append(f"Supervised model has better macro F1-score (+{f1_diff:.2%})")
        else:
            insights.append(f"Unsupervised clustering achieves competitive F1-score (difference: {f1_diff:.2%})")
        
        # Confidence comparison
        if sup["confidence"]["mean"] > unsup["confidence"]["mean"]:
            insights.append(f"Supervised model produces more confident predictions")
        else:
            insights.append(f"Unsupervised approach shows surprising confidence levels")
        
        # Speed comparison
        if sup["avg_inference_time"] < unsup["avg_inference_time"]:
            speedup = unsup["avg_inference_time"] / sup["avg_inference_time"]
            insights.append(f"Supervised model is {speedup:.1f}x faster")
        else:
            speedup = sup["avg_inference_time"] / unsup["avg_inference_time"]
            insights.append(f"Unsupervised approach is {speedup:.1f}x faster")
        
        # Per-class insights
        for role in self.ROLES:
            sup_f1 = sup["per_class"][role]["f1_score"]
            unsup_f1 = unsup["per_class"][role]["f1_score"]
            
            if abs(sup_f1 - unsup_f1) > 0.1:
                better = "supervised" if sup_f1 > unsup_f1 else "unsupervised"
                insights.append(f"Role '{role}': {better} performs significantly better")
        
        return insights
    
    def plot_comparison(self):
        """Generate comparison visualizations"""
        logger.info("Generating comparison plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Supervised vs Unsupervised Role Classification Comparison", fontsize=16)
        
        # 1. Overall Metrics Comparison
        ax = axes[0, 0]
        metrics_to_plot = ['accuracy', 'f1_macro', 'f1_weighted']
        sup_values = [self.results["supervised"][m] for m in metrics_to_plot]
        unsup_values = [self.results["unsupervised"][m] for m in metrics_to_plot]
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        ax.bar(x - width/2, sup_values, width, label='Supervised', color='skyblue')
        ax.bar(x + width/2, unsup_values, width, label='Unsupervised', color='lightcoral')
        ax.set_ylabel('Score')
        ax.set_title('Overall Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Per-Role F1 Scores
        ax = axes[0, 1]
        roles = self.ROLES
        sup_f1 = [self.results["supervised"]["per_class"][r]["f1_score"] for r in roles]
        unsup_f1 = [self.results["unsupervised"]["per_class"][r]["f1_score"] for r in roles]
        
        x = np.arange(len(roles))
        ax.bar(x - width/2, sup_f1, width, label='Supervised', color='skyblue')
        ax.bar(x + width/2, unsup_f1, width, label='Unsupervised', color='lightcoral')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Role F1 Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(roles, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Confidence Distribution
        ax = axes[0, 2]
        sup_conf = self.results["supervised"]["confidence"]
        unsup_conf = self.results["unsupervised"]["confidence"]
        
        conf_metrics = ['mean', 'median', 'std']
        sup_conf_values = [sup_conf[m] for m in conf_metrics]
        unsup_conf_values = [unsup_conf[m] for m in conf_metrics]
        
        x = np.arange(len(conf_metrics))
        ax.bar(x - width/2, sup_conf_values, width, label='Supervised', color='skyblue')
        ax.bar(x + width/2, unsup_conf_values, width, label='Unsupervised', color='lightcoral')
        ax.set_ylabel('Confidence')
        ax.set_title('Confidence Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels([m.title() for m in conf_metrics])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Confusion Matrix - Supervised
        ax = axes[1, 0]
        cm_sup = np.array(self.results["supervised"]["confusion_matrix"])
        sns.heatmap(cm_sup, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=roles, yticklabels=roles, cbar=False)
        ax.set_title('Confusion Matrix: Supervised')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 5. Confusion Matrix - Unsupervised
        ax = axes[1, 1]
        cm_unsup = np.array(self.results["unsupervised"]["confusion_matrix"])
        sns.heatmap(cm_unsup, annot=True, fmt='d', cmap='Reds', ax=ax,
                   xticklabels=roles, yticklabels=roles, cbar=False)
        ax.set_title('Confusion Matrix: Unsupervised')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 6. Inference Time Comparison
        ax = axes[1, 2]
        times = ['Avg Time (s)', 'Total Time (s)']
        sup_times = [
            self.results["supervised"]["avg_inference_time"],
            self.results["supervised"]["total_inference_time"]
        ]
        unsup_times = [
            self.results["unsupervised"]["avg_inference_time"],
            self.results["unsupervised"]["total_inference_time"]
        ]
        
        x = np.arange(len(times))
        ax.bar(x - width/2, sup_times, width, label='Supervised', color='skyblue')
        ax.bar(x + width/2, unsup_times, width, label='Unsupervised', color='lightcoral')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Inference Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(times)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_plots.png", dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plots saved to {self.output_dir / 'comparison_plots.png'}")
        
        plt.show()
    
    def print_summary(self):
        """Print comparison summary"""
        print("\n" + "="*80)
        print("ROLE CLASSIFICATION COMPARISON: SUPERVISED VS UNSUPERVISED")
        print("="*80)
        
        comparison = self.results["comparison"]
        
        print("\n📊 OVERALL METRICS:")
        print("-"*80)
        print(f"{'Metric':<20} {'Supervised':>15} {'Unsupervised':>15} {'Winner':>15}")
        print("-"*80)
        
        summary = comparison["summary"]
        for metric in ["accuracy", "f1_macro", "f1_weighted", "avg_confidence", "inference_time"]:
            sup_val = summary["supervised"][metric]
            unsup_val = summary["unsupervised"][metric]
            winner = "🏆 Supervised" if sup_val > unsup_val else "🏆 Unsupervised"
            
            if metric == "inference_time":
                winner = "🏆 Supervised" if sup_val < unsup_val else "🏆 Unsupervised"
                print(f"{metric:<20} {sup_val:>14.4f}s {unsup_val:>14.4f}s {winner:>15}")
            else:
                print(f"{metric:<20} {sup_val:>15.4f} {unsup_val:>15.4f} {winner:>15}")
        
        print("\n🏆 OVERALL WINNER:", comparison["winner"]["overall"].upper())
        
        print("\n💡 KEY INSIGHTS:")
        print("-"*80)
        for i, insight in enumerate(comparison["insights"], 1):
            print(f"{i}. {insight}")
        
        print("\n" + "="*80)


def main():
    """Main comparison workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare supervised vs unsupervised role classification")
    parser.add_argument(
        "--test-dir",
        type=str,
        default="dataset/Hier_BiLSTM_CRF/test",
        help="Test data directory"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help="Maximum number of test files to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparison = ApproachComparison(args.test_dir, args.output_dir)
    
    # Load test data
    comparison.load_test_data(max_files=args.max_files)
    
    # Train unsupervised model
    comparison.train_unsupervised(use_train_data=True)
    
    # Evaluate both approaches
    comparison.evaluate_supervised()
    comparison.evaluate_unsupervised()
    
    # Generate report
    comparison.generate_comparison_report()
    
    # Create visualizations
    comparison.plot_comparison()
    
    # Print summary
    comparison.print_summary()


if __name__ == "__main__":
    main()
