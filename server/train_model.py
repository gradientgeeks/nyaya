#!/usr/bin/env python3
"""
Training Script for Legal Document Role Classification Model

This script trains the rhetorical role classifier using the existing dataset
and saves the model in the models folder.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src" / "models" / "training"))

from src.models.training.train import RoleClassifierTrainer

def main():
    """Main training function"""
    
    # Training Configuration
    config = {
        # Data paths - NOTE: val/test files don't have labels, so we use None
        # This will cause automatic train/val split from training data
        "train_data": str(PROJECT_ROOT / "dataset" / "Hier_BiLSTM_CRF" / "train"),
        "val_data": None,  # Val files don't have labels - will auto-split from train
        "test_data": None,  # Test files don't have labels - will auto-split from train
        
        # Model configuration
        "model_type": "inlegalbert",  # Options: "inlegalbert", "bilstm_crf"
        "model_name": "law-ai/InLegalBERT",
        "context_mode": "prev",  # Options: "single", "prev", "prev_two", "surrounding"
        
        # Training hyperparameters
        "batch_size": 32,
        "num_epochs": 5,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "max_length": 512,
        "warmup_steps": 500,
        "dataset_sample_ratio": 0.8,  # Use 80% of training data
        
        # Device
        "device": "cuda",  # Change to "cpu" if no GPU available
    }
    
    print("="*70)
    print("LEGAL DOCUMENT ROLE CLASSIFIER TRAINING")
    print("="*70)
    print(f"\n📋 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create output directory in models folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "src" / "models" / "trained_models" / f"{config['model_type']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Output directory: {output_dir}\n")
    
    # Verify data paths exist
    for key in ["train_data", "val_data", "test_data"]:
        if config[key] is None:
            print(f"⚠️  {key}: Not provided (will use automatic split)")
            continue
        data_path = Path(config[key])
        if not data_path.exists():
            print(f"❌ Error: {key} path does not exist: {data_path}")
            return
        print(f"✅ Found {key}: {data_path}")
    
    print("\n" + "="*70)
    print("INITIALIZING TRAINER")
    print("="*70 + "\n")
    
    # Initialize trainer
    trainer = RoleClassifierTrainer(
        model_type=config["model_type"],
        model_name=config["model_name"],
        device=config["device"],
        output_dir=str(output_dir),
        num_labels=7  # 7 rhetorical roles
    )
    
    print(f"✅ Trainer initialized successfully!")
    print(f"🖥️  Using device: {config['device']}")
    print(f"🤖 Model type: {config['model_type']}")
    print(f"📏 Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"⏱️  Training for {config['num_epochs']} epochs")
    print(f"📚 Batch size: {config['batch_size']}")
    print(f"🧠 Learning rate: {config['learning_rate']}")
    print(f"📝 Context mode: {config['context_mode']}")
    print("="*70 + "\n")
    
    # Train the model
    try:
        trainer.train(
            train_data_path=config["train_data"],
            val_data_path=config["val_data"],
            test_data_path=config["test_data"],
            context_mode=config["context_mode"],
            batch_size=config["batch_size"],
            num_epochs=config["num_epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            warmup_steps=config["warmup_steps"],
            max_length=config["max_length"],
            save_best_model=True,
            dataset_sample_ratio=config["dataset_sample_ratio"]
        )
        
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📂 Model saved at: {output_dir}")
        print(f"📄 Best model: {output_dir / 'best_model.pt'}")
        print(f"📄 Final model: {output_dir / 'final_model.pt'}")
        print(f"📊 Training history: {output_dir / 'training_history.json'}")
        print(f"📈 Training plots: {output_dir / 'training_curves.png'}")
        
        print("\n✅ You can now use this model in your Nyaya system!")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
