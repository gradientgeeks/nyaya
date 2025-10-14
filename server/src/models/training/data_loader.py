"""
Data Loader for Custom Legal Document Role Classification Dataset

This module provides utilities for loading and preprocessing custom datasets
for training the role classifier.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)

class LegalDocumentDataset(Dataset):
    """
    Dataset class for legal document role classification
    
    Supports different data formats:
    1. Text files with sentence\trole format (like your current dataset)
    2. CSV files with columns: sentence, role, context_mode
    3. JSON files with structured data
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer_name: str = "law-ai/InLegalBERT",
                 context_mode: str = "single",
                 max_length: int = 512,
                 test_split: bool = False,
                 dataset_sample_ratio: float = 1.0):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the dataset directory or file
            tokenizer_name: Name of the tokenizer to use
            context_mode: Context mode for training ("single", "prev", "prev_two", "surrounding")
            max_length: Maximum sequence length
            test_split: Whether this is test split (affects processing)
            dataset_sample_ratio: Ratio of dataset to use (0.0-1.0), randomly samples files
        """
        self.data_path = Path(data_path)
        self.context_mode = context_mode
        self.max_length = max_length
        self.test_split = test_split
        self.dataset_sample_ratio = dataset_sample_ratio
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Role labels mapping
        self.roles = [
            "Facts", "Issue", "Arguments of Petitioner", "Arguments of Respondent", 
            "Reasoning", "Decision", "None"
        ]
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.roles)
        
        # Load and process data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        logger.info(f"Role distribution: {self._get_role_distribution()}")
    
    def _load_data(self) -> List[Dict]:
        """Load data from various formats"""
        if self.data_path.is_file():
            # Single file
            if self.data_path.suffix == '.txt':
                return self._load_txt_file(self.data_path)
            elif self.data_path.suffix == '.csv':
                return self._load_csv_file(self.data_path)
            elif self.data_path.suffix == '.json':
                return self._load_json_file(self.data_path)
        elif self.data_path.is_dir():
            # Directory with multiple files
            return self._load_directory(self.data_path)
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")
    
    def _load_txt_file(self, file_path: Path) -> List[Dict]:
        """Load data from text file in sentence\trole format"""
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Split by lines
        lines = content.split('\n')
        current_document = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_document:
                    # Process current document
                    data.extend(self._process_document(current_document))
                    current_document = []
                continue
            
            # Split sentence and role
            parts = line.split('\t')
            if len(parts) >= 2:
                sentence = parts[0].strip()
                role = parts[1].strip()
                
                # Validate role
                if role not in self.roles:
                    logger.warning(f"Unknown role '{role}' found, mapping to 'None'")
                    role = "None"
                
                current_document.append({
                    'sentence': sentence,
                    'role': role,
                    'sentence_index': len(current_document)
                })
        
        # Process last document if exists
        if current_document:
            data.extend(self._process_document(current_document))
        
        return data
    
    def _load_directory(self, dir_path: Path) -> List[Dict]:
        """Load data from directory containing multiple text files"""
        data = []
        
        # Get all text files recursively (handles nested directories like val/val/)
        txt_files = list(dir_path.rglob("*.txt"))
        
        if len(txt_files) == 0:
            logger.warning(f"No .txt files found in {dir_path}")
            return data
        
        # Apply dataset sampling if ratio < 1.0
        if self.dataset_sample_ratio < 1.0 and not self.test_split:
            import random
            num_files_to_use = int(len(txt_files) * self.dataset_sample_ratio)
            random.seed(42)  # For reproducibility
            txt_files = random.sample(txt_files, num_files_to_use)
            logger.info(f"Sampling {num_files_to_use}/{len(list(dir_path.rglob('*.txt')))} files ({self.dataset_sample_ratio*100}% of dataset)")
        
        for file_path in txt_files:
            try:
                file_data = self._load_txt_file(file_path)
                data.extend(file_data)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                continue
        
        return data
    
    def _load_csv_file(self, file_path: Path) -> List[Dict]:
        """Load data from CSV file"""
        df = pd.read_csv(file_path)
        
        required_columns = ['sentence', 'role']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        data = []
        # Group by document if document_id column exists
        if 'document_id' in df.columns:
            for doc_id, group in df.groupby('document_id'):
                doc_data = []
                for idx, row in group.iterrows():
                    doc_data.append({
                        'sentence': row['sentence'],
                        'role': row['role'],
                        'sentence_index': len(doc_data)
                    })
                data.extend(self._process_document(doc_data))
        else:
            # Treat as single document
            doc_data = []
            for idx, row in df.iterrows():
                doc_data.append({
                    'sentence': row['sentence'],
                    'role': row['role'],
                    'sentence_index': len(doc_data)
                })
            data.extend(self._process_document(doc_data))
        
        return data
    
    def _load_json_file(self, file_path: Path) -> List[Dict]:
        """Load data from JSON file"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        data = []
        
        if isinstance(json_data, list):
            # List of documents
            for doc in json_data:
                if 'sentences' in doc:
                    doc_data = []
                    for sent_data in doc['sentences']:
                        doc_data.append({
                            'sentence': sent_data['sentence'],
                            'role': sent_data['role'],
                            'sentence_index': len(doc_data)
                        })
                    data.extend(self._process_document(doc_data))
        
        return data
    
    def _process_document(self, document: List[Dict]) -> List[Dict]:
        """Process a single document to create training samples with context"""
        processed_data = []
        
        for i, item in enumerate(document):
            # Create context based on context_mode
            context_text = self._create_context(document, i)
            
            processed_item = {
                'text': context_text,
                'label': item['role'],
                'sentence_index': i,
                'original_sentence': item['sentence']
            }
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def _create_context(self, document: List[Dict], sentence_idx: int) -> str:
        """Create context text based on context mode"""
        if self.context_mode == "single":
            return document[sentence_idx]['sentence']
        
        elif self.context_mode == "prev":
            context_sentences = []
            if sentence_idx > 0:
                context_sentences.append(document[sentence_idx - 1]['sentence'])
            context_sentences.append(document[sentence_idx]['sentence'])
            return " [SEP] ".join(context_sentences)
        
        elif self.context_mode == "prev_two":
            context_sentences = []
            if sentence_idx > 1:
                context_sentences.append(document[sentence_idx - 2]['sentence'])
            if sentence_idx > 0:
                context_sentences.append(document[sentence_idx - 1]['sentence'])
            context_sentences.append(document[sentence_idx]['sentence'])
            return " [SEP] ".join(context_sentences)
        
        elif self.context_mode == "surrounding":
            context_sentences = []
            if sentence_idx > 0:
                context_sentences.append(document[sentence_idx - 1]['sentence'])
            context_sentences.append(document[sentence_idx]['sentence'])
            if sentence_idx < len(document) - 1:
                context_sentences.append(document[sentence_idx + 1]['sentence'])
            return " [SEP] ".join(context_sentences)
        
        else:
            return document[sentence_idx]['sentence']
    
    def _get_role_distribution(self) -> Dict[str, int]:
        """Get distribution of roles in the dataset"""
        role_counts = {}
        for item in self.data:
            role = item['label']
            role_counts[role] = role_counts.get(role, 0) + 1
        return role_counts
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Encode label
        label = self.label_encoder.transform([item['label']])[0]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': item['text'],
            'original_sentence': item['original_sentence']
        }

def create_data_loaders(
    train_path: str,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    tokenizer_name: str = "law-ai/InLegalBERT",
    context_mode: str = "prev",
    batch_size: int = 16,
    max_length: int = 512,
    train_split_ratio: float = 0.8,
    dataset_sample_ratio: float = 1.0
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        test_path: Path to test data (optional)
        tokenizer_name: Name of tokenizer to use
        context_mode: Context mode for training
        batch_size: Batch size
        max_length: Maximum sequence length
        train_split_ratio: Ratio for train/val split if val_path not provided
        dataset_sample_ratio: Ratio of dataset files to use (0.0-1.0)
    
    Returns:
        Dictionary containing data loaders
    """
    data_loaders = {}
    
    # Load training dataset
    train_dataset = LegalDocumentDataset(
        train_path, tokenizer_name, context_mode, max_length,
        dataset_sample_ratio=dataset_sample_ratio
    )
    
    # Split train/val if validation path not provided
    if val_path is None:
        train_size = int(train_split_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        data_loaders['train'] = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        data_loaders['val'] = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )
    else:
        data_loaders['train'] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_dataset = LegalDocumentDataset(
            val_path, tokenizer_name, context_mode, max_length
        )
        data_loaders['val'] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
    
    # Load test dataset if provided
    if test_path:
        test_dataset = LegalDocumentDataset(
            test_path, tokenizer_name, context_mode, max_length, test_split=True
        )
        data_loaders['test'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
    
    return data_loaders

# Example usage
if __name__ == "__main__":
    # Example for your dataset structure
    train_path = "/path/to/your/train/directory"
    test_path = "/path/to/your/test/directory"
    
    # Create data loaders
    data_loaders = create_data_loaders(
        train_path=train_path,
        test_path=test_path,
        context_mode="prev",
        batch_size=16
    )
    
    print(f"Training batches: {len(data_loaders['train'])}")
    print(f"Validation batches: {len(data_loaders['val'])}")
    if 'test' in data_loaders:
        print(f"Test batches: {len(data_loaders['test'])}")
    
    # Show example batch
    batch = next(iter(data_loaders['train']))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")