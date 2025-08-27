"""
Dataset Converter and Preprocessor

This script helps convert your existing dataset to different formats
and perform preprocessing operations.
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from sklearn.model_selection import train_test_split
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetConverter:
    """
    Utility class for converting and preprocessing legal document datasets
    """
    
    def __init__(self):
        self.role_mapping = {
            "Facts": "Facts",
            "Issue": "Issue", 
            "Arguments of Petitioner": "Arguments of Petitioner",
            "Arguments of Respondent": "Arguments of Respondent",
            "Reasoning": "Reasoning",
            "Decision": "Decision",
            "None": "None"
        }
    
    def convert_txt_to_csv(self, input_dir: str, output_file: str):
        """
        Convert text files to CSV format
        
        Args:
            input_dir: Directory containing text files
            output_file: Output CSV file path
        """
        logger.info(f"Converting text files from {input_dir} to CSV...")
        
        all_data = []
        input_path = Path(input_dir)
        
        # Process each text file
        for file_path in input_path.glob("*.txt"):
            logger.info(f"Processing {file_path.name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Parse file content
                lines = content.split('\n')
                document_id = file_path.stem
                
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split sentence and role
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        sentence = parts[0].strip()
                        role = parts[1].strip()
                        
                        # Validate and normalize role
                        if role not in self.role_mapping:
                            logger.warning(f"Unknown role '{role}' in {file_path.name}, mapping to 'None'")
                            role = "None"
                        
                        all_data.append({
                            'document_id': document_id,
                            'sentence_index': line_idx,
                            'sentence': sentence,
                            'role': role,
                            'file_source': file_path.name
                        })
            
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Converted {len(all_data)} sentences from {len(list(input_path.glob('*.txt')))} files")
        logger.info(f"Saved to {output_file}")
        
        # Print statistics
        role_counts = df['role'].value_counts()
        logger.info("Role distribution:")
        for role, count in role_counts.items():
            logger.info(f"  {role}: {count}")
        
        return df
    
    def split_dataset(self, 
                     csv_file: str, 
                     output_dir: str,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     split_by_document: bool = True,
                     random_seed: int = 42):
        """
        Split dataset into train/validation/test sets
        
        Args:
            csv_file: Input CSV file
            output_dir: Output directory for splits
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            split_by_document: Whether to split by documents (recommended)
            random_seed: Random seed for reproducibility
        """
        logger.info(f"Splitting dataset from {csv_file}")
        
        # Load data
        df = pd.read_csv(csv_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        random.seed(random_seed)
        
        if split_by_document:
            # Split by documents to avoid data leakage
            document_ids = df['document_id'].unique()
            random.shuffle(document_ids)
            
            n_docs = len(document_ids)
            train_end = int(n_docs * train_ratio)
            val_end = train_end + int(n_docs * val_ratio)
            
            train_docs = document_ids[:train_end]
            val_docs = document_ids[train_end:val_end]
            test_docs = document_ids[val_end:]
            
            train_df = df[df['document_id'].isin(train_docs)]
            val_df = df[df['document_id'].isin(val_docs)]
            test_df = df[df['document_id'].isin(test_docs)]
            
        else:
            # Split by sentences (not recommended for legal documents)
            train_df, temp_df = train_test_split(
                df, test_size=(val_ratio + test_ratio), 
                random_state=random_seed, stratify=df['role']
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=test_ratio/(val_ratio + test_ratio),
                random_state=random_seed, stratify=temp_df['role']
            )
        
        # Save splits
        train_df.to_csv(output_path / "train.csv", index=False)
        val_df.to_csv(output_path / "val.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        
        # Log statistics
        logger.info(f"Dataset split completed:")
        logger.info(f"  Train: {len(train_df)} samples from {train_df['document_id'].nunique()} documents")
        logger.info(f"  Val: {len(val_df)} samples from {val_df['document_id'].nunique()} documents")
        logger.info(f"  Test: {len(test_df)} samples from {test_df['document_id'].nunique()} documents")
        
        # Role distribution in each split
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            logger.info(f"\n{split_name} role distribution:")
            role_counts = split_df['role'].value_counts()
            for role, count in role_counts.items():
                percentage = (count / len(split_df)) * 100
                logger.info(f"  {role}: {count} ({percentage:.1f}%)")
    
    def convert_to_training_format(self, 
                                  csv_file: str, 
                                  output_dir: str,
                                  format_type: str = "txt"):
        """
        Convert CSV to training format (text files with sentence\trole)
        
        Args:
            csv_file: Input CSV file
            output_dir: Output directory
            format_type: Output format ("txt" or "json")
        """
        logger.info(f"Converting {csv_file} to {format_type} format")
        
        df = pd.read_csv(csv_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format_type == "txt":
            # Group by document and create text files
            for doc_id, group in df.groupby('document_id'):
                output_file = output_path / f"{doc_id}.txt"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for _, row in group.iterrows():
                        f.write(f"{row['sentence']}\t{row['role']}\n")
        
        elif format_type == "json":
            # Create JSON format
            documents = []
            
            for doc_id, group in df.groupby('document_id'):
                sentences = []
                for _, row in group.iterrows():
                    sentences.append({
                        'sentence': row['sentence'],
                        'role': row['role'],
                        'sentence_index': row['sentence_index']
                    })
                
                documents.append({
                    'document_id': doc_id,
                    'sentences': sentences
                })
            
            output_file = output_path / "dataset.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversion completed. Output saved to {output_path}")
    
    def analyze_dataset(self, data_path: str):
        """
        Analyze dataset and provide statistics
        
        Args:
            data_path: Path to dataset (CSV file or directory)
        """
        logger.info(f"Analyzing dataset: {data_path}")
        
        data_path = Path(data_path)
        
        if data_path.is_file() and data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.is_dir():
            # Load from text files
            all_data = []
            for file_path in data_path.glob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                lines = content.split('\n')
                document_id = file_path.stem
                
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        sentence = parts[0].strip()
                        role = parts[1].strip()
                        
                        all_data.append({
                            'document_id': document_id,
                            'sentence_index': line_idx,
                            'sentence': sentence,
                            'role': role
                        })
            
            df = pd.DataFrame(all_data)
        else:
            raise ValueError(f"Unsupported data path: {data_path}")
        
        # Analysis
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        
        print(f"Total samples: {len(df)}")
        print(f"Total documents: {df['document_id'].nunique()}")
        print(f"Average sentences per document: {len(df) / df['document_id'].nunique():.2f}")
        
        # Role distribution
        print("\nRole Distribution:")
        role_counts = df['role'].value_counts()
        for role, count in role_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {role}: {count} ({percentage:.1f}%)")
        
        # Sentence length statistics
        df['sentence_length'] = df['sentence'].str.len()
        print(f"\nSentence Length Statistics:")
        print(f"  Mean: {df['sentence_length'].mean():.2f} characters")
        print(f"  Median: {df['sentence_length'].median():.2f} characters")
        print(f"  Min: {df['sentence_length'].min()} characters")
        print(f"  Max: {df['sentence_length'].max()} characters")
        
        # Document length statistics
        doc_lengths = df.groupby('document_id').size()
        print(f"\nDocument Length Statistics:")
        print(f"  Mean: {doc_lengths.mean():.2f} sentences")
        print(f"  Median: {doc_lengths.median():.2f} sentences")
        print(f"  Min: {doc_lengths.min()} sentences")
        print(f"  Max: {doc_lengths.max()} sentences")
        
        print("="*60)
        
        return df

def main():
    """Main function for dataset preprocessing"""
    parser = argparse.ArgumentParser(description="Legal Document Dataset Preprocessor")
    
    parser.add_argument("--task", type=str, required=True,
                       choices=["convert", "split", "analyze", "format"],
                       help="Task to perform")
    
    # Convert arguments
    parser.add_argument("--input_dir", type=str,
                       help="Input directory (for convert task)")
    parser.add_argument("--output_file", type=str,
                       help="Output file (for convert task)")
    
    # Split arguments
    parser.add_argument("--csv_file", type=str,
                       help="CSV file to split")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                       help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                       help="Test set ratio")
    parser.add_argument("--split_by_document", action="store_true", default=True,
                       help="Split by documents")
    
    # Analysis arguments
    parser.add_argument("--data_path", type=str,
                       help="Path to dataset for analysis")
    
    # Format arguments
    parser.add_argument("--format_type", type=str, default="txt",
                       choices=["txt", "json"],
                       help="Output format type")
    
    args = parser.parse_args()
    
    converter = DatasetConverter()
    
    if args.task == "convert":
        if not args.input_dir or not args.output_file:
            raise ValueError("convert task requires --input_dir and --output_file")
        converter.convert_txt_to_csv(args.input_dir, args.output_file)
    
    elif args.task == "split":
        if not args.csv_file or not args.output_dir:
            raise ValueError("split task requires --csv_file and --output_dir")
        converter.split_dataset(
            args.csv_file, args.output_dir,
            args.train_ratio, args.val_ratio, args.test_ratio,
            args.split_by_document
        )
    
    elif args.task == "analyze":
        if not args.data_path:
            raise ValueError("analyze task requires --data_path")
        converter.analyze_dataset(args.data_path)
    
    elif args.task == "format":
        if not args.csv_file or not args.output_dir:
            raise ValueError("format task requires --csv_file and --output_dir")
        converter.convert_to_training_format(
            args.csv_file, args.output_dir, args.format_type
        )

if __name__ == "__main__":
    main()