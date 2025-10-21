"""
Document Preprocessing Service

Handles:
1. PDF text extraction
2. Parsing sentence\trole format from training data
3. Text cleaning and validation
"""

import re
from typing import List, Tuple, Optional
from pathlib import Path
import PyPDF2
from app.models.schemas import RhetoricalRole


class DocumentPreprocessor:
    """Preprocess documents for classification and embedding."""
    
    def __init__(self):
        self.valid_roles = {role.value for role in RhetoricalRole}
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Extracted text
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def parse_training_format(
        self,
        file_path: str,
        validate_roles: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Parse training data in sentence\trole format.
        
        Format:
        ```
        The petitioner filed a writ petition.	Facts
        The main issue is constitutional validity.	Issue
        
        The respondent filed an appeal.	Facts
        ```
        
        Args:
            file_path: Path to training file
            validate_roles: Whether to validate role labels
        
        Returns:
            List of (sentence, role) tuples
        
        Raises:
            ValueError: If format is invalid
        """
        sentences_with_roles = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip blank lines (document separators)
                    if not line:
                        continue
                    
                    # Split by tab
                    parts = line.split('\t')
                    
                    if len(parts) != 2:
                        raise ValueError(
                            f"Line {line_num}: Expected 'sentence\\trole' format, "
                            f"got {len(parts)} parts. "
                            f"Ensure you're using TAB (not spaces) as separator."
                        )
                    
                    sentence, role = parts
                    sentence = sentence.strip()
                    role = role.strip()
                    
                    # Validate
                    if not sentence:
                        raise ValueError(f"Line {line_num}: Empty sentence")
                    
                    if validate_roles and role not in self.valid_roles:
                        raise ValueError(
                            f"Line {line_num}: Invalid role '{role}'. "
                            f"Valid roles: {', '.join(self.valid_roles)}"
                        )
                    
                    sentences_with_roles.append((sentence, role))
            
            return sentences_with_roles
        
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
        except UnicodeDecodeError:
            raise ValueError(f"File encoding error. Ensure UTF-8 encoding: {file_path}")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple rules.
        
        For production, consider using spaCy or NLTK for better accuracy.
        
        Args:
            text: Input text
        
        Returns:
            List of sentences
        """
        # Simple sentence splitter (works for most legal text)
        # Splits on period followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace, special characters, etc.
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def validate_classified_data(
        self,
        sentences_with_roles: List[Tuple[str, str, float]]
    ) -> bool:
        """
        Validate classified data (sentence, role, confidence).
        
        Args:
            sentences_with_roles: List of (sentence, role, confidence) tuples
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If invalid
        """
        if not sentences_with_roles:
            raise ValueError("Empty classification result")
        
        for i, (sentence, role, confidence) in enumerate(sentences_with_roles):
            # Check sentence
            if not isinstance(sentence, str) or not sentence.strip():
                raise ValueError(f"Sentence {i}: Invalid or empty sentence")
            
            # Check role
            if role not in self.valid_roles:
                raise ValueError(f"Sentence {i}: Invalid role '{role}'")
            
            # Check confidence
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                raise ValueError(f"Sentence {i}: Invalid confidence {confidence}")
        
        return True
    
    def format_for_embedding(
        self,
        sentence: str,
        case_id: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Format sentence for embedding with title prefix.
        
        Following EmbeddingGemma best practices:
        "title: {case_id} | text: {sentence}"
        
        Args:
            sentence: The sentence text
            case_id: Case identifier
            metadata: Optional additional metadata
        
        Returns:
            Formatted string for embedding
        """
        return f"title: {case_id} | text: {sentence}"


# Singleton instance
preprocessor = DocumentPreprocessor()
