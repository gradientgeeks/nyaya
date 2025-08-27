"""
Document Processor for Legal Documents

This module handles the ingestion, cleaning, and preprocessing of legal documents
including PDFs, text files, and other formats commonly used in legal practice.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Union, BinaryIO
from pathlib import Path
from io import BytesIO

import fitz  # PyMuPDF
from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from unstructured.staging.base import dict_to_elements
import spacy
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class DocumentMetadata(BaseModel):
    """Metadata for processed legal documents"""
    filename: str
    file_type: str
    case_name: Optional[str] = None
    court: Optional[str] = None
    date: Optional[str] = None
    citation: Optional[str] = None
    parties: Dict[str, str] = {}
    page_count: int = 0
    word_count: int = 0
    processing_status: str = "pending"
    error_message: Optional[str] = None

class ProcessedDocument(BaseModel):
    """Processed legal document with extracted content and metadata"""
    content: str
    metadata: DocumentMetadata
    sections: List[Dict[str, Any]] = []
    extracted_entities: Dict[str, List[str]] = {}

class LegalDocumentProcessor:
    """
    Comprehensive document processor for legal documents
    Supports multiple formats and extracts legal-specific metadata
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the document processor
        
        Args:
            use_gpu: Whether to use GPU for processing (if available)
        """
        self.use_gpu = use_gpu
        
        # Load spaCy model for NER and text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Some features may be limited.")
            self.nlp = None
        
        # Legal document patterns
        self.case_name_patterns = [
            r"([A-Z][a-zA-Z\s&\.]+)\s+v\.?\s+([A-Z][a-zA-Z\s&\.]+)",
            r"([A-Z][a-zA-Z\s&\.]+)\s+vs\.?\s+([A-Z][a-zA-Z\s&\.]+)",
            r"([A-Z][a-zA-Z\s&\.]+)\s+versus\s+([A-Z][a-zA-Z\s&\.]+)"
        ]
        
        self.court_patterns = [
            r"Supreme\s+Court\s+of\s+India",
            r"High\s+Court\s+of\s+[A-Za-z\s]+",
            r"District\s+Court\s+of\s+[A-Za-z\s]+",
            r"[A-Za-z\s]+\s+High\s+Court",
            r"Hon'ble\s+[A-Za-z\s]+\s+Court"
        ]
        
        self.citation_patterns = [
            r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",
            r"\d{4}\s+\(\d+\)\s+SCC\s+\d+",
            r"AIR\s+\d{4}\s+SC\s+\d+",
            r"\d{4}\s+AIR\s+\d+",
            r"WP\s*\([C]?\)\s*No\.\s*\d+/\d{4}"
        ]
        
        logger.info("Legal Document Processor initialized")
    
    def extract_text_from_pdf(self, file_path: Union[str, BytesIO]) -> Tuple[str, int]:
        """
        Extract text from PDF using multiple methods for robustness
        
        Args:
            file_path: Path to PDF file or BytesIO object
            
        Returns:
            Tuple of (extracted_text, page_count)
        """
        try:
            # Method 1: Try PyMuPDF (fitz) first - better for complex layouts
            if isinstance(file_path, str):
                doc = fitz.open(file_path)
            else:
                doc = fitz.open(stream=file_path.read(), filetype="pdf")
            
            text_content = ""
            for page in doc:
                text_content += page.get_text()
            
            page_count = len(doc)
            doc.close()
            
            if text_content.strip():
                return text_content, page_count
                
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        try:
            # Method 2: Try pypdf as fallback
            if isinstance(file_path, str):
                with open(file_path, 'rb') as file:
                    reader = PdfReader(file)
                    text_content = ""
                    for page in reader.pages:
                        text_content += page.extract_text()
                    page_count = len(reader.pages)
            else:
                file_path.seek(0)
                reader = PdfReader(file_path)
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text()
                page_count = len(reader.pages)
            
            if text_content.strip():
                return text_content, page_count
                
        except Exception as e:
            logger.warning(f"pypdf extraction failed: {e}")
        
        try:
            # Method 3: Try unstructured as final fallback
            if isinstance(file_path, str):
                elements = partition_pdf(filename=file_path)
            else:
                file_path.seek(0)
                elements = partition_pdf(file=file_path)
            
            text_content = "\n".join([str(element) for element in elements])
            page_count = 1  # Unstructured doesn't provide page count easily
            
            return text_content, page_count
            
        except Exception as e:
            logger.error(f"All PDF extraction methods failed: {e}")
            return "", 0
    
    def extract_text_from_txt(self, file_path: Union[str, BytesIO]) -> str:
        """
        Extract text from plain text file
        
        Args:
            file_path: Path to text file or BytesIO object
            
        Returns:
            Extracted text content
        """
        try:
            if isinstance(file_path, str):
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            else:
                file_path.seek(0)
                return file_path.read().decode('utf-8')
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                if isinstance(file_path, str):
                    with open(file_path, 'r', encoding='latin-1') as file:
                        return file.read()
                else:
                    file_path.seek(0)
                    return file_path.read().decode('latin-1')
            except Exception as e:
                logger.error(f"Failed to read text file: {e}")
                return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize legal document text
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove unusual characters and normalize quotes
        text = re.sub(r'["\'“”‘’]', '"', text)
        text = re.sub(r'[–—]', '-', text)
        
        # Fix common OCR errors in legal documents
        text = re.sub(r'\bvs\b', 'v.', text)
        text = re.sub(r'\bVs\b', 'V.', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def extract_case_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract legal-specific metadata from document text
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {
            "case_name": None,
            "court": None,
            "citation": None,
            "parties": {},
            "date": None
        }
        
        # Extract case name
        for pattern in self.case_name_patterns:
            match = re.search(pattern, text[:2000])  # Search in first 2000 chars
            if match:
                metadata["case_name"] = match.group(0)
                metadata["parties"] = {
                    "petitioner": match.group(1).strip(),
                    "respondent": match.group(2).strip()
                }
                break
        
        # Extract court information
        for pattern in self.court_patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                metadata["court"] = match.group(0)
                break
        
        # Extract citation
        for pattern in self.citation_patterns:
            match = re.search(pattern, text[:3000])
            if match:
                metadata["citation"] = match.group(0)
                break
        
        # Extract date patterns
        date_patterns = [
            r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b',
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                metadata["date"] = match.group(0)
                break
        
        return metadata
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities relevant to legal documents
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # Geopolitical entities
            "LAW": [],
            "DATE": [],
            "MONEY": []
        }
        
        if not self.nlp:
            return entities
        
        # Process text in chunks to handle large documents
        chunk_size = 100000  # 100k characters per chunk
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            doc = self.nlp(chunk)
            
            for ent in doc.ents:
                if ent.label_ in entities:
                    entity_text = ent.text.strip()
                    if entity_text and entity_text not in entities[ent.label_]:
                        entities[ent.label_].append(entity_text)
        
        # Limit entities to avoid overwhelming storage
        for key in entities:
            entities[key] = entities[key][:50]  # Keep top 50 entities per type
        
        return entities
    
    def identify_document_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify and extract major sections of a legal document
        
        Args:
            text: Document text
            
        Returns:
            List of document sections with metadata
        """
        sections = []
        
        # Common legal document section headers
        section_patterns = [
            (r"FACTS?\s*:?", "Facts"),
            (r"BACKGROUND\s*:?", "Background"),
            (r"ISSUES?\s*:?", "Issues"),
            (r"ARGUMENTS?\s*:?", "Arguments"),
            (r"PETITIONER'S\s+ARGUMENTS?\s*:?", "Petitioner Arguments"),
            (r"RESPONDENT'S\s+ARGUMENTS?\s*:?", "Respondent Arguments"),
            (r"ANALYSIS\s*:?", "Analysis"),
            (r"REASONING\s*:?", "Reasoning"),
            (r"DISCUSSION\s*:?", "Discussion"),
            (r"DECISION\s*:?", "Decision"),
            (r"JUDGMENT\s*:?", "Judgment"),
            (r"CONCLUSION\s*:?", "Conclusion"),
            (r"ORDER\s*:?", "Order"),
            (r"HELD\s*:?", "Held")
        ]
        
        text_lines = text.split('\n')
        current_section = None
        current_content = []
        
        for i, line in enumerate(text_lines):
            line_stripped = line.strip()
            
            # Check if this line is a section header
            section_found = False
            for pattern, section_name in section_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # Save previous section if exists
                    if current_section and current_content:
                        sections.append({
                            "title": current_section,
                            "content": '\n'.join(current_content).strip(),
                            "start_line": sections[-1]["end_line"] + 1 if sections else 0,
                            "end_line": i - 1
                        })
                    
                    current_section = section_name
                    current_content = []
                    section_found = True
                    break
            
            if not section_found and current_section:
                current_content.append(line)
            elif not current_section:
                # Before any section headers, treat as introduction/header
                if not sections:
                    sections.append({
                        "title": "Header",
                        "content": line,
                        "start_line": i,
                        "end_line": i
                    })
                else:
                    sections[0]["content"] += f'\n{line}'
                    sections[0]["end_line"] = i
        
        # Add final section
        if current_section and current_content:
            sections.append({
                "title": current_section,
                "content": '\n'.join(current_content).strip(),
                "start_line": sections[-1]["end_line"] + 1 if sections else 0,
                "end_line": len(text_lines) - 1
            })
        
        return sections
    
    def process_document(self, 
                        file_path: Union[str, BinaryIO], 
                        filename: str = None,
                        additional_metadata: Dict[str, Any] = None) -> ProcessedDocument:
        """
        Main document processing function
        
        Args:
            file_path: Path to file or file object
            filename: Name of the file (required if file_path is BinaryIO)
            additional_metadata: Additional metadata to include
            
        Returns:
            ProcessedDocument object
        """
        if additional_metadata is None:
            additional_metadata = {}
        
        # Determine filename
        if isinstance(file_path, str):
            filename = filename or Path(file_path).name
            file_type = Path(file_path).suffix.lower()
        else:
            if not filename:
                raise ValueError("filename must be provided when using file object")
            file_type = Path(filename).suffix.lower()
        
        # Initialize metadata
        metadata = DocumentMetadata(
            filename=filename,
            file_type=file_type,
            processing_status="processing"
        )
        
        try:
            # Extract text based on file type
            if file_type == '.pdf':
                content, page_count = self.extract_text_from_pdf(file_path)
                metadata.page_count = page_count
            elif file_type in ['.txt', '.text']:
                content = self.extract_text_from_txt(file_path)
                metadata.page_count = 1
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            if not content.strip():
                raise ValueError("No text content extracted from document")
            
            # Clean the text
            content = self.clean_text(content)
            metadata.word_count = len(content.split())
            
            # Extract legal metadata
            legal_metadata = self.extract_case_metadata(content)
            metadata.case_name = legal_metadata.get("case_name")
            metadata.court = legal_metadata.get("court")
            metadata.date = legal_metadata.get("date")
            metadata.citation = legal_metadata.get("citation")
            metadata.parties = legal_metadata.get("parties", {})
            
            # Extract named entities
            entities = self.extract_named_entities(content)
            
            # Identify document sections
            sections = self.identify_document_sections(content)
            
            # Mark as successfully processed
            metadata.processing_status = "completed"
            
            # Add additional metadata
            for key, value in additional_metadata.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            
            return ProcessedDocument(
                content=content,
                metadata=metadata,
                sections=sections,
                extracted_entities=entities
            )
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            metadata.processing_status = "failed"
            metadata.error_message = str(e)
            
            return ProcessedDocument(
                content="",
                metadata=metadata,
                sections=[],
                extracted_entities={}
            )
    
    def batch_process_documents(self, 
                              file_paths: List[Union[str, BinaryIO]], 
                              filenames: List[str] = None) -> List[ProcessedDocument]:
        """
        Process multiple documents in batch
        
        Args:
            file_paths: List of file paths or file objects
            filenames: List of filenames (required if using file objects)
            
        Returns:
            List of ProcessedDocument objects
        """
        if filenames and len(filenames) != len(file_paths):
            raise ValueError("Number of filenames must match number of file paths")
        
        processed_docs = []
        
        for i, file_path in enumerate(file_paths):
            filename = filenames[i] if filenames else None
            try:
                doc = self.process_document(file_path, filename)
                processed_docs.append(doc)
                logger.info(f"Successfully processed: {doc.metadata.filename}")
            except Exception as e:
                logger.error(f"Failed to process document {i}: {e}")
                # Create failed document entry
                failed_doc = ProcessedDocument(
                    content="",
                    metadata=DocumentMetadata(
                        filename=filename or f"document_{i}",
                        file_type="unknown",
                        processing_status="failed",
                        error_message=str(e)
                    ),
                    sections=[],
                    extracted_entities={}
                )
                processed_docs.append(failed_doc)
        
        logger.info(f"Batch processing completed: {len(processed_docs)} documents")
        return processed_docs

# Example usage
if __name__ == "__main__":
    processor = LegalDocumentProcessor()
    
    # Example with text content
    sample_text = """
    Ram Kumar v. State of Maharashtra
    
    Supreme Court of India
    Civil Appeal No. 123/2023
    Date: 15th March, 2024
    
    FACTS:
    The petitioner was arrested on charges of theft. The arrest was made without a warrant.
    
    ISSUES:
    Whether the arrest without warrant was constitutional?
    
    REASONING:
    The court analyzed the provisions of Article 21 and found that the arrest violated due process.
    
    DECISION:
    The arrest is declared unconstitutional and the petitioner is released.
    """
    
    # Create a temporary file for testing
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        temp_path = f.name
    
    try:
        # Process the document
        result = processor.process_document(temp_path)
        
        print("Document processing completed:")
        print(f"Case Name: {result.metadata.case_name}")
        print(f"Court: {result.metadata.court}")
        print(f"Word Count: {result.metadata.word_count}")
        print(f"Sections Found: {[s['title'] for s in result.sections]}")
        print(f"Entities: {result.extracted_entities}")
        
    finally:
        # Clean up
        os.unlink(temp_path)