"""
Classification Service

Handles rhetorical role classification using InLegalBERT.
This is the first stage of Nyaya's two-stage pipeline.
"""

import logging
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..core.config import Settings
from ..models.schemas import RhetoricalRole, ClassificationResult

logger = logging.getLogger(__name__)


class ClassificationService:
    """
    Wrapper for InLegalBERT rhetorical role classifier.
    
    Key responsibilities:
    - Load trained model.pt file
    - Classify sentences into 7 rhetorical roles
    - Return confidence scores
    - Handle batch processing
    
    Expected model: Fine-tuned InLegalBERT (law-ai/InLegalBERT) on rhetorical role data.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Role mapping (must match training labels)
        self.id2label = {
            0: RhetoricalRole.FACTS,
            1: RhetoricalRole.ISSUE,
            2: RhetoricalRole.ARGUMENTS_PETITIONER,
            3: RhetoricalRole.ARGUMENTS_RESPONDENT,
            4: RhetoricalRole.REASONING,
            5: RhetoricalRole.DECISION,
            6: RhetoricalRole.NONE
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        self._load_model()
    
    def _load_model(self):
        """
        Load trained InLegalBERT model.
        
        Expected file structure:
        - backend/models/inlegalbert_classifier.pt (or path from settings)
        
        The model should be a fine-tuned InLegalBERT saved with torch.save().
        """
        try:
            logger.info(f"📥 Loading classification model from: {self.settings.classifier_model_path}")
            
            # Load tokenizer (always from InLegalBERT base)
            self.tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
            
            # Load model
            # Option 1: If saved as state_dict
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "law-ai/InLegalBERT",
                    num_labels=7,
                    id2label=self.id2label,
                    label2id=self.label2id
                )
                
                # Load trained weights
                state_dict = torch.load(
                    self.settings.classifier_model_path,
                    map_location=self.device,
                    weights_only=True  # Security: only load weights, not code
                )
                self.model.load_state_dict(state_dict)
                
            except Exception as e:
                logger.warning(f"⚠️  Could not load as state_dict, trying full model: {e}")
                
                # Option 2: If saved as full model
                self.model = torch.load(
                    self.settings.classifier_model_path,
                    map_location=self.device,
                    weights_only=False  # Allow full model loading
                )
            
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(
                f"✅ Model loaded successfully! "
                f"Device: {self.device}, Labels: {len(self.id2label)}"
            )
            
        except FileNotFoundError:
            logger.error(
                f"❌ Model file not found: {self.settings.classifier_model_path}\n"
                f"💡 Tip: Train the model first using the notebooks in docs/\n"
                f"   or provide a pre-trained model.pt file"
            )
            raise
        
        except Exception as e:
            logger.error(f"❌ Failed to load classification model: {e}")
            raise
    
    def classify_sentence(
        self,
        sentence: str,
        return_all_scores: bool = False
    ) -> ClassificationResult:
        """
        Classify a single sentence into rhetorical role.
        
        Args:
            sentence: Input sentence text
            return_all_scores: Whether to return scores for all roles
        
        Returns:
            ClassificationResult with predicted role and confidence
        """
        if not sentence.strip():
            return ClassificationResult(
                role=RhetoricalRole.NONE,
                confidence=1.0,
                all_scores={}
            )
        
        # Tokenize
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get prediction
        predicted_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_id].item()
        predicted_role = self.id2label[predicted_id]
        
        result = ClassificationResult(
            role=predicted_role,
            confidence=float(confidence)
        )
        
        # Add all scores if requested
        if return_all_scores:
            result.all_scores = {
                role.value: float(probs[0, role_id].item())
                for role_id, role in self.id2label.items()
            }
        
        return result
    
    def classify_document(
        self,
        sentences: List[str],
        batch_size: int = 16,
        return_all_scores: bool = False
    ) -> List[ClassificationResult]:
        """
        Classify multiple sentences (batch processing).
        
        Args:
            sentences: List of sentence texts
            batch_size: Batch size for inference
            return_all_scores: Whether to return scores for all roles
        
        Returns:
            List of ClassificationResult objects
        """
        if not sentences:
            return []
        
        logger.info(f"🔄 Classifying {len(sentences)} sentences...")
        
        results = []
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Process each prediction in batch
            for j in range(len(batch)):
                predicted_id = torch.argmax(probs[j]).item()
                confidence = probs[j, predicted_id].item()
                predicted_role = self.id2label[predicted_id]
                
                result = ClassificationResult(
                    role=predicted_role,
                    confidence=float(confidence)
                )
                
                if return_all_scores:
                    result.all_scores = {
                        role.value: float(probs[j, role_id].item())
                        for role_id, role in self.id2label.items()
                    }
                
                results.append(result)
            
            logger.info(
                f"📊 Batch {i//batch_size + 1}/{(len(sentences) + batch_size - 1)//batch_size} complete"
            )
        
        logger.info(f"✅ Classification complete!")
        
        return results
    
    def get_role_distribution(
        self,
        classifications: List[ClassificationResult]
    ) -> dict:
        """
        Get distribution of roles in a classified document.
        
        Args:
            classifications: List of ClassificationResult objects
        
        Returns:
            Dict with role counts and percentages
        """
        from collections import Counter
        
        role_counts = Counter([c.role for c in classifications])
        total = len(classifications)
        
        distribution = {
            role.value: {
                "count": role_counts.get(role, 0),
                "percentage": (role_counts.get(role, 0) / total * 100) if total > 0 else 0
            }
            for role in RhetoricalRole
        }
        
        return distribution
