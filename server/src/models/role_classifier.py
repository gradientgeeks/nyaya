"""
Rhetorical Role Classifier for Legal Documents

This module implements multiple approaches for classifying rhetorical roles
in legal documents as described in the LegalSeg paper.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import numpy as np
from enum import Enum
import spacy
import logging

logger = logging.getLogger(__name__)

class RhetoricalRole(Enum):
    """Rhetorical roles in legal documents"""
    FACTS = "Facts"
    ISSUE = "Issue"
    ARGUMENTS_PETITIONER = "Arguments of Petitioner"
    ARGUMENTS_RESPONDENT = "Arguments of Respondent"
    REASONING = "Reasoning"
    DECISION = "Decision"
    NONE = "None"

class InLegalBERTClassifier(nn.Module):
    """
    InLegalBERT-based classifier for rhetorical role classification
    Supports different context configurations as described in the paper
    """
    
    def __init__(self, model_name: str = "law-ai/InLegalBERT", 
                 num_labels: int = 7, context_mode: str = "single"):
        super().__init__()
        self.context_mode = context_mode  # "single", "prev", "prev_two", "surrounding"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class BiLSTMCRFClassifier(nn.Module):
    """
    Hierarchical BiLSTM-CRF model as described in the LegalSeg paper
    """
    
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 256, 
                 num_labels: int = 7, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True,
            dropout=0.1
        )
        
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        
        # CRF transitions
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))
        
    def _forward_alg(self, feats):
        """Forward algorithm for CRF"""
        batch_size, seq_len, num_tags = feats.size()
        
        # Initialize alpha at timestep 0
        alpha = self.start_transitions.view(1, -1) + feats[:, 0]
        
        for i in range(1, seq_len):
            alpha_t = []
            for next_tag in range(num_tags):
                emit_score = feats[:, i, next_tag].view(batch_size, 1)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = alpha + trans_score + emit_score
                alpha_t.append(torch.logsumexp(next_tag_var, dim=1))
            alpha = torch.stack(alpha_t, dim=1)
        
        terminal_var = alpha + self.end_transitions.view(1, -1)
        scores = torch.logsumexp(terminal_var, dim=1)
        
        return scores
    
    def _score_sentence(self, feats, tags):
        """Score a sentence given tags"""
        batch_size, seq_len = tags.size()
        score = torch.zeros(batch_size)
        
        # Start transition
        score += self.start_transitions[tags[:, 0]]
        
        # Emission scores
        for i in range(seq_len):
            score += feats[range(batch_size), i, tags[:, i]]
            
        # Transition scores
        for i in range(1, seq_len):
            score += self.transitions[tags[:, i], tags[:, i-1]]
            
        # End transition
        score += self.end_transitions[tags[:, -1]]
        
        return score
    
    def neg_log_likelihood(self, sentence_embeddings, tags):
        """Calculate negative log likelihood for training"""
        feats = self._get_lstm_features(sentence_embeddings)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.mean(forward_score - gold_score)
    
    def _get_lstm_features(self, sentence_embeddings):
        """Get LSTM features from sentence embeddings"""
        lstm_out, _ = self.lstm(sentence_embeddings)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def forward(self, sentence_embeddings):
        """Forward pass for inference"""
        lstm_feats = self._get_lstm_features(sentence_embeddings)
        return self._viterbi_decode(lstm_feats)
    
    def _viterbi_decode(self, feats):
        """Viterbi decoding for best path"""
        batch_size, seq_len, num_tags = feats.size()
        
        # Initialize backpointers and path scores
        backpointers = []
        
        # Initialize the viterbi variables in log space
        init_vvars = self.start_transitions.view(1, -1) + feats[:, 0]
        forward_var = init_vvars
        
        for i in range(1, seq_len):
            bptrs_t = []
            viterbivars_t = []
            
            for next_tag in range(num_tags):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var, dim=1)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[range(batch_size), best_tag_id])
            
            forward_var = torch.stack(viterbivars_t, dim=1) + feats[:, i]
            backpointers.append(torch.stack(bptrs_t, dim=1))
        
        # Transition to STOP_TAG
        terminal_var = forward_var + self.end_transitions.view(1, -1)
        best_tag_id = torch.argmax(terminal_var, dim=1)
        
        # Follow the back pointers to decode the best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[range(batch_size), best_tag_id]
            best_path.append(best_tag_id)
        
        # Remove the last tag (start tag)
        best_path.reverse()
        return torch.stack(best_path, dim=1)

class RoleClassifier:
    """
    Main role classifier interface supporting multiple models
    """
    
    def __init__(self, model_type: str = "inlegalbert", device: str = "cpu"):
        self.model_type = model_type
        self.device = device
        self.nlp = spacy.load("en_core_web_sm")
        self.model = None
        self.role_to_id = {role.value: i for i, role in enumerate(RhetoricalRole)}
        self.id_to_role = {i: role.value for i, role in enumerate(RhetoricalRole)}
        
        self._load_model()
    
    def _load_model(self):
        """Load the specified model"""
        if self.model_type == "inlegalbert":
            self.model = InLegalBERTClassifier()
            self.tokenizer = self.model.tokenizer
        elif self.model_type == "bilstm_crf":
            self.model = BiLSTMCRFClassifier()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.to(self.device)
        logger.info(f"Loaded {self.model_type} model on {self.device}")
    
    def preprocess_document(self, document_text: str) -> List[str]:
        """
        Preprocess legal document and extract sentences
        """
        doc = self.nlp(document_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def prepare_input(self, sentences: List[str], sentence_idx: int, 
                     context_mode: str = "single") -> Dict:
        """
        Prepare input for the model based on context mode
        """
        if context_mode == "single":
            text = sentences[sentence_idx]
        elif context_mode == "prev":
            if sentence_idx > 0:
                text = f"{sentences[sentence_idx-1]} [SEP] {sentences[sentence_idx]}"
            else:
                text = sentences[sentence_idx]
        elif context_mode == "prev_two":
            context_sentences = []
            if sentence_idx > 1:
                context_sentences.append(sentences[sentence_idx-2])
            if sentence_idx > 0:
                context_sentences.append(sentences[sentence_idx-1])
            context_sentences.append(sentences[sentence_idx])
            text = " [SEP] ".join(context_sentences)
        elif context_mode == "surrounding":
            context_sentences = []
            if sentence_idx > 0:
                context_sentences.append(sentences[sentence_idx-1])
            context_sentences.append(sentences[sentence_idx])
            if sentence_idx < len(sentences) - 1:
                context_sentences.append(sentences[sentence_idx+1])
            text = " [SEP] ".join(context_sentences)
        else:
            text = sentences[sentence_idx]
        
        if self.model_type == "inlegalbert":
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            return encoding
        else:
            # For BiLSTM-CRF, we need sentence embeddings
            # This would typically use sent2vec or another sentence encoder
            return {"text": text}
    
    def classify_document(self, document_text: str, 
                         context_mode: str = "single") -> List[Dict[str, str]]:
        """
        Classify rhetorical roles for all sentences in a document
        """
        sentences = self.preprocess_document(document_text)
        results = []
        
        self.model.eval()
        with torch.no_grad():
            for i, sentence in enumerate(sentences):
                if self.model_type == "inlegalbert":
                    inputs = self.prepare_input(sentences, i, context_mode)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    logits = self.model(**inputs)
                    predicted_id = torch.argmax(logits, dim=-1).item()
                    predicted_role = self.id_to_role[predicted_id]
                    confidence = torch.softmax(logits, dim=-1).max().item()
                    
                    results.append({
                        "sentence": sentence,
                        "role": predicted_role,
                        "confidence": confidence,
                        "sentence_index": i
                    })
        
        return results
    
    def load_pretrained_weights(self, weights_path: str):
        """Load pretrained weights"""
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pretrained weights from {weights_path}")
    
    def save_model(self, save_path: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'role_to_id': self.role_to_id
        }, save_path)
        logger.info(f"Model saved to {save_path}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    classifier = RoleClassifier(model_type="inlegalbert")
    
    # Sample legal text
    sample_text = """
    The petitioner filed a writ petition challenging the constitutional validity of Section 377.
    The main issue in this case is whether Section 377 violates fundamental rights.
    The petitioner argues that Section 377 is discriminatory and violates Article 14.
    The respondent contends that Section 377 is constitutionally valid.
    The court finds that Section 377 infringes upon the right to privacy and equality.
    Therefore, Section 377 is hereby declared unconstitutional.
    """
    
    # Classify rhetorical roles
    results = classifier.classify_document(sample_text, context_mode="prev")
    
    for result in results:
        print(f"Sentence: {result['sentence'][:50]}...")
        print(f"Role: {result['role']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("-" * 50)