"""
Unsupervised Clustering Approach for Rhetorical Role Classification

This module implements clustering-based (unsupervised) methods for identifying
rhetorical roles in legal documents as an alternative to supervised classification.

Approaches:
1. K-Means clustering on sentence embeddings
2. Hierarchical clustering
3. DBSCAN for density-based clustering
4. Topic modeling (LDA)

Can be compared with supervised InLegalBERT classifier for performance analysis.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json

import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

from pydantic import BaseModel
import spacy

logger = logging.getLogger(__name__)


class ClusterResult(BaseModel):
    """Result of clustering operation"""
    sentence: str
    cluster_id: int
    confidence: float  # Distance to cluster center (0-1, higher is better)
    predicted_role: str
    embedding: Optional[List[float]] = None


class ClusteringConfig(BaseModel):
    """Configuration for clustering approach"""
    num_clusters: int = 7  # One per rhetorical role
    embedding_model: str = "all-MiniLM-L6-v2"
    clustering_algorithm: str = "kmeans"  # kmeans, hierarchical, dbscan, hdbscan
    dimensionality_reduction: Optional[str] = None  # umap, pca, tsne
    use_tfidf_features: bool = False
    min_cluster_size: int = 5  # For HDBSCAN


class ClusteringRoleClassifier:
    """
    Unsupervised clustering-based role classifier
    
    Maps clusters to rhetorical roles using various heuristics:
    1. Keyword matching
    2. Sentence position patterns
    3. Linguistic features
    """
    
    # Rhetorical roles
    ROLES = [
        "None",
        "Facts",
        "Issue",
        "Arguments of Petitioner",
        "Arguments of Respondent",
        "Reasoning",
        "Decision"
    ]
    
    # Keywords for role identification (for cluster labeling)
    ROLE_KEYWORDS = {
        "Facts": [
            "petitioner", "filed", "appellant", "case", "incident", "alleged",
            "facts", "background", "circumstances", "events", "occurred",
            "party", "parties", "between", "against"
        ],
        "Issue": [
            "issue", "question", "whether", "matter", "dispute", "challenge",
            "raised", "contention", "point", "controversy", "arises"
        ],
        "Arguments of Petitioner": [
            "petitioner argues", "petitioner contends", "petitioner submits",
            "counsel for petitioner", "learned counsel", "it is argued",
            "it is contended", "submission", "grounds"
        ],
        "Arguments of Respondent": [
            "respondent argues", "respondent contends", "respondent submits",
            "counsel for respondent", "it is submitted", "reply", "counter"
        ],
        "Reasoning": [
            "court", "view", "opinion", "analyzed", "considered", "examined",
            "find", "hold", "observed", "noted", "reasoning", "analysis",
            "principles", "law", "held that"
        ],
        "Decision": [
            "decided", "dismissed", "allowed", "disposed", "ordered", "held",
            "judgment", "conclusion", "result", "accordingly", "therefore",
            "thus", "final order", "decree"
        ],
        "None": []
    }
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize clustering-based classifier
        
        Args:
            config: Configuration for clustering approach
        """
        self.config = config or ClusteringConfig()
        
        # Load spaCy for preprocessing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        
        # TF-IDF vectorizer
        if self.config.use_tfidf_features:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 3),
                stop_words='english'
            )
        
        # Clustering model (initialized on fit)
        self.clustering_model = None
        self.scaler = StandardScaler()
        self.dim_reducer = None
        
        # Cluster to role mapping (learned after clustering)
        self.cluster_to_role = {}
        
        logger.info(f"Clustering classifier initialized with {self.config.clustering_algorithm}")
    
    def preprocess_document(self, document_text: str) -> List[str]:
        """Extract sentences from document"""
        if self.nlp:
            doc = self.nlp(document_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Simple sentence splitting
            sentences = [s.strip() for s in document_text.split('.') if s.strip()]
        
        return sentences
    
    def extract_features(self, sentences: List[str]) -> np.ndarray:
        """
        Extract features from sentences
        
        Args:
            sentences: List of sentences
            
        Returns:
            Feature matrix (n_sentences, n_features)
        """
        # Semantic embeddings
        logger.info(f"Generating embeddings for {len(sentences)} sentences...")
        embeddings = self.embedding_model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Optionally add TF-IDF features
        if self.config.use_tfidf_features:
            logger.info("Extracting TF-IDF features...")
            tfidf_features = self.tfidf_vectorizer.fit_transform(sentences).toarray()
            
            # Combine embeddings and TF-IDF
            features = np.hstack([embeddings, tfidf_features])
        else:
            features = embeddings
        
        # Normalize features
        features = self.scaler.fit_transform(features)
        
        # Dimensionality reduction if configured
        if self.config.dimensionality_reduction:
            logger.info(f"Applying {self.config.dimensionality_reduction} dimensionality reduction...")
            features = self._apply_dimensionality_reduction(features)
        
        return features
    
    def _apply_dimensionality_reduction(self, features: np.ndarray, 
                                       n_components: int = 50) -> np.ndarray:
        """Apply dimensionality reduction"""
        if self.config.dimensionality_reduction == "umap":
            self.dim_reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1
            )
        elif self.config.dimensionality_reduction == "pca":
            self.dim_reducer = TruncatedSVD(n_components=n_components, random_state=42)
        else:
            return features
        
        reduced_features = self.dim_reducer.fit_transform(features)
        return reduced_features
    
    def fit(self, sentences: List[str], ground_truth_roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fit clustering model on sentences
        
        Args:
            sentences: List of sentences to cluster
            ground_truth_roles: Optional ground truth roles for evaluation
            
        Returns:
            Clustering metrics and statistics
        """
        logger.info(f"Fitting clustering model on {len(sentences)} sentences...")
        
        # Extract features
        features = self.extract_features(sentences)
        
        # Initialize and fit clustering model
        if self.config.clustering_algorithm == "kmeans":
            self.clustering_model = KMeans(
                n_clusters=self.config.num_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            cluster_labels = self.clustering_model.fit_predict(features)
            
        elif self.config.clustering_algorithm == "hierarchical":
            self.clustering_model = AgglomerativeClustering(
                n_clusters=self.config.num_clusters,
                linkage='ward'
            )
            cluster_labels = self.clustering_model.fit_predict(features)
            
        elif self.config.clustering_algorithm == "dbscan":
            self.clustering_model = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            cluster_labels = self.clustering_model.fit_predict(features)
            
        elif self.config.clustering_algorithm == "hdbscan":
            self.clustering_model = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=3
            )
            cluster_labels = self.clustering_model.fit_predict(features)
        
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.config.clustering_algorithm}")
        
        logger.info(f"Clustering complete. Found {len(np.unique(cluster_labels))} clusters")
        
        # Map clusters to roles
        self.cluster_to_role = self._map_clusters_to_roles(sentences, cluster_labels)
        
        # Calculate clustering metrics
        metrics = self._calculate_metrics(features, cluster_labels, sentences, ground_truth_roles)
        
        return metrics
    
    def _map_clusters_to_roles(self, sentences: List[str], cluster_labels: np.ndarray) -> Dict[int, str]:
        """
        Map cluster IDs to rhetorical roles using keyword matching and heuristics
        
        Args:
            sentences: Original sentences
            cluster_labels: Cluster assignments
            
        Returns:
            Mapping from cluster ID to role name
        """
        cluster_to_role = {}
        
        # For each cluster, analyze its sentences
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Noise in DBSCAN/HDBSCAN
                cluster_to_role[cluster_id] = "None"
                continue
            
            # Get sentences in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_sentences = [sentences[i] for i in range(len(sentences)) if cluster_mask[i]]
            
            # Score each role based on keyword matches
            role_scores = {}
            for role, keywords in self.ROLE_KEYWORDS.items():
                score = 0
                for sentence in cluster_sentences:
                    sentence_lower = sentence.lower()
                    score += sum(1 for keyword in keywords if keyword in sentence_lower)
                
                role_scores[role] = score / max(len(cluster_sentences), 1)
            
            # Assign role with highest score
            if role_scores:
                best_role = max(role_scores.items(), key=lambda x: x[1])[0]
            else:
                best_role = "None"
            
            cluster_to_role[cluster_id] = best_role
            logger.info(f"Cluster {cluster_id} -> {best_role} (score: {role_scores.get(best_role, 0):.2f})")
        
        return cluster_to_role
    
    def _calculate_metrics(self, features: np.ndarray, cluster_labels: np.ndarray, 
                          sentences: List[str], ground_truth_roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate clustering quality metrics"""
        metrics = {}
        
        # Filter out noise points (label -1)
        valid_mask = cluster_labels != -1
        if valid_mask.sum() > 0:
            valid_features = features[valid_mask]
            valid_labels = cluster_labels[valid_mask]
            
            # Internal metrics (unsupervised)
            if len(np.unique(valid_labels)) > 1:
                metrics["silhouette_score"] = silhouette_score(valid_features, valid_labels)
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(valid_features, valid_labels)
                metrics["davies_bouldin_score"] = davies_bouldin_score(valid_features, valid_labels)
        
        # Cluster statistics
        unique_clusters = np.unique(cluster_labels)
        metrics["num_clusters"] = len(unique_clusters)
        metrics["cluster_sizes"] = {
            int(c): int((cluster_labels == c).sum()) 
            for c in unique_clusters
        }
        
        # External metrics (if ground truth available)
        if ground_truth_roles:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure
            
            # Map roles to numeric labels
            role_to_id = {role: i for i, role in enumerate(self.ROLES)}
            gt_labels = np.array([role_to_id.get(role, 0) for role in ground_truth_roles])
            
            metrics["adjusted_rand_score"] = adjusted_rand_score(gt_labels, cluster_labels)
            metrics["normalized_mutual_info"] = normalized_mutual_info_score(gt_labels, cluster_labels)
            
            h, c, v = homogeneity_completeness_v_measure(gt_labels, cluster_labels)
            metrics["homogeneity"] = h
            metrics["completeness"] = c
            metrics["v_measure"] = v
        
        return metrics
    
    def predict(self, sentences: List[str]) -> List[ClusterResult]:
        """
        Predict roles for new sentences
        
        Args:
            sentences: List of sentences to classify
            
        Returns:
            List of clustering results with predicted roles
        """
        if self.clustering_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract features
        embeddings = self.embedding_model.encode(sentences, convert_to_numpy=True)
        features = self.scaler.transform(embeddings)
        
        if self.dim_reducer:
            features = self.dim_reducer.transform(features)
        
        # Predict clusters
        if hasattr(self.clustering_model, 'predict'):
            cluster_labels = self.clustering_model.predict(features)
        else:
            # For models without predict (like hierarchical), use closest cluster
            from scipy.spatial.distance import cdist
            cluster_centers = self._get_cluster_centers(features, 
                                                        self.clustering_model.labels_)
            distances = cdist(features, cluster_centers, metric='euclidean')
            cluster_labels = np.argmin(distances, axis=1)
        
        # Calculate confidence based on distance to cluster center
        confidences = self._calculate_confidence(features, cluster_labels)
        
        # Map to roles
        results = []
        for i, (sentence, cluster_id, confidence) in enumerate(zip(sentences, cluster_labels, confidences)):
            predicted_role = self.cluster_to_role.get(int(cluster_id), "None")
            
            results.append(ClusterResult(
                sentence=sentence,
                cluster_id=int(cluster_id),
                confidence=float(confidence),
                predicted_role=predicted_role,
                embedding=embeddings[i].tolist()
            ))
        
        return results
    
    def _get_cluster_centers(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers"""
        unique_labels = np.unique(labels)
        centers = []
        
        for label in unique_labels:
            if label != -1:  # Skip noise
                mask = labels == label
                center = features[mask].mean(axis=0)
                centers.append(center)
        
        return np.array(centers)
    
    def _calculate_confidence(self, features: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """
        Calculate confidence scores based on distance to cluster center
        
        Confidence = 1 / (1 + normalized_distance)
        """
        confidences = np.zeros(len(features))
        
        # Get cluster centers
        if hasattr(self.clustering_model, 'cluster_centers_'):
            cluster_centers = self.clustering_model.cluster_centers_
        else:
            cluster_centers = self._get_cluster_centers(features, cluster_labels)
        
        # Calculate distance to assigned cluster center
        for i, (feature, label) in enumerate(zip(features, cluster_labels)):
            if label >= 0 and label < len(cluster_centers):
                distance = np.linalg.norm(feature - cluster_centers[label])
                # Normalize to 0-1 range (higher is better)
                confidences[i] = 1.0 / (1.0 + distance)
            else:
                confidences[i] = 0.0  # Noise point
        
        return confidences
    
    def classify_document(self, document_text: str) -> List[ClusterResult]:
        """
        Classify all sentences in a document
        
        Args:
            document_text: Full document text
            
        Returns:
            List of clustering results for each sentence
        """
        sentences = self.preprocess_document(document_text)
        return self.predict(sentences)
    
    def save_model(self, save_path: str):
        """Save clustering model and mappings"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save clustering model
        with open(save_path / "clustering_model.pkl", 'wb') as f:
            pickle.dump(self.clustering_model, f)
        
        # Save scaler
        with open(save_path / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save dimensionality reducer
        if self.dim_reducer:
            with open(save_path / "dim_reducer.pkl", 'wb') as f:
                pickle.dump(self.dim_reducer, f)
        
        # Save cluster to role mapping
        with open(save_path / "cluster_to_role.json", 'w') as f:
            json.dump(self.cluster_to_role, f, indent=2)
        
        # Save configuration
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config.dict(), f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load clustering model and mappings"""
        load_path = Path(load_path)
        
        # Load clustering model
        with open(load_path / "clustering_model.pkl", 'rb') as f:
            self.clustering_model = pickle.load(f)
        
        # Load scaler
        with open(load_path / "scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load dimensionality reducer
        if (load_path / "dim_reducer.pkl").exists():
            with open(load_path / "dim_reducer.pkl", 'rb') as f:
                self.dim_reducer = pickle.load(f)
        
        # Load cluster to role mapping
        with open(load_path / "cluster_to_role.json", 'r') as f:
            self.cluster_to_role = {int(k): v for k, v in json.load(f).items()}
        
        # Load configuration
        with open(load_path / "config.json", 'r') as f:
            self.config = ClusteringConfig(**json.load(f))
        
        logger.info(f"Model loaded from {load_path}")


# Example usage
if __name__ == "__main__":
    # Sample legal document
    sample_text = """
    The petitioner filed a writ petition challenging the constitutional validity of Section 377.
    The appellant contends that the impugned provision violates fundamental rights.
    The main issue in this case is whether Section 377 violates fundamental rights under Articles 14, 15, 19 and 21.
    The petitioner argues that Section 377 is discriminatory and violates Article 14.
    The petitioner further submits that the provision infringes upon the right to privacy.
    The respondent contends that Section 377 is constitutionally valid and necessary.
    The court has analyzed the constitutional provisions and previous judgments.
    The court finds that Section 377 infringes upon the right to privacy and equality.
    After considering all submissions, we hold that Section 377 is unconstitutional.
    Therefore, Section 377 is hereby declared unconstitutional and struck down.
    """
    
    # Initialize clustering classifier
    config = ClusteringConfig(
        num_clusters=7,
        clustering_algorithm="kmeans",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    classifier = ClusteringRoleClassifier(config)
    
    # Fit on sample data
    sentences = classifier.preprocess_document(sample_text)
    metrics = classifier.fit(sentences)
    
    print("\n" + "="*80)
    print("CLUSTERING METRICS")
    print("="*80)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    print("\n" + "="*80)
    print("CLUSTER TO ROLE MAPPING")
    print("="*80)
    for cluster_id, role in classifier.cluster_to_role.items():
        print(f"Cluster {cluster_id} -> {role}")
    
    # Predict roles
    print("\n" + "="*80)
    print("PREDICTIONS")
    print("="*80)
    results = classifier.classify_document(sample_text)
    
    for result in results:
        print(f"\nSentence: {result.sentence[:60]}...")
        print(f"Cluster: {result.cluster_id}")
        print(f"Role: {result.predicted_role}")
        print(f"Confidence: {result.confidence:.3f}")
        print("-" * 80)
