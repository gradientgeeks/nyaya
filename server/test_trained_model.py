"""
Test Script for Trained Role Classifier Model

This script tests your trained model on sample legal text to verify it works correctly.
Run this after downloading your trained model from the remote GPU.
"""

import sys
import os
from pathlib import Path

# Add server to path
sys.path.append(str(Path(__file__).parent))

from src.models.role_classifier import RoleClassifier
import torch

def test_trained_model(model_path: str, device: str = "cpu"):
    """
    Test the trained model with sample legal texts
    
    Args:
        model_path: Path to the trained model file (best_model.pt)
        device: Device to run on ("cpu" or "cuda")
    """
    
    print("=" * 80)
    print("🧪 TESTING TRAINED ROLE CLASSIFIER MODEL")
    print("=" * 80)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please ensure you've downloaded the trained model from remote GPU.")
        return False
    
    print(f"\n📂 Model Path: {model_path}")
    print(f"🖥️  Device: {device}")
    
    # Check model file size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📊 Model Size: {model_size_mb:.2f} MB")
    
    try:
        # Initialize classifier
        print("\n🔧 Initializing Role Classifier...")
        classifier = RoleClassifier(
            model_type="inlegalbert",
            device=device
        )
        print("✅ Classifier initialized")
        
        # Load trained weights
        print(f"\n📥 Loading trained weights from {model_path}...")
        classifier.load_pretrained_weights(model_path)
        print("✅ Trained weights loaded successfully")
        
        # Test with sample legal text
        print("\n" + "=" * 80)
        print("📝 TEST CASE 1: Typical Legal Document Structure")
        print("=" * 80)
        
        sample_text_1 = """
The petitioner filed a writ petition challenging the constitutional validity of Section 377 of the Indian Penal Code.
The primary issue in this case is whether Section 377 violates the fundamental rights guaranteed under Articles 14, 15, 19, and 21 of the Constitution.
The petitioner argues that Section 377 is discriminatory and violates the right to equality under Article 14 of the Constitution.
The petitioner further contends that the provision infringes upon the right to freedom of expression and personal liberty.
The respondent argues that Section 377 is constitutionally valid and serves a legitimate state interest in maintaining public morality.
The respondent contends that the provision does not violate any fundamental rights and is a reasonable restriction.
The Court finds that Section 377 infringes upon the right to privacy and dignity guaranteed under Article 21.
The Court observes that consensual sexual conduct between adults in private is protected by the right to privacy.
The Court reasons that Section 377 perpetuates discrimination against LGBTQ+ individuals and violates their dignity.
Therefore, the Court declares that Section 377, insofar as it criminalizes consensual sexual conduct between adults, is unconstitutional.
Accordingly, Section 377 is hereby struck down to the extent mentioned above.
        """
        
        results_1 = classifier.classify_document(sample_text_1, context_mode="prev")
        
        print("\n🎯 Classification Results:")
        print("-" * 80)
        for i, result in enumerate(results_1, 1):
            sentence = result['sentence'].strip()
            if sentence:
                print(f"\n{i}. Sentence (first 70 chars):")
                print(f"   {sentence[:70]}...")
                print(f"   👉 Predicted Role: {result['role']}")
                print(f"   📊 Confidence: {result['confidence']:.3f}")
        
        # Test with edge cases
        print("\n" + "=" * 80)
        print("📝 TEST CASE 2: Short Legal Statements")
        print("=" * 80)
        
        sample_text_2 = """
The appellant filed an appeal against the judgment of the High Court.
Whether the provisions of Section 10A are applicable to the present case?
The appellant submits that the impugned order is contrary to law.
The respondent maintains that the order is in accordance with the statute.
We find that the provisions of Section 10A are indeed applicable.
The appeal is dismissed.
        """
        
        results_2 = classifier.classify_document(sample_text_2, context_mode="prev")
        
        print("\n🎯 Classification Results:")
        print("-" * 80)
        for i, result in enumerate(results_2, 1):
            sentence = result['sentence'].strip()
            if sentence:
                print(f"\n{i}. {sentence}")
                print(f"   👉 Role: {result['role']} (Confidence: {result['confidence']:.3f})")
        
        # Calculate statistics
        print("\n" + "=" * 80)
        print("📊 CLASSIFICATION STATISTICS")
        print("=" * 80)
        
        all_results = results_1 + results_2
        
        # Count predictions by role
        role_counts = {}
        confidence_by_role = {}
        
        for result in all_results:
            role = result['role']
            confidence = result['confidence']
            
            role_counts[role] = role_counts.get(role, 0) + 1
            
            if role not in confidence_by_role:
                confidence_by_role[role] = []
            confidence_by_role[role].append(confidence)
        
        print("\n📈 Role Distribution:")
        for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_results)) * 100
            avg_confidence = sum(confidence_by_role[role]) / len(confidence_by_role[role])
            print(f"  {role:30} | Count: {count:2} ({percentage:5.1f}%) | Avg Confidence: {avg_confidence:.3f}")
        
        print(f"\n📊 Total Sentences Classified: {len(all_results)}")
        print(f"📊 Overall Average Confidence: {sum(r['confidence'] for r in all_results) / len(all_results):.3f}")
        
        # Check for low confidence predictions
        low_confidence = [r for r in all_results if r['confidence'] < 0.5]
        if low_confidence:
            print(f"\n⚠️  {len(low_confidence)} predictions with confidence < 0.5:")
            for result in low_confidence[:3]:  # Show first 3
                print(f"  - {result['sentence'][:50]}... | {result['role']} ({result['confidence']:.3f})")
        
        print("\n" + "=" * 80)
        print("✅ MODEL TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    # Configuration
    MODEL_PATH = "./trained_models/best_model.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Allow command line override
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        DEVICE = sys.argv[2]
    
    print("\n🚀 Starting Model Testing...")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Run test
    success = test_trained_model(MODEL_PATH, DEVICE)
    
    if success:
        print("\n✅ All tests passed!")
        print("\n📚 Next Steps:")
        print("  1. Review the classification results above")
        print("  2. If results look good, integrate the model into your server")
        print("  3. Update main.py to use the trained model")
        print("  4. Test with your own legal documents")
        print("  5. Deploy to production")
        return 0
    else:
        print("\n❌ Testing failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
