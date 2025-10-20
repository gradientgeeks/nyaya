#!/bin/bash
# Quick setup script for Pinecone + EmbeddingGemma

set -e  # Exit on error

echo "🚀 Setting up Pinecone + EmbeddingGemma for Nyaya..."
echo ""

# Check if we're in the backend directory
if [ ! -f "test_pinecone_embedding.py" ]; then
    echo "❌ Error: Please run this from the backend/ directory"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python packages..."
uv pip install sentence-transformers python-dotenv pinecone

echo "📦 Installing transformers with EmbeddingGemma support..."
uv pip install "git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview"

echo ""
echo "✅ Dependencies installed successfully!"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Get Hugging Face access:"
echo "   - Go to: https://huggingface.co/google/embeddinggemma-300M"
echo "   - Click 'Acknowledge license'"
echo "   - Get token from: https://huggingface.co/settings/tokens"
echo ""
echo "2. Login to Hugging Face (choose one):"
echo "   Option A: Interactive"
echo "     python -c 'from huggingface_hub import login; login()'"
echo ""
echo "   Option B: Environment variable"
echo "     export HF_TOKEN=your_token_here"
echo ""
echo "   Option C: Add to .env"
echo "     echo 'HF_TOKEN=your_token_here' >> .env"
echo ""
echo "3. Run the test script:"
echo "   python test_pinecone_embedding.py"
echo ""
echo "📚 See PINECONE_SETUP.md for detailed documentation"
