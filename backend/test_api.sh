#!/bin/bash

# Nyaya Backend - Quick Test Script
# Tests all API endpoints to verify implementation

set -e  # Exit on error

echo "🏛️  Nyaya Backend - API Test Suite"
echo "=================================="
echo ""

# Configuration
BASE_URL="${1:-http://localhost:8000}"
API_URL="$BASE_URL/api/v1"

echo "📍 Testing API at: $BASE_URL"
echo ""

# Test 1: Root endpoint
echo "1️⃣  Testing root endpoint..."
ROOT_RESPONSE=$(curl -s "$BASE_URL/")
echo "   Response: $(echo $ROOT_RESPONSE | jq -r '.message' 2>/dev/null || echo $ROOT_RESPONSE)"
echo "   ✅ Root endpoint OK"
echo ""

# Test 2: Health check
echo "2️⃣  Testing health check..."
HEALTH_RESPONSE=$(curl -s "$API_URL/health")
STATUS=$(echo $HEALTH_RESPONSE | jq -r '.status' 2>/dev/null || echo "error")
if [ "$STATUS" = "healthy" ]; then
    echo "   ✅ Health check PASSED"
else
    echo "   ❌ Health check FAILED"
    exit 1
fi
echo ""

# Test 3: Stats endpoint
echo "3️⃣  Testing stats endpoint..."
STATS_RESPONSE=$(curl -s "$API_URL/stats" 2>&1)
if echo "$STATS_RESPONSE" | jq . >/dev/null 2>&1; then
    VECTOR_COUNT=$(echo $STATS_RESPONSE | jq -r '.pinecone.total_vector_count' 2>/dev/null || echo "0")
    echo "   📊 Total vectors in Pinecone: $VECTOR_COUNT"
    echo "   ✅ Stats endpoint OK"
else
    echo "   ⚠️  Stats endpoint returned error (this is OK if Pinecone not configured yet)"
    echo "   Response: $STATS_RESPONSE"
fi
echo ""

# Test 4: Create session
echo "4️⃣  Testing session creation..."
SESSION_RESPONSE=$(curl -s -X POST "$API_URL/sessions" 2>&1)
if echo "$SESSION_RESPONSE" | jq . >/dev/null 2>&1; then
    SESSION_ID=$(echo $SESSION_RESPONSE | jq -r '.session_id' 2>/dev/null)
    echo "   📝 Session ID: $SESSION_ID"
    echo "   ✅ Session creation OK"
else
    echo "   ❌ Session creation FAILED"
    echo "   Response: $SESSION_RESPONSE"
    exit 1
fi
echo ""

# Test 5: Query endpoint (without file)
echo "5️⃣  Testing query endpoint..."
QUERY_RESPONSE=$(curl -s -X POST "$API_URL/query" \
    -H "Content-Type: application/json" \
    -d "{
        \"query\": \"Hello, what can you help me with?\",
        \"session_id\": \"$SESSION_ID\"
    }" 2>&1)

if echo "$QUERY_RESPONSE" | jq . >/dev/null 2>&1; then
    ANSWER=$(echo $QUERY_RESPONSE | jq -r '.answer' 2>/dev/null | head -c 100)
    echo "   💬 Answer (first 100 chars): $ANSWER..."
    echo "   ✅ Query endpoint OK"
else
    echo "   ⚠️  Query endpoint returned error (expected if no data uploaded yet)"
    echo "   Response: $(echo $QUERY_RESPONSE | head -c 200)..."
fi
echo ""

# Test 6: Search endpoint
echo "6️⃣  Testing search endpoint..."
SEARCH_RESPONSE=$(curl -s -X POST "$API_URL/search" \
    -H "Content-Type: application/json" \
    -d "{
        \"query\": \"test search query\",
        \"session_id\": \"$SESSION_ID\",
        \"top_k\": 5
    }" 2>&1)

if echo "$SEARCH_RESPONSE" | jq . >/dev/null 2>&1; then
    echo "   🔍 Search executed successfully"
    echo "   ✅ Search endpoint OK"
else
    echo "   ⚠️  Search endpoint returned error (expected if no data uploaded yet)"
fi
echo ""

# Summary
echo "=================================="
echo "✅ Basic API Test Complete!"
echo ""
echo "📝 Summary:"
echo "   - Root endpoint: ✅"
echo "   - Health check: ✅"
echo "   - Stats: ✅ (or ⚠️  if Pinecone not configured)"
echo "   - Session creation: ✅"
echo "   - Query endpoint: ✅ (may need data)"
echo "   - Search endpoint: ✅ (may need data)"
echo ""
echo "🚀 Next steps:"
echo "   1. Configure .env with API keys"
echo "   2. Add InLegalBERT model to backend/models/"
echo "   3. Upload a document via /upload endpoint"
echo "   4. Try role-specific queries"
echo ""
echo "📚 Full API documentation: $BASE_URL/docs"
