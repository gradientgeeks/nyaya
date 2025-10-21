#!/bin/bash
# Nyaya Setup Checker - Verifies environment setup

echo "🔍 Nyaya Setup Checker"
echo "======================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

check_pass=0
check_fail=0
check_warn=0

# Function to check
check_item() {
    local name="$1"
    local command="$2"
    local type="${3:-required}"  # required or optional
    
    if eval "$command" &> /dev/null; then
        echo -e "${GREEN}✅ $name${NC}"
        ((check_pass++))
        return 0
    else
        if [ "$type" = "required" ]; then
            echo -e "${RED}❌ $name${NC}"
            ((check_fail++))
        else
            echo -e "${YELLOW}⚠️  $name${NC}"
            ((check_warn++))
        fi
        return 1
    fi
}

# System Requirements
echo -e "${BLUE}System Requirements:${NC}"
check_item "Python 3.12+" "python3 --version | grep -E 'Python 3\.(1[2-9]|[2-9][0-9])'" "optional"
check_item "Node.js 18+" "node --version | grep -E 'v(1[8-9]|[2-9][0-9])\.'"
check_item "npm" "command -v npm"
echo ""

# Backend Setup
echo -e "${BLUE}Backend Setup:${NC}"
check_item "Backend .env exists" "test -f backend/.env"
if [ -f backend/.env ]; then
    check_item "PINECONE_API_KEY set" "grep -q '^PINECONE_API_KEY=' backend/.env"
    check_item "HF_TOKEN set" "grep -q '^HF_TOKEN=' backend/.env"
    check_item "GOOGLE_CLOUD_PROJECT set" "grep -q '^GOOGLE_CLOUD_PROJECT=' backend/.env" "optional"
fi
echo ""

# Frontend Setup
echo -e "${BLUE}Frontend Setup:${NC}"
check_item "Client .env exists" "test -f client/.env"
check_item "node_modules installed" "test -d client/node_modules"
echo ""

# Port Availability
echo -e "${BLUE}Port Availability:${NC}"
check_item "Port 8000 available" "! lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null"
check_item "Port 5173 available" "! lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null"
echo ""

# API Keys (if .env exists)
if [ -f backend/.env ]; then
    echo -e "${BLUE}API Key Validation:${NC}"
    
    # Check if keys look valid (not example values)
    if grep -q "PINECONE_API_KEY=pcsk_xxxxx" backend/.env 2>/dev/null; then
        echo -e "${RED}❌ PINECONE_API_KEY is still example value${NC}"
        ((check_fail++))
    elif grep -q "PINECONE_API_KEY=pcsk_" backend/.env 2>/dev/null; then
        echo -e "${GREEN}✅ PINECONE_API_KEY looks valid${NC}"
        ((check_pass++))
    fi
    
    if grep -q "HF_TOKEN=hf_xxxxx" backend/.env 2>/dev/null; then
        echo -e "${RED}❌ HF_TOKEN is still example value${NC}"
        ((check_fail++))
    elif grep -q "HF_TOKEN=hf_" backend/.env 2>/dev/null; then
        echo -e "${GREEN}✅ HF_TOKEN looks valid${NC}"
        ((check_pass++))
    fi
    echo ""
fi

# Summary
echo "======================"
echo -e "${BLUE}Summary:${NC}"
echo -e "${GREEN}✅ Passed: $check_pass${NC}"
echo -e "${RED}❌ Failed: $check_fail${NC}"
echo -e "${YELLOW}⚠️  Warnings: $check_warn${NC}"
echo ""

if [ $check_fail -eq 0 ]; then
    echo -e "${GREEN}🎉 Ready to run! Execute: ./start.sh${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Please fix the failed checks before running.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  1. Copy environment files:"
    echo "     cp backend/.env.example backend/.env"
    echo "     cp client/.env.example client/.env"
    echo ""
    echo "  2. Add your API keys to backend/.env:"
    echo "     - PINECONE_API_KEY"
    echo "     - HF_TOKEN"
    echo "     - GOOGLE_CLOUD_PROJECT (optional)"
    echo ""
    echo "  3. Install frontend dependencies:"
    echo "     cd client && npm install"
    echo ""
    echo "See INTEGRATION_GUIDE.md for detailed setup instructions."
    exit 1
fi
