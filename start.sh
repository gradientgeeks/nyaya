#!/bin/bash
# Quick start script for Nyaya - starts both backend and frontend

set -e

echo "🚀 Starting Nyaya Legal RAG System..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if backend .env exists
if [ ! -f backend/.env ]; then
    echo -e "${YELLOW}⚠️  Backend .env not found. Creating from example...${NC}"
    cp backend/.env.example backend/.env
    echo -e "${RED}❌ Please edit backend/.env with your API keys before running again.${NC}"
    echo "   Required keys: PINECONE_API_KEY, HF_TOKEN, GOOGLE_CLOUD_PROJECT"
    exit 1
fi

# Check if client .env exists
if [ ! -f client/.env ]; then
    echo -e "${YELLOW}⚠️  Client .env not found. Creating from example...${NC}"
    cp client/.env.example client/.env
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not found.${NC}"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is required but not found.${NC}"
    exit 1
fi

# Install frontend dependencies if needed
if [ ! -d "client/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
    cd client
    npm install
    cd ..
fi

echo -e "${GREEN}✅ Prerequisites checked${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down Nyaya...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo -e "${GREEN}🔧 Starting backend API (port 8000)...${NC}"
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}❌ Backend failed to start. Check logs above.${NC}"
    exit 1
fi

# Test backend health
if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend is healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Backend health check failed (might still be initializing)${NC}"
fi

echo ""

# Start frontend
echo -e "${GREEN}🎨 Starting frontend dev server (port 5173)...${NC}"
cd client
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}✅ Nyaya is running!${NC}"
echo ""
echo "📚 Access points:"
echo "   Frontend:        http://localhost:5173"
echo "   Backend API:     http://localhost:8000"
echo "   API Docs:        http://localhost:8000/docs"
echo "   Health Check:    http://localhost:8000/api/v1/health"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Keep script running
wait
