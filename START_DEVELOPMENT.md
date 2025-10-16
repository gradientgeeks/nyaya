# Quick Start Guide - Development Mode

This guide helps you quickly start both the backend and frontend services for development.

## Prerequisites Check

Before starting, ensure you have:

- [ ] Python 3.10+ installed (`python --version`)
- [ ] Node.js 18+ installed (`node --version`)
- [ ] uv or pip available for Python packages
- [ ] npm installed for Node packages
- [ ] Google Cloud credentials configured (optional for basic testing)

## Step-by-Step Startup

### Terminal 1: Start Backend

```bash
# Navigate to server directory
cd server

# Install dependencies (first time only)
# Option A: Using uv (recommended)
uv sync

# Option B: Using pip
pip install -r requirements.txt

# Set environment variables (optional)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export HOST=0.0.0.0
export PORT=8000
export RELOAD=true

# Start the server
python main.py
```

**Expected Output:**
```
INFO:     Initializing Legal Document Analysis System...
INFO:     Loading Agent Orchestrator...
INFO:     Loading RAG System...
INFO:     System initialization completed successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verify Backend:**
- Open http://localhost:8000/health - Should show system status
- Open http://localhost:8000/docs - Should show API documentation

### Terminal 2: Start Frontend

```bash
# Navigate to client directory
cd client

# Install dependencies (first time only)
npm install

# Copy environment file (first time only)
cp .env.example .env

# Start the development server
npm run dev
```

**Expected Output:**
```
VITE v7.1.3  ready in XXX ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

**Access Application:**
- Open http://localhost:5173 in your browser

## Quick Test

### 1. Upload a Document
1. Click the "+" button in the chat input
2. Select a PDF or TXT legal document
3. You'll see: "📄 **filename.pdf** is ready for upload. Please ask a question about this document, and I'll process it for you."

### 2. Ask a Question
Type a question like:
- "What are the main facts of this case?"
- "Summarize the key arguments"
- "What was the final decision?"

### 3. Follow-up Questions
After the first response, ask follow-up questions:
- "Can you explain the reasoning in more detail?"
- "What precedents were cited?"
- "What are the implications of this decision?"

## Troubleshooting

### Backend Won't Start

**Error: `ModuleNotFoundError: No module named 'fastapi'`**
```bash
# Reinstall dependencies
cd server
pip install -r requirements.txt
```

**Error: `Port 8000 already in use`**
```bash
# Check what's using the port
lsof -i :8000

# Use a different port
PORT=8080 python main.py
# Don't forget to update client .env: VITE_API_BASE_URL=http://localhost:8080
```

**Error: Google Cloud credentials not found**
```bash
# For testing without Vertex AI, you may need to modify the code or
# Set up credentials:
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"

# Or use gcloud auth
gcloud auth application-default login
```

### Frontend Won't Start

**Error: `Cannot find module`**
```bash
# Clear and reinstall
cd client
rm -rf node_modules package-lock.json
npm install
```

**Error: `Cannot connect to backend`**
- Ensure backend is running on port 8000
- Check `http://localhost:8000/health` directly
- Verify `.env` has correct `VITE_API_BASE_URL`

### Connection Issues

**Backend starts but frontend can't connect:**
1. Verify backend is accessible:
   ```bash
   curl http://localhost:8000/health
   ```
2. Check Vite proxy configuration in `client/vite.config.ts`
3. Look for CORS errors in browser console (F12)

## Development Tips

### Hot Reload
Both services support hot reload:
- **Backend**: Auto-reloads when Python files change (if `RELOAD=true`)
- **Frontend**: Vite provides instant HMR for React changes

### Viewing Logs
- **Backend**: Logs appear in Terminal 1
- **Frontend**: Logs appear in Terminal 2 and browser console (F12)

### Testing API Directly
Visit http://localhost:8000/docs to test API endpoints interactively.

### Environment Variables

**Backend** (server/.env):
```bash
HOST=0.0.0.0
PORT=8000
RELOAD=true
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

**Frontend** (client/.env):
```bash
VITE_API_BASE_URL=http://localhost:8000
```

## Next Steps

Once both services are running:

1. **Try Different Queries**: Test various types of questions
2. **Upload Multiple Documents**: Build a document library
3. **Test Multi-turn Conversations**: See how context is maintained
4. **Check the Documentation**: See `docs/INTEGRATION_GUIDE.md` for more details

## Production Deployment

For production deployment instructions, see:
- `docs/INTEGRATION_GUIDE.md` - Full deployment guide
- `client/README.md` - Frontend production build
- `server/README.md` - Backend production setup

## Getting Help

If you encounter issues:
1. Check the logs in both terminals
2. Review `docs/INTEGRATION_GUIDE.md` troubleshooting section
3. Verify all prerequisites are installed
4. Check that ports 8000 and 5173 are available
5. Ensure Google Cloud credentials are properly configured (if using Vertex AI)

## Quick Commands Reference

```bash
# Backend
cd server && python main.py

# Frontend  
cd client && npm run dev

# Build Frontend
cd client && npm run build

# Test Backend API
curl http://localhost:8000/health

# View API Docs
open http://localhost:8000/docs

# Access Application
open http://localhost:5173
```

---

**Happy Coding! 🚀**
