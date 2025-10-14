# Integration Guide: Client-Server Setup

This guide explains how to set up and run the integrated Nyaya system with both the React frontend and FastAPI backend working together.

## Overview

The Nyaya system consists of:
- **Backend (Server)**: FastAPI application on `http://localhost:8000`
- **Frontend (Client)**: React + Vite application on `http://localhost:5173`

The client communicates with the server via REST API endpoints, with Vite's proxy handling cross-origin requests during development.

## Prerequisites

### Backend Requirements
- Python 3.10 or higher
- **uv** package manager (recommended) or pip
- Google Cloud credentials for Vertex AI (for embeddings and LLM)

### Frontend Requirements
- Node.js 18 or higher
- npm or yarn

## Setup Instructions

### 1. Backend Setup

Navigate to the server directory:

```bash
cd server
```

#### Option A: Using uv (Recommended)

```bash
# Install dependencies (uv handles virtual environment automatically)
uv sync

# Set up environment variables (optional)
cp .env.example .env
# Edit .env to configure Vertex AI credentials if needed
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Configure Google Cloud Credentials

The system requires Google Cloud Vertex AI for embeddings and LLM generation:

```bash
# Set the path to your service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"

# Or authenticate using gcloud CLI
gcloud auth application-default login
```

### 2. Frontend Setup

Navigate to the client directory:

```bash
cd client
```

Install dependencies:

```bash
npm install
```

Configure API endpoint (optional):

```bash
# Copy the example environment file
cp .env.example .env

# The default API URL is http://localhost:8000
# You can change it in .env if needed:
VITE_API_BASE_URL=http://localhost:8000
```

## Running the Application

### Start the Backend

In the `server` directory:

```bash
python main.py
```

The server will start on `http://localhost:8000`. You can verify it's running by visiting:
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

Default configuration:
- Host: `0.0.0.0` (configurable via `HOST` env var)
- Port: `8000` (configurable via `PORT` env var)
- Auto-reload: `true` (configurable via `RELOAD` env var)

### Start the Frontend

In a new terminal, navigate to the `client` directory:

```bash
npm run dev
```

The client will start on `http://localhost:5173`. The Vite proxy will automatically forward API requests to the backend.

### Access the Application

Open your browser and navigate to:
```
http://localhost:5173
```

## How the Integration Works

### 1. API Communication

The client uses axios to communicate with the backend through the `src/services/api.ts` service layer:

- **Upload and Ask**: Upload a document and immediately ask a question
- **Follow-up Questions**: Continue conversation using session context
- **Error Handling**: Automatic error extraction and user-friendly messages

### 2. Development Proxy

Vite's proxy configuration (`vite.config.ts`) forwards requests to the backend:

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
    '/health': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

This prevents CORS issues during development.

### 3. Authentication

The backend uses optional Bearer token authentication. For demo purposes, authentication is not required:

- The client checks `localStorage` for an `authToken`
- If present, it's automatically included in request headers
- If absent, the backend returns a mock user for demo purposes

To add authentication:

```typescript
// Store token after login
localStorage.setItem('authToken', 'your-jwt-token');

// Remove token on logout
localStorage.removeItem('authToken');
```

### 4. Session Management

Each document conversation has a unique session:

1. **Upload**: Client uploads a file with a question
2. **Session Created**: Backend returns a `session_id`
3. **Follow-ups**: Client uses `session_id` for subsequent questions
4. **Context Maintained**: Backend tracks conversation history and document context

## API Endpoints Used

### Primary Endpoints

#### Upload Document and Ask Question
```
POST /api/document-query/upload-and-ask
Content-Type: multipart/form-data

Parameters:
- file: File (PDF, TXT)
- query: string
- session_id: string (optional)
- role_filter: JSON string (optional)

Response:
{
  "success": true,
  "document_id": "...",
  "filename": "...",
  "session_id": "...",
  "answer": "...",
  "confidence": 0.95,
  "sources": [...],
  "classification": {...}
}
```

#### Ask Follow-up Question
```
POST /api/document-query/ask-followup
Content-Type: multipart/form-data

Parameters:
- query: string
- session_id: string (required)
- role_filter: JSON string (optional)

Response:
{
  "answer": "...",
  "session_id": "...",
  "confidence": 0.92,
  "sources": [...],
  "classification": {...}
}
```

### Health Check
```
GET /health

Response:
{
  "status": "healthy",
  "timestamp": "2025-10-12T16:00:00.000Z",
  "system": "Legal Document Analysis API",
  "version": "1.0.0"
}
```

## Workflow Example

### 1. User Uploads Document

1. User clicks the "+" button and selects a PDF file
2. Client sets the file as `pendingUpload` and shows prompt:
   > "📄 **filename.pdf** is ready for upload. Please ask a question about this document, and I'll process it for you."

### 2. User Asks Question

1. User types: "What are the main facts of this case?"
2. Client calls `uploadDocumentAndAsk(file, query)`
3. Backend:
   - Processes the document
   - Classifies sentences by rhetorical role
   - Stores role-aware embeddings
   - Creates a session
   - Generates answer using RAG
4. Client displays the answer and stores the session

### 3. User Asks Follow-up

1. User types: "What was the final decision?"
2. Client calls `askFollowUpQuestion(query, sessionId)`
3. Backend:
   - Retrieves conversation context
   - Uses role-aware retrieval
   - Generates answer with context
4. Client displays the answer

## Troubleshooting

### Backend Issues

**Problem**: Server fails to start

```bash
# Check if port 8000 is in use
lsof -i :8000

# Try a different port
PORT=8080 python main.py
```

**Problem**: "System not initialized" error

- Ensure Vertex AI credentials are properly configured
- Check server logs for initialization errors

**Problem**: Model loading errors

- Ensure you have the required models downloaded
- Check `server/embeddings` directory for cached embeddings

### Frontend Issues

**Problem**: Cannot connect to backend

- Verify backend is running on port 8000
- Check Vite proxy configuration
- Try accessing `http://localhost:8000/health` directly

**Problem**: CORS errors (in production)

- Update CORS settings in `server/main.py`
- Configure `allow_origins` for your production domain

**Problem**: Build fails

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Try building again
npm run build
```

## Production Deployment

### Backend

1. Configure environment variables:
   ```bash
   export HOST=0.0.0.0
   export PORT=8000
   export RELOAD=false
   ```

2. Use a production server (e.g., Gunicorn):
   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

3. Set up proper CORS origins in `main.py`:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

### Frontend

1. Build the production bundle:
   ```bash
   cd client
   npm run build
   ```

2. Configure the API URL:
   ```bash
   # In .env.production
   VITE_API_BASE_URL=https://api.yourdomain.com
   ```

3. Serve the built files with a web server (nginx, Apache, etc.)

4. Example nginx configuration:
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;
       
       root /path/to/client/dist;
       index index.html;
       
       location / {
           try_files $uri $uri/ /index.html;
       }
       
       location /api {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## Development Tips

### Hot Reload

Both backend and frontend support hot reload during development:

- **Backend**: Automatically reloads when Python files change (if `RELOAD=true`)
- **Frontend**: Vite provides instant HMR for React components

### Debugging API Calls

1. Open browser DevTools (F12)
2. Go to Network tab
3. Filter by "XHR" or "Fetch"
4. Inspect request/response payloads

### Testing the API

Use the interactive API docs:
```
http://localhost:8000/docs
```

You can test endpoints directly from the browser.

### Adding New API Endpoints

1. **Backend**: Add route in `server/src/api/`
2. **Frontend**: Add function in `client/src/services/api.ts`
3. **Types**: Update types in `client/src/types/index.ts`
4. **Component**: Use the new function in your React components

## Further Reading

- [DOCUMENT_QUERY_API.md](./DOCUMENT_QUERY_API.md) - Detailed API documentation
- [EXAMPLE_FLOW_1.md](./EXAMPLE_FLOW_1.md) - Complete system workflow
- [QUICK_START_GUIDE.md](./QUICK_START_GUIDE.md) - Getting started guide
- [CLAUDE.md](../CLAUDE.md) - Project overview and architecture

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review server logs for detailed error messages
3. Consult the API documentation at `/docs`
4. Check the GitHub issues for similar problems
