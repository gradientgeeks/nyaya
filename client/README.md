# Nyaya Client - Legal Document Analysis Frontend

React 19 + Vite + Tailwind v4 frontend for the Nyaya Legal RAG System.

## Features

- 🎨 Modern UI with dark mode support
- 💬 Real-time chat interface for legal document queries
- 📄 Document upload and classification
- 🔍 Role-aware question answering (Facts, Reasoning, Decision, etc.)
- 🔮 Case outcome prediction
- 📊 Visual analysis of legal documents
- ⚡ Optimized with Vite for fast development

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Backend API running (see `/backend/README.md`)

### Installation

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Edit .env if needed (default uses localhost:8000)
```

### Development

```bash
# Start dev server (with hot reload)
npm run dev

# Access at http://localhost:5173
```

The development server will automatically proxy API requests to the backend at `http://localhost:8000`.

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Environment Configuration

Create a `.env` file:

```bash
# Backend API URL
VITE_API_URL=http://localhost:8000
```

**Note:** In development, API requests are proxied through Vite's dev server (configured in `vite.config.ts`).

## Project Structure

```
client/
├── src/
│   ├── components/          # UI components
│   │   ├── AnalysisView.tsx    # Display classification results
│   │   ├── ChatInput.tsx       # Message input with file upload
│   │   ├── ChatMessage.tsx     # Chat message display
│   │   ├── LoadingIndicator.tsx
│   │   ├── PredictionView.tsx  # Display predictions
│   │   ├── Sidebar.tsx         # Document list sidebar
│   │   └── ThemeToggle.tsx     # Dark mode toggle
│   ├── contexts/            # React contexts
│   │   └── ThemeContext.tsx    # Dark mode context
│   ├── data/                # Mock data (fallback)
│   │   └── mockData.ts
│   ├── services/            # API integration
│   │   ├── api.ts              # Backend API client
│   │   └── index.ts
│   ├── types/               # TypeScript definitions
│   │   └── index.ts
│   ├── App.tsx              # Main application component
│   └── main.tsx             # Application entry point
├── public/                  # Static assets
├── .env.example             # Environment template
├── .env                     # Environment variables (not committed)
├── vite.config.ts           # Vite configuration
├── tailwind.config.js       # Tailwind CSS configuration
└── package.json
```

## Backend Integration

### API Service

The client integrates with the backend through `src/services/api.ts`:

```typescript
import { api } from './services';

// Create session
const session = await api.createSession();

// Upload document
const result = await api.uploadDocument(file, sessionId);

// Query with role-aware RAG
const response = await api.query({
  query: "What were the facts of the case?",
  session_id: sessionId,
  case_id: caseId
});

// Search for similar cases
const searchResults = await api.search({
  query: "Find cases about contract disputes",
  session_id: sessionId
});

// Predict outcome
const prediction = await api.predict({
  case_description: "Contract dispute...",
  session_id: sessionId
});
```

### Graceful Fallback

The application automatically falls back to mock data if the backend is unavailable:

- Shows a warning banner: "⚠️ Backend not connected - using mock data"
- Uses local mock data from `src/data/mockData.ts`
- Allows UI development without backend dependency

### Session Management

The app automatically:
1. Creates a session on mount
2. Stores session ID for all requests
3. Maintains conversation context
4. Handles session failures gracefully

## Available Scripts

```bash
# Development
npm run dev          # Start dev server with HMR
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint

# Type checking
npm run type-check   # Check TypeScript types
```

## Development Workflow

### 1. Start Backend

```bash
cd ../backend
uvicorn app.main:app --reload
# Backend runs at http://localhost:8000
```

### 2. Start Frontend

```bash
npm run dev
# Frontend runs at http://localhost:5173
```

### 3. Test Integration

- Upload a legal document (PDF/TXT)
- Ask questions: "What were the facts?", "Show me the reasoning"
- Request predictions for pending cases
- Search for similar cases

## Key Technologies

- **React 19**: Latest React with improved performance
- **Vite 7**: Next-generation frontend tooling
- **Tailwind CSS 4**: Utility-first CSS framework
- **TypeScript**: Type-safe development
- **Lucide React**: Modern icon library

## API Endpoints Used

The client connects to these backend endpoints:

```
POST /api/v1/sessions          - Create session
POST /api/v1/upload            - Upload & classify document
POST /api/v1/query             - Role-aware Q&A
POST /api/v1/search            - Find similar cases
POST /api/v1/predict           - Predict outcomes
GET  /api/v1/health            - Health check
GET  /api/v1/stats             - System statistics
```

## Error Handling

The application handles errors gracefully:

- Network errors → Show error message in chat
- Backend unavailable → Fall back to mock data
- Upload failures → Display error with details
- API errors → Show user-friendly messages

## Troubleshooting

### Backend Connection Issues

If you see "Backend not connected" warning:

1. Ensure backend is running: `cd ../backend && uvicorn app.main:app --reload`
2. Check backend URL in `.env` matches backend port
3. Verify CORS is enabled in backend (already configured)

### Build Errors

```bash
# Clear cache and reinstall
rm -rf node_modules dist
npm install
npm run build
```

### Type Errors

```bash
# Check TypeScript configuration
npm run type-check
```

## Contributing

When adding new features:

1. Update TypeScript types in `src/types/index.ts`
2. Add API methods in `src/services/api.ts`
3. Create reusable components in `src/components/`
4. Test with both backend and mock data
5. Run linter before committing: `npm run lint`

## License

Part of the Nyaya Legal RAG System project.
