# Architecture: Client-Server Integration

This document provides a visual overview of how the Nyaya client and server components integrate.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User's Browser                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    React Frontend                          │  │
│  │                  (http://localhost:5173)                   │  │
│  │                                                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │  │
│  │  │   Sidebar    │  │  ChatWindow  │  │   ChatInput     │ │  │
│  │  │              │  │              │  │                 │ │  │
│  │  │ Documents    │  │  Messages    │  │ [Type/Upload]   │ │  │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘ │  │
│  │                                                            │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │         API Service Layer (api.ts)                 │   │  │
│  │  │  - uploadDocumentAndAsk()                          │   │  │
│  │  │  - askFollowUpQuestion()                           │   │  │
│  │  │  - getErrorMessage()                               │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────┬───────────────────────────────┘  │
└────────────────────────────────┼──────────────────────────────────┘
                                 │
                          HTTP Requests
                          (via Axios)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Development Proxy                           │
│                     (Vite Dev Server)                            │
│  Forwards /api/* and /health to backend                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend Server                        │
│                   (http://localhost:8000)                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                      API Endpoints                          │ │
│  │                                                             │ │
│  │  /api/document-query/upload-and-ask    ← Upload + Query    │ │
│  │  /api/document-query/ask-followup      ← Follow-up Q       │ │
│  │  /api/query                            ← General Query      │ │
│  │  /health                               ← Health Check       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Core System Components                         │ │
│  │                                                             │ │
│  │  ┌──────────────────┐  ┌──────────────────┐               │ │
│  │  │ Agent            │  │ Conversation     │               │ │
│  │  │ Orchestrator     │  │ Manager          │               │ │
│  │  │                  │  │                  │               │ │
│  │  │ - Route queries  │  │ - Track sessions │               │ │
│  │  │ - Select tools   │  │ - Maintain ctx   │               │ │
│  │  └────────┬─────────┘  └────────┬─────────┘               │ │
│  │           │                     │                          │ │
│  │           ▼                     ▼                          │ │
│  │  ┌──────────────────────────────────────────┐             │ │
│  │  │         Legal RAG System                 │             │ │
│  │  │                                          │             │ │
│  │  │  ┌────────────┐  ┌──────────────────┐   │             │ │
│  │  │  │ Role       │  │ Vector Store     │   │             │ │
│  │  │  │ Classifier │  │ (ChromaDB)       │   │             │ │
│  │  │  │            │  │                  │   │             │ │
│  │  │  │ InLegal-   │  │ Role-aware       │   │             │ │
│  │  │  │ BERT       │  │ embeddings       │   │             │ │
│  │  │  └────────────┘  └──────────────────┘   │             │ │
│  │  └──────────────────────────────────────────┘             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              External Services                              │ │
│  │                                                             │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │         Google Cloud Vertex AI                        │  │
│  │  │                                                        │  │
│  │  │  - Embeddings (text-embedding-005)                    │  │
│  │  │  - LLM Generation (Gemini 2.5 Flash)                  │  │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Request Flow: Upload and Ask

```
┌──────────┐                                                    ┌──────────┐
│  Client  │                                                    │  Server  │
└────┬─────┘                                                    └────┬─────┘
     │                                                               │
     │ 1. User uploads file                                          │
     │    and types question                                         │
     │                                                               │
     │ 2. POST /api/document-query/upload-and-ask                   │
     │    FormData: { file, query }                                  │
     ├──────────────────────────────────────────────────────────────►│
     │                                                               │
     │                                                               │ 3. Document Processing
     │                                                               │    - Extract text
     │                                                               │    - Clean & segment
     │                                                               │
     │                                                               │ 4. Role Classification
     │                                                               │    - Label sentences
     │                                                               │    - Identify roles
     │                                                               │
     │                                                               │ 5. Create Embeddings
     │                                                               │    - Generate vectors
     │                                                               │    - Store in ChromaDB
     │                                                               │
     │                                                               │ 6. Start Session
     │                                                               │    - Create session_id
     │                                                               │    - Initialize context
     │                                                               │
     │                                                               │ 7. Query Processing
     │                                                               │    - Classify query
     │                                                               │    - Retrieve segments
     │                                                               │    - Generate answer
     │                                                               │
     │ 8. Response with answer                                       │
     │    and session_id                                             │
     │◄──────────────────────────────────────────────────────────────┤
     │                                                               │
     │ 9. Display answer,                                            │
     │    store session_id                                           │
     │                                                               │
```

## Request Flow: Follow-up Question

```
┌──────────┐                                                    ┌──────────┐
│  Client  │                                                    │  Server  │
└────┬─────┘                                                    └────┬─────┘
     │                                                               │
     │ 1. User types                                                 │
     │    follow-up question                                         │
     │                                                               │
     │ 2. POST /api/document-query/ask-followup                     │
     │    FormData: { query, session_id }                            │
     ├──────────────────────────────────────────────────────────────►│
     │                                                               │
     │                                                               │ 3. Load Session Context
     │                                                               │    - Get conversation history
     │                                                               │    - Get document context
     │                                                               │
     │                                                               │ 4. Query Processing
     │                                                               │    - Classify query intent
     │                                                               │    - Retrieve relevant segments
     │                                                               │    - Use conversation context
     │                                                               │
     │                                                               │ 5. Generate Answer
     │                                                               │    - Context-aware response
     │                                                               │    - Update session history
     │                                                               │
     │ 6. Response with                                              │
     │    contextual answer                                          │
     │◄──────────────────────────────────────────────────────────────┤
     │                                                               │
     │ 7. Display answer                                             │
     │                                                               │
```

## Data Flow: Document to Answer

```
┌────────────┐
│   User     │
│  Uploads   │
│  Document  │
└─────┬──────┘
      │
      ▼
┌─────────────────┐
│  PDF/TXT File   │
└─────┬───────────┘
      │
      ▼
┌─────────────────────────────┐
│  Document Processor         │
│  - Extract text             │
│  - Clean formatting         │
│  - Segment into sentences   │
└─────┬───────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Role Classifier            │
│  (InLegalBERT)              │
│  - Label each sentence:     │
│    • Facts                  │
│    • Issues                 │
│    • Arguments (AoP/AoR)    │
│    • Reasoning              │
│    • Decision               │
└─────┬───────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Embedding Generator        │
│  (Vertex AI)                │
│  - Create vector embeddings │
│  - Preserve role metadata   │
└─────┬───────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Vector Store (ChromaDB)    │
│  - Store embeddings         │
│  - Index by role            │
│  - Store metadata           │
└─────┬───────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Ready for Querying         │
│  - Role-aware retrieval     │
│  - Semantic search          │
│  - Context-aware responses  │
└─────────────────────────────┘
```

## Component Interactions

### Frontend Components

```
App.tsx (Main Controller)
├── State Management
│   ├── documents: LegalDocument[]
│   ├── activeDocument: LegalDocument | null
│   ├── messages: ChatMessage[]
│   ├── pendingUpload: File | null
│   └── sessionId: string (stored in activeDocument)
│
├── Event Handlers
│   ├── handleFileUpload(file)
│   │   └── Sets pendingUpload, prompts for question
│   │
│   ├── handleSendMessage()
│   │   ├── If pendingUpload: uploadDocumentAndAsk()
│   │   └── Else: askFollowUpQuestion()
│   │
│   └── handleDocumentSelect(doc)
│       └── Switch to document's session
│
└── Child Components
    ├── Sidebar (Document list)
    ├── ChatMessage (Display messages)
    ├── LoadingIndicator (Show processing)
    └── ChatInput (User input)
```

### Backend Components

```
main.py (FastAPI App)
├── Startup
│   ├── Initialize AgentOrchestrator
│   ├── Initialize ConversationManager
│   ├── Initialize RoleClassifier
│   └── Set component references
│
├── API Routes
│   ├── /api/document-query/upload-and-ask
│   │   └── Calls orchestrator.upload_document()
│   │       └── Calls orchestrator.process_query()
│   │
│   ├── /api/document-query/ask-followup
│   │   └── Calls orchestrator.process_query()
│   │       └── Uses session context
│   │
│   └── /health
│       └── Returns system status
│
└── Core Components
    ├── AgentOrchestrator
    │   ├── Upload document
    │   ├── Process queries
    │   └── Route to tools
    │
    ├── LegalRAGSystem
    │   ├── Store embeddings
    │   ├── Retrieve segments
    │   └── Generate answers
    │
    └── ConversationManager
        ├── Create sessions
        ├── Track history
        └── Maintain context
```

## Session Management

```
┌─────────────────────────────────────────────────────────────┐
│                      Session Lifecycle                       │
└─────────────────────────────────────────────────────────────┘

1. Document Upload + First Question
   ┌────────────────────────────────────┐
   │  POST /upload-and-ask              │
   │  - No session_id provided          │
   │                                    │
   │  Server:                           │
   │  - Creates new session             │
   │  - Returns session_id: "abc123"    │
   │                                    │
   │  Client:                           │
   │  - Stores session_id in document   │
   │  - Uses for all future queries     │
   └────────────────────────────────────┘

2. Follow-up Questions
   ┌────────────────────────────────────┐
   │  POST /ask-followup                │
   │  - Includes session_id: "abc123"   │
   │                                    │
   │  Server:                           │
   │  - Loads session context           │
   │  - Retrieves conversation history  │
   │  - Generates contextual answer     │
   │  - Updates session                 │
   │                                    │
   │  Client:                           │
   │  - Displays answer                 │
   │  - Maintains same session_id       │
   └────────────────────────────────────┘

3. Switch Documents
   ┌────────────────────────────────────┐
   │  User clicks different document    │
   │                                    │
   │  Client:                           │
   │  - Loads document's session_id     │
   │  - Loads conversation history      │
   │  - Continues from where left off   │
   └────────────────────────────────────┘
```

## Authentication Flow

```
┌──────────┐                                    ┌──────────┐
│  Client  │                                    │  Server  │
└────┬─────┘                                    └────┬─────┘
     │                                               │
     │ 1. Check localStorage for token               │
     │    token = localStorage.getItem('authToken')  │
     │                                               │
     │ 2. API Request with optional token            │
     │    Headers: { Authorization: "Bearer <token>" }│
     ├──────────────────────────────────────────────►│
     │                                               │
     │                                               │ 3. auth.py
     │                                               │    - Token optional (auto_error=False)
     │                                               │    - Returns demo user if no token
     │                                               │    - Would validate in production
     │                                               │
     │ 4. Response (regardless of auth)              │
     │◄──────────────────────────────────────────────┤
     │                                               │
```

## Error Handling

```
Client Side (api.ts)
└── Try-Catch Blocks
    ├── Axios Interceptors
    │   ├── Request: Add auth token
    │   └── Response: Log errors
    │
    └── getErrorMessage()
        ├── Extract error from axios
        ├── Check response.data.message
        ├── Check response.data.detail
        └── Return user-friendly message

Server Side (main.py)
└── Exception Handlers
    ├── HTTPException Handler
    │   └── Return structured error
    │
    └── General Exception Handler
        └── Log and return 500 error

User Experience
└── Error Display
    ├── Show error message in chat
    ├── Keep conversation history
    └── Allow user to retry
```

## Technology Stack Summary

### Frontend
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite 7
- **HTTP Client**: Axios
- **Styling**: Tailwind CSS v4
- **Icons**: Lucide React

### Backend
- **Framework**: FastAPI
- **Server**: Uvicorn
- **ML Models**: InLegalBERT, BiLSTM-CRF
- **Vector DB**: ChromaDB
- **LLM**: Google Vertex AI (Gemini 2.5 Flash)
- **Embeddings**: Vertex AI (text-embedding-005)

### Communication
- **Protocol**: HTTP/REST
- **Format**: JSON + multipart/form-data
- **Development**: Vite proxy (CORS-free)
- **Production**: CORS middleware configured

---

For more details, see:
- [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) - Setup and configuration
- [DOCUMENT_QUERY_API.md](./DOCUMENT_QUERY_API.md) - API reference
- [EXAMPLE_FLOW_1.md](./EXAMPLE_FLOW_1.md) - Complete workflow example
