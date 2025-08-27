# Document Upload with Simultaneous Query API

## Overview

The system now supports uploading a document and asking questions about it simultaneously, with full support for follow-up questions in a conversational manner.

## New Endpoints

### 1. Upload Document with Immediate Query

**POST** `/api/document-query/upload-and-ask`

Upload a legal document and ask a question about it in a single request.

**Request:**
- `file`: PDF or TXT file (multipart/form-data)
- `query`: Question about the document (form field)
- `session_id`: Optional conversation session ID (form field)
- `role_filter`: Optional JSON array of roles to filter (form field)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/document-query/upload-and-ask" \
  -F "file=@legal_case.pdf" \
  -F "query=What are the main facts of this case?" \
  -F "session_id=optional-session-id" \
  -F "role_filter=[\"Facts\", \"Issue\"]"
```

**Response:**
```json
{
  "success": true,
  "document_id": "uuid-string",
  "filename": "legal_case.pdf",
  "session_id": "session-uuid",
  "answer": "Based on the document 'legal_case.pdf' you just uploaded:\n\nThe main facts of the case are...",
  "document_metadata": {
    "case_name": "Ram Kumar v. State",
    "court": "Supreme Court",
    "date": "2024-03-15"
  },
  "sources": [...],
  "classification": {...},
  "tools_used": ["document_processor", "role_classifier", "document_specific_rag"],
  "confidence": 0.89
}
```

### 2. Follow-up Questions

**POST** `/api/document-query/ask-followup`

Ask follow-up questions about previously uploaded documents in the same conversation.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/document-query/ask-followup" \
  -F "query=What was the court's reasoning?" \
  -F "session_id=session-uuid-from-upload" \
  -F "role_filter=[\"Reasoning\", \"Decision\"]"
```

**Response:**
```json
{
  "success": true,
  "session_id": "session-uuid",
  "answer": "Continuing analysis of 'legal_case.pdf':\n\nThe court's reasoning was based on...",
  "sources": [...],
  "classification": {...},
  "tools_used": ["document_specific_rag", "role_classifier"],
  "confidence": 0.92
}
```

## User Workflow Examples

### Example 1: Complete Document Analysis

1. **Upload and Ask Initial Question:**
   ```
   Upload: judgment.pdf
   Question: "What are the facts of this case?"
   ```

2. **Follow-up Questions:**
   ```
   "What were the petitioner's arguments?"
   "What was the court's decision?"
   "What legal precedents were cited?"
   ```

### Example 2: Role-Specific Analysis

1. **Upload with Role Filter:**
   ```
   Upload: case.pdf
   Question: "Summarize the reasoning"
   Role Filter: ["Reasoning", "Decision"]
   ```

2. **Multi-role Follow-up:**
   ```
   "Now show me the arguments from both sides"
   Role Filter: ["Arguments of Petitioner", "Arguments of Respondent"]
   ```

## Features

### ✅ Implemented Features:

1. **Simultaneous Upload + Query**: Single API call for document upload and immediate questioning
2. **Conversation Continuity**: Follow-up questions maintain context from the uploaded document
3. **Role-Aware Filtering**: Users can specify which rhetorical roles to focus on
4. **Document Context**: System remembers which document is being discussed
5. **Automatic Role Detection**: System detects relevant roles even without explicit specification
6. **Session Management**: Maintains conversation history and document associations

### 🔧 Technical Implementation:

1. **Document Processing Pipeline**:
   - File upload and validation
   - Text extraction and cleaning
   - Rhetorical role classification
   - Vector embedding generation
   - Storage in role-aware vector database

2. **Query Processing**:
   - Query intent classification
   - Role detection from natural language
   - Document-specific retrieval
   - Context-aware response generation

3. **Conversation Management**:
   - Session-based context tracking
   - Document metadata preservation
   - Multi-turn dialogue support

## Integration Points

### Frontend Integration

```javascript
// Upload document with question
const formData = new FormData();
formData.append('file', documentFile);
formData.append('query', 'What are the facts?');
formData.append('session_id', sessionId);

const response = await fetch('/api/document-query/upload-and-ask', {
  method: 'POST',
  body: formData
});

// Follow-up question
const followupData = new FormData();
followupData.append('query', 'What was the decision?');
followupData.append('session_id', sessionId);

const followupResponse = await fetch('/api/document-query/ask-followup', {
  method: 'POST', 
  body: followupData
});
```

### Backend Integration

The new functionality integrates with existing components:

- **Agent Orchestrator**: Enhanced with document-specific query handling
- **Conversation Manager**: Tracks document uploads and context
- **Legal RAG System**: Provides document-aware retrieval
- **Role Classifier**: Enables role-specific querying

## Error Handling

The system handles various error scenarios:

- **Unsupported file formats**: Clear error messages with supported formats
- **Processing failures**: Graceful fallback with retry suggestions
- **Session issues**: Automatic session recovery or creation
- **Query errors**: Context-aware error responses with suggestions

## Benefits

1. **Improved User Experience**: Single-step document upload and analysis
2. **Context Preservation**: Natural conversation flow about specific documents
3. **Precise Information Retrieval**: Role-aware filtering for targeted answers
4. **Efficient Workflow**: Eliminates need for separate upload and query steps
5. **Scalable Architecture**: Supports multiple concurrent document discussions

This implementation fully addresses the requirement for simultaneous file upload with questions and seamless follow-up queries, making the system much more user-friendly for legal document analysis.