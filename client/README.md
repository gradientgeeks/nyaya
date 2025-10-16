# Nyaya Legal Analysis - Client

The frontend for the Nyaya intelligent legal document analysis system, built with React, TypeScript, and Vite.

## Features

- 📄 **Document Upload**: Upload legal documents (PDF, TXT) for analysis
- 💬 **Conversational Interface**: Chat-based interaction with the analysis system
- 🔍 **Role-Aware Querying**: Ask about specific aspects (facts, arguments, reasoning, decision)
- 🔄 **Multi-turn Conversations**: Maintain context across multiple questions
- 🌓 **Dark Mode**: Toggle between light and dark themes
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices

## Tech Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS v4** - Styling
- **Axios** - HTTP client
- **Lucide React** - Icon library

## Quick Start

### Prerequisites

- Node.js 18 or higher
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env if you need to change the API URL (default: http://localhost:8000)

# Start development server
npm run dev
```

The application will be available at `http://localhost:5173`.

### Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Development Scripts

```bash
# Start development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## Project Structure

```
client/
├── src/
│   ├── components/       # React components
│   │   ├── ChatInput.tsx
│   │   ├── ChatMessage.tsx
│   │   ├── LoadingIndicator.tsx
│   │   ├── Sidebar.tsx
│   │   └── ThemeToggle.tsx
│   ├── contexts/         # React contexts
│   │   └── ThemeContext.tsx
│   ├── data/            # Mock data (for reference)
│   │   └── mockData.ts
│   ├── services/        # API service layer
│   │   └── api.ts
│   ├── types/           # TypeScript type definitions
│   │   └── index.ts
│   ├── App.tsx          # Main application component
│   ├── App.css          # Application styles
│   └── main.tsx         # Application entry point
├── public/              # Static assets
├── .env.example         # Environment variables template
├── vite.config.ts       # Vite configuration
├── tailwind.config.js   # Tailwind CSS configuration
└── tsconfig.json        # TypeScript configuration
```

## Configuration

### Environment Variables

Create a `.env` file in the client directory:

```env
# API endpoint (default: http://localhost:8000)
VITE_API_BASE_URL=http://localhost:8000
```

### Vite Proxy

The development server is configured to proxy API requests to the backend:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': 'http://localhost:8000',
    '/health': 'http://localhost:8000',
  },
}
```

This prevents CORS issues during development.

## API Integration

The client communicates with the FastAPI backend through the `src/services/api.ts` service layer.

### Key API Functions

- `uploadDocumentAndAsk()` - Upload a document and ask a question
- `askFollowUpQuestion()` - Ask follow-up questions using session context
- `checkHealth()` - Check backend health status

### Example Usage

```typescript
import { uploadDocumentAndAsk, askFollowUpQuestion } from './services/api';

// Upload and ask
const response = await uploadDocumentAndAsk(file, 'What are the main facts?');
console.log(response.answer);
console.log(response.session_id); // Save for follow-ups

// Follow-up question
const followUp = await askFollowUpQuestion(
  'What was the final decision?',
  sessionId
);
console.log(followUp.answer);
```

## Workflow

1. **Upload Document**: User selects a PDF or TXT file
2. **Ask Question**: User types a question about the document
3. **Processing**: Backend processes the document and generates an answer
4. **Display Answer**: Answer is displayed in the chat interface
5. **Follow-up**: User can ask additional questions using the same session

## Components

### App.tsx
Main application component that manages:
- Document list
- Active document state
- Conversation messages
- File upload workflow
- Session management

### ChatInput.tsx
Input component for user messages and file uploads:
- Text input with auto-resize
- File upload button
- Send button
- Keyboard shortcuts (Enter to send)

### ChatMessage.tsx
Displays individual messages with:
- User/bot differentiation
- Support for text, analysis, and prediction messages
- Markdown rendering
- Timestamp

### Sidebar.tsx
Document list sidebar:
- List of uploaded documents
- Active document highlighting
- Minimize/expand functionality
- Responsive design

### ThemeToggle.tsx
Theme switcher component for light/dark mode.

## Styling

The application uses Tailwind CSS v4 with:
- Custom color schemes for light and dark modes
- Responsive design breakpoints
- Custom animations and transitions
- Dark mode support via `dark:` variants

## Backend Integration

This frontend requires the Nyaya backend to be running. See the [Integration Guide](../docs/INTEGRATION_GUIDE.md) for setup instructions.

### Backend Requirements

- FastAPI server running on `http://localhost:8000`
- Google Cloud Vertex AI credentials configured
- Role classifier model loaded
- RAG system initialized

## Contributing

When adding new features:

1. Add TypeScript types in `src/types/index.ts`
2. Create API functions in `src/services/api.ts`
3. Build React components in `src/components/`
4. Update this README with new features

## Troubleshooting

### Cannot connect to backend

- Ensure the backend is running on port 8000
- Check the `VITE_API_BASE_URL` in `.env`
- Verify the Vite proxy configuration

### Build errors

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### CORS issues

- In development, the Vite proxy should handle CORS
- In production, ensure backend CORS settings allow your domain

## Further Reading

- [Integration Guide](../docs/INTEGRATION_GUIDE.md) - Full setup instructions
- [Backend README](../server/README.md) - Backend documentation
- [API Documentation](../docs/DOCUMENT_QUERY_API.md) - API reference

## License

See the main project README for license information.
