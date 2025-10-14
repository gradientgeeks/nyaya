import { useState, useEffect, useRef } from 'react';
import './App.css';
import type { 
  LegalDocument, 
  ChatMessage as ChatMessageType
} from './types';
import { 
  Sidebar, 
  ChatMessage, 
  LoadingIndicator, 
  ChatInput,
  ThemeToggle
} from './components';
import { ThemeProvider } from './contexts/ThemeContext';
import { uploadDocumentAndAsk, askFollowUpQuestion, getErrorMessage } from './services/api';

function AppContent() {
  const [documents, setDocuments] = useState<LegalDocument[]>([]);
  const [activeDocument, setActiveDocument] = useState<LegalDocument | null>(null);
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarMinimized, setIsSidebarMinimized] = useState(false);
  const [pendingUpload, setPendingUpload] = useState<File | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Set initial greeting message when active document changes
    if (activeDocument && !pendingUpload) {
      setMessages([{
        id: `greeting-${activeDocument.id}`,
        sender: 'bot',
        type: 'greeting',
        text: `Analyzing **${activeDocument.name}**. How can I assist you with this case? You can ask for a full summary, specific facts, arguments, or a probable outcome if the case is pending.`,
        timestamp: new Date()
      }]);
    }
  }, [activeDocument, pendingUpload]);

  useEffect(() => {
    // Auto-scroll to the latest message
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (input.trim() === '' || isLoading) return;

    const userMessage: ChatMessageType = { 
      id: `user-${Date.now()}`,
      sender: 'user', 
      text: input,
      timestamp: new Date()
    };
    
    const currentInput = input;
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // If there's a pending upload, upload the file with the query
      if (pendingUpload) {
        const response = await uploadDocumentAndAsk(
          pendingUpload,
          currentInput,
          undefined,
          undefined
        );

        // Create the document record
        const newDoc: LegalDocument = {
          id: documents.length + 1,
          name: response.filename,
          uploadDate: new Date(),
          size: pendingUpload.size,
          sessionId: response.session_id,
          documentId: response.document_id,
        };

        setDocuments(prev => [...prev, newDoc]);
        setActiveDocument(newDoc);
        setPendingUpload(null);

        const botResponse: ChatMessageType = {
          id: `bot-${Date.now()}`,
          sender: 'bot',
          type: 'text',
          text: response.answer,
          timestamp: new Date()
        };

        setMessages(prev => [...prev, botResponse]);
      } 
      // If there's an active document with a session, send follow-up
      else if (activeDocument?.sessionId) {
        const response = await askFollowUpQuestion(
          currentInput,
          activeDocument.sessionId,
          undefined
        );

        const botResponse: ChatMessageType = {
          id: `bot-${Date.now()}`,
          sender: 'bot',
          type: 'text',
          text: response.answer,
          timestamp: new Date()
        };

        setMessages(prev => [...prev, botResponse]);
      }
      // No document selected
      else {
        const botResponse: ChatMessageType = {
          id: `bot-${Date.now()}`,
          sender: 'bot',
          type: 'text',
          text: 'Please upload a document first to start a conversation.',
          timestamp: new Date()
        };

        setMessages(prev => [...prev, botResponse]);
      }
    } catch (error) {
      console.error('Error processing query:', error);
      const errorMessage: ChatMessageType = {
        id: `error-${Date.now()}`,
        sender: 'bot',
        type: 'text',
        text: `Error: ${getErrorMessage(error)}`,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = (file: File) => {
    // Set the pending upload and prompt user to ask a question
    setPendingUpload(file);
    setMessages([{
      id: `upload-prompt-${Date.now()}`,
      sender: 'bot',
      type: 'greeting',
      text: `📄 **${file.name}** is ready for upload. Please ask a question about this document, and I'll process it for you.`,
      timestamp: new Date()
    }]);
  };

  const handleDocumentSelect = (document: LegalDocument) => {
    setActiveDocument(document);
  };

  const handleToggleSidebar = () => {
    setIsSidebarMinimized(!isSidebarMinimized);
  };

  return (
    <div className="font-sans bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 flex h-screen overflow-hidden">
      {/* Mobile Backdrop */}
      {!isSidebarMinimized && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-20 lg:hidden"
          onClick={handleToggleSidebar}
        />
      )}

      <Sidebar
        documents={documents}
        activeDocument={activeDocument}
        onDocumentSelect={handleDocumentSelect}
        isMinimized={isSidebarMinimized}
        onToggleMinimize={handleToggleSidebar}
      />

      {/* Main Chat Interface */}
      <main className={`${isSidebarMinimized ? 'w-[calc(100%-4rem)]' : 'w-[calc(100%-20rem)]'} lg:${isSidebarMinimized ? 'w-[calc(100%-4rem)]' : 'w-[calc(100%-20rem)]'} w-full flex flex-col h-screen bg-gray-50 dark:bg-gray-900 transition-all duration-300`}>
        {/* Mobile Header */}
        <div className="lg:hidden bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex items-center justify-between">
          <button
            onClick={handleToggleSidebar}
            className="p-2 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <h1 className="text-lg font-semibold">Legalysis AI</h1>
          <div className="w-10 flex justify-center">
            <ThemeToggle isMinimized={true} />
          </div>
        </div>
        <div id="chat-window" className="flex-1 p-4 sm:p-6 overflow-y-auto">
          <div className="max-w-4xl mx-auto space-y-4 sm:space-y-6">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}
            {isLoading && <LoadingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        </div>
        
        <ChatInput
          input={input}
          activeDocument={activeDocument}
          isLoading={isLoading}
          onInputChange={setInput}
          onSendMessage={handleSendMessage}
          onFileUpload={handleFileUpload}
        />
      </main>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}
