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
import { mockCases } from './data/mockData';
import { ThemeProvider } from './contexts/ThemeContext';

function AppContent() {
  const [documents, setDocuments] = useState<LegalDocument[]>([
    { id: 1, name: "Case_A_vs_B_2023.pdf" },
    { id: 2, name: "Case_C_vs_D_Pending.pdf" },
  ]);
  const [activeDocument, setActiveDocument] = useState<LegalDocument | null>(documents[0]);
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarMinimized, setIsSidebarMinimized] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Set initial greeting message when active document changes
    if (activeDocument) {
      setMessages([{
        id: `greeting-${activeDocument.id}`,
        sender: 'bot',
        type: 'greeting',
        text: `Analyzing **${activeDocument.name}**. How can I assist you with this case? You can ask for a full summary, specific facts, arguments, or a probable outcome if the case is pending.`,
        timestamp: new Date()
      }]);
    }
  }, [activeDocument]);

  useEffect(() => {
    // Auto-scroll to the latest message
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = () => {
    if (input.trim() === '' || isLoading || !activeDocument) return;

    const userMessage: ChatMessageType = { 
      id: `user-${Date.now()}`,
      sender: 'user', 
      text: input,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Simulate API call and agent response
    setTimeout(() => {
      const docData = mockCases[activeDocument.name];
      let botResponse: ChatMessageType;

      const lowerInput = input.toLowerCase();

      if (lowerInput.includes("summary")) {
        botResponse = { 
          id: `bot-${Date.now()}`,
          sender: 'bot', 
          type: 'analysis', 
          data: docData.summary,
          timestamp: new Date()
        };
      } else if (lowerInput.includes("fact")) {
        botResponse = { 
          id: `bot-${Date.now()}`,
          sender: 'bot', 
          type: 'analysis', 
          data: { facts: docData.summary.facts },
          timestamp: new Date()
        };
      } else if (lowerInput.includes("petitioner") || lowerInput.includes("aop")) {
        botResponse = { 
          id: `bot-${Date.now()}`,
          sender: 'bot', 
          type: 'analysis', 
          data: { petitionerArgs: docData.summary.petitionerArgs },
          timestamp: new Date()
        };
      } else if (lowerInput.includes("respondent") || lowerInput.includes("aor")) {
        botResponse = { 
          id: `bot-${Date.now()}`,
          sender: 'bot', 
          type: 'analysis', 
          data: { respondentArgs: docData.summary.respondentArgs },
          timestamp: new Date()
        };
      } else if (lowerInput.includes("reasoning")) {
        botResponse = { 
          id: `bot-${Date.now()}`,
          sender: 'bot', 
          type: 'analysis', 
          data: { reasoning: docData.summary.reasoning },
          timestamp: new Date()
        };
      } else if (lowerInput.includes("decision")) {
        botResponse = { 
          id: `bot-${Date.now()}`,
          sender: 'bot', 
          type: 'analysis', 
          data: { decision: docData.summary.decision },
          timestamp: new Date()
        };
      } else if (lowerInput.includes("predict") || lowerInput.includes("outcome")) {
        if (docData.type === 'pending' && docData.prediction) {
          botResponse = { 
            id: `bot-${Date.now()}`,
            sender: 'bot', 
            type: 'prediction', 
            data: docData.prediction,
            timestamp: new Date()
          };
        } else {
          botResponse = { 
            id: `bot-${Date.now()}`,
            sender: 'bot', 
            type: 'text', 
            text: "This case has already been judged. I can provide the final decision and reasoning.",
            timestamp: new Date()
          };
        }
      } else {
        botResponse = { 
          id: `bot-${Date.now()}`,
          sender: 'bot', 
          type: 'text', 
          text: "I'm not sure how to respond to that. You can ask me to summarize the case, show specific parts like 'facts' or 'arguments', or 'predict the outcome' for pending cases.",
          timestamp: new Date()
        };
      }
      
      setMessages(prev => [...prev, botResponse]);
      setIsLoading(false);
    }, 1500);
  };

  const handleFileUpload = (file: File) => {
    const newDoc: LegalDocument = { 
      id: documents.length + 1, 
      name: file.name,
      uploadDate: new Date(),
      size: file.size
    };
    setDocuments(prev => [...prev, newDoc]);
    setActiveDocument(newDoc);
    // Add a mock entry for the new file
    (mockCases as any)[file.name] = mockCases["Case_C_vs_D_Pending.pdf"]; // Use a default mock
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
