import React from 'react';
import { User, Bot } from 'lucide-react';
import type { ChatMessage as ChatMessageType, CaseSummary, CasePrediction } from '../types';
import { AnalysisView } from './AnalysisView';
import { PredictionView } from './PredictionView';

interface ChatMessageProps {
  message: ChatMessageType;
}

// Type guard functions
const isCaseSummary = (data: any): data is CaseSummary => {
  return data && typeof data === 'object' && 
    ('facts' in data || 'petitionerArgs' in data || 'respondentArgs' in data || 'reasoning' in data || 'decision' in data);
};

const isCasePrediction = (data: any): data is CasePrediction => {
  return data && typeof data === 'object' && 'probabilities' in data && 'precedents' in data;
};

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.sender === 'user';

  if (isUser) {
    return (
      <div className="flex justify-end items-start gap-3 mb-4">
        <div className="bg-blue-600 text-white rounded-xl rounded-br-none p-4 max-w-xs sm:max-w-sm md:max-w-lg shadow-sm">
          <p className="text-sm sm:text-base">{message.text}</p>
        </div>
        <div className="bg-gray-300 dark:bg-gray-700 rounded-full h-8 w-8 sm:h-10 sm:w-10 flex items-center justify-center flex-shrink-0">
          <User size={16} className="sm:w-5 sm:h-5" />
        </div>
      </div>
    );
  }

  // Bot message rendering
  return (
    <div className="flex justify-start items-start gap-3 mb-4">
      <div className="bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300 rounded-full h-8 w-8 sm:h-10 sm:w-10 flex items-center justify-center flex-shrink-0">
        <Bot size={16} className="sm:w-5 sm:h-5" />
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-xl rounded-bl-none p-4 max-w-xs sm:max-w-md md:max-w-2xl border border-gray-200 dark:border-gray-700 shadow-sm">
        {message.type === 'greeting' && (
          <p className="text-sm sm:text-base" dangerouslySetInnerHTML={{ 
            __html: message.text?.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') || ''
          }} />
        )}
        {message.type === 'text' && <p className="text-sm sm:text-base">{message.text}</p>}
        {message.type === 'analysis' && message.data && isCaseSummary(message.data) && (
          <AnalysisView data={message.data} />
        )}
        {message.type === 'prediction' && message.data && isCasePrediction(message.data) && (
          <PredictionView data={message.data} />
        )}
      </div>
    </div>
  );
};