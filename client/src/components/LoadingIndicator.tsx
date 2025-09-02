import React from 'react';
import { Loader2, Bot } from 'lucide-react';

interface LoadingIndicatorProps {
  message?: string;
}

export const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({ 
  message = 'Analyzing...' 
}) => (
  <div className="flex justify-start items-start gap-3 mb-4">
    <div className="bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300 rounded-full h-8 w-8 sm:h-10 sm:w-10 flex items-center justify-center flex-shrink-0">
      <Bot size={16} className="sm:w-5 sm:h-5" />
    </div>
    <div className="bg-white dark:bg-gray-800 rounded-xl rounded-bl-none p-4 border border-gray-200 dark:border-gray-700 shadow-sm">
      <div className="flex items-center gap-2">
        <Loader2 className="animate-spin text-blue-500" size={18} />
        <span className="text-sm sm:text-base text-gray-600 dark:text-gray-300">{message}</span>
      </div>
    </div>
  </div>
);