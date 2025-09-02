import React, { useRef, useEffect } from 'react';
import { Send, Plus } from 'lucide-react';
import type { LegalDocument } from '../types';

interface ChatInputProps {
  input: string;
  activeDocument: LegalDocument | null;
  isLoading: boolean;
  onInputChange: (value: string) => void;
  onSendMessage: () => void;
  onFileUpload: (file: File) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  input,
  activeDocument,
  isLoading,
  onInputChange,
  onSendMessage,
  onFileUpload
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea based on content
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      // Reset height to auto to get the correct scrollHeight
      textarea.style.height = 'auto';
      // Set height based on scrollHeight, with min and max constraints
      const newHeight = Math.min(Math.max(textarea.scrollHeight, 44), 120); // Min 44px, Max 120px
      textarea.style.height = `${newHeight}px`;
    }
  }, [input]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSendMessage();
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileUpload(file);
      // Reset the input value to allow uploading the same file again
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4 sm:p-6">
      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept=".pdf,.doc,.docx,.txt"
      />
      
      <div className="max-w-4xl mx-auto flex items-end gap-2 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-2xl py-2 px-3 focus-within:ring-2 focus-within:ring-blue-500 transition-all duration-300 shadow-sm">
        <button
          onClick={handleUploadClick}
          className="p-2 text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex-shrink-0 mb-1"
          aria-label="Upload document"
        >
          <Plus size={22} />
        </button>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={`Ask about ${activeDocument?.name || 'the case'}...`}
          className="w-full bg-transparent focus:outline-none text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 min-w-0 resize-none overflow-hidden min-h-[44px] max-h-[120px] py-2"
          rows={1}
        />
        <button
          onClick={onSendMessage}
          disabled={isLoading || input.trim() === ''}
          className="bg-blue-600 text-white p-2.5 rounded-full hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-all duration-300 shadow-sm flex-shrink-0 disabled:opacity-50 mb-1"
          aria-label="Send message"
        >
          <Send size={18} />
        </button>
      </div>
    </div>
  );
};