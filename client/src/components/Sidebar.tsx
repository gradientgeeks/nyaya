import React from 'react';
import { FileText, ChevronsRight, ChevronLeft, ChevronRight } from 'lucide-react';
import type { LegalDocument } from '../types';
import { ThemeToggle } from './ThemeToggle';

interface SidebarProps {
  documents: LegalDocument[];
  activeDocument: LegalDocument | null;
  onDocumentSelect: (document: LegalDocument) => void;
  isMinimized: boolean;
  onToggleMinimize: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  documents,
  activeDocument,
  onDocumentSelect,
  isMinimized,
  onToggleMinimize
}) => {
  return (
    <aside className={`${isMinimized ? 'w-16' : 'w-80'} ${isMinimized ? '-translate-x-full lg:translate-x-0 lg:w-16' : 'translate-x-0 lg:w-80'} fixed lg:relative z-30 min-h-screen bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 transition-all duration-300 flex flex-col`}>
      {/* Toggle Button */}
      <button
        onClick={onToggleMinimize}
        className="absolute -right-3 top-6 z-10 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-full p-1.5 shadow-md hover:shadow-lg transition-all duration-200 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hidden lg:flex items-center justify-center"
        aria-label={isMinimized ? 'Expand sidebar' : 'Minimize sidebar'}
      >
        {isMinimized ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>

      <div className={`${isMinimized ? 'px-2' : 'px-6'} py-6 transition-all duration-300`}>
        {/* Header */}
        <div className={`flex items-center gap-3 mb-6 ${isMinimized ? 'justify-center' : ''}`}>
          <div className="p-2 bg-blue-600/10 text-blue-600 dark:bg-blue-500/10 dark:text-blue-400 rounded-lg flex-shrink-0">
            <FileText size={24} />
          </div>
          {!isMinimized && (
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Legalysis AI</h1>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className={`${isMinimized ? 'px-2' : 'px-6'} flex-1 overflow-y-auto`}>
        {!isMinimized && (
          <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            Case Files
          </h2>
        )}
        <ul className="space-y-2">
          {documents.map(doc => (
            <li key={doc.id}>
              <button
                onClick={() => onDocumentSelect(doc)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg transition-colors duration-200 text-left group ${
                  activeDocument?.id === doc.id 
                    ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 font-semibold' 
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700/50 text-gray-700 dark:text-gray-300'
                } ${isMinimized ? 'justify-center px-2' : ''}`}
                title={isMinimized ? doc.name : ''}
              >
                <FileText size={20} className="flex-shrink-0" />
                {!isMinimized && (
                  <>
                    <span className="truncate flex-1">{doc.name}</span>
                    {activeDocument?.id === doc.id && <ChevronsRight size={16} />}
                  </>
                )}
                {isMinimized && activeDocument?.id === doc.id && (
                  <div className="absolute left-full ml-2 bg-gray-900 dark:bg-gray-700 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-50">
                    {doc.name}
                  </div>
                )}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      {/* Footer */}
      {/* <div className={`${isMinimized ? 'px-2' : 'px-6'} py-4 border-t border-gray-200 dark:border-gray-700`}>
        <ThemeToggle isMinimized={isMinimized} />
        {!isMinimized && (
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center mt-3">
            Legal Document Analysis System
          </p>
        )}
      </div> */}
    </aside>
  );
};