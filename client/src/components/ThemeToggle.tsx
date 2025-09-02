import React from 'react';
import { Sun, Moon } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

interface ThemeToggleProps {
  isMinimized?: boolean;
}

export const ThemeToggle: React.FC<ThemeToggleProps> = ({ isMinimized = false }) => {
  const { theme, toggleTheme } = useTheme();

  const handleToggle = () => {
    console.log('Theme toggle clicked. Current theme:', theme);
    toggleTheme();
  };

  return (
    <button
      onClick={handleToggle}
      className={`flex items-center gap-2 p-3 rounded-lg transition-colors duration-200 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700/50 w-full ${
        isMinimized ? 'justify-center' : 'justify-start'
      }`}
      title={isMinimized ? (theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode') : ''}
    >
      {theme === 'light' ? (
        <Moon size={20} className="flex-shrink-0" />
      ) : (
        <Sun size={20} className="flex-shrink-0" />
      )}
      {!isMinimized && (
        <span className="text-sm font-medium">
          {theme === 'light' ? 'Dark Mode' : 'Light Mode'}
        </span>
      )}
    </button>
  );
};