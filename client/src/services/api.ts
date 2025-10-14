/**
 * API Service Layer
 * Handles all HTTP requests to the backend server
 */

import axios, { type AxiosInstance, type AxiosError } from 'axios';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds for document processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - add auth token if available
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - handle errors globally
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Type definitions matching backend responses
export interface DocumentQueryResponse {
  success: boolean;
  document_id: string;
  filename: string;
  session_id: string;
  answer: string;
  document_metadata?: Record<string, any>;
  sources?: Array<Record<string, any>>;
  classification?: Record<string, any>;
  tools_used?: string[];
  confidence?: number;
}

export interface FollowUpQueryResponse {
  answer: string;
  session_id: string;
  confidence?: number;
  sources?: Array<Record<string, any>>;
  classification?: Record<string, any>;
  tools_used?: string[];
}

export interface QueryResponse {
  answer: string;
  session_id: string;
  confidence?: number;
  sources?: Array<Record<string, any>>;
  classification?: Record<string, any>;
  tools_used?: string[];
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  system: string;
  version: string;
}

/**
 * Upload a document and immediately ask a question about it
 */
export const uploadDocumentAndAsk = async (
  file: File,
  query: string,
  sessionId?: string,
  roleFilter?: string[]
): Promise<DocumentQueryResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('query', query);
  
  if (sessionId) {
    formData.append('session_id', sessionId);
  }
  
  if (roleFilter && roleFilter.length > 0) {
    formData.append('role_filter', JSON.stringify(roleFilter));
  }

  const response = await apiClient.post<DocumentQueryResponse>(
    '/api/document-query/upload-and-ask',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};

/**
 * Ask a follow-up question in an existing document conversation
 */
export const askFollowUpQuestion = async (
  query: string,
  sessionId: string,
  roleFilter?: string[]
): Promise<FollowUpQueryResponse> => {
  const formData = new FormData();
  formData.append('query', query);
  formData.append('session_id', sessionId);
  
  if (roleFilter && roleFilter.length > 0) {
    formData.append('role_filter', JSON.stringify(roleFilter));
  }

  const response = await apiClient.post<FollowUpQueryResponse>(
    '/api/document-query/ask-followup',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};

/**
 * Process a general query across all indexed documents
 */
export const processQuery = async (
  query: string,
  sessionId?: string,
  context?: Record<string, any>,
  roleFilter?: string[]
): Promise<QueryResponse> => {
  const response = await apiClient.post<QueryResponse>('/api/query', {
    query,
    session_id: sessionId,
    context,
    role_filter: roleFilter,
  });

  return response.data;
};

/**
 * Health check endpoint
 */
export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
};

/**
 * Helper to handle API errors and extract error messages
 */
export const getErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ message?: string; detail?: string }>;
    return (
      axiosError.response?.data?.message ||
      axiosError.response?.data?.detail ||
      axiosError.message ||
      'An unexpected error occurred'
    );
  }
  
  if (error instanceof Error) {
    return error.message;
  }
  
  return 'An unexpected error occurred';
};

export default apiClient;
