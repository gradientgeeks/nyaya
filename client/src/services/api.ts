/**
 * API Service for Nyaya Backend Integration
 * 
 * Handles all communication with the FastAPI backend.
 */

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_VERSION = '/api/v1';

// ============================================================================
// Types (matching backend schemas)
// ============================================================================

export interface SessionResponse {
  session_id: string;
  message: string;
}

export interface QueryRequest {
  query: string;
  session_id: string;
  case_id?: string;
  role_filter?: string[];
}

export interface SearchRequest {
  query: string;
  session_id: string;
  case_id?: string;
  top_k?: number;
}

export interface PredictOutcomeRequest {
  case_description: string;
  session_id: string;
  case_id?: string;
  relevant_laws?: string[];
}

export interface QueryResponse {
  answer: string;
  intent?: string;
  classification_result?: {
    case_id: string;
    total_sentences: number;
    role_distribution: Record<string, number>;
    sentences: Array<{
      text: string;
      role: string;
      confidence: number;
      sentence_index: number;
    }>;
    processing_time: number;
    timestamp: string;
  };
  search_results?: Array<{
    case_id: string;
    case_title?: string;
    overall_similarity: number;
    role_scores: Record<string, number>;
    matching_roles: string[];
    snippet?: string;
  }>;
  prediction?: {
    case_id: string;
    predicted_outcome: string;
    confidence: number;
    based_on_cases: number;
    key_factors: string[];
    similar_precedents: Array<{
      case_id: string;
      case_title?: string;
      overall_similarity: number;
    }>;
    processing_time: number;
  };
  rag_response?: {
    query: string;
    answer: string;
    sources: Array<{
      text: string;
      role: string;
      case_id: string;
      similarity: number;
    }>;
    role_filter_applied?: string[];
    confidence: number;
    processing_time: number;
  };
  session_id: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface StatsResponse {
  pinecone: {
    total_vector_count: number;
    dimension: number;
    index_fullness: number;
  };
  sessions: {
    active_count: number;
  };
}

// ============================================================================
// Error Handling
// ============================================================================

export class APIError extends Error {
  status: number;
  detail?: string;

  constructor(
    message: string,
    status: number,
    detail?: string
  ) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.detail = detail;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new APIError(
      errorData.error || 'API request failed',
      response.status,
      errorData.detail || response.statusText
    );
  }
  return response.json();
}

// ============================================================================
// API Client Class
// ============================================================================

class NyayaAPIClient {
  private baseUrl: string;
  private defaultHeaders: HeadersInit;

  constructor() {
    this.baseUrl = `${API_BASE_URL}${API_VERSION}`;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  /**
   * Create a new session
   */
  async createSession(userId?: string): Promise<SessionResponse> {
    const response = await fetch(`${this.baseUrl}/sessions`, {
      method: 'POST',
      headers: this.defaultHeaders,
      body: JSON.stringify({ user_id: userId }),
    });
    return handleResponse<SessionResponse>(response);
  }

  /**
   * Upload a document for classification
   */
  async uploadDocument(
    file: File,
    sessionId: string,
    caseId?: string
  ): Promise<QueryResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);
    if (caseId) {
      formData.append('case_id', caseId);
    }

    const response = await fetch(`${this.baseUrl}/upload`, {
      method: 'POST',
      // Don't set Content-Type header - browser will set it with boundary for FormData
      body: formData,
    });
    return handleResponse<QueryResponse>(response);
  }

  /**
   * Query documents (role-aware RAG)
   */
  async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/query`, {
      method: 'POST',
      headers: this.defaultHeaders,
      body: JSON.stringify(request),
    });
    return handleResponse<QueryResponse>(response);
  }

  /**
   * Search for similar cases
   */
  async searchSimilarCases(request: SearchRequest): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/search`, {
      method: 'POST',
      headers: this.defaultHeaders,
      body: JSON.stringify(request),
    });
    return handleResponse<QueryResponse>(response);
  }

  /**
   * Predict case outcome
   */
  async predictOutcome(request: PredictOutcomeRequest): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/predict`, {
      method: 'POST',
      headers: this.defaultHeaders,
      body: JSON.stringify(request),
    });
    return handleResponse<QueryResponse>(response);
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`);
    return handleResponse<HealthResponse>(response);
  }

  /**
   * Get system statistics
   */
  async getStats(): Promise<StatsResponse> {
    const response = await fetch(`${this.baseUrl}/stats`);
    return handleResponse<StatsResponse>(response);
  }
}

// Export singleton instance
export const apiClient = new NyayaAPIClient();

// Export convenience methods
export const api = {
  createSession: (userId?: string) => apiClient.createSession(userId),
  uploadDocument: (file: File, sessionId: string, caseId?: string) =>
    apiClient.uploadDocument(file, sessionId, caseId),
  query: (request: QueryRequest) => apiClient.query(request),
  search: (request: SearchRequest) => apiClient.searchSimilarCases(request),
  predict: (request: PredictOutcomeRequest) => apiClient.predictOutcome(request),
  healthCheck: () => apiClient.healthCheck(),
  getStats: () => apiClient.getStats(),
};
