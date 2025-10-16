export interface LegalDocument {
  id: number;
  name: string;
  type?: 'judged' | 'pending';
  uploadDate?: Date;
  size?: number;
  sessionId?: string;
  documentId?: string;
}

export interface CaseSummary {
  facts?: string;
  petitionerArgs?: string;
  respondentArgs?: string;
  reasoning?: string | null;
  decision?: string | null;
  issues?: string;
}

export interface PredictionOutcome {
  outcome: string;
  probability: string;
}

export interface CasePrediction {
  probabilities: PredictionOutcome[];
  precedents: string[];
}

export interface CaseData {
  type: 'judged' | 'pending';
  summary: CaseSummary;
  prediction?: CasePrediction;
}

export interface ChatMessage {
  id?: string;
  sender: 'user' | 'bot';
  text?: string;
  type?: 'greeting' | 'text' | 'analysis' | 'prediction';
  data?: CaseSummary | CasePrediction;
  timestamp?: Date;
}

export interface AnalysisSection {
  title: string;
  key: keyof CaseSummary;
  content?: string | null;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface ConversationContext {
  documentId: number;
  conversationId: string;
  history: ChatMessage[];
}

export interface LoadingState {
  isLoading: boolean;
  operation?: 'upload' | 'analysis' | 'prediction' | 'chat';
  message?: string;
}