/* ui/src/types.ts — Shared TypeScript types for picasso-rag-chat UI */

export interface ExtractedAnswer {
  value: string;
  confidence: number;
  turn_index: number;
}

export interface MissingField {
  field_code: string;
  description: string;
  enum_values?: string[];
}

export interface SessionSnapshot {
  session_id: string;
  profile_id: string;
  status: string;
  extracted_answers: Record<string, ExtractedAnswer>;
  missing_fields: MissingField[];
  model_believes_complete: boolean;
  is_complete: boolean;
  turn_count: number;
  /** Set by the backend when a vertical is resolved (auto-detected or pre-selected). */
  resolved_vertical: string | null;
  /** Set by the backend when a template key is resolved (auto-detected or pre-selected). */
  resolved_template_key: string | null;
  /** Optional: topic the model suggests as next to cover. */
  suggested_next_topic: string | null;
}


export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  timestamp: Date;
}
