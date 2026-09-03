/* ui/src/useChat.ts — SSE chat hook */
import { useState, useRef, useCallback } from "react";
import type { ChatMessage, SessionSnapshot } from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface UseChatReturn {
  messages: ChatMessage[];
  snapshot: SessionSnapshot | null;
  isStreaming: boolean;
  error: string | null;
  sendMessage: (text: string, context?: ChatContext, hiddenUserMessage?: boolean) => Promise<void>;
  addLocalMessage: (role: "user" | "assistant", content: string) => void;
  uploadDocuments: (files: File[]) => Promise<void>;
  directFieldWrite: (fieldCode: string, value: string) => Promise<{ status: string; snapshot: SessionSnapshot | null }>;
  sessionId: string | null;
  clearSession: () => void;
}

/** Pre-chat selection context passed into the conversation on the first turn. */
export interface ChatContext {
  vertical: string;
  template_key: string;
}

function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  const clearSession = useCallback(() => {
    sessionIdRef.current = null;
    setMessages([]);
    setSnapshot(null);
    setError(null);
  }, []);

  const addLocalMessage = useCallback((role: "user" | "assistant", content: string) => {
    const msg: ChatMessage = {
      id: generateId(),
      role,
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, msg]);
  }, []);

  const sendMessage = useCallback(async (text: string, context?: ChatContext, hiddenUserMessage?: boolean) => {
    if (isStreaming) return;
    setError(null);

    // Append user message if not hidden
    if (!hiddenUserMessage) {
      const userMsg: ChatMessage = {
        id: generateId(),
        role: "user",
        content: text,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);
    }

    // Placeholder for streaming assistant message
    const assistantId = generateId();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, assistantMsg]);
    setIsStreaming(true);

    try {
      const isFirstMessage = !sessionIdRef.current;
      const body: Record<string, unknown> = {
        session_id: sessionIdRef.current ?? undefined,
        user_message: text,
      };
      // On the very first message of a session, attach the pre-chat selections.
      // The backend uses these to set resolved_vertical / resolved_template_key
      // before the first LLM turn, bypassing text-sniffing auto-detection.
      if (isFirstMessage && context?.vertical) {
        body.vertical = context.vertical;
      }
      if (isFirstMessage && context?.template_key) {
        body.template_key = context.template_key;
      }
      const res = await fetch(`${API_BASE}/conversation/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        accumulated += decoder.decode(value, { stream: true });
        const lines = accumulated.split("\n\n");
        accumulated = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const json = line.slice(6).trim();
          if (!json) continue;

          try {
            const evt = JSON.parse(json);
            if (evt.chunk !== undefined) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + evt.chunk }
                    : m
                )
              );
            } else if (evt.done && evt.snapshot) {
              sessionIdRef.current = evt.snapshot.session_id;
              setSnapshot(evt.snapshot);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, streaming: false } : m
                )
              );
            }
          } catch {
            // partial line — accumulate and continue
          }
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: m.content || "[Error: " + msg + "]", streaming: false }
            : m
        )
      );
    } finally {
      setIsStreaming(false);
    }
  }, [isStreaming]);

  const uploadDocuments = useCallback(async (files: File[]) => {
    if (!sessionIdRef.current || isStreaming || files.length === 0) return;
    setError(null);
    setIsStreaming(true);

    const fileNames = files.map(f => f.name).join(", ");
    const userMsg: ChatMessage = {
      id: generateId(),
      role: "user",
      content: `[Uploaded document(s): ${fileNames}]`,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append("files", file); // Must match backend 'files' param
      });

      const res = await fetch(`${API_BASE}/conversation/session/${sessionIdRef.current}/document`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      
      if (data.snapshot) {
        setSnapshot(data.snapshot);
      }
      
      // Auto-trigger AI acknowledgement via hidden message
      const hiddenMessage = `__hidden_upload_success__: ${files.length} file(s) uploaded successfully. Acknowledge the upload and ask the next question.`;
      
      // Disable streaming temporarily so sendMessage can run
      setIsStreaming(false);
      
      await sendMessage(hiddenMessage, undefined, true);

    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error during upload";
      setError(msg);
      setIsStreaming(false);
    }
  }, [isStreaming, sendMessage]);

  // Bug 5 fix: directFieldWrite — bypass LLM extraction for UI chip selections.
  // Posts directly to the backend's direct_field_write endpoint at confidence=1.0.
  // This is the reliable path for enum field UI selections: no LLM involved = no hallucination.
  const directFieldWrite = useCallback(async (
    fieldCode: string,
    value: string,
  ): Promise<{ status: string; snapshot: SessionSnapshot | null }> => {
    const sid = sessionIdRef.current;
    if (!sid) return { status: "no_session", snapshot: null };
    try {
      const res = await fetch(
        `${API_BASE}/conversation/session/${sid}/direct_field_write`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ field_code: fieldCode, value }),
        }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      // If the backend returns a snapshot, update local state
      if (data.snapshot) {
        setSnapshot(data.snapshot);
      }
      return { status: data.status ?? "saved", snapshot: data.snapshot ?? null };
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      console.error("directFieldWrite failed:", msg);
      return { status: "error", snapshot: null };
    }
  }, []);

  return {
    messages,
    snapshot,
    isStreaming,
    error,
    sendMessage,
    addLocalMessage,
    uploadDocuments,
    directFieldWrite,
    sessionId: sessionIdRef.current,
    clearSession,
  };
}
