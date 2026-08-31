/* ui/src/App.tsx — Main application with pre-chat selection flow */
import { useState, useRef, useEffect } from "react";
import { useChat } from "./useChat";
import type { ChatContext } from "./useChat";
import type { SessionSnapshot, ChatMessage } from "./types";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// ── Vertical & Content-Type Configuration ────────────────────────────────────
// Keys are the exact folder names in app/project_profiles/picasso_fusion/field_sets/
// and the YAML template_key values within each folder.

interface VerticalOption {
  key: string;
  label: string;
  icon: string;
  desc: string;
}

interface ContentTypeOption {
  key: string;      // matches template_key in field_sets YAML
  label: string;
  icon: string;
  desc: string;
}

const VERTICALS: VerticalOption[] = [
  { key: "restaurant",  label: "Restaurant & Café",   icon: "🍽️", desc: "Food, beverage & dining" },
  { key: "realestate",  label: "Real Estate",          icon: "🏠", desc: "Property listings & agency" },
  { key: "ecommerce",   label: "E-Commerce",           icon: "🛒", desc: "Online retail & products" },
  { key: "fitness",     label: "Fitness & Wellness",   icon: "💪", desc: "Gyms, studios & wellness brands" },
  { key: "retail",      label: "Retail & Boutique",    icon: "🏪", desc: "In-store & boutique brands" },
  { key: "startup",     label: "Startup & Entrepreneur",icon: "🚀", desc: "Pitches, decks & investor docs" },
  { key: "technology",  label: "Technology Services",  icon: "💻", desc: "SaaS, IT & tech businesses" },
];

// Content types keyed by vertical — template_key must exactly match the YAML filename
// (without .yaml extension) in field_sets/<vertical>/
const CONTENT_TYPES: Record<string, ContentTypeOption[]> = {
  restaurant: [
    { key: "restaurant_cafe_static_post",    label: "Static Post",    icon: "🖼️",  desc: "Single image or graphic" },
    { key: "restaurant_cafe_carousel",       label: "Carousel",       icon: "📚",  desc: "Multi-slide carousel" },
    { key: "restaurant_cafe_reel",           label: "Reel / Video",   icon: "🎬",  desc: "Short-form vertical video" },
    { key: "restaurant_cafe_digital_addon",  label: "Digital Asset",  icon: "📱",  desc: "Banner, ad or digital file" },
    { key: "restaurant_cafe_print_addon",    label: "Print",          icon: "🖨️",  desc: "Menu, flyer or print material" },
    { key: "restaurant_cafe_video_addon",    label: "Video Edit",     icon: "🎞️",  desc: "Professional video edit" },
  ],
  realestate: [
    { key: "real_estate_listing_sheet",      label: "Listing Sheet",     icon: "🏡",  desc: "Property listing one-pager" },
    { key: "real_estate_flyer_brochure",     label: "Flyer / Brochure",  icon: "📋",  desc: "Print flyer or brochure" },
    { key: "real_estate_buyer_seller_guide", label: "Buyer/Seller Guide", icon: "📘",  desc: "Client education guide" },
    { key: "real_estate_market_report",      label: "Market Report",      icon: "📊",  desc: "Area or suburb market data" },
    { key: "real_estate_cma_presentation",   label: "CMA Presentation",   icon: "🗂️",  desc: "Comparative market analysis" },
    { key: "real_estate_open_home_signboard",label: "Open Home Signboard", icon: "🪧",  desc: "Physical signboard" },
    { key: "real_estate_testimonial_set",    label: "Testimonial Set",    icon: "⭐",  desc: "Client review graphics" },
    { key: "real_estate_email_template",     label: "Email Template",     icon: "📧",  desc: "Branded email layout" },
    { key: "real_estate_linkedin_banner",    label: "LinkedIn Banner",    icon: "💼",  desc: "Professional profile banner" },
    { key: "real_estate_youtube_banner",     label: "YouTube Banner",     icon: "🎥",  desc: "Channel header art" },
  ],
  ecommerce: [
    { key: "ecommerce_static_post",    label: "Static Post",     icon: "🖼️",  desc: "Single social media post" },
    { key: "ecommerce_carousel",       label: "Carousel",        icon: "📚",  desc: "Multi-slide carousel" },
    { key: "ecommerce_reel",           label: "Reel / Video",    icon: "🎬",  desc: "Short-form video" },
    { key: "ecommerce_product_mockup", label: "Product Mockup",  icon: "📦",  desc: "Product visual mockup" },
    { key: "ecommerce_ad_banner",      label: "Ad Banner",       icon: "📣",  desc: "Digital advertising banner" },
    { key: "ecommerce_digital_addon",  label: "Digital Asset",   icon: "📱",  desc: "Digital marketing asset" },
    { key: "ecommerce_print_addon",    label: "Print",           icon: "🖨️",  desc: "Packaging or print material" },
    { key: "ecommerce_video_addon",    label: "Video Edit",      icon: "🎞️",  desc: "Professional video edit" },
  ],
  fitness: [
    { key: "fitness_wellness_static_post_set",  label: "Static Post",    icon: "🖼️",  desc: "Single social graphic" },
    { key: "fitness_wellness_carousel_set",     label: "Carousel",       icon: "📚",  desc: "Multi-slide carousel" },
    { key: "fitness_wellness_reel",             label: "Reel / Video",   icon: "🎬",  desc: "Short-form video" },
    { key: "fitness_wellness_digital_addon",    label: "Digital Asset",  icon: "📱",  desc: "Digital marketing asset" },
    { key: "fitness_wellness_print_addon",      label: "Print",          icon: "🖨️",  desc: "Poster or flyer" },
    { key: "fitness_wellness_video_addon",      label: "Video Edit",     icon: "🎞️",  desc: "Professional video edit" },
  ],
  retail: [
    { key: "retail_boutique_static_post",   label: "Static Post",   icon: "🖼️",  desc: "Single social graphic" },
    { key: "retail_boutique_carousel",      label: "Carousel",      icon: "📚",  desc: "Multi-slide carousel" },
    { key: "retail_boutique_reel",          label: "Reel / Video",  icon: "🎬",  desc: "Short-form video" },
    { key: "retail_boutique_digital_addon", label: "Digital Asset", icon: "📱",  desc: "Digital marketing asset" },
    { key: "retail_boutique_print_addon",   label: "Print",         icon: "🖨️",  desc: "In-store print material" },
    { key: "retail_boutique_video_addon",   label: "Video Edit",    icon: "🎞️",  desc: "Professional video edit" },
  ],
  startup: [
    { key: "startup_entrepreneur_pitch_deck_investor",    label: "Pitch Deck",         icon: "📊",  desc: "Investor presentation" },
    { key: "startup_entrepreneur_one_pager",              label: "One-Pager",          icon: "📄",  desc: "Startup summary doc" },
    { key: "startup_entrepreneur_sales_deck",             label: "Sales Deck",         icon: "🗂️",  desc: "Sales presentation" },
    { key: "startup_entrepreneur_proposal_template",      label: "Proposal Template",  icon: "📝",  desc: "Client proposal" },
    { key: "startup_entrepreneur_presentation_template",  label: "Presentation",       icon: "🖥️",  desc: "Brand presentation" },
    { key: "startup_entrepreneur_press_kit",              label: "Press Kit",          icon: "📰",  desc: "Media & PR kit" },
    { key: "startup_entrepreneur_business_card",          label: "Business Card",      icon: "💳",  desc: "Professional card design" },
    { key: "startup_entrepreneur_linkedin_banner",        label: "LinkedIn Banner",    icon: "💼",  desc: "Profile header banner" },
    { key: "startup_entrepreneur_email_signature",        label: "Email Signature",    icon: "📧",  desc: "Branded email footer" },
  ],
  technology: [
    { key: "technology_services_pitch_deck_investor",  label: "Pitch Deck",      icon: "📊",  desc: "Investor presentation" },
    { key: "technology_services_sales_deck",           label: "Sales Deck",      icon: "🗂️",  desc: "Sales presentation" },
    { key: "technology_services_product_one_pager",    label: "Product One-Pager", icon: "📄",  desc: "Product summary" },
    { key: "technology_services_case_study",           label: "Case Study",      icon: "🔬",  desc: "Client success story" },
    { key: "technology_services_white_paper",          label: "White Paper",     icon: "📋",  desc: "Technical white paper" },
    { key: "technology_services_web_banner_set",       label: "Web Banner Set",  icon: "🌐",  desc: "Display ad banners" },
    { key: "technology_services_email_template",       label: "Email Template",  icon: "📧",  desc: "Branded email layout" },
    { key: "technology_services_linkedin_banner",      label: "LinkedIn Banner", icon: "💼",  desc: "Profile header banner" },
  ],
};

// ── Picasso AI Glyph ─────────────────────────────────────────────────────────
function AIGlyph({ size = 30 }: { size?: number }) {
  return (
    <div className="ai-glyph" style={{ width: size, height: size }} aria-hidden>
      <img
        src="/AIbubble.svg"
        alt="Picasso AI"
        style={{ width: size, height: size, objectFit: "cover", display: "block" }}
      />
    </div>
  );
}

// ── Typing Dots Animation ─────────────────────────────────────────────────────
function TypingDots() {
  return (
    <div className="msg-in">
      <div className="ai-bubble-row">
        <AIGlyph />
        <div className="ai-bubble-wrapper">
          <div className="ai-author-label">Picasso AI</div>
          <div className="streaming-dots">
            <span className="dot tdot" style={{ animationDelay: "0s" }} />
            <span className="dot tdot" style={{ animationDelay: "0.2s" }} />
            <span className="dot tdot" style={{ animationDelay: "0.4s" }} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Message Bubbles ───────────────────────────────────────────────────────────
function MessageItem({ message }: { message: ChatMessage }) {
  if (message.role === "assistant") {
    return (
      <div className="msg-in ai-bubble-row">
        <AIGlyph />
        <div className="ai-bubble-wrapper">
          <div className="ai-author-label">Picasso AI</div>
          <div className="ai-bubble-content">
            {message.content}
            {message.streaming && (
              <span className="tdot inline-block ml-1 opacity-70">▍</span>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="msg-in user-bubble-row">
      <div className="user-bubble-content">{message.content}</div>
      <div className="user-avatar" title="User">U</div>
    </div>
  );
}

// ── Pre-Chat Selection Screen ─────────────────────────────────────────────────
type SelectionStep = "vertical" | "contentType";

interface SelectionScreenProps {
  onStart: (vertical: VerticalOption, contentType: ContentTypeOption) => void;
}

function SelectionScreen({ onStart }: SelectionScreenProps) {
  const [step, setStep] = useState<SelectionStep>("vertical");
  const [selectedVertical, setSelectedVertical] = useState<VerticalOption | null>(null);

  const handleVerticalClick = (v: VerticalOption) => {
    setSelectedVertical(v);
    setStep("contentType");
  };

  const handleContentTypeClick = (ct: ContentTypeOption) => {
    if (selectedVertical) {
      onStart(selectedVertical, ct);
    }
  };

  const handleBack = () => {
    setStep("vertical");
    setSelectedVertical(null);
  };

  if (step === "vertical") {
    return (
      <div className="selection-screen">
        <div className="selection-header">
          <AIGlyph size={40} />
          <span className="selection-eyebrow">Picasso AI · Debug Testbed</span>
          <h2 className="selection-title">What type of business are you briefing for?</h2>
          <p className="selection-subtitle">
            Choose the vertical that best matches your client. This determines the
            specific brief fields we'll capture in the conversation.
          </p>
        </div>
        <div className="selection-grid">
          {VERTICALS.map((v) => (
            <button
              key={v.key}
              className="selection-card"
              onClick={() => handleVerticalClick(v)}
              id={`vertical-${v.key}`}
            >
              <span className="selection-card-icon">{v.icon}</span>
              <span className="selection-card-label">{v.label}</span>
              <span className="selection-card-desc">{v.desc}</span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  // Step 2 — Content Type
  const contentTypes = CONTENT_TYPES[selectedVertical?.key ?? ""] ?? [];

  return (
    <div className="selection-screen">
      <div className="selection-back-row">
        <button className="selection-back-btn" onClick={handleBack} id="btn-back-to-vertical">
          ← Back
        </button>
        {selectedVertical && (
          <span className="selection-context-pill">
            {selectedVertical.icon} {selectedVertical.label}
          </span>
        )}
      </div>
      <div className="selection-header">
        <span className="selection-eyebrow">Step 2 of 2</span>
        <h2 className="selection-title">Select a content type</h2>
        <p className="selection-subtitle">
          Choose the specific deliverable for this brief. The AI will then ask
          all the fields needed for this exact content type.
        </p>
      </div>
      <div className="selection-grid">
        {contentTypes.map((ct) => (
          <button
            key={ct.key}
            className="selection-card"
            onClick={() => handleContentTypeClick(ct)}
            id={`content-type-${ct.key}`}
          >
            <span className="selection-card-icon">{ct.icon}</span>
            <span className="selection-card-label">{ct.label}</span>
            <span className="selection-card-desc">{ct.desc}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Brief Modal ───────────────────────────────────────────────────────────────
interface BriefModalProps {
  briefMarkdown: string;
  onClose: () => void;
}

function BriefModal({ briefMarkdown, onClose }: BriefModalProps) {
  // Render the markdown as simple HTML paragraphs / lists
  // (No external markdown lib — render section headings and bullet points natively)
  const renderLine = (line: string, idx: number) => {
    if (line.startsWith("## ")) {
      return <h2 key={idx} className="brief-modal-h2">{line.slice(3)}</h2>;
    }
    if (line.startsWith("### ")) {
      return <h3 key={idx} className="brief-modal-h3">{line.slice(4)}</h3>;
    }
    if (line.startsWith("- **")) {
      // "- **Label**: value" pattern
      const content = line.slice(2);
      const match = content.match(/^\*\*(.+?)\*\*:\s*(.*)/);
      if (match) {
        return (
          <div key={idx} className="brief-modal-field">
            <span className="brief-modal-field-label">{match[1]}</span>
            <span className="brief-modal-field-value">{match[2] || <em>not yet provided</em>}</span>
          </div>
        );
      }
    }
    if (line.startsWith("- ")) {
      return <li key={idx} className="brief-modal-li">{line.slice(2)}</li>;
    }
    if (line.startsWith("*") && line.endsWith("*")) {
      return <p key={idx} className="brief-modal-meta">{line.slice(1, -1)}</p>;
    }
    if (line.trim() === "") {
      return <div key={idx} style={{ height: 4 }} />;
    }
    return <p key={idx} className="brief-modal-p">{line}</p>;
  };

  return (
    <div className="brief-modal-overlay" id="brief-modal-overlay" onClick={onClose}>
      <div
        className="brief-modal"
        id="brief-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Full Brief Summary"
      >
        <div className="brief-modal-header">
          <span className="brief-modal-title">📋 Full Brief Summary</span>
          <button
            className="brief-modal-close"
            onClick={onClose}
            id="btn-close-brief-modal"
            aria-label="Close brief"
          >
            ✕
          </button>
        </div>
        <div className="brief-modal-body">
          {briefMarkdown.split("\n").map((line, idx) => renderLine(line, idx))}
        </div>
        <div className="brief-modal-footer">
          <button
            className="brief-modal-submit-btn"
            id="btn-submit-brief-from-modal"
            onClick={() => alert("Brief submission would trigger here once the submission backend is wired in.")}
          >
            🚀 Submit Brief
          </button>
          <button
            className="brief-modal-close-btn"
            onClick={onClose}
            id="btn-close-brief-modal-footer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Completion CTA Banner ─────────────────────────────────────────────────────
interface CompletionBannerProps {
  onViewBrief: () => void;
  isLoadingBrief: boolean;
}

function CompletionBanner({ onViewBrief, isLoadingBrief }: CompletionBannerProps) {
  return (
    <div className="completion-banner" id="completion-banner" role="status" aria-live="polite">
      <div className="completion-banner-content">
        <div className="completion-banner-left">
          <span className="completion-banner-icon">✅</span>
          <div>
            <div className="completion-banner-title">Brief Complete!</div>
            <div className="completion-banner-sub">
              All required fields have been captured. You can still ask questions or request changes.
            </div>
          </div>
        </div>
        <div className="completion-banner-actions">
          <button
            className="completion-view-btn"
            onClick={onViewBrief}
            disabled={isLoadingBrief}
            id="btn-view-brief"
          >
            {isLoadingBrief ? "Loading…" : "📄 View Full Brief"}
          </button>
          <button
            className="completion-submit-btn"
            id="btn-submit-brief"
            onClick={() => alert("Brief submission would trigger here once the submission backend is wired in.")}
          >
            🚀 Submit Brief
          </button>
        </div>
      </div>
    </div>
  );
}

// ── State Sidebar Inspector ───────────────────────────────────────────────────
function StatePanel({ snapshot }: { snapshot: SessionSnapshot | null }) {
  if (!snapshot) {
    return (
      <div className="state-panel">
        <div className="state-header">
          <span className="state-header-icon">🎨</span>
          <div>
            <div className="state-header-title">Session State</div>
            <div className="state-header-sub">FastAPI RAG Live Feed</div>
          </div>
        </div>
        <div className="state-content">
          <div className="v1-card">
            <p className="empty-state-text">
              No active session yet. Select a vertical and content type, then send a message.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const capturedEntries = Object.entries(snapshot.extracted_answers);
  const missing = snapshot.missing_fields;

  return (
    <div className="state-panel">
      <div className="state-header">
        <span className="state-header-icon">🎨</span>
        <div>
          <div className="state-header-title">Session State</div>
          <div className="state-header-sub">FastAPI RAG Live Feed</div>
        </div>
      </div>

      <div className="state-content">
        {/* Metadata Card */}
        <div className="v1-card">
          <div className="card-label">Session Metadata</div>
          <div className="meta-row">
            <span className="meta-key">Session ID</span>
            <span className="meta-val mono-tag">{snapshot.session_id.slice(0, 8)}…</span>
          </div>
          <div className="meta-row">
            <span className="meta-key">Profile</span>
            <span className="meta-val">{snapshot.profile_id}</span>
          </div>
          {snapshot.resolved_vertical && (
            <div className="meta-row">
              <span className="meta-key">Vertical</span>
              <span className="meta-val">{snapshot.resolved_vertical}</span>
            </div>
          )}
          {snapshot.resolved_template_key && (
            <div className="meta-row">
              <span className="meta-key">Template</span>
              <span className="meta-val mono-tag" style={{ fontSize: 10 }}>
                {snapshot.resolved_template_key}
              </span>
            </div>
          )}
          <div className="meta-row">
            <span className="meta-key">Turn Count</span>
            <span className="meta-val">{snapshot.turn_count}</span>
          </div>

          <div className="meta-row" style={{ marginTop: 4 }}>
            <span className="meta-key">Status</span>
            {snapshot.is_complete ? (
              <span className="status-pill complete">
                <span className="status-dot" />
                Brief Complete
              </span>
            ) : snapshot.model_believes_complete ? (
              <span className="status-pill model-complete">
                <span className="status-dot" />
                Model Thinks Done
              </span>
            ) : (
              <span className="status-pill in-progress">
                <span className="status-dot" />
                In Progress
              </span>
            )}
          </div>
        </div>

        {/* Missing Fields Card */}
        <div className="v1-card">
          <div className="card-label missing">
            Missing Fields ({missing.length})
          </div>
          {missing.length === 0 ? (
            <p className="empty-state-text">No missing fields remaining.</p>
          ) : (
            <div className="field-chips-grid">
              {missing.map((mf) => (
                <div
                  key={mf.field_code}
                  className="field-chip-missing"
                  title={mf.description}
                >
                  {mf.field_code}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Captured Fields Card */}
        <div className="v1-card">
          <div className="card-label captured">
            Captured Fields ({capturedEntries.length})
          </div>
          {capturedEntries.length === 0 ? (
            <p className="empty-state-text">No fields extracted yet.</p>
          ) : (
            <div className="captured-list">
              {capturedEntries.map(([code, data]) => {
                const confPercent = Math.round(data.confidence * 100);
                const confClass =
                  data.confidence >= 0.8 ? "high" :
                  data.confidence >= 0.6 ? "medium" : "low";

                return (
                  <div key={code} className="captured-card">
                    <div className="captured-header">
                      <span className="captured-field-code">{code}</span>
                      <span className={`confidence-pill ${confClass}`}>
                        {confPercent}%
                      </span>
                    </div>
                    <div className="captured-val">{data.value}</div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
type AppStep = "selection" | "chat";

export default function App() {
  const {
    messages,
    snapshot,
    isStreaming,
    error,
    sendMessage,
    uploadDocument,
    sessionId,
    clearSession,
  } = useChat();
  const [input, setInput] = useState("");
  const [appStep, setAppStep] = useState<AppStep>("selection");
  const [chatContext, setChatContext] = useState<ChatContext | null>(null);
  const [contextLabel, setContextLabel] = useState<string>("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Brief modal state
  const [briefModalOpen, setBriefModalOpen] = useState(false);
  const [briefContent, setBriefContent] = useState<string>("");
  const [isLoadingBrief, setIsLoadingBrief] = useState(false);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  // Focus textarea when not streaming in chat view
  useEffect(() => {
    if (appStep === "chat" && !isStreaming) {
      textareaRef.current?.focus();
    }
  }, [isStreaming, appStep]);

  const handleSelectionComplete = (
    vertical: VerticalOption,
    contentType: ContentTypeOption,
  ) => {
    const ctx: ChatContext = {
      vertical: vertical.key,
      template_key: contentType.key,
    };
    setChatContext(ctx);
    setContextLabel(`${vertical.icon} ${vertical.label} · ${contentType.label}`);
    setAppStep("chat");

    // Send a hidden trigger so the backend generates the first greeting dynamically.
    // hiddenUserMessage=true keeps the trigger out of the visible chat.
    // This also ensures the backend knows the vertical+template from turn 1,
    // so it can extract fields from the very first user reply. 
    sendMessage("__start__", ctx, true);
  };

  const handleNewSession = () => {
    clearSession();
    setChatContext(null);
    setContextLabel("");
    setInput("");
    setBriefModalOpen(false);
    setBriefContent("");
    setAppStep("selection");
  };

  const handleSend = () => {
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput("");
    // Reset textarea height after send
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    // Pass context on first send (context is cleared after session is created)
    sendMessage(text, chatContext ?? undefined);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadDocument(file);
    }
    // reset input so same file can be uploaded again if needed
    e.target.value = '';
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleViewBrief = async () => {
    if (!sessionId) return;
    setIsLoadingBrief(true);
    try {
      const res = await fetch(`${API_BASE}/conversation/session/${sessionId}/brief`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setBriefContent(data.brief ?? "No brief content available.");
      setBriefModalOpen(true);
    } catch (err) {
      setBriefContent("Failed to load brief. Please try again.");
      setBriefModalOpen(true);
    } finally {
      setIsLoadingBrief(false);
    }
  };

  const isComplete = snapshot?.is_complete ?? false;

  return (
    <div className="app-layout">
      {/* ── Chat Panel ─────────────────────────────────────────────────────── */}
      <div className="chat-panel">
        {/* Header */}
        <header className="chat-header">
          <div className="header-left">
            <AIGlyph size={28} />
            <div className="header-title-group">
              <span className="header-title">Picasso RAG Chat</span>
              {appStep === "chat" && contextLabel ? (
                <span className="header-subtitle-pill">{contextLabel}</span>
              ) : (
                <span className="header-subtitle-pill">Debug Testbed</span>
              )}
            </div>
          </div>
          <button
            className="new-session-btn"
            onClick={handleNewSession}
            title="Clear session and start a new brief"
            id="btn-new-session"
          >
            <span>+</span> New Session
          </button>
        </header>

        {/* Selection Screen or Chat */}
        {appStep === "selection" ? (
          <SelectionScreen onStart={handleSelectionComplete} />
        ) : (
          <>
            {/* Message Stream */}
            <div className="messages-container">
              <div className="messages-inner">
                {messages.length === 0 && (
                  <div className="welcome-card msg-in">
                    <div className="welcome-glyph">
                      <AIGlyph size={48} />
                    </div>
                    <h2 className="welcome-title">Picasso AI</h2>
                    <p className="welcome-desc">
                      {contextLabel
                        ? `Ready to capture a ${contextLabel} brief. Type your first message below.`
                        : "Start a conversation to capture a creative brief."}
                    </p>
                  </div>
                )}

                {messages.map((msg) => (
                  <MessageItem key={msg.id} message={msg} />
                ))}

                {isStreaming &&
                  messages.length > 0 &&
                  messages[messages.length - 1]?.role !== "assistant" && (
                    <TypingDots />
                  )}

                {error && (
                  <div className="error-banner">
                    <span>⚠️</span>
                    <span>{error}</span>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            </div>

            {/* Completion CTA — shown above input bar when brief is complete */}
            {isComplete && (
              <CompletionBanner
                onViewBrief={handleViewBrief}
                isLoadingBrief={isLoadingBrief}
              />
            )}

            {/* Input Bar */}
            <div className="input-area-container">
              {/* Enum Option Chips */}
              {snapshot && snapshot.missing_fields.length > 0 && snapshot.missing_fields[0].enum_values && !isComplete && (
                <div style={{ padding: '8px 16px', display: 'flex', flexWrap: 'wrap', gap: '8px', background: 'rgba(0,0,0,0.02)', borderRadius: '12px 12px 0 0', border: '1px solid #e0e0e0', borderBottom: 'none', width: '100%', maxWidth: '820px', maxHeight: '120px', overflowY: 'auto' }}>
                  <span style={{ fontSize: '12px', fontWeight: 600, color: '#666', marginRight: '8px', display: 'flex', alignItems: 'center' }}>
                    Options for {snapshot.missing_fields[0].field_code}:
                  </span>
                  {snapshot.missing_fields[0].enum_values.map(val => (
                    <button 
                      key={val} 
                      style={{ padding: '4px 10px', fontSize: '12px', borderRadius: '16px', border: '1px solid #ccc', background: '#fff', cursor: 'pointer' }}
                      onClick={() => setInput((prev) => prev ? prev + ", " + val : val)}
                    >
                      {val}
                    </button>
                  ))}
                </div>
              )}
              <div className="input-bar-pill" style={{ borderRadius: (snapshot && snapshot.missing_fields.length > 0 && snapshot.missing_fields[0].enum_values && !isComplete) ? '0 0 24px 24px' : '24px' }}>
                <AIGlyph size={26} />
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isStreaming}
                  aria-label="Upload document"
                  title="Upload Brief Document"
                  style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '20px', padding: '0 8px', opacity: isStreaming ? 0.5 : 0.8 }}
                >
                  📁
                </button>
                <textarea
                  ref={textareaRef}
                  className="input-bar-textarea no-sb"
                  value={input}
                  onChange={(e) => {
                    setInput(e.target.value);
                    // Auto-resize: reset to auto then set to scrollHeight so it grows with content
                    const el = e.target;
                    el.style.height = "auto";
                    el.style.height = Math.min(el.scrollHeight, 180) + "px";
                  }}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    isComplete
                      ? "Ask a follow-up question or request changes…"
                      : "Type your answer… (Enter to send, Shift+Enter for newline)"
                  }
                  disabled={isStreaming}
                  rows={1}
                  id="chat-input"
                  style={{ overflowY: input.length > 0 && textareaRef.current && textareaRef.current.scrollHeight > 180 ? "auto" : "hidden", resize: "none" }}
                />
                <button
                  type="button"
                  className="send-btn"
                  onClick={handleSend}
                  disabled={isStreaming || !input.trim()}
                  aria-label="Send message"
                  id="btn-send"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M5 12h14" />
                    <path d="m12 5 7 7-7 7" />
                  </svg>
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* ── State Sidebar Inspector ────────────────────────────────────────── */}
      <StatePanel snapshot={snapshot} />

      {/* ── Brief Modal ───────────────────────────────────────────────────── */}
      {briefModalOpen && (
        <BriefModal
          briefMarkdown={briefContent}
          onClose={() => setBriefModalOpen(false)}
        />
      )}
    </div>
  );
}


