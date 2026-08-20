// A brief-spec JSON file describes a fixed set of "must-ask" questions for one
// content type (e.g. a real estate listing sheet). The AI chat interview
// (ai-chat.service.ts) weaves these into its own dynamically-invented
// questions using the same {code, kind, options} shape it already uses for
// self-authored questions — so answers save and persist exactly like any
// other question (see ensureQuestionExists in ai-chat.service.ts).
//
// To add a new brief type: drop a new JSON file matching this shape next to
// this one, then register it in ./index.ts.

export type BriefSpecKind = 'radio' | 'checkbox' | 'color' | 'file' | 'text';

export interface BriefSpecOption {
  label: string;
  value: string;
}

export interface BriefSpecQuestion {
  code: string;
  kind: BriefSpecKind;
  question: string;
  hint?: string;
  options?: BriefSpecOption[];
  required?: boolean; // defaults to true
}

export interface BriefSpecSection {
  name: string;
  questions: BriefSpecQuestion[];
}

export interface BriefSpec {
  key: string;
  label: string;
  // Matched against BriefTemplate.vertical.key (exact match).
  verticalKey?: string;
  // Matched as case-insensitive substrings against BriefTemplate.name.
  templateNameMatch?: string[];
  sections: BriefSpecSection[];
}
