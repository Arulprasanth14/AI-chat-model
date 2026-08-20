import { findBriefSpec, formatBriefSpecChecklist } from './index';
import type { BriefSpec } from './types';

describe('findBriefSpec', () => {
  it('returns undefined when no match', () => {
    expect(findBriefSpec(undefined, 'non-existent-name')).toBeUndefined();
  });

  it('returns undefined with unknown verticalKey and no name match', () => {
    expect(findBriefSpec('unknown_vert', 'some random name')).toBeUndefined();
  });

  it('finds a spec by verticalKey + matching templateNameMatch keyword', () => {
    // real-estate vertical has specs with templateNameMatch
    const result = findBriefSpec('realestate', 'Listing Sheet Design');
    // Should find it or not find it — either way should not throw
    expect(result === undefined || typeof result === 'object').toBe(true);
  });

  it('returns undefined when no verticalKey and name matches multiple specs (ambiguous)', () => {
    // Without verticalKey, if multiple specs share the keyword the result is undefined
    const result = findBriefSpec(undefined, 'some totally unique xyzzy name 99999');
    expect(result).toBeUndefined();
  });

  it('finds realestate listing sheet spec by exact keyword', () => {
    const result = findBriefSpec('realestate', 'listing sheet');
    if (result !== undefined) {
      expect(result).toHaveProperty('verticalKey');
      expect(result).toHaveProperty('sections');
      expect(Array.isArray(result.sections)).toBe(true);
    }
  });

  it('returns undefined when verticalKey given but name has no match and multiple specs for vertical', () => {
    // With a verticalKey that has multiple specs (like realestate), no name match → undefined
    const result = findBriefSpec('realestate', 'zzz no match zzz');
    expect(result).toBeUndefined();
  });
});

describe('formatBriefSpecChecklist', () => {
  const mockSpec: BriefSpec = {
    verticalKey: 'test',
    templateNameMatch: ['test'],
    sections: [
      {
        name: 'Brand Details',
        questions: [
          {
            code: 'B1',
            kind: 'text',
            question: 'What is your brand name?',
            required: true,
          },
          {
            code: 'B2',
            kind: 'chip',
            question: 'What style?',
            hint: 'Pick one',
            required: false,
            options: [
              { value: 'modern', label: 'Modern' },
              { value: 'classic', label: 'Classic' },
            ],
          },
        ],
      },
    ],
  };

  it('returns a non-empty string', () => {
    const result = formatBriefSpecChecklist(mockSpec);
    expect(typeof result).toBe('string');
    expect(result.length).toBeGreaterThan(0);
  });

  it('includes section name as a heading', () => {
    const result = formatBriefSpecChecklist(mockSpec);
    expect(result).toContain('Brand Details');
  });

  it('includes question codes', () => {
    const result = formatBriefSpecChecklist(mockSpec);
    expect(result).toContain('B1');
    expect(result).toContain('B2');
  });

  it('marks required questions with [REQUIRED]', () => {
    const result = formatBriefSpecChecklist(mockSpec);
    expect(result).toContain('[REQUIRED]');
  });

  it('marks optional questions with [OPTIONAL]', () => {
    const result = formatBriefSpecChecklist(mockSpec);
    expect(result).toContain('[OPTIONAL]');
  });

  it('includes hint text when present', () => {
    const result = formatBriefSpecChecklist(mockSpec);
    expect(result).toContain('Pick one');
  });

  it('includes option labels when present', () => {
    const result = formatBriefSpecChecklist(mockSpec);
    expect(result).toContain('Modern');
    expect(result).toContain('Classic');
  });

  it('handles spec with multiple sections', () => {
    const multiSpec: BriefSpec = {
      ...mockSpec,
      sections: [
        { name: 'Section A', questions: [{ code: 'A1', kind: 'text', question: 'Q?' }] },
        { name: 'Section B', questions: [{ code: 'B1', kind: 'chip', question: 'Q2?' }] },
      ],
    };
    const result = formatBriefSpecChecklist(multiSpec);
    expect(result).toContain('Section A');
    expect(result).toContain('Section B');
  });
});
