import type { TriviaCard, TriviaGradingConcept } from './trivia-decks';

export type TriviaGradeStatus = 'correct' | 'close' | 'needs-work';

export interface TriviaGrade {
  status: TriviaGradeStatus;
  label: 'Correct' | 'Close' | 'Needs work';
  explanation: string;
  matchedConcepts: string[];
  missingConcepts: string[];
}

const STOP_WORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'because', 'been', 'being', 'both',
  'by', 'can', 'called', 'does', 'each', 'for', 'from', 'has', 'have', 'if', 'in',
  'into', 'is', 'it', 'its', 'may', 'no', 'normally', 'not', 'of', 'on', 'only', 'or',
  'so', 'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore',
  'they', 'this', 'through', 'to', 'use', 'used', 'uses', 'using', 'when', 'where',
  'whether', 'which', 'while', 'with', 'without', 'would', 'asks', 'another', 'same',
  'function', 'values', 'value', 'name', 'names', 'make', 'makes', 'made', 'returns',
  'return', 'receives', 'receive', 'produces', 'produce', 'lets', 'let', 'means', 'yes',
]);

const PHRASE_ALIASES: Array<[RegExp, string]> = [
  [/pass(?:ed)? by assignment/g, 'passbyassignment'],
  [/object sharing/g, 'passbyassignment'],
  [/exact(?:ly)? the same object/g, 'identity'],
  [/exact(?:ly)? same object/g, 'identity'],
  [/same object/g, 'identity'],
  [/compare(?:s|d)? equal/g, 'equality'],
  [/in[ -]place/g, 'inplace'],
  [/log[ -]softmax/g, 'logsoftmax'],
  [/negative log[ -]likelihood/g, 'nll'],
  [/state[ _-]dict/g, 'statedict'],
  [/requires[ _-]grad/g, 'requiresgrad'],
  [/no[ _-]grad/g, 'nograd'],
];

const WORD_ALIASES: Record<string, string> = {
  assigned: 'assignment',
  assigns: 'assignment',
  bind: 'binding',
  binds: 'binding',
  bound: 'binding',
  changed: 'mutation',
  changes: 'mutation',
  changing: 'mutation',
  equal: 'equality',
  equals: 'equality',
  gradients: 'gradient',
  grads: 'gradient',
  identical: 'identity',
  indices: 'index',
  logits: 'logit',
  modifies: 'mutation',
  modified: 'mutation',
  modifying: 'mutation',
  mutate: 'mutation',
  mutated: 'mutation',
  mutates: 'mutation',
  mutating: 'mutation',
  probabilities: 'probability',
  references: 'reference',
  reassign: 'rebinding',
  reassigned: 'rebinding',
  reassigning: 'rebinding',
  reassignment: 'rebinding',
  rebind: 'rebinding',
  tensors: 'tensor',
};

function canonicalToken(token: string): string {
  if (WORD_ALIASES[token]) return WORD_ALIASES[token];
  if (token.length > 5 && token.endsWith('ies')) return `${token.slice(0, -3)}y`;
  if (token.length > 5 && token.endsWith('ing')) return token.slice(0, -3);
  if (token.length > 4 && token.endsWith('ed')) return token.slice(0, -2);
  if (token.length > 4 && token.endsWith('s')) return token.slice(0, -1);
  return token;
}

function tokenize(text: string): string[] {
  let normalized = text.toLowerCase().replaceAll('`', '');
  for (const [pattern, replacement] of PHRASE_ALIASES) {
    normalized = normalized.replace(pattern, replacement);
  }

  return (normalized.match(/[a-z0-9_+.-]+/g) ?? [])
    .map((token) => token.replace(/^[.+-]+|[.+-]+$/g, ''))
    .map(canonicalToken)
    .filter((token) => token.length > 1 && !STOP_WORDS.has(token));
}

function matchesCuratedConcept(answerTokens: Set<string>, concept: TriviaGradingConcept): boolean {
  return concept.anyOf.some((pattern) => {
    const patternTokens = tokenize(pattern);
    return patternTokens.length > 0 && patternTokens.every((token) => answerTokens.has(token));
  });
}

function conceptLabel(clause: string): string {
  const clean = clause.replaceAll('`', '').replace(/\s+/g, ' ').trim();
  if (clean.length <= 100) return clean;
  return `${clean.slice(0, 97).trimEnd()}…`;
}

function derivedConcepts(referenceAnswer: string): Array<{ label: string; terms: Set<string> }> {
  const clauses = referenceAnswer
    .split(/(?:\.(?:\s+|$)|[;:]|\bbut\b|\bwhereas\b|\bhowever\b)/i)
    .map((clause) => clause.trim())
    .filter(Boolean)
    .map((clause) => ({ label: conceptLabel(clause), terms: new Set(tokenize(clause)) }))
    .filter((concept) => concept.terms.size > 0);

  return clauses.length > 0 ? clauses : [{
    label: conceptLabel(referenceAnswer),
    terms: new Set(tokenize(referenceAnswer)),
  }];
}

function gradeFromCounts(
  answerTokenCount: number,
  matchedConcepts: string[],
  missingConcepts: string[],
  singleConceptTermMatches = 0,
): TriviaGradeStatus {
  const conceptCount = matchedConcepts.length + missingConcepts.length;
  const coverage = conceptCount === 0 ? 0 : matchedConcepts.length / conceptCount;
  const minimumMatched = conceptCount === 1 ? 1 : 2;
  const singleConceptPasses = conceptCount !== 1 || singleConceptTermMatches >= 2;

  if (
    answerTokenCount >= 3
    && matchedConcepts.length >= minimumMatched
    && coverage >= 0.6
    && singleConceptPasses
  ) {
    return 'correct';
  }
  if (matchedConcepts.length > 0 || singleConceptTermMatches > 0) return 'close';
  return 'needs-work';
}

export function gradeTriviaAnswer(card: TriviaCard, answer: string): TriviaGrade {
  const answerTokens = new Set(tokenize(answer));
  const matchedConcepts: string[] = [];
  const missingConcepts: string[] = [];
  let singleConceptTermMatches = 0;

  if (card.grading) {
    for (const concept of card.grading.concepts) {
      if (matchesCuratedConcept(answerTokens, concept)) matchedConcepts.push(concept.label);
      else missingConcepts.push(concept.label);
    }
  } else {
    const concepts = derivedConcepts(card.answer);
    for (const concept of concepts) {
      const termMatches = Array.from(concept.terms)
        .filter((term) => answerTokens.has(term)).length;
      if (termMatches > 0) matchedConcepts.push(concept.label);
      else missingConcepts.push(concept.label);
      if (concepts.length === 1) singleConceptTermMatches = termMatches;
    }
  }

  const status = gradeFromCounts(
    answerTokens.size,
    matchedConcepts,
    missingConcepts,
    singleConceptTermMatches,
  );

  if (status === 'correct') {
    return {
      status,
      label: 'Correct',
      explanation: 'Your answer covers the main ideas in the reference answer.',
      matchedConcepts,
      missingConcepts,
    };
  }
  if (status === 'close') {
    return {
      status,
      label: 'Close',
      explanation: 'You have part of the answer. Add the missing idea before treating this card as learned.',
      matchedConcepts,
      missingConcepts,
    };
  }
  return {
    status,
    label: 'Needs work',
    explanation: answerTokens.size === 0
      ? 'No answer was entered. Use the reference answer to learn the required ideas.'
      : 'Your answer does not yet explain a required idea from the reference answer.',
    matchedConcepts,
    missingConcepts,
  };
}
