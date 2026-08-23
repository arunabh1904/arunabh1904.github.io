import type { TriviaCard } from './trivia-decks';

export type TriviaGradeStatus = 'correct' | 'close' | 'needs-work';

export interface TriviaGrade {
  status: TriviaGradeStatus;
  label: 'Correct' | 'Close' | 'Needs work';
  explanation: string;
  matchedConcepts: string[];
  missingConcepts: string[];
}

function answerWords(value: string): string[] {
  const literal = value.trim().replaceAll('`', '');
  if (literal === '[]') return ['emptylist'];
  if (literal === '==') return ['doubleequals'];
  if (literal === '/') return ['slash'];
  if (literal === '*') return ['star'];
  return (literal.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter(Boolean);
}

function normalizedAnswer(value: string): string {
  return answerWords(value).join('');
}

function editDistance(left: string, right: string): number {
  const previous = Array.from({ length: right.length + 1 }, (_, index) => index);

  for (let leftIndex = 1; leftIndex <= left.length; leftIndex += 1) {
    const current = [leftIndex];
    for (let rightIndex = 1; rightIndex <= right.length; rightIndex += 1) {
      const substitution = previous[rightIndex - 1]
        + (left[leftIndex - 1] === right[rightIndex - 1] ? 0 : 1);
      current[rightIndex] = Math.min(
        previous[rightIndex] + 1,
        current[rightIndex - 1] + 1,
        substitution,
      );
    }
    previous.splice(0, previous.length, ...current);
  }

  return previous[right.length];
}

export function gradeTriviaAnswer(card: TriviaCard, answer: string): TriviaGrade {
  const candidate = normalizedAnswer(answer);
  const accepted = [card.answer, ...(card.acceptedAnswers ?? [])]
    .map(normalizedAnswer)
    .filter(Boolean);

  if (candidate.length === 0) {
    return {
      status: 'needs-work',
      label: 'Needs work',
      explanation: 'No answer was entered. Recall the short answer, then try this card again.',
      matchedConcepts: [],
      missingConcepts: [card.answer],
    };
  }

  if (accepted.includes(candidate)) {
    return {
      status: 'correct',
      label: 'Correct',
      explanation: 'Exact concept match.',
      matchedConcepts: [card.answer],
      missingConcepts: [],
    };
  }

  const candidateWords = new Set(answerWords(answer));
  const referenceWords = answerWords(card.answer);
  const sharedWords = referenceWords.filter((word) => candidateWords.has(word));
  const nearMiss = accepted.some(
    (expected) => expected.length >= 4 && editDistance(candidate, expected) === 1,
  );

  if (sharedWords.length > 0 || nearMiss) {
    return {
      status: 'close',
      label: 'Close',
      explanation: nearMiss
        ? 'That looks like a one-character typo. Check the reference spelling.'
        : 'You recalled part of the term. Supply the complete short answer.',
      matchedConcepts: sharedWords,
      missingConcepts: [card.answer],
    };
  }

  return {
    status: 'needs-work',
    label: 'Needs work',
    explanation: 'The response does not match the expected short answer.',
    matchedConcepts: [],
    missingConcepts: [card.answer],
  };
}
