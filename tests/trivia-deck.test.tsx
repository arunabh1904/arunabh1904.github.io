// @vitest-environment jsdom

import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import TriviaDeck from '../src/components/TriviaDeck';
import { gradeTriviaAnswer } from '../src/lib/trivia-grader';
import {
  pythonTriviaDeck,
  pytorchTriviaDeck,
  type TriviaDeckData,
} from '../src/lib/trivia-decks';

const deck: TriviaDeckData = {
  id: 'test-deck',
  title: 'Test trivia',
  cards: [
    {
      id: 'one',
      topic: 'Semantics',
      question: 'Question `one`?',
      answer: 'A complete answer covers object identity and value equality.',
      grading: {
        concepts: [
          { label: 'object identity', anyOf: ['object identity'] },
          { label: 'value equality', anyOf: ['value equality'] },
        ],
      },
    },
    { id: 'two', topic: 'Runtime', question: 'Question two?', answer: 'Answer two.' },
  ],
};

describe('TriviaDeck', () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  async function renderDeck() {
    await act(async () => root.render(<TriviaDeck deck={deck} />));
  }

  function click(label: string) {
    const button = Array.from(container.querySelectorAll('button'))
      .find((candidate) => candidate.textContent === label);
    expect(button).not.toBeUndefined();
    act(() => button?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  }

  function enterAnswer(answer: string) {
    const textarea = container.querySelector('textarea');
    expect(textarea).not.toBeNull();
    act(() => {
      const valueSetter = Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype,
        'value',
      )?.set;
      valueSetter?.call(textarea, answer);
      textarea?.dispatchEvent(new Event('input', { bubbles: true }));
    });
  }

  it('hides the answer until grading and renders the automatic result accessibly', async () => {
    await renderDeck();

    expect(container.textContent).toContain('Question one?');
    expect(container.textContent).not.toContain('complete answer');
    expect(container.querySelector('[role="heading"] code')?.textContent).toBe('one');
    expect(container.querySelector('textarea')).not.toBeNull();

    enterAnswer('My candidate answer.');

    click('Grade answer');

    expect(container.textContent).toContain('complete answer');
    expect(container.textContent).toContain('My candidate answer.');
    expect(container.querySelector('[aria-label="Automatic grade"]')).not.toBeNull();
  });

  it('marks a passing answer correct and persists the automatic score', async () => {
    await renderDeck();
    enterAnswer('Object identity is distinct from value equality.');
    click('Grade answer');

    expect(container.textContent).toContain('Correct');
    expect(container.textContent).toContain('1/1 right');
    const saved = window.localStorage.getItem('trivia-progress:test-deck');
    expect(saved).toContain('Object identity');
    expect(saved).toContain('"attemptedCardIds":["one"]');
    expect(saved).toContain('"correctCardIds":["one"]');
  });

  it('rejects a vague answer and identifies concepts to review', async () => {
    await renderDeck();
    enterAnswer('reference');
    click('Grade answer');

    expect(container.textContent).toContain('Needs work');
    expect(container.textContent).toContain('Review:');
    expect(container.textContent).toContain('object identity');
    expect(container.textContent).toContain('0/1 right');
  });

  it('migrates the previous got-it progress into the new score', async () => {
    window.localStorage.setItem(
      'trivia-progress:test-deck',
      JSON.stringify({ knownCardIds: ['one'] }),
    );

    await renderDeck();

    expect(container.textContent).toContain('1/1 right');
  });

  it('filters by topic and resets the visible position', async () => {
    await renderDeck();
    click('Next');

    const select = container.querySelector('select');
    await act(async () => {
      if (!select) return;
      select.value = 'Semantics';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });

    expect(container.textContent).toContain('Card 1 of 1');
    expect(container.textContent).toContain('Question one?');
  });

  it('scopes the score to the selected topic', async () => {
    await renderDeck();
    enterAnswer('Object identity differs from value equality.');
    click('Grade answer');

    const select = container.querySelector('select');
    await act(async () => {
      if (!select) return;
      select.value = 'Runtime';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });

    expect(container.textContent).toContain('0/0 right');
  });
});

describe('automatic trivia grading', () => {
  it.each([...pythonTriviaDeck.cards, ...pytorchTriviaDeck.cards])(
    'accepts the full reference answer for $id',
    (card) => {
      expect(gradeTriviaAnswer(card, card.answer).status).toBe('correct');
    },
  );

  it.each([...pythonTriviaDeck.cards, ...pytorchTriviaDeck.cards])(
    'never passes a one-word response for $id',
    (card) => {
      expect(gradeTriviaAnswer(card, 'reference').status).not.toBe('correct');
    },
  );

  it('rejects the one-word answer from the reported Python example', () => {
    const card = pythonTriviaDeck.cards.find(
      (candidate) => candidate.id === 'python-pass-by-assignment',
    );
    expect(card).toBeDefined();

    const grade = gradeTriviaAnswer(card!, 'reference');

    expect(grade.status).toBe('needs-work');
    expect(grade.missingConcepts).toContain('pass-by-assignment or object sharing');
    expect(grade.missingConcepts).toContain('rebinding stays local');
  });

  it('accepts a strong paraphrase of pass-by-assignment', () => {
    const card = pythonTriviaDeck.cards.find(
      (candidate) => candidate.id === 'python-pass-by-assignment',
    );
    expect(card).toBeDefined();

    const grade = gradeTriviaAnswer(
      card!,
      'Python passes a reference to the same object. Mutation is visible outside, but reassignment stays local.',
    );

    expect(grade.status).toBe('correct');
  });

  it('marks a partial explanation close and names the missing distinction', () => {
    const card = pythonTriviaDeck.cards.find(
      (candidate) => candidate.id === 'python-pass-by-assignment',
    );
    expect(card).toBeDefined();

    const grade = gradeTriviaAnswer(card!, 'Python uses object sharing with the same object.');

    expect(grade.status).toBe('close');
    expect(grade.missingConcepts).toContain('mutation can affect the caller');
    expect(grade.missingConcepts).toContain('rebinding stays local');
  });

  it('accepts concise Python identity and equality terminology', () => {
    const card = pythonTriviaDeck.cards.find(
      (candidate) => candidate.id === 'python-is-equality',
    );
    expect(card).toBeDefined();

    expect(gradeTriviaAnswer(card!, '`is` checks identity; `==` checks equality.').status)
      .toBe('correct');
  });

  it('accepts the core PyTorch cross-entropy contract', () => {
    const card = pytorchTriviaDeck.cards.find(
      (candidate) => candidate.id === 'torch-cross-entropy-logits',
    );
    expect(card).toBeDefined();

    const grade = gradeTriviaAnswer(
      card!,
      'Pass logits, not softmax probabilities. The loss fuses log-softmax with NLL for numerical stability.',
    );

    expect(grade.status).toBe('correct');
  });
});

describe('published trivia data', () => {
  it.each([pythonTriviaDeck, pytorchTriviaDeck])(
    'keeps stable unique card ids in $title',
    (publishedDeck) => {
      const ids = publishedDeck.cards.map((card) => card.id);
      expect(new Set(ids).size).toBe(ids.length);
      expect(publishedDeck.cards.every((card) => card.question && card.answer && card.topic)).toBe(true);
    },
  );

  it('does not describe Pydantic coercion as strict-by-default validation', () => {
    const comparison = pythonTriviaDeck.cards.find(
      (card) => card.id === 'python-pydantic-dataclass',
    );
    expect(comparison?.answer).toContain('Strict rejection is configurable—not the default.');
  });
});
