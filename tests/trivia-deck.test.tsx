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
      answer: 'Object identity',
      acceptedAnswers: ['identity'],
      explanation: 'Identity asks whether two names refer to the same object.',
    },
    {
      id: 'two',
      topic: 'Runtime',
      question: 'Question two?',
      answer: 'Runtime',
      explanation: 'The runtime executes the program.',
      code: 'result = service.run()',
    },
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
    const input = container.querySelector('input');
    expect(input).not.toBeNull();
    act(() => {
      const valueSetter = Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        'value',
      )?.set;
      valueSetter?.call(input, answer);
      input?.dispatchEvent(new Event('input', { bubbles: true }));
    });
  }

  it('hides the answer until grading and renders the automatic result accessibly', async () => {
    await renderDeck();

    expect(container.textContent).toContain('Question one?');
    expect(container.textContent).not.toContain('same object');
    expect(container.querySelector('[role="heading"] code')?.textContent).toBe('one');
    expect(container.querySelector('input')).not.toBeNull();

    enterAnswer('My candidate answer.');

    click('Grade answer');

    expect(container.textContent).toContain('same object');
    expect(container.textContent).toContain('My candidate answer.');
    expect(container.querySelector('[aria-label="Automatic grade"]')).not.toBeNull();
  });

  it('marks a passing answer correct and persists the automatic score', async () => {
    await renderDeck();
    enterAnswer('identity');
    click('Grade answer');

    expect(container.textContent).toContain('Correct');
    expect(container.textContent).toContain('1/1 right');
    const saved = window.localStorage.getItem('trivia-progress:test-deck');
    expect(saved).toContain('identity');
    expect(saved).toContain('"attemptedCardIds":["one"]');
    expect(saved).toContain('"correctCardIds":["one"]');
  });

  it('shows a production snippet on the front of its own card', async () => {
    await renderDeck();
    click('Next');

    expect(container.querySelector('[aria-label="Production code scenario"]')?.textContent)
      .toContain('service.run()');
    expect(container.textContent).not.toContain('The runtime executes the program.');

    enterAnswer('Runtime');
    click('Grade answer');

    expect(container.textContent).toContain('The runtime executes the program.');
    expect(container.querySelectorAll('[aria-label="Production code scenario"]')).toHaveLength(1);
  });

  it('rejects a vague answer and identifies concepts to review', async () => {
    await renderDeck();
    enterAnswer('reference');
    click('Grade answer');

    expect(container.textContent).toContain('Needs work');
    expect(container.textContent).toContain('Review:');
    expect(container.textContent).toContain('Object identity');
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
    enterAnswer('Object identity');
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

  it('accepts configured short-answer aliases', () => {
    const card = pythonTriviaDeck.cards.find(
      (candidate) => candidate.id === 'python-argument-model',
    );
    expect(card).toBeDefined();

    const grade = gradeTriviaAnswer(card!, 'pass-by-assignment');

    expect(grade.status).toBe('correct');
  });

  it('marks a partial two-word term close', () => {
    const card = pythonTriviaDeck.cards.find(
      (candidate) => candidate.id === 'python-argument-model',
    );
    expect(card).toBeDefined();

    const grade = gradeTriviaAnswer(card!, 'object');

    expect(grade.status).toBe('close');
    expect(grade.missingConcepts).toContain('Object sharing');
  });

  it('marks a one-character typo close', () => {
    const card = pytorchTriviaDeck.cards.find(
      (candidate) => candidate.id === 'torch-contiguous-copy',
    );
    expect(card).toBeDefined();

    const grade = gradeTriviaAnswer(card!, 'contigous');

    expect(grade.status).toBe('close');
  });

  it('rejects an unrelated short answer', () => {
    const card = pythonTriviaDeck.cards.find(
      (candidate) => candidate.id === 'python-argument-model',
    );
    expect(card).toBeDefined();

    expect(gradeTriviaAnswer(card!, 'reference').status).toBe('needs-work');
  });
});

describe('published trivia data', () => {
  it.each([pythonTriviaDeck, pytorchTriviaDeck])(
    'keeps stable unique card ids in $title',
    (publishedDeck) => {
      const ids = publishedDeck.cards.map((card) => card.id);
      expect(new Set(ids).size).toBe(ids.length);
      expect(publishedDeck.cards.every((card) => card.question && card.answer && card.topic)).toBe(true);
      expect(publishedDeck.cards.every((card) => card.answer.trim().split(/\s+/).length <= 2))
        .toBe(true);
      expect(publishedDeck.cards.every((card) => Boolean(card.explanation))).toBe(true);
    },
  );

  it.each([pythonTriviaDeck, pytorchTriviaDeck])(
    'publishes production snippets only as standalone code cards in $title',
    (publishedDeck) => {
      const codeCards = publishedDeck.cards.filter((card) => card.code);
      expect(codeCards.length).toBeGreaterThanOrEqual(15);
      expect(codeCards.every((card) => card.id.endsWith('-code'))).toBe(true);
      expect(codeCards.every((card) => card.topic === 'Code scenarios')).toBe(true);
    },
  );

  it('does not describe Pydantic coercion as strict-by-default validation', () => {
    const comparison = pythonTriviaDeck.cards.find(
      (card) => card.id === 'python-pydantic-validation',
    );
    expect(comparison?.explanation).toContain('Strict rejection is configurable—not the default.');
  });
});
