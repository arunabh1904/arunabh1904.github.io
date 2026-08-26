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
      detail: 'Names are labels for objects, so identity compares the referenced objects instead of comparing their values.',
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
      .find((candidate) => (
        candidate.textContent === label || candidate.getAttribute('aria-label') === label
      ));
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
    expect(container.textContent).toContain('Why it works');
    expect(container.textContent).toContain('Mental model');
    expect(container.textContent).toContain('Names are labels for objects');
    expect(container.querySelector('[aria-label="Automatic grade"]')).not.toBeNull();
  });

  it('uses one consistent treatment for every deck action', async () => {
    await renderDeck();

    const buttons = Array.from(container.querySelectorAll('button'));
    expect(buttons).toHaveLength(5);
    expect(buttons.every((button) => button.classList.contains('trivia-deck__action')))
      .toBe(true);
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
    click('Next trivia card');

    expect(container.querySelector('[aria-label="Code scenario"]')?.textContent)
      .toContain('service.run()');
    expect(container.textContent).not.toContain('The runtime executes the program.');

    enterAnswer('Runtime');
    click('Grade answer');

    expect(container.textContent).toContain('The runtime executes the program.');
    expect(container.querySelectorAll('[aria-label="Code scenario"]')).toHaveLength(1);
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
    click('Next trivia card');

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
      (candidate) => candidate.id === 'python-shared-mutable-default',
    );
    expect(card).toBeDefined();

    const grade = gradeTriviaAnswer(card!, 'default evaluated once');

    expect(grade.status).toBe('correct');
  });

  it('marks a partial two-word term close', () => {
    const card = pythonTriviaDeck.cards.find(
      (candidate) => candidate.id === 'python-shared-mutable-default',
    );
    expect(card).toBeDefined();

    const grade = gradeTriviaAnswer(card!, 'mutable');

    expect(grade.status).toBe('close');
    expect(grade.missingConcepts).toContain('Shared mutable default');
  });

  it('marks a one-character typo close', () => {
    const card = pytorchTriviaDeck.cards.find(
      (candidate) => candidate.id === 'torch-contiguous-copy',
    );
    expect(card).toBeDefined();

    const grade = gradeTriviaAnswer(card!, 'contigous');

    expect(grade.status).toBe('close');
  });

  it('grades an empty-list literal as a real answer', () => {
    const card = pythonTriviaDeck.cards.find(
      (candidate) => candidate.id === 'python-generator-exhaustion',
    );
    expect(card).toBeDefined();
    expect(gradeTriviaAnswer(card!, '[]').status).toBe('correct');
  });

  it('rejects an unrelated short answer', () => {
    const card = pythonTriviaDeck.cards.find(
      (candidate) => candidate.id === 'python-shared-mutable-default',
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
      expect(publishedDeck.cards.every((card) => !card.answer.includes('\n'))).toBe(true);
      expect(publishedDeck.cards.every((card) => Boolean(card.explanation))).toBe(true);
      const shallowMentalModels = publishedDeck.cards
        .filter((card) => (card.detail?.trim().split(/\s+/).length ?? 0) < 20)
        .map((card) => card.id);
      expect(shallowMentalModels).toEqual([]);
    },
  );

  it('publishes the comprehensive practical Python core', () => {
    expect(pythonTriviaDeck).toMatchObject({
      id: 'python-interview-trivia-v4',
      title: 'Practical Python for ML engineering',
    });
    expect(pythonTriviaDeck.cards).toHaveLength(230);
    expect(new Set(pythonTriviaDeck.cards.map((card) => card.topic))).toEqual(new Set([
      'References & values',
      'Collections',
      'Functions & iteration',
      'Classes & interfaces',
      'Typing',
      'Dataclasses',
      'Pydantic v2',
      'Reliability & I/O',
      'Syntax & control flow',
      'Collections & algorithms',
      'Object model & OOP',
      'Concurrency & parallelism',
      'Performance & optimization',
      'Runtime, imports & testing',
    ]));
  });

  it('covers the requested Python revision domains with substantial topic depth', () => {
    const minimumCardsByTopic: Record<string, number> = {
      'References & values': 10,
      Collections: 10,
      'Functions & iteration': 20,
      'Classes & interfaces': 10,
      Typing: 20,
      Dataclasses: 8,
      'Pydantic v2': 30,
      'Reliability & I/O': 20,
      'Syntax & control flow': 12,
      'Collections & algorithms': 14,
      'Object model & OOP': 15,
      'Concurrency & parallelism': 20,
      'Performance & optimization': 10,
      'Runtime, imports & testing': 10,
    };

    for (const [topic, minimum] of Object.entries(minimumCardsByTopic)) {
      expect(
        pythonTriviaDeck.cards.filter((card) => card.topic === topic).length,
        `${topic} coverage`,
      ).toBeGreaterThanOrEqual(minimum);
    }
  });

  it('keeps every expanded Python card concrete and explanatory', () => {
    const expandedCards = pythonTriviaDeck.cards.slice(88);
    expect(expandedCards.length).toBeGreaterThanOrEqual(110);
    expect(expandedCards.every((card) => Boolean(card.code) || card.question.includes('`'))).toBe(true);
    expect(expandedCards.every((card) => (card.explanation?.split(/\s+/).length ?? 0) >= 8)).toBe(true);
    expect(expandedCards.every((card) => (card.detail?.split(/\s+/).length ?? 0) >= 20)).toBe(true);
  });

  it('keeps code and concept on the same practical Python card', () => {
    const codeCards = pythonTriviaDeck.cards.filter((card) => card.code);
    expect(codeCards.length).toBeGreaterThanOrEqual(60);
    expect(codeCards.every((card) => !card.id.endsWith('-code'))).toBe(true);
    expect(pythonTriviaDeck.cards.every((card) => card.topic !== 'Code scenarios')).toBe(true);
  });

  it('grounds most Python prompts in code or a named API', () => {
    const concreteCards = pythonTriviaDeck.cards.filter(
      (card) => Boolean(card.code) || card.question.includes('`'),
    );
    expect(concreteCards.length / pythonTriviaDeck.cards.length).toBeGreaterThanOrEqual(0.65);
  });

  it('does not describe Pydantic coercion as strict-by-default validation', () => {
    const comparison = pythonTriviaDeck.cards.find(
      (card) => card.id === 'python-pydantic-default-coercion',
    );
    expect(comparison).toMatchObject({
      answer: '`10` as an `int`',
      explanation: expect.stringContaining('coercive validation by default'),
    });
  });

  it('distinguishes Pydantic arbitrary-type checks from tensor validation', () => {
    const tensorCard = pythonTriviaDeck.cards.find(
      (card) => card.id === 'python-pydantic-arbitrary-tensor',
    );
    expect(tensorCard).toMatchObject({
      answer: 'No',
      explanation: expect.stringContaining('not tensor shape, dtype, values, layout, or device'),
    });
  });

  it('publishes every prompt as a complete question', () => {
    const completeQuestion = /^(What|Which|When|Where|How|Why|Does|Do|Is|Are|Can|Will|Should)\b.*\?$/;
    const incompleteQuestions = [...pythonTriviaDeck.cards, ...pytorchTriviaDeck.cards]
      .filter((card) => !completeQuestion.test(card.question) || card.question.includes('…'))
      .map((card) => `${card.id}: ${card.question}`);
    expect(incompleteQuestions).toEqual([]);
  });

  it('keeps falsy values distinct from missing configuration', () => {
    const missingValueCard = pythonTriviaDeck.cards.find(
      (card) => card.id === 'python-falsy-valid-value',
    );
    expect(missingValueCard).toMatchObject({
      answer: 'Explicit `None` check',
      explanation: expect.stringContaining('valid value `0.0`'),
    });
  });

  it('keeps the dangerous serialization boundary explicit', () => {
    const pickleCard = pythonTriviaDeck.cards.find(
      (card) => card.id === 'python-untrusted-pickle',
    );
    expect(pickleCard?.answer).toBe('No');
    expect(pickleCard?.explanation).toContain('execute attacker-controlled code');
  });

  it('leaves PyTorch mechanics to the separate deck', () => {
    const pythonText = pythonTriviaDeck.cards
      .map((card) => `${card.question} ${card.explanation}`)
      .join(' ');
    expect(pythonText).not.toMatch(/\bautograd\b|\bDataLoader\b|distributed training/i);
  });
});
