// @vitest-environment jsdom

import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import TriviaDeck from '../src/components/TriviaDeck';
import {
  pythonTriviaDeck,
  pytorchTriviaDeck,
  type TriviaDeckData,
} from '../src/lib/trivia-decks';

const deck: TriviaDeckData = {
  id: 'test-deck',
  title: 'Test trivia',
  cards: [
    { id: 'one', topic: 'Semantics', question: 'Question `one`?', answer: 'Answer `one`.' },
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

  it('hides the answer until reveal and renders inline code accessibly', async () => {
    await renderDeck();

    expect(container.textContent).toContain('Question one?');
    expect(container.textContent).not.toContain('Answer one.');
    expect(container.querySelector('[role="heading"] code')?.textContent).toBe('one');
    expect(container.querySelector('textarea')).not.toBeNull();

    enterAnswer('My candidate answer.');

    click('Show answer');

    expect(container.textContent).toContain('Answer one.');
    expect(container.textContent).toContain('My candidate answer.');
    expect(container.querySelector('.trivia-card__answer code')?.textContent).toBe('one');
  });

  it('marks a card correct, persists the answer and score, and advances', async () => {
    await renderDeck();
    enterAnswer('My answer.');
    click('Show answer');
    click('Got it right');

    expect(container.textContent).toContain('Question two?');
    expect(container.textContent).toContain('1/1 right');
    const saved = window.localStorage.getItem('trivia-progress:test-deck');
    expect(saved).toContain('My answer.');
    expect(saved).toContain('"attemptedCardIds":["one"]');
    expect(saved).toContain('"correctCardIds":["one"]');
  });

  it('counts an incorrect self-grade as attempted but not correct', async () => {
    await renderDeck();
    click('Show answer');
    click('Not quite');

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
    click('Show answer');
    click('Got it right');

    const select = container.querySelector('select');
    await act(async () => {
      if (!select) return;
      select.value = 'Runtime';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });

    expect(container.textContent).toContain('0/0 right');
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
