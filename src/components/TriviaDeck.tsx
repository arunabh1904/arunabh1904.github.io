import React, { useEffect, useMemo, useState } from 'react';
import type { KeyboardEvent, ReactNode } from 'react';
import type { TriviaCard, TriviaDeckData } from '../lib/trivia-decks';

interface TriviaDeckProps {
  deck: TriviaDeckData;
}

interface SavedProgress {
  answers?: Record<string, string>;
  attemptedCardIds?: string[];
  correctCardIds?: string[];
  knownCardIds?: string[];
}

function inlineCode(text: string): ReactNode[] {
  return text.split(/(`[^`]+`)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={`${part}-${index}`}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

function shuffled<T>(items: readonly T[]): T[] {
  const result = [...items];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }
  return result;
}

function clampPosition(position: number, length: number) {
  return Math.max(0, Math.min(position, Math.max(0, length - 1)));
}

export default function TriviaDeck({ deck }: TriviaDeckProps) {
  const storageKey = `trivia-progress:${deck.id}`;
  const topics = useMemo(
    () => ['All topics', ...Array.from(new Set(deck.cards.map((card) => card.topic)))],
    [deck.cards],
  );
  const [topic, setTopic] = useState('All topics');
  const [orderedIds, setOrderedIds] = useState(() => deck.cards.map((card) => card.id));
  const [position, setPosition] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [attemptedCardIds, setAttemptedCardIds] = useState<Set<string>>(() => new Set());
  const [correctCardIds, setCorrectCardIds] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(storageKey);
      if (!saved) return;
      const parsed = JSON.parse(saved) as SavedProgress;
      const validIds = new Set(deck.cards.map((card) => card.id));
      const legacyKnownIds = Array.isArray(parsed.knownCardIds) ? parsed.knownCardIds : [];
      const savedCorrectIds = Array.isArray(parsed.correctCardIds)
        ? parsed.correctCardIds
        : legacyKnownIds;
      const savedAttemptedIds = Array.isArray(parsed.attemptedCardIds)
        ? parsed.attemptedCardIds
        : savedCorrectIds;
      const savedAnswers = parsed.answers && typeof parsed.answers === 'object'
        ? Object.fromEntries(
          Object.entries(parsed.answers).filter(
            ([id, answer]) => validIds.has(id) && typeof answer === 'string',
          ),
        )
        : {};

      setAnswers(savedAnswers);
      setAttemptedCardIds(new Set(savedAttemptedIds.filter((id) => validIds.has(id))));
      setCorrectCardIds(new Set(savedCorrectIds.filter((id) => validIds.has(id))));
    } catch {
      // A corrupt or unavailable local store should never block the study deck.
    }
  }, [deck.cards, storageKey]);

  const cardsById = useMemo(
    () => new Map(deck.cards.map((card) => [card.id, card])),
    [deck.cards],
  );
  const visibleCards = useMemo(
    () => orderedIds
      .map((id) => cardsById.get(id))
      .filter((card): card is TriviaCard => Boolean(card))
      .filter((card) => topic === 'All topics' || card.topic === topic),
    [cardsById, orderedIds, topic],
  );
  const currentPosition = clampPosition(position, visibleCards.length);
  const card = visibleCards[currentPosition];
  const attemptedVisibleCount = visibleCards.filter((item) => attemptedCardIds.has(item.id)).length;
  const correctVisibleCount = visibleCards.filter((item) => correctCardIds.has(item.id)).length;

  function persist(
    nextAnswers: Record<string, string>,
    nextAttempted: Set<string>,
    nextCorrect: Set<string>,
  ) {
    try {
      window.localStorage.setItem(
        storageKey,
        JSON.stringify({
          answers: nextAnswers,
          attemptedCardIds: Array.from(nextAttempted),
          correctCardIds: Array.from(nextCorrect),
        } satisfies SavedProgress),
      );
    } catch {
      // Progress persistence is optional in private or storage-restricted browsing modes.
    }
  }

  function move(delta: number) {
    if (visibleCards.length === 0) return;
    setPosition((current) => {
      const next = current + delta;
      if (next < 0) return visibleCards.length - 1;
      if (next >= visibleCards.length) return 0;
      return next;
    });
    setRevealed(false);
  }

  function updateAnswer(answer: string) {
    if (!card) return;
    const nextAnswers = { ...answers, [card.id]: answer };
    setAnswers(nextAnswers);
    persist(nextAnswers, attemptedCardIds, correctCardIds);
  }

  function markAnswer(isCorrect: boolean) {
    if (!card) return;
    const nextAttempted = new Set(attemptedCardIds).add(card.id);
    const nextCorrect = new Set(correctCardIds);
    if (isCorrect) nextCorrect.add(card.id);
    else nextCorrect.delete(card.id);
    setAttemptedCardIds(nextAttempted);
    setCorrectCardIds(nextCorrect);
    persist(answers, nextAttempted, nextCorrect);
    move(1);
  }

  function handleDeckKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.target !== event.currentTarget) return;
    if (event.key === ' ' || event.key === 'Enter') {
      event.preventDefault();
      setRevealed(true);
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      move(-1);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      move(1);
    }
  }

  function selectTopic(nextTopic: string) {
    setTopic(nextTopic);
    setPosition(0);
    setRevealed(false);
  }

  function shuffleCards() {
    setOrderedIds((ids) => shuffled(ids));
    setPosition(0);
    setRevealed(false);
  }

  function resetProgress() {
    setAnswers({});
    setAttemptedCardIds(new Set());
    setCorrectCardIds(new Set());
    try {
      window.localStorage.removeItem(storageKey);
    } catch {
      // Progress persistence is optional in private or storage-restricted browsing modes.
    }
    setPosition(0);
    setRevealed(false);
  }

  if (!card) return null;

  const progressPercent = visibleCards.length === 0
    ? 0
    : Math.round((correctVisibleCount / visibleCards.length) * 100);
  const currentAnswer = answers[card.id] ?? '';

  return (
    <section
      className="trivia-deck"
      aria-label={deck.title}
      onKeyDown={handleDeckKeyDown}
      tabIndex={0}
    >
      <div className="trivia-deck__toolbar">
        <label>
          <span>Topic</span>
          <select value={topic} onChange={(event) => selectTopic(event.target.value)}>
            {topics.map((item) => <option key={item}>{item}</option>)}
          </select>
        </label>
        <button type="button" className="trivia-deck__quiet-action" onClick={shuffleCards}>
          Shuffle
        </button>
      </div>

      <div className="trivia-deck__status">
        <span>Card {currentPosition + 1} of {visibleCards.length}</span>
        <span>{correctVisibleCount}/{attemptedVisibleCount} right</span>
      </div>
      <div
        className="trivia-deck__progress"
        role="progressbar"
        aria-label="Cards answered correctly"
        aria-valuemin={0}
        aria-valuemax={visibleCards.length}
        aria-valuenow={correctVisibleCount}
      >
        <span style={{ width: `${progressPercent}%` }} />
      </div>

      <article className={`trivia-card${revealed ? ' trivia-card--revealed' : ''}`} aria-live="polite">
        <p className="trivia-card__topic">{card.topic}</p>
        <p className="trivia-card__question" role="heading" aria-level={2}>
          {inlineCode(card.question)}
        </p>

        {!revealed ? (
          <div className="trivia-card__response">
            <label htmlFor={`${deck.id}-${card.id}-answer`}>Your answer</label>
            <textarea
              id={`${deck.id}-${card.id}-answer`}
              value={currentAnswer}
              onChange={(event) => updateAnswer(event.target.value)}
              placeholder="Write or outline your answer here…"
              rows={4}
            />
            <button
              type="button"
              className="trivia-card__reveal"
              onClick={() => setRevealed(true)}
              aria-expanded="false"
            >
              Show answer
            </button>
          </div>
        ) : (
          <div className="trivia-card__comparison">
            <section aria-label="Your answer">
              <h3>Your answer</h3>
              <p className={currentAnswer.trim() ? '' : 'trivia-card__empty-answer'}>
                {currentAnswer.trim() || 'No answer entered.'}
              </p>
            </section>
            <section className="trivia-card__answer" aria-label="Reference answer">
              <h3>Reference answer</h3>
              <p>{inlineCode(card.answer)}</p>
              {card.code && <pre><code>{card.code}</code></pre>}
              {card.detail && <p className="trivia-card__detail">{inlineCode(card.detail)}</p>}
            </section>
          </div>
        )}
      </article>

      <div className="trivia-deck__navigation">
        <button type="button" onClick={() => move(-1)} aria-label="Previous trivia card">
          Previous
        </button>
        {revealed ? (
          <div className="trivia-deck__rating" aria-label="Rate this answer">
            <button type="button" onClick={() => markAnswer(false)}>Not quite</button>
            <button type="button" className="trivia-deck__primary" onClick={() => markAnswer(true)}>
              Got it right
            </button>
          </div>
        ) : (
          <button type="button" className="trivia-deck__primary" onClick={() => move(1)}>
            Next
          </button>
        )}
      </div>

      <div className="trivia-deck__footer">
        <span>Space reveals · arrows move</span>
        <button type="button" onClick={resetProgress}>Reset progress</button>
      </div>
    </section>
  );
}
