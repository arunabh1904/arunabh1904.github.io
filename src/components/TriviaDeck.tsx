import React, { useEffect, useMemo, useState } from 'react';
import type { KeyboardEvent, ReactNode } from 'react';
import type { TriviaCard, TriviaDeckData } from '../lib/trivia-decks';

interface TriviaDeckProps {
  deck: TriviaDeckData;
}

interface SavedProgress {
  knownCardIds: string[];
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
  const [knownCardIds, setKnownCardIds] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(storageKey);
      if (!saved) return;
      const parsed = JSON.parse(saved) as SavedProgress;
      const validIds = new Set(deck.cards.map((card) => card.id));
      setKnownCardIds(new Set(parsed.knownCardIds.filter((id) => validIds.has(id))));
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
  const knownVisibleCount = visibleCards.filter((item) => knownCardIds.has(item.id)).length;

  function persist(nextKnown: Set<string>) {
    setKnownCardIds(nextKnown);
    try {
      window.localStorage.setItem(
        storageKey,
        JSON.stringify({ knownCardIds: Array.from(nextKnown) } satisfies SavedProgress),
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

  function markKnown(isKnown: boolean) {
    if (!card) return;
    const nextKnown = new Set(knownCardIds);
    if (isKnown) nextKnown.add(card.id);
    else nextKnown.delete(card.id);
    persist(nextKnown);
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
    persist(new Set());
    setPosition(0);
    setRevealed(false);
  }

  if (!card) return null;

  const progressPercent = visibleCards.length === 0
    ? 0
    : Math.round((knownVisibleCount / visibleCards.length) * 100);

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
        <span>{knownVisibleCount} got it</span>
      </div>
      <div
        className="trivia-deck__progress"
        role="progressbar"
        aria-label="Cards marked got it"
        aria-valuemin={0}
        aria-valuemax={visibleCards.length}
        aria-valuenow={knownVisibleCount}
      >
        <span style={{ width: `${progressPercent}%` }} />
      </div>

      <article className={`trivia-card${revealed ? ' trivia-card--revealed' : ''}`} aria-live="polite">
        <p className="trivia-card__topic">{card.topic}</p>
        <p className="trivia-card__question" role="heading" aria-level={2}>
          {inlineCode(card.question)}
        </p>

        {!revealed ? (
          <button
            type="button"
            className="trivia-card__reveal"
            onClick={() => setRevealed(true)}
            aria-expanded="false"
          >
            Show answer
          </button>
        ) : (
          <div className="trivia-card__answer">
            <p>{inlineCode(card.answer)}</p>
            {card.code && <pre><code>{card.code}</code></pre>}
            {card.detail && <p className="trivia-card__detail">{inlineCode(card.detail)}</p>}
          </div>
        )}
      </article>

      <div className="trivia-deck__navigation">
        <button type="button" onClick={() => move(-1)} aria-label="Previous trivia card">
          Previous
        </button>
        {revealed ? (
          <div className="trivia-deck__rating" aria-label="Rate this answer">
            <button type="button" onClick={() => markKnown(false)}>Again</button>
            <button type="button" className="trivia-deck__primary" onClick={() => markKnown(true)}>
              Got it
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
