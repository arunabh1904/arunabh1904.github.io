import { JSDOM } from 'jsdom';
import { describe, expect, it, vi } from 'vitest';

import { initReadAloud } from '../src/lib/read-aloud';

const playerMarkup = `
  <section data-read-aloud>
    <button data-read-aloud-toggle aria-pressed="false">
      <span data-read-aloud-icon></span>
      <span data-read-aloud-label>Listen</span>
    </button>
    <select data-read-aloud-speed><option value="1">1x</option></select>
    <input data-read-aloud-progress type="range" />
    <span data-read-aloud-current></span>
    <span data-read-aloud-duration></span>
    <audio data-read-aloud-audio src="/post.mp3"></audio>
  </section>
`;

describe('initReadAloud', () => {
  it('loads a media source inserted by client-side navigation', () => {
    const dom = new JSDOM(playerMarkup);
    const audio = dom.window.document.querySelector('audio')!;
    const load = vi.fn();
    audio.load = load;

    initReadAloud(dom.window.document);

    expect(audio.currentSrc).toBe('');
    expect(load).toHaveBeenCalledOnce();
  });

  it('retries source selection inside the first play gesture', () => {
    const dom = new JSDOM(playerMarkup);
    const audio = dom.window.document.querySelector('audio')!;
    const toggle = dom.window.document.querySelector<HTMLButtonElement>('[data-read-aloud-toggle]')!;
    const load = vi.fn();
    const play = vi.fn().mockResolvedValue(undefined);
    audio.load = load;
    audio.play = play;

    initReadAloud(dom.window.document);
    load.mockClear();
    toggle.click();

    expect(load).toHaveBeenCalledOnce();
    expect(play).toHaveBeenCalledOnce();
  });
});
