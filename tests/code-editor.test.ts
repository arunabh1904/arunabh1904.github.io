// @vitest-environment jsdom

import { EditorState } from '@codemirror/state';
import { EditorView } from '@codemirror/view';
import { afterEach, describe, expect, it } from 'vitest';
import { codeEditorExtensions, codeEditorKeyBindings } from '../src/lib/code-editor';

function getKeyBinding(key: string) {
  const binding = codeEditorKeyBindings.find((entry) => entry.key === key);
  expect(binding).toBeDefined();
  return binding;
}

describe('codeEditorExtensions', () => {
  const mountedParents: HTMLDivElement[] = [];
  const mountedViews: EditorView[] = [];

  afterEach(() => {
    mountedViews.forEach((view) => view.destroy());
    mountedParents.forEach((parent) => parent.remove());
    mountedViews.length = 0;
    mountedParents.length = 0;
  });

  function createView(doc: string) {
    const parent = document.createElement('div');
    document.body.appendChild(parent);
    mountedParents.push(parent);

    const view = new EditorView({
      parent,
      state: EditorState.create({
        doc,
        selection: { anchor: 0, head: doc.length },
        extensions: codeEditorExtensions,
      }),
    });

    mountedViews.push(view);
    return view;
  }

  it('registers the IDE-style shortcut bindings we rely on', () => {
    expect(codeEditorKeyBindings.map((binding) => binding.key)).toEqual(
      expect.arrayContaining(['Tab', 'Shift-Tab', 'Mod-/']),
    );
  });

  it('indents and outdents the current selection', () => {
    const view = createView('print("starter")');
    const indentBinding = getKeyBinding('Tab');
    const outdentBinding = getKeyBinding('Shift-Tab');

    expect(indentBinding?.run?.(view)).toBe(true);
    expect(view.state.doc.toString()).toBe('    print("starter")');

    view.dispatch({
      selection: { anchor: 0, head: view.state.doc.length },
    });

    expect(outdentBinding?.run?.(view)).toBe(true);
    expect(view.state.doc.toString()).toBe('print("starter")');
  });

  it('toggles python comments across selected lines', () => {
    const view = createView('x = 1\ny = 2');
    const commentBinding = getKeyBinding('Mod-/');

    expect(commentBinding?.run?.(view)).toBe(true);
    expect(view.state.doc.toString()).toBe('# x = 1\n# y = 2');

    view.dispatch({
      selection: { anchor: 0, head: view.state.doc.length },
    });

    expect(commentBinding?.run?.(view)).toBe(true);
    expect(view.state.doc.toString()).toBe('x = 1\ny = 2');
  });
});
