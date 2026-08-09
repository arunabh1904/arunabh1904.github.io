import {
  defaultKeymap,
  history,
  historyKeymap,
  indentLess,
  indentWithTab,
  toggleComment,
} from '@codemirror/commands';
import { python } from '@codemirror/lang-python';
import {
  bracketMatching,
  defaultHighlightStyle,
  indentOnInput,
  indentUnit,
  syntaxHighlighting,
} from '@codemirror/language';
import { lintKeymap } from '@codemirror/lint';
import { Prec, EditorState } from '@codemirror/state';
import { highlightSelectionMatches, searchKeymap } from '@codemirror/search';
import {
  crosshairCursor,
  drawSelection,
  dropCursor,
  EditorView,
  highlightActiveLine,
  highlightActiveLineGutter,
  highlightSpecialChars,
  keymap,
  lineNumbers,
  rectangularSelection,
  type KeyBinding,
} from '@codemirror/view';

export const CODE_EDITOR_INDENT = '    ';

export const codeEditorKeyBindings: readonly KeyBinding[] = [
  { key: 'Mod-/', run: toggleComment },
  { key: 'Shift-Tab', run: indentLess },
  indentWithTab,
];

/**
 * Adds the workspace run shortcut without changing the editor's shared
 * indentation and comment behavior. Keep both bindings: `Mod` gives Windows
 * and Linux users Ctrl+Enter (and macOS users Cmd+Enter), while the explicit
 * Ctrl binding keeps Ctrl+Enter available on macOS too.
 */
export function createRunCodeKeyBindings(runCode: () => void): readonly KeyBinding[] {
  const run = () => {
    runCode();
    return true;
  };

  return [
    { key: 'Ctrl-Enter', run },
    { key: 'Mod-Enter', run },
  ];
}

export const codeEditorExtensions = [
  lineNumbers(),
  highlightActiveLineGutter(),
  highlightSpecialChars(),
  history(),
  drawSelection(),
  dropCursor(),
  EditorState.allowMultipleSelections.of(true),
  indentOnInput(),
  syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
  bracketMatching(),
  rectangularSelection(),
  crosshairCursor(),
  highlightActiveLine(),
  highlightSelectionMatches(),
  indentUnit.of(CODE_EDITOR_INDENT),
  EditorState.tabSize.of(4),
  python(),
  EditorView.lineWrapping,
  Prec.highest(keymap.of([...codeEditorKeyBindings])),
  keymap.of([...defaultKeymap, ...historyKeymap, ...searchKeymap, ...lintKeymap]),
];

export function getCodeEditorThemeName(documentTheme: string | null | undefined) {
  return documentTheme === 'dark' ? 'dark' : 'light';
}

export type CodeEditorThemeName = ReturnType<typeof getCodeEditorThemeName>;

export function createCodeEditorThemeObserver(onThemeChange: (theme: CodeEditorThemeName) => void) {
  if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') {
    return () => {};
  }

  const root = document.documentElement;
  const observer = new MutationObserver(() => {
    onThemeChange(getCodeEditorThemeName(root.getAttribute('data-theme')));
  });

  observer.observe(root, {
    attributes: true,
    attributeFilter: ['data-theme'],
  });

  return () => {
    observer.disconnect();
  };
}
