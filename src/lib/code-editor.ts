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
