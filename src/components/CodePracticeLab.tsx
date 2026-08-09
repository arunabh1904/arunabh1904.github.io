import React, { startTransition, useEffect, useMemo, useRef, useState } from 'react';
import { Prec } from '@codemirror/state';
import { EditorView, keymap } from '@codemirror/view';
import CodeMirror from '@uiw/react-codemirror';
import { githubDark, githubLight } from '@uiw/codemirror-theme-github';
import { loadPyodideRuntime } from '../lib/pyodide-loader';
import type { PyodideRuntime } from '../lib/pyodide-loader';
import {
  codeEditorExtensions,
  createCodeEditorThemeObserver,
  createRunCodeKeyBindings,
  getCodeEditorThemeName,
} from '../lib/code-editor';
import { augmentCodeWithSolution } from '../lib/code-solution';
import { runPythonSnippet } from '../lib/python-runner';
import type { CodePracticeProblem } from '../lib/code-practice';

interface CodePracticeLabProps {
  problem: CodePracticeProblem;
}

function createHintCommentBlock(hints: readonly string[]) {
  if (hints.length === 0) {
    return '';
  }

  return ['# Hints', ...hints.map((hint, index) => `# ${index + 1}. ${hint}`)].join('\n');
}

export default function CodePracticeLab({ problem }: CodePracticeLabProps) {
  const containerRef = useRef<HTMLElement | null>(null);
  const runtimeRef = useRef<PyodideRuntime | null>(null);
  const loadingRef = useRef(false);
  const isRunningRef = useRef(false);
  const runHandlerRef = useRef<() => void>(() => {});
  const [code, setCode] = useState(problem.starterCode);
  const [output, setOutput] = useState('');
  const [errorOutput, setErrorOutput] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState('Loading Python...');
  const [isRunning, setIsRunning] = useState(false);
  const [editorTheme, setEditorTheme] = useState(() =>
    getCodeEditorThemeName(
      typeof document === 'undefined' ? 'light' : document.documentElement.getAttribute('data-theme'),
    ),
  );

  const hintCommentBlock = useMemo(() => createHintCommentBlock(problem.hint), [problem.hint]);
  const runShortcutExtension = useMemo(
    () =>
      Prec.highest([
        keymap.of(createRunCodeKeyBindings(() => runHandlerRef.current())),
        // Some browser contenteditable paths bypass the keymap handler. Bridge
        // the same command at the editor DOM boundary so Ctrl/Cmd+Enter is reliable.
        EditorView.domEventHandlers({
          keydown(event) {
            if (event.key !== 'Enter' || (!event.ctrlKey && !event.metaKey)) {
              return false;
            }

            event.preventDefault();
            runHandlerRef.current();
            return true;
          },
        }),
      ]),
    [],
  );
  const editorExtensions = useMemo(
    () => [...codeEditorExtensions, runShortcutExtension],
    [runShortcutExtension],
  );

  useEffect(() => {
    setCode(problem.starterCode);
    setOutput('');
    setErrorOutput('');
  }, [problem]);

  useEffect(() => {
    let didCancel = false;

    async function bootstrapRuntime() {
      if (runtimeRef.current || loadingRef.current) {
        return;
      }

      loadingRef.current = true;
      setStatus('loading');
      setStatusMessage('Loading Python...');

      try {
        const runtime = await loadPyodideRuntime();
        if (didCancel) {
          return;
        }

        runtimeRef.current = runtime;
        setStatus('ready');
        setStatusMessage('Python ready');
      } catch (error) {
        if (didCancel) {
          return;
        }

        setStatus('error');
        setStatusMessage('Python unavailable');
        setErrorOutput(error instanceof Error ? error.message : 'The Python runtime failed to load.');
      } finally {
        loadingRef.current = false;
      }
    }

    if (typeof window === 'undefined') {
      return () => {
        didCancel = true;
      };
    }

    const node = containerRef.current;
    if (!node) {
      return () => {
        didCancel = true;
      };
    }

    if (typeof window.IntersectionObserver !== 'function') {
      void bootstrapRuntime();
      return () => {
        didCancel = true;
      };
    }

    const observer = new window.IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry?.isIntersecting) {
          observer.disconnect();
          void bootstrapRuntime();
        }
      },
      { rootMargin: '160px 0px' },
    );

    observer.observe(node);

    return () => {
      didCancel = true;
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    setEditorTheme(
      getCodeEditorThemeName(
        typeof document === 'undefined' ? 'light' : document.documentElement.getAttribute('data-theme'),
      ),
    );

    return createCodeEditorThemeObserver(setEditorTheme);
  }, []);

  const editorId = `${problem.id}-editor`;
  const editorThemeExtension = editorTheme === 'dark' ? githubDark : githubLight;

  async function handleRun() {
    if (isRunningRef.current) {
      return;
    }

    if (!runtimeRef.current) {
      setStatusMessage(status === 'error' ? 'Python unavailable' : 'Python is loading...');
      return;
    }

    isRunningRef.current = true;
    setIsRunning(true);
    setOutput('');
    setErrorOutput('');

    try {
      const result = await runPythonSnippet(
        runtimeRef.current,
        code,
        problem.packages ?? [],
      );

      startTransition(() => {
        setOutput(result.stdout.trimEnd());
        setErrorOutput(result.stderr.trimEnd());
      });
    } catch (error) {
      setErrorOutput(error instanceof Error ? error.message : 'Unknown execution error.');
    } finally {
      isRunningRef.current = false;
      setIsRunning(false);
    }
  }

  function handleReset() {
    setCode(problem.starterCode);
    setOutput('');
    setErrorOutput('');
  }

  function handleAddHints() {
    if (!hintCommentBlock) {
      return;
    }

    setCode((currentCode) =>
      currentCode.startsWith(hintCommentBlock) ? currentCode : `${hintCommentBlock}\n\n${currentCode}`,
    );
  }

  function handleLoadSolution() {
    setCode((currentCode) => augmentCodeWithSolution(problem, currentCode));
    setOutput('');
    setErrorOutput('');
  }

  // The CodeMirror keymap is created once, so route it through a ref to the
  // current controlled editor state instead of rebuilding the editor on each keystroke.
  runHandlerRef.current = () => {
    void handleRun();
  };

  return (
    <section className="code-practice-lab" ref={containerRef}>
      <article className="code-practice-lab__problem">
        <header className="code-practice-lab__problem-header">
          <div className="code-practice-lab__title-block">
            <p className="code-practice-lab__eyebrow">{`Problem ${String(problem.order).padStart(2, '0')}`}</p>
            <h1>{problem.title}</h1>
          </div>
          <span className="code-practice-lab__difficulty">{problem.difficulty}</span>
        </header>

        <div className="code-practice-lab__copy">
          {problem.prompt.map((paragraph) => (
            <p key={paragraph}>{paragraph}</p>
          ))}
        </div>

        <div className="code-practice-lab__specs">
          <section className="code-practice-lab__spec-card">
            <p className="code-practice-lab__section-label">Implement</p>
            <pre>
              <code>{problem.signature}</code>
            </pre>
          </section>

          <section className="code-practice-lab__spec-card">
            <p className="code-practice-lab__section-label">Requirements</p>
            <ul className="code-practice-lab__list">
              {problem.requirements.map((requirement) => (
                <li key={requirement}>{requirement}</li>
              ))}
            </ul>
          </section>

          <section className="code-practice-lab__spec-card">
            <p className="code-practice-lab__section-label">Examples</p>
            <div className="code-practice-lab__examples">
              {problem.examples.map((example) => (
                <div key={example.label} className="code-practice-lab__example">
                  <p>{example.label}</p>
                  <pre>
                    <code>{`${example.lines.join('\n')}\n\n${example.result}`}</code>
                  </pre>
                </div>
              ))}
            </div>
          </section>
        </div>
      </article>

      <article className="code-practice-lab__workspace">
        <header className="code-practice-lab__workspace-header">
          <div>
            <p className="code-practice-lab__eyebrow">Workspace</p>
            <h2>Your solution</h2>
          </div>
          <div className="code-practice-lab__workspace-controls">
            <button
              className="code-practice-lab__button code-practice-lab__button--secondary"
              type="button"
              onClick={handleAddHints}
            >
              Add hints
            </button>
            <button
              className="code-practice-lab__button code-practice-lab__button--solution"
              type="button"
              onClick={handleLoadSolution}
            >
              Load solution
            </button>
          </div>
        </header>

        <div className="code-practice-lab__workspace-meta">
          <p
            className={`code-practice-lab__status code-practice-lab__status--${status}`}
            aria-live="polite"
          >
            {statusMessage}
          </p>
          <p className="code-practice-lab__shortcut">
            <kbd>Ctrl / Cmd + Enter</kbd>
            <span>runs code</span>
          </p>
        </div>

        <label className="code-practice-lab__editor-label" htmlFor={editorId}>
          solution.py
        </label>
        <div className="code-practice-lab__editor-shell">
          <CodeMirror
            id={editorId}
            className="code-practice-lab__editor"
            aria-label="Python solution editor"
            basicSetup={false}
            extensions={editorExtensions}
            theme={editorThemeExtension}
            height="36rem"
            editable
            indentWithTab={false}
            value={code}
            onChange={(value) => setCode(value)}
          />
        </div>

        <div className="code-practice-lab__actions">
          <button
            className="code-practice-lab__button code-practice-lab__button--primary"
            type="button"
            aria-label="Run code"
            aria-keyshortcuts="Control+Enter Meta+Enter"
            onClick={() => void handleRun()}
            disabled={status !== 'ready' || isRunning}
          >
            <span>{isRunning ? 'Running...' : 'Run'}</span>
            {!isRunning && <kbd>Ctrl / Cmd + Enter</kbd>}
          </button>
          <button
            className="code-practice-lab__button code-practice-lab__button--secondary"
            type="button"
            onClick={handleReset}
          >
            Reset
          </button>
        </div>

        <div className="code-practice-lab__output" aria-live="polite">
          <div>
            <p>Output</p>
            <pre>{output || 'Output appears here.'}</pre>
          </div>
          <div>
            <p>Errors</p>
            <pre>{errorOutput || 'Errors appear here.'}</pre>
          </div>
        </div>
      </article>
    </section>
  );
}
