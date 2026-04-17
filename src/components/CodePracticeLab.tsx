import React, { startTransition, useEffect, useRef, useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { githubDark, githubLight } from '@uiw/codemirror-theme-github';
import { loadPyodideRuntime } from '../lib/pyodide-loader';
import type { PyodideRuntime } from '../lib/pyodide-loader';
import {
  codeEditorExtensions,
  createCodeEditorThemeObserver,
  getCodeEditorThemeName,
} from '../lib/code-editor';
import { runPythonSnippet } from '../lib/python-runner';
import type { CodePracticeProblem } from '../lib/code-practice';

interface CodePracticeLabProps {
  problem: CodePracticeProblem;
}

function formatPackages(packages: readonly string[] = []) {
  return packages.map((packageName) => {
    if (packageName === 'numpy') {
      return 'NumPy';
    }
    return packageName;
  });
}

function getRuntimeNote(packages: readonly string[] = []) {
  const packageLabels = formatPackages(packages);
  if (packageLabels.length === 0) {
    return 'The in-browser Python runtime executes this workspace locally in the browser.';
  }

  if (packageLabels.length === 1) {
    return `${packageLabels[0]} is available and will load automatically the first time you run the code.`;
  }

  const leadingPackages = packageLabels.slice(0, -1).join(', ');
  const trailingPackage = packageLabels.at(-1);
  return `${leadingPackages}, and ${trailingPackage} are available and will load automatically the first time you run the code.`;
}

export default function CodePracticeLab({ problem }: CodePracticeLabProps) {
  const containerRef = useRef<HTMLElement | null>(null);
  const runtimeRef = useRef<PyodideRuntime | null>(null);
  const loadingRef = useRef(false);
  const [code, setCode] = useState(problem.starterCode);
  const [output, setOutput] = useState('');
  const [errorOutput, setErrorOutput] = useState('');
  const [showHint, setShowHint] = useState(false);
  const [showSolution, setShowSolution] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState(
    'Python runtime will load when this lab scrolls into view.',
  );
  const [isRunning, setIsRunning] = useState(false);
  const [editorTheme, setEditorTheme] = useState(() =>
    getCodeEditorThemeName(
      typeof document === 'undefined' ? 'light' : document.documentElement.getAttribute('data-theme'),
    ),
  );

  useEffect(() => {
    setCode(problem.starterCode);
    setOutput('');
    setErrorOutput('');
    setShowHint(false);
    setShowSolution(false);
  }, [problem]);

  useEffect(() => {
    let didCancel = false;

    async function bootstrapRuntime() {
      if (runtimeRef.current || loadingRef.current) {
        return;
      }

      loadingRef.current = true;
      setStatus('loading');
      setStatusMessage('Preparing the in-browser Python runtime...');

      try {
        const runtime = await loadPyodideRuntime();
        if (didCancel) {
          return;
        }

        runtimeRef.current = runtime;
        setStatus('ready');
        setStatusMessage('Python is ready. Run the starter code or edit your solution.');
      } catch (error) {
        if (didCancel) {
          return;
        }

        setStatus('error');
        setStatusMessage(
          error instanceof Error ? error.message : 'The Python runtime failed to load.',
        );
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
    if (!runtimeRef.current) {
      setStatusMessage('Python is still loading. Try again in a moment.');
      return;
    }

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
      setIsRunning(false);
    }
  }

  function handleReset() {
    setCode(problem.starterCode);
    setOutput('');
    setErrorOutput('');
  }

  return (
    <section className="code-practice-lab" ref={containerRef}>
      <article className="code-practice-lab__hero">
        <div className="code-practice-lab__hero-header">
          <div className="code-practice-lab__title-block">
            <p className="code-practice-lab__eyebrow">{`Problem ${String(problem.order).padStart(2, '0')}`}</p>
            <h1>{problem.title}</h1>
            <p className="code-practice-lab__summary">{problem.summary}</p>
          </div>

          <div className="code-practice-lab__hero-actions">
            <span className="code-practice-lab__difficulty">{problem.difficulty}</span>
            <button
              type="button"
              aria-expanded={showHint}
              onClick={() => setShowHint((current) => !current)}
            >
              {showHint ? 'Hide hint' : 'Hint'}
            </button>
            <button
              type="button"
              aria-expanded={showSolution}
              onClick={() => setShowSolution((current) => !current)}
            >
              {showSolution ? 'Hide solution' : 'Solution'}
            </button>
          </div>
        </div>

        {showHint && (
          <div className="code-practice-lab__hint-panel">
            <p className="code-practice-lab__section-label">Hint</p>
            <ul className="code-practice-lab__list">
              {problem.hint.map((hintLine) => (
                <li key={hintLine}>{hintLine}</li>
              ))}
            </ul>
          </div>
        )}

        {showSolution && (
          <div className="code-practice-lab__solution-panel">
            <div className="code-practice-lab__solution-header">
              <p className="code-practice-lab__section-label">Solution</p>
              <button type="button" onClick={() => setShowSolution(false)}>
                Collapse
              </button>
            </div>

            <div className="code-practice-lab__copy">
              {problem.solutionNotes.map((note) => (
                <p key={note}>{note}</p>
              ))}
            </div>
            <pre>
              <code>{problem.solutionCode}</code>
            </pre>
          </div>
        )}

        <div className="code-practice-lab__prompt">
          <div className="code-practice-lab__copy">
            {problem.prompt.map((paragraph) => (
              <p key={paragraph}>{paragraph}</p>
            ))}
          </div>

          <div className="code-practice-lab__spec-grid">
            <article className="code-practice-lab__spec-card">
              <p className="code-practice-lab__section-label">Implement</p>
              <pre>
                <code>{problem.signature}</code>
              </pre>
            </article>

            <article className="code-practice-lab__spec-card">
              <p className="code-practice-lab__section-label">Requirements</p>
              <ul className="code-practice-lab__list">
                {problem.requirements.map((requirement) => (
                  <li key={requirement}>{requirement}</li>
                ))}
              </ul>
            </article>

            <article className="code-practice-lab__spec-card">
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
            </article>
          </div>
        </div>
      </article>

      <article className="code-practice-lab__workspace">
        <div className="code-practice-lab__header code-practice-lab__header--workspace">
          <div>
            <p className="code-practice-lab__eyebrow">Python Workspace</p>
            <h2>Run your solution</h2>
          </div>
          <p
            className={`code-practice-lab__status code-practice-lab__status--${status}`}
            aria-live="polite"
          >
            {statusMessage}
          </p>
        </div>

        <p className="code-practice-lab__runtime-note">{getRuntimeNote(problem.packages ?? [])}</p>

        <div className="code-practice-lab__promptbar">
          <span className="code-practice-lab__dot" />
          <span className="code-practice-lab__dot" />
          <span className="code-practice-lab__dot" />
          <span className="code-practice-lab__prompt">python interview_practice.py</span>
        </div>

        <label className="code-practice-lab__editor-label" htmlFor={editorId}>
          Editable starter code
        </label>
        <p className="code-practice-lab__editor-help">
          `Tab` indents, `Shift+Tab` outdents, and `Cmd/Ctrl + /` toggles comments.
        </p>
        <div className="code-practice-lab__editor-shell">
          <CodeMirror
            id={editorId}
            className="code-practice-lab__editor"
            aria-label="Editable starter code"
            basicSetup={false}
            extensions={codeEditorExtensions}
            theme={editorThemeExtension}
            height="34rem"
            editable
            indentWithTab={false}
            value={code}
            onChange={(value) => setCode(value)}
          />
        </div>

        <div className="code-practice-lab__actions">
          <button type="button" onClick={() => void handleRun()} disabled={status !== 'ready' || isRunning}>
            {isRunning ? 'Running...' : 'Run code'}
          </button>
          <button type="button" onClick={handleReset}>
            Reset starter
          </button>
        </div>

        <div className="code-practice-lab__output">
          <div>
            <p>stdout</p>
            <pre>{output || 'Run the starter code to see your printed output here.'}</pre>
          </div>
          <div>
            <p>stderr</p>
            <pre>{errorOutput || 'Execution errors will appear here.'}</pre>
          </div>
        </div>
      </article>
    </section>
  );
}
