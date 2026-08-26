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

/**
 * The exercise scaffold used to repeat the guidance in a large TODO comment
 * block above the placeholder. That guidance now lives beside the editor, so
 * keep only the executable placeholder in the code surface.
 */
function removeTodoCommentBlocks(source: string) {
  const lines = source.split('\n');
  const cleaned: string[] = [];
  let removingTodoComments = false;

  for (const line of lines) {
    const trimmed = line.trim();

    if (/^# TODO\b/.test(trimmed)) {
      removingTodoComments = true;
      continue;
    }

    if (removingTodoComments && (trimmed === '' || trimmed.startsWith('#'))) {
      continue;
    }

    removingTodoComments = false;
    cleaned.push(line);
  }

  return cleaned.join('\n');
}

function startsFromBlankFile(problem: CodePracticeProblem) {
  return problem.editorStart === 'blank';
}

function getInitialEditorCode(problem: CodePracticeProblem) {
  return startsFromBlankFile(problem) ? '' : removeTodoCommentBlocks(problem.starterCode);
}

function removeReferenceAnnotations(source: string) {
  return source
    .split('\n')
    .filter((line) => {
      const trimmed = line.trimStart();
      return (
        !trimmed.startsWith('# Reference solution') &&
        !trimmed.startsWith('# Original placeholder:')
      );
    })
    .join('\n');
}

function getReferenceEditorCode(problem: CodePracticeProblem, currentCode: string) {
  if (startsFromBlankFile(problem)) {
    // Blank-start exercises should still reveal a complete runnable file. Use
    // the stored scaffold only at reveal time so the sample inputs and print
    // calls remain available without giving the learner an initial template.
    return removeReferenceAnnotations(
      augmentCodeWithSolution(problem, removeTodoCommentBlocks(problem.starterCode)),
    );
  }

  return removeReferenceAnnotations(augmentCodeWithSolution(problem, currentCode));
}

export default function CodePracticeLab({ problem }: CodePracticeLabProps) {
  const isBrowserRunnable = (problem.environment ?? 'browser') === 'browser';
  const interviewDuration =
    problem.interview?.durationMinutes ??
    ({ Easy: 20, Medium: 30, Hard: 45 } as const)[problem.difficulty];
  const evaluationCriteria = problem.interview?.evaluationCriteria ?? [
    'Clarify tensor shapes, return values, and failure cases before coding.',
    'Keep the implementation small enough to explain while you write it.',
    'Run the example and add one edge-case check before calling it done.',
  ];
  const containerRef = useRef<HTMLElement | null>(null);
  const runtimeRef = useRef<PyodideRuntime | null>(null);
  const loadingRef = useRef(false);
  const isRunningRef = useRef(false);
  const runHandlerRef = useRef<() => void>(() => {});
  const [code, setCode] = useState(() => getInitialEditorCode(problem));
  const [output, setOutput] = useState('');
  const [errorOutput, setErrorOutput] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>(() =>
    isBrowserRunnable ? 'idle' : 'ready',
  );
  const [statusMessage, setStatusMessage] = useState(() =>
    isBrowserRunnable ? 'Loading Python...' : 'Local PyTorch',
  );
  const [isRunning, setIsRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [solutionLanguage, setSolutionLanguage] = useState<'torch' | 'numpy' | null>(null);
  const [isExplanationOpen, setIsExplanationOpen] = useState(false);
  const [explanationStep, setExplanationStep] = useState(0);
  const [editorTheme, setEditorTheme] = useState(() =>
    getCodeEditorThemeName(
      typeof document === 'undefined' ? 'light' : document.documentElement.getAttribute('data-theme'),
    ),
  );

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
    () =>
      isBrowserRunnable
        ? [...codeEditorExtensions, runShortcutExtension]
        : [...codeEditorExtensions],
    [isBrowserRunnable, runShortcutExtension],
  );

  useEffect(() => {
    setCode(getInitialEditorCode(problem));
    setOutput('');
    setErrorOutput('');
    setHasRun(false);
    setSolutionLanguage(null);
    setIsExplanationOpen(false);
    setExplanationStep(0);
  }, [problem]);

  useEffect(() => {
    if (!isExplanationOpen || typeof document === 'undefined') {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsExplanationOpen(false);
      }
    };

    document.body.style.overflow = 'hidden';
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isExplanationOpen]);

  useEffect(() => {
    let didCancel = false;

    if (!isBrowserRunnable) {
      setStatus('ready');
      setStatusMessage('Local PyTorch');
      return () => {
        didCancel = true;
      };
    }

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
        setStatusMessage('');
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
  }, [isBrowserRunnable]);

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
  const isReferenceLoaded = solutionLanguage !== null;
  const hasExecutionResult = hasRun || Boolean(errorOutput);
  const explanationCode =
    solutionLanguage === 'numpy' && problem.numpyAlternative
      ? problem.numpyAlternative.code
      : getReferenceEditorCode(problem, getInitialEditorCode(problem));
  const explanationNotes =
    solutionLanguage === 'numpy'
      ? [...problem.solutionNotes, ...(problem.numpyAlternative?.memory ?? [])]
      : problem.solutionNotes;
  const outputSummary =
    problem.requirements.find((requirement) => requirement.startsWith('Return')) ??
    problem.examples[0]?.result ??
    'Return the value required by the API contract.';

  async function handleRun() {
    if (!isBrowserRunnable || isRunningRef.current) {
      return;
    }

    if (!runtimeRef.current) {
      setStatusMessage(status === 'error' ? 'Python unavailable' : 'Python is loading...');
      return;
    }

    isRunningRef.current = true;
    setIsRunning(true);
    setHasRun(true);
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
    setCode(getInitialEditorCode(problem));
    setOutput('');
    setErrorOutput('');
    setHasRun(false);
    setSolutionLanguage(null);
    setIsExplanationOpen(false);
    setExplanationStep(0);
  }

  function handleLoadSolution(language: 'torch' | 'numpy') {
    if (language === 'numpy' && problem.numpyAlternative) {
      setCode(problem.numpyAlternative.code);
    } else {
      setCode(getReferenceEditorCode(problem, getInitialEditorCode(problem)));
    }
    setOutput('');
    setErrorOutput('');
    setHasRun(false);
    setSolutionLanguage(language);
    setIsExplanationOpen(true);
    setExplanationStep(0);
  }

  function handleOpenExplanation() {
    if (solutionLanguage === null) {
      handleLoadSolution('torch');
      return;
    }

    setIsExplanationOpen(true);
  }

  // The CodeMirror keymap is created once, so route it through a ref to the
  // current controlled editor state instead of rebuilding the editor on each keystroke.
  runHandlerRef.current = () => {
    if (isBrowserRunnable) {
      void handleRun();
    }
  };

  return (
    <section
      className={`code-practice-lab${isReferenceLoaded ? ' code-practice-lab--reference' : ''}`}
      ref={containerRef}
    >
      <article className="code-practice-lab__problem">
        <header className="code-practice-lab__problem-header">
          <div className="code-practice-lab__title-block">
            <p className="code-practice-lab__eyebrow">{`Problem ${String(problem.order).padStart(2, '0')}`}</p>
            <h1>{problem.title}</h1>
          </div>
          <span className="code-practice-lab__difficulty">{problem.difficulty}</span>
        </header>

        <p className="code-practice-lab__section-label">Interview prompt</p>
        <div className="code-practice-lab__copy">
          {problem.prompt.map((paragraph) => (
            <p key={paragraph}>{paragraph}</p>
          ))}
        </div>

        <section className="code-practice-lab__interview" aria-label="Interview format">
          <div className="code-practice-lab__interview-heading">
            <p className="code-practice-lab__section-label">What good looks like</p>
            <strong>{interviewDuration} min</strong>
          </div>
          <ul className="code-practice-lab__list">
            {evaluationCriteria.map((criterion) => (
              <li key={criterion}>{criterion}</li>
            ))}
          </ul>
          {problem.interview && problem.interview.followUps.length > 0 && (
            <div className="code-practice-lab__follow-ups">
              <p className="code-practice-lab__section-label">Likely follow-ups</p>
              <ul className="code-practice-lab__list">
                {problem.interview.followUps.map((followUp) => (
                  <li key={followUp}>{followUp}</li>
                ))}
              </ul>
            </div>
          )}
        </section>

        {problem.reasoning && problem.reasoning.length > 0 && (
          <section
            className="code-practice-lab__reasoning"
            aria-labelledby={`${problem.id}-reasoning-title`}
          >
            <div className="code-practice-lab__reasoning-heading">
              <p className="code-practice-lab__section-label">Reason through the system</p>
              <h2 id={`${problem.id}-reasoning-title`}>Defend the tradeoffs</h2>
            </div>
            <div className="code-practice-lab__reasoning-grid">
              {problem.reasoning.map((point) => (
                <article key={point.axis} className="code-practice-lab__reasoning-card">
                  <h3>{point.axis}</h3>
                  <p>{point.detail}</p>
                </article>
              ))}
            </div>
          </section>
        )}

        {problem.visual && (
          <figure className="code-practice-lab__visual">
            <img src={problem.visual.src} alt={problem.visual.alt} loading="lazy" />
            <figcaption>{problem.visual.caption}</figcaption>
          </figure>
        )}

        <div className="code-practice-lab__specs">
          <section className="code-practice-lab__spec-card">
            <p className="code-practice-lab__section-label">API contract</p>
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
            <p className="code-practice-lab__section-label">Acceptance checks</p>
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

      <article
        className={`code-practice-lab__workspace${isBrowserRunnable ? '' : ' code-practice-lab__workspace--local'}`}
      >
        <header className="code-practice-lab__workspace-header">
          <div className="code-practice-lab__workspace-identity">
            {(!isBrowserRunnable || status !== 'ready') && (
              <p
                className={`code-practice-lab__status code-practice-lab__status--${status}`}
                aria-live="polite"
              >
                {statusMessage}
              </p>
            )}
          </div>
          <div className="code-practice-lab__workspace-controls">
            {problem.numpyAlternative ? (
              <div className="code-practice-lab__solution-picker" aria-label="Solution language">
                <span>Solution</span>
                <button
                  type="button"
                  aria-pressed={solutionLanguage === 'torch'}
                  onClick={() => handleLoadSolution('torch')}
                >
                  Torch
                </button>
                <button
                  type="button"
                  aria-pressed={solutionLanguage === 'numpy'}
                  onClick={() => handleLoadSolution('numpy')}
                >
                  NumPy
                </button>
              </div>
            ) : (
              <button
                className="code-practice-lab__button code-practice-lab__button--solution"
                type="button"
                onClick={() => handleLoadSolution('torch')}
              >
                Solution
              </button>
            )}
            <button
              className="code-practice-lab__explanation-button"
              type="button"
              onClick={handleOpenExplanation}
            >
              Explanation
            </button>
            {isBrowserRunnable && (
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
            )}
            <button
              className="code-practice-lab__button code-practice-lab__button--secondary"
              type="button"
              onClick={handleReset}
            >
              Reset
            </button>
          </div>
        </header>

        {!isBrowserRunnable && (
          <p className="code-practice-lab__runtime-note code-practice-lab__runtime-note--local">
            This exercise uses the full <code>torch.nn</code> API. Write here, then copy
            <code> solution.py </code> into a local PyTorch environment to run the included smoke test.
          </p>
        )}

        <div className="code-practice-lab__editor-layout">
          <div className="code-practice-lab__editor-column">
            <label className="code-practice-lab__editor-label" htmlFor={editorId}>
              solution.py editor
            </label>
            <div className="code-practice-lab__editor-shell">
              <CodeMirror
                id={editorId}
                className="code-practice-lab__editor"
                aria-label="Python solution editor"
                basicSetup={false}
                extensions={editorExtensions}
                theme={editorThemeExtension}
                height="100%"
                editable
                indentWithTab={false}
                value={code}
                onChange={(value) => setCode(value)}
              />
            </div>
          </div>

        </div>

        {isBrowserRunnable && hasExecutionResult && (
          <div className="code-practice-lab__output" aria-live="polite">
            <div>
              <p>{errorOutput ? 'Errors' : 'Output'}</p>
              <pre>{errorOutput || output || 'Program finished with no output.'}</pre>
            </div>
          </div>
        )}
      </article>

      {isExplanationOpen && solutionLanguage && (
        <div
          className="code-practice-lab__explanation-backdrop"
          onMouseDown={(event) => {
            if (event.currentTarget === event.target) {
              setIsExplanationOpen(false);
            }
          }}
        >
          <section
            className="code-practice-lab__explanation-popout"
            role="dialog"
            aria-modal="true"
            aria-labelledby={`${problem.id}-explanation-title`}
          >
            <header className="code-practice-lab__explanation-header">
              <div>
                <p className="code-practice-lab__section-label">
                  {solutionLanguage === 'numpy' ? 'NumPy explanation' : 'Torch explanation'}
                </p>
                <h2 id={`${problem.id}-explanation-title`}>{problem.title}</h2>
              </div>
              <button
                type="button"
                aria-label="Close explanation"
                onClick={() => setIsExplanationOpen(false)}
              >
                Close
              </button>
            </header>

            <nav className="code-practice-lab__explanation-steps" aria-label="Explanation steps">
              {['Contract', 'Mechanism', 'Shapes', 'Code & checks'].map((label, index) => (
                <button
                  key={label}
                  type="button"
                  aria-current={explanationStep === index ? 'step' : undefined}
                  onClick={() => setExplanationStep(index)}
                >
                  <span>{index + 1}</span>
                  {label}
                </button>
              ))}
            </nav>

            <div className="code-practice-lab__explanation-scroll">
              {explanationStep === 0 && (
                <section className="code-practice-lab__explanation-step" aria-labelledby={`${problem.id}-contract-title`}>
                  <p className="code-practice-lab__section-label">Step 1 · Contract</p>
                  <h3 id={`${problem.id}-contract-title`}>Name the input and output first</h3>
                  <p className="code-practice-lab__explanation-lead">{problem.prompt[0]}</p>
                  <div className="code-practice-lab__concept-flow">
                    <article>
                      <span>Input</span>
                      <p>{problem.requirements[0]}</p>
                    </article>
                    <span aria-hidden="true">→</span>
                    <article>
                      <span>Core operation</span>
                      <p>{problem.summary}</p>
                    </article>
                    <span aria-hidden="true">→</span>
                    <article>
                      <span>Output</span>
                      <p>{outputSummary}</p>
                    </article>
                  </div>
                  <pre className="code-practice-lab__explanation-signature">
                    <code>{problem.signature}</code>
                  </pre>
                </section>
              )}

              {explanationStep === 1 && (
                <section className="code-practice-lab__explanation-step" aria-labelledby={`${problem.id}-mechanism-title`}>
                  <p className="code-practice-lab__section-label">Step 2 · Mechanism</p>
                  <h3 id={`${problem.id}-mechanism-title`}>
                    {solutionLanguage === 'numpy'
                      ? 'Build the NumPy version from the same math'
                      : 'Build the result from the math'}
                  </h3>
                  {problem.visual && (
                    <figure className="code-practice-lab__explanation-visual">
                      <img src={problem.visual.src} alt={problem.visual.alt} />
                      <figcaption>{problem.visual.caption}</figcaption>
                    </figure>
                  )}
                  {solutionLanguage === 'numpy' && (
                    <p className="code-practice-lab__explanation-lead">
                      The formula and shapes do not change. The final notes name the NumPy syntax
                      worth remembering.
                    </p>
                  )}
                  <div className="code-practice-lab__explanation-copy">
                    {explanationNotes.map((note) => (
                      <p key={note}>{note}</p>
                    ))}
                  </div>
                </section>
              )}

              {explanationStep === 2 && (
                <section className="code-practice-lab__explanation-step" aria-labelledby={`${problem.id}-shapes-title`}>
                  <p className="code-practice-lab__section-label">Step 3 · Shapes</p>
                  <h3 id={`${problem.id}-shapes-title`}>Track what changes at each operation</h3>
                  {problem.solutionDiagram ? (
                    <pre className="code-practice-lab__solution-diagram">
                      <code>{problem.solutionDiagram}</code>
                    </pre>
                  ) : (
                    <ul className="code-practice-lab__list code-practice-lab__shape-checklist">
                      {problem.requirements.map((requirement) => (
                        <li key={requirement}>{requirement}</li>
                      ))}
                    </ul>
                  )}
                  {problem.reasoning && problem.reasoning.length > 0 && (
                    <div className="code-practice-lab__explanation-reasoning">
                      {problem.reasoning.map((point) => (
                        <article key={point.axis}>
                          <strong>{point.axis}</strong>
                          <p>{point.detail}</p>
                        </article>
                      ))}
                    </div>
                  )}
                </section>
              )}

              {explanationStep === 3 && (
                <section className="code-practice-lab__explanation-step" aria-labelledby={`${problem.id}-code-title`}>
                  <p className="code-practice-lab__section-label">Step 4 · Code and checks</p>
                  <h3 id={`${problem.id}-code-title`}>
                    {solutionLanguage === 'numpy' ? 'NumPy' : 'Torch'} reference
                  </h3>
                  <pre className="code-practice-lab__explanation-code">
                    <code>{explanationCode}</code>
                  </pre>
                  <h3>What to test and explain</h3>
                  <ul className="code-practice-lab__list">
                    {problem.hint.map((hint) => (
                      <li key={hint}>{hint}</li>
                    ))}
                  </ul>
                  <div className="code-practice-lab__explanation-example">
                    <strong>{problem.examples[0]?.label ?? 'Example'}</strong>
                    <pre>
                      <code>{`${problem.examples[0]?.lines.join('\n') ?? ''}\n\n${problem.examples[0]?.result ?? ''}`}</code>
                    </pre>
                  </div>
                </section>
              )}
            </div>

            <footer className="code-practice-lab__explanation-footer">
              <span>Step {explanationStep + 1} of 4</span>
              <div>
                <button
                  type="button"
                  onClick={() => setExplanationStep((step) => Math.max(0, step - 1))}
                  disabled={explanationStep === 0}
                >
                  Previous
                </button>
                <button
                  type="button"
                  onClick={() => setExplanationStep((step) => Math.min(3, step + 1))}
                  disabled={explanationStep === 3}
                >
                  Next
                </button>
              </div>
            </footer>
          </section>
        </div>
      )}
    </section>
  );
}
