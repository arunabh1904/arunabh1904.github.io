import React, { startTransition, useEffect, useRef, useState } from 'react';
import { loadPyodideRuntime } from '../lib/pyodide-loader';
import type { PyodideRuntime } from '../lib/pyodide-loader';
import { runPythonSnippet } from '../lib/python-runner';
import type { CodePracticeProblem } from '../lib/code-practice';

interface CodePracticeLabProps {
  problems: readonly CodePracticeProblem[];
}

export default function CodePracticeLab({ problems }: CodePracticeLabProps) {
  const containerRef = useRef<HTMLElement | null>(null);
  const runtimeRef = useRef<PyodideRuntime | null>(null);
  const loadingRef = useRef(false);
  const [selectedProblemIndex, setSelectedProblemIndex] = useState(0);
  const selectedProblem = problems[selectedProblemIndex] ?? problems[0];
  const [code, setCode] = useState(selectedProblem?.starterCode ?? '');
  const [output, setOutput] = useState('');
  const [errorOutput, setErrorOutput] = useState('');
  const [showHint, setShowHint] = useState(false);
  const [showSolution, setShowSolution] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState(
    'Python runtime will load when this lab scrolls into view.',
  );
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    if (!selectedProblem) {
      return;
    }

    setCode(selectedProblem.starterCode);
    setOutput('');
    setErrorOutput('');
    setShowHint(false);
    setShowSolution(false);
  }, [selectedProblem]);

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

  if (!selectedProblem) {
    return null;
  }

  const editorId = `${selectedProblem.id}-editor`;

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
        selectedProblem.packages ?? [],
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
    setCode(selectedProblem.starterCode);
    setOutput('');
    setErrorOutput('');
  }

  return (
    <section className="code-practice-lab" ref={containerRef}>
      <div className="code-practice-lab__problem-strip" aria-label="Practice problems">
        {problems.map((problem, index) => (
          <button
            key={problem.id}
            className={index === selectedProblemIndex ? 'is-active' : undefined}
            type="button"
            onClick={() => setSelectedProblemIndex(index)}
          >
            <span>{`Problem ${String(problem.order).padStart(2, '0')}`}</span>
            <strong>{problem.title}</strong>
          </button>
        ))}
      </div>

      <div className="code-practice-lab__workspace">
        <article className="code-practice-lab__panel code-practice-lab__panel--problem">
          <div className="code-practice-lab__header">
            <div>
              <p className="code-practice-lab__eyebrow">Interview Practice</p>
              <h3>{`Problem ${selectedProblem.order}: ${selectedProblem.title}`}</h3>
              <p className="code-practice-lab__summary">{selectedProblem.summary}</p>
            </div>
            <span className="code-practice-lab__difficulty">{selectedProblem.difficulty}</span>
          </div>

          {selectedProblem.tags && selectedProblem.tags.length > 0 && (
            <div className="code-practice-lab__tags" aria-label="Problem topics">
              {selectedProblem.tags.map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
          )}

          <div className="code-practice-lab__copy">
            {selectedProblem.prompt.map((paragraph) => (
              <p key={paragraph}>{paragraph}</p>
            ))}
          </div>

          <div className="code-practice-lab__block">
            <p className="code-practice-lab__section-label">Implement</p>
            <pre>
              <code>{selectedProblem.signature}</code>
            </pre>
          </div>

          <div className="code-practice-lab__block">
            <p className="code-practice-lab__section-label">Where</p>
            <ul className="code-practice-lab__list">
              {selectedProblem.requirements.map((requirement) => (
                <li key={requirement}>{requirement}</li>
              ))}
            </ul>
          </div>

          <div className="code-practice-lab__block">
            <p className="code-practice-lab__section-label">Examples</p>
            <div className="code-practice-lab__examples">
              {selectedProblem.examples.map((example) => (
                <div key={example.label} className="code-practice-lab__example">
                  <p>{example.label}</p>
                  <pre>
                    <code>{`${example.lines.join('\n')}\n\n${example.result}`}</code>
                  </pre>
                </div>
              ))}
            </div>
          </div>

          <div className="code-practice-lab__reveal-actions">
            <button
              type="button"
              aria-expanded={showHint}
              onClick={() => setShowHint((current) => !current)}
            >
              {showHint ? 'Hide hint' : 'Show hint'}
            </button>
            <button
              type="button"
              aria-expanded={showSolution}
              onClick={() => setShowSolution((current) => !current)}
            >
              {showSolution ? 'Hide solution' : 'Reveal solution'}
            </button>
          </div>

          {showHint && (
            <div className="code-practice-lab__reveal-panel">
              <p className="code-practice-lab__section-label">Hint</p>
              <ul className="code-practice-lab__list">
                {selectedProblem.hint.map((hintLine) => (
                  <li key={hintLine}>{hintLine}</li>
                ))}
              </ul>
            </div>
          )}

          {showSolution && (
            <div className="code-practice-lab__reveal-panel">
              <p className="code-practice-lab__section-label">Solution</p>
              <div className="code-practice-lab__copy">
                {selectedProblem.solutionNotes.map((note) => (
                  <p key={note}>{note}</p>
                ))}
              </div>
              <pre>
                <code>{selectedProblem.solutionCode}</code>
              </pre>
            </div>
          )}
        </article>

        <article className="code-practice-lab__panel code-practice-lab__panel--editor">
          <div className="code-practice-lab__header code-practice-lab__header--editor">
            <div>
              <p className="code-practice-lab__eyebrow">Python Workspace</p>
              <h4>Run your solution</h4>
            </div>
            <p
              className={`code-practice-lab__status code-practice-lab__status--${status}`}
              aria-live="polite"
            >
              {statusMessage}
            </p>
          </div>

          <p className="code-practice-lab__runtime-note">
            NumPy is available for this problem and will load automatically the first time you run
            it.
          </p>

          <div className="code-practice-lab__promptbar">
            <span className="code-practice-lab__dot" />
            <span className="code-practice-lab__dot" />
            <span className="code-practice-lab__dot" />
            <span className="code-practice-lab__prompt">python interview_practice.py</span>
          </div>

          <label className="code-practice-lab__editor-label" htmlFor={editorId}>
            Editable starter code
          </label>
          <textarea
            id={editorId}
            className="code-practice-lab__editor"
            spellCheck={false}
            value={code}
            onChange={(event) => setCode(event.target.value)}
          />

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
      </div>
    </section>
  );
}
