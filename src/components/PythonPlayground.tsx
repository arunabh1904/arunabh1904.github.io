import React, {
  startTransition,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Prec } from '@codemirror/state';
import { EditorView, keymap } from '@codemirror/view';
import CodeMirror from '@uiw/react-codemirror';
import { githubDark, githubLight } from '@uiw/codemirror-theme-github';
import {
  codeEditorExtensions,
  createCodeEditorThemeObserver,
  createRunCodeKeyBindings,
  getCodeEditorThemeName,
} from '../lib/code-editor';
import { loadPyodideRuntime } from '../lib/pyodide-loader';
import type { PyodideRuntime } from '../lib/pyodide-loader';
import type { PythonPlaygroundProps } from '../lib/python-playground';
import { runPythonSnippet } from '../lib/python-runner';

function createHeadingId(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^\p{L}\p{N}\s-]/gu, '')
    .replace(/[\s-]+/g, '-');
}

export default function PythonPlayground({
  title,
  initialCode,
  samples,
  walkthroughSteps = [],
  notes,
  compact = false,
}: PythonPlaygroundProps) {
  const containerRef = useRef<HTMLElement | null>(null);
  const runtimeRef = useRef<PyodideRuntime | null>(null);
  const loadingRef = useRef(false);
  const isRunningRef = useRef(false);
  const runHandlerRef = useRef<() => void>(() => {});
  const generatedEditorId = useId();
  const editorId = `${createHeadingId(title)}-${generatedEditorId.replace(/:/g, '')}`;
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState('');
  const [errorOutput, setErrorOutput] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState(
    compact ? 'Loading Python…' : 'Python runtime will load when this block scrolls into view.',
  );
  const [isRunning, setIsRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [selectedSample, setSelectedSample] = useState(0);
  const [activeStep, setActiveStep] = useState(0);
  const [editorTheme, setEditorTheme] = useState(() =>
    getCodeEditorThemeName(
      typeof document === 'undefined' ? 'light' : document.documentElement.getAttribute('data-theme'),
    ),
  );

  const runShortcutExtension = useMemo(
    () =>
      Prec.highest([
        keymap.of(createRunCodeKeyBindings(() => runHandlerRef.current())),
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
  const editorThemeExtension = editorTheme === 'dark' ? githubDark : githubLight;

  useEffect(() => {
    let didCancel = false;

    async function bootstrapRuntime() {
      if (runtimeRef.current || loadingRef.current) {
        return;
      }

      loadingRef.current = true;
      setStatus('loading');
      setStatusMessage(compact ? 'Loading Python…' : 'Preparing the in-browser Python runtime...');

      try {
        const runtime = await loadPyodideRuntime();
        if (didCancel) {
          return;
        }
        runtimeRef.current = runtime;
        setStatus('ready');
        setStatusMessage(compact ? 'Ready' : 'Python is ready. Edit the code and run it.');
      } catch (error) {
        if (didCancel) {
          return;
        }
        setStatus('error');
        setStatusMessage(compact ? 'Unavailable' : 'The Python runtime failed to load.');
        setErrorOutput(error instanceof Error ? error.message : 'The Python runtime failed to load.');
        if (compact) {
          setHasRun(true);
        }
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
  }, [compact]);

  useEffect(() => {
    setEditorTheme(
      getCodeEditorThemeName(
        typeof document === 'undefined' ? 'light' : document.documentElement.getAttribute('data-theme'),
      ),
    );

    return createCodeEditorThemeObserver(setEditorTheme);
  }, []);

  const currentStep = walkthroughSteps[activeStep];

  async function handleRun() {
    if (isRunningRef.current) {
      return;
    }

    if (!runtimeRef.current) {
      setStatusMessage(status === 'error' ? 'Unavailable' : 'Loading Python…');
      return;
    }

    isRunningRef.current = true;
    setIsRunning(true);
    setHasRun(true);
    setOutput('');
    setErrorOutput('');

    try {
      const result = await runPythonSnippet(runtimeRef.current, code);
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
    setCode(initialCode);
    setOutput('');
    setErrorOutput('');
    setHasRun(false);
    setSelectedSample(0);
    setActiveStep(0);
  }

  function handleLoadSample(index: number) {
    setSelectedSample(index);
    setCode(samples[index]?.code ?? initialCode);
    setOutput('');
    setErrorOutput('');
    setHasRun(false);
    setActiveStep(0);
  }

  runHandlerRef.current = () => {
    void handleRun();
  };

  if (compact) {
    return (
      <section
        className="python-playground python-playground--compact"
        ref={containerRef}
        aria-label={`${title} editable Python scratchpad`}
      >
        <header className="python-playground__workspace-header">
          <div className="python-playground__workspace-controls">
            <button
              className="python-playground__button python-playground__button--primary"
              type="button"
              aria-label="Run code"
              aria-keyshortcuts="Control+Enter Meta+Enter"
              onClick={() => void handleRun()}
              disabled={status !== 'ready' || isRunning}
            >
              <span>{isRunning ? 'Running…' : 'Run'}</span>
              {!isRunning && <kbd>Ctrl / Cmd + Enter</kbd>}
            </button>
            <button
              className="python-playground__button"
              type="button"
              onClick={handleReset}
            >
              Reset
            </button>
          </div>
        </header>

        <label className="python-playground__editor-label" htmlFor={editorId}>
          {title} Python editor
        </label>
        <div className="python-playground__editor-shell">
          <CodeMirror
            id={editorId}
            className="python-playground__code-editor"
            aria-label={`${title} Python editor`}
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

        {hasRun && (
          <div className="python-playground__compact-output" aria-live="polite">
            <p>{errorOutput ? 'Errors' : 'Output'}</p>
            <pre>{errorOutput || output || 'Program finished with no output.'}</pre>
          </div>
        )}
      </section>
    );
  }

  return (
    <section className="python-playground" ref={containerRef}>
      <div className="python-playground__header">
        <div>
          <p className="python-playground__eyebrow">Interactive Python</p>
          <h3 id={createHeadingId(title)}>{title}</h3>
        </div>
        <p
          className={`python-playground__status python-playground__status--${status}`}
          aria-live="polite"
        >
          {statusMessage}
        </p>
      </div>

      <div className="python-playground__samples" aria-label="Example presets">
        {samples.map((sample, index) => (
          <button
            key={sample.label}
            className={index === selectedSample ? 'is-active' : undefined}
            type="button"
            onClick={() => handleLoadSample(index)}
          >
            {sample.label}
          </button>
        ))}
      </div>

      {samples[selectedSample]?.description && (
        <p className="python-playground__sample-description">
          {samples[selectedSample].description}
        </p>
      )}

      <div className="python-playground__terminal">
        <div className="python-playground__promptbar">
          <span className="python-playground__dot" />
          <span className="python-playground__dot" />
          <span className="python-playground__dot" />
          <span className="python-playground__prompt">python lesson.py</span>
        </div>

        <label className="python-playground__editor-label" htmlFor={editorId}>
          Editable Python snippet
        </label>
        <textarea
          id={editorId}
          className="python-playground__editor"
          spellCheck={false}
          value={code}
          onChange={(event) => setCode(event.target.value)}
        />

        <div className="python-playground__actions">
          <button type="button" onClick={() => void handleRun()} disabled={status !== 'ready' || isRunning}>
            {isRunning ? 'Running...' : 'Run'}
          </button>
          <button type="button" onClick={handleReset}>
            Reset
          </button>
          <button type="button" onClick={() => handleLoadSample(selectedSample)}>
            Load Example
          </button>
        </div>

        <div className="python-playground__output">
          <div>
            <p>stdout</p>
            <pre>{output || 'Run the snippet to see printed output here.'}</pre>
          </div>
          <div>
            <p>stderr</p>
            <pre>{errorOutput || 'Execution errors will appear here.'}</pre>
          </div>
        </div>
      </div>

      {walkthroughSteps.length > 0 && currentStep && (
        <div className="python-playground__walkthrough">
          <div className="python-playground__walkthrough-header">
            <div>
              <p className="python-playground__eyebrow">Guided Trace</p>
              <h4 id={createHeadingId(currentStep.label)}>{currentStep.label}</h4>
            </div>
            <p>
              Step {activeStep + 1} of {walkthroughSteps.length}
              {currentStep.lineHint ? ` · line ${currentStep.lineHint}` : ''}
            </p>
          </div>

          <div className="python-playground__variables">
            {Object.entries(currentStep.variables).map(([name, value]) => (
              <div key={name} className="python-playground__variable-card">
                <p>{name}</p>
                <code>{value}</code>
              </div>
            ))}
          </div>

          {currentStep.output && <p className="python-playground__walkthrough-note">{currentStep.output}</p>}

          <div className="python-playground__walkthrough-actions">
            <button
              type="button"
              onClick={() => setActiveStep((step) => Math.max(step - 1, 0))}
              disabled={activeStep === 0}
            >
              Previous Step
            </button>
            <button
              type="button"
              onClick={() =>
                setActiveStep((step) => Math.min(step + 1, walkthroughSteps.length - 1))
              }
              disabled={activeStep === walkthroughSteps.length - 1}
            >
              Next Step
            </button>
          </div>
        </div>
      )}

      {notes && <p className="python-playground__notes">{notes}</p>}
    </section>
  );
}
