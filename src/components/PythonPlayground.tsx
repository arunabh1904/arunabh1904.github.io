import React, { startTransition, useEffect, useRef, useState } from 'react';
import { loadPyodideRuntime } from '../lib/pyodide-loader';
import type { PyodideRuntime } from '../lib/pyodide-loader';
import type { PythonPlaygroundProps } from '../lib/python-playground';
import { runPythonSnippet } from '../lib/python-runner';

export default function PythonPlayground({
  title,
  initialCode,
  samples,
  walkthroughSteps = [],
  notes,
}: PythonPlaygroundProps) {
  const containerRef = useRef<HTMLElement | null>(null);
  const runtimeRef = useRef<PyodideRuntime | null>(null);
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState('');
  const [errorOutput, setErrorOutput] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState('Python runtime will load when this block scrolls into view.');
  const [isRunning, setIsRunning] = useState(false);
  const [selectedSample, setSelectedSample] = useState(0);
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    let didCancel = false;

    async function bootstrapRuntime() {
      if (runtimeRef.current || status === 'loading') {
        return;
      }

      setStatus('loading');
      setStatusMessage('Preparing the in-browser Python runtime...');

      try {
        const runtime = await loadPyodideRuntime();
        if (didCancel) {
          return;
        }
        runtimeRef.current = runtime;
        setStatus('ready');
        setStatusMessage('Python is ready. Edit the code and run it.');
      } catch (error) {
        if (didCancel) {
          return;
        }
        setStatus('error');
        setStatusMessage(
          error instanceof Error ? error.message : 'The Python runtime failed to load.',
        );
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

  const currentStep = walkthroughSteps[activeStep];

  async function handleRun() {
    if (!runtimeRef.current) {
      setStatusMessage('Python is still loading. Try again in a moment.');
      return;
    }

    setIsRunning(true);
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
      setIsRunning(false);
    }
  }

  function handleReset() {
    setCode(initialCode);
    setOutput('');
    setErrorOutput('');
    setSelectedSample(0);
    setActiveStep(0);
  }

  function handleLoadSample(index: number) {
    setSelectedSample(index);
    setCode(samples[index]?.code ?? initialCode);
    setOutput('');
    setErrorOutput('');
    setActiveStep(0);
  }

  return (
    <section className="python-playground" ref={containerRef}>
      <div className="python-playground__header">
        <div>
          <p className="python-playground__eyebrow">Interactive Python</p>
          <h3>{title}</h3>
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

        <label className="python-playground__editor-label" htmlFor={title}>
          Editable Python snippet
        </label>
        <textarea
          id={title}
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
              <h4>{currentStep.label}</h4>
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
