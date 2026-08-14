import React, { startTransition, useEffect, useMemo, useRef, useState } from 'react';
import { Prec, RangeSetBuilder } from '@codemirror/state';
import { EditorView, GutterMarker, gutter, keymap } from '@codemirror/view';
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

interface SolutionWalkthroughAnnotation {
  lineNumber: number;
  code: string;
  explanation: string;
  formula?: string;
  overview?: string;
}

function normalizeCodeLine(line: string) {
  return line.replace(/\s+/g, ' ').trim();
}

function inlineCode(text: string) {
  return text.split(/(`[^`]+`)/g).map((part, index) => {
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={`${part}-${index}`}>{part.slice(1, -1)}</code>;
    }

    return part;
  });
}

function formulaForSolutionLine(line: string) {
  const normalized = normalizeCodeLine(line);

  if (/torch\.(amax|max)\(/.test(normalized)) {
    return String.raw`m_i = \max_j z_{ij},\qquad z'_{ij} = z_{ij} - m_i`;
  }

  if (/torch\.exp\(/.test(normalized)) {
    return String.raw`w_{ij} = \exp(z'_{ij}),\qquad z'_{ij} \leq 0`;
  }

  if (/normalizer|normalizers|sum\(/i.test(normalized) && /exp|exponent/i.test(normalized)) {
    return String.raw`Z_i = \sum_j \exp(z'_{ij})`;
  }

  if (/torch\.log\(|log\(/.test(normalized) && /loss|normalizer|target|logit/i.test(normalized)) {
    return String.raw`\ell_i = \log Z_i - z'_{i,y_i} = -\log p_{i,y_i}`;
  }

  if (/torch\.mean\(|mean\(/.test(normalized)) {
    return String.raw`L = \frac{1}{N}\sum_i \ell_i`;
  }

  if (/intersection|inter_area/.test(normalized) && /union|area/.test(normalized)) {
    return String.raw`\operatorname{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|}`;
  }

  if (/union\s*=/.test(normalized)) {
    return String.raw`|A \cup B| = |A| + |B| - |A \cap B|`;
  }

  if (/(?:@\s*torch\.transpose|squared_distances|deltas \* deltas)/.test(normalized)) {
    return String.raw`\lVert x_i-y_j\rVert^2 = \lVert x_i\rVert^2 + \lVert y_j\rVert^2 - 2x_i\cdot y_j`;
  }

  if (/torch\.arg(?:max|min)\(/.test(normalized)) {
    return normalized.includes('argmax')
      ? String.raw`j^* = \operatorname*{arg\,max}_j\; s_j`
      : String.raw`j^* = \operatorname*{arg\,min}_j\; d_j^2`;
  }

  if (/torch\.where\(/.test(normalized) && /delta|magnitude|error/.test(normalized)) {
    return String.raw`H_\delta(e)=\begin{cases}\frac{1}{2}e^2,& |e|\leq\delta\\ \delta(|e|-\frac{1}{2}\delta),& |e|>\delta\end{cases}`;
  }

  if (/matmul\(attention|attention\s*=/.test(normalized)) {
    return String.raw`\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\mathsf{T}}{\sqrt{d_h}}\right)V`;
  }

  if (/unsqueeze|None|reshape|view\(|transpose|permute/.test(normalized)) {
    return 'Broadcasting creates one tensor axis for each pair before the reduction.';
  }

  return undefined;
}

function getCommentGroups(solutionCode: string) {
  const lines = solutionCode.split('\n');
  const groups: Array<{ comments: string[]; code: string }> = [];

  for (let index = 0; index < lines.length; index += 1) {
    const trimmed = lines[index].trim();
    if (!trimmed.startsWith('#') || /^#\s*(TODO|Original placeholder)/i.test(trimmed)) {
      continue;
    }

    const comments: string[] = [];
    while (index < lines.length) {
      const comment = lines[index].trim();
      if (!comment.startsWith('#') || /^#\s*(TODO|Original placeholder)/i.test(comment)) {
        break;
      }

      comments.push(comment.replace(/^#\s?/, ''));
      index += 1;
    }

    while (index < lines.length && !lines[index].trim()) {
      index += 1;
    }

    if (comments.length > 0 && index < lines.length) {
      groups.push({ comments, code: lines[index] });
    }

    index -= 1;
  }

  return groups;
}

function findReferenceLineNumber(code: string, sourceLine: string, startAt: number) {
  const lines = code.split('\n');
  const markerIndex = lines.findIndex((line) => line.trim() === '# Reference solution');
  const source = normalizeCodeLine(sourceLine);
  const firstReferenceLine = markerIndex >= 0 ? markerIndex + 1 : 0;

  for (let index = Math.max(firstReferenceLine, startAt); index < lines.length; index += 1) {
    if (normalizeCodeLine(lines[index]) === source) {
      return index + 1;
    }
  }

  // Compacting turns simple if/raise guards into assert statements. If an
  // exact match is unavailable, keep the annotation attached to the next
  // executable reference line instead of dropping the explanation.
  for (let index = Math.max(firstReferenceLine, startAt); index < lines.length; index += 1) {
    const candidate = normalizeCodeLine(lines[index]);
    const meaningfulWords = source.split(/\W+/).filter((word) => word.length > 3);
    if (meaningfulWords.length > 0 && meaningfulWords.filter((word) => candidate.includes(word)).length >= 2) {
      return index + 1;
    }
  }

  return Math.min(Math.max(firstReferenceLine + 1, startAt + 1), lines.length);
}

function getSolutionWalkthroughAnnotations(problem: CodePracticeProblem, code: string) {
  if (!code.includes('# Reference solution')) {
    return [];
  }

  const lines = code.split('\n');
  const groups = getCommentGroups(problem.walkthroughCode ?? problem.solutionCode);
  let searchFrom = lines.findIndex((line) => line.trim() === '# Reference solution') + 1;
  const annotations: SolutionWalkthroughAnnotation[] = [];

  for (const group of groups) {
    const lineNumber = findReferenceLineNumber(code, group.code, Math.max(searchFrom, 0));
    const line = lines[lineNumber - 1] ?? group.code;
    const explanation = group.comments.join(' ');
    const previous = annotations.at(-1);

    if (previous?.lineNumber === lineNumber) {
      previous.explanation = `${previous.explanation} ${explanation}`;
      continue;
    }

    annotations.push({
      lineNumber,
      code: line.trim(),
      explanation,
      formula: formulaForSolutionLine(line),
      overview: annotations.length === 0 ? problem.solutionNotes.join(' ') : undefined,
    });
    searchFrom = lineNumber;
  }

  if (annotations.length === 0) {
    const firstLine = lines.findIndex(
      (line, index) => index > searchFrom && line.trim() && !line.trim().startsWith('#'),
    );
    if (firstLine >= 0) {
      annotations.push({
        lineNumber: firstLine + 1,
        code: lines[firstLine].trim(),
        explanation: problem.solutionNotes.join(' '),
        formula: formulaForSolutionLine(lines[firstLine]),
      });
    }
  }

  return annotations;
}

class SolutionLineMarker extends GutterMarker {
  constructor(
    private readonly lineNumber: number,
    private readonly isOpen: boolean,
    private readonly onToggle: (lineNumber: number) => void,
  ) {
    super();
  }

  eq(other: GutterMarker) {
    return other instanceof SolutionLineMarker && other.lineNumber === this.lineNumber && other.isOpen === this.isOpen;
  }

  toDOM() {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `cm-solution-line-toggle${this.isOpen ? ' is-open' : ''}`;
    button.textContent = this.isOpen ? '⌄' : '›';
    button.setAttribute('aria-label', `${this.isOpen ? 'Collapse' : 'Explain'} solution line ${this.lineNumber}`);
    button.setAttribute('aria-expanded', String(this.isOpen));
    button.addEventListener('mousedown', (event) => event.preventDefault());
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      this.onToggle(this.lineNumber);
    });
    return button;
  }
}

function createSolutionWalkthroughGutter(
  annotations: readonly SolutionWalkthroughAnnotation[],
  openLine: number | null,
  onToggle: (lineNumber: number) => void,
) {
  return gutter({
    class: 'cm-solution-walkthrough-gutter',
    markers: (view) => {
      const builder = new RangeSetBuilder<GutterMarker>();
      for (const annotation of annotations) {
        if (annotation.lineNumber > view.state.doc.lines) {
          continue;
        }

        const line = view.state.doc.line(annotation.lineNumber);
        builder.add(
          line.from,
          line.from,
          new SolutionLineMarker(annotation.lineNumber, annotation.lineNumber === openLine, onToggle),
        );
      }
      return builder.finish();
    },
  });
}

export default function CodePracticeLab({ problem }: CodePracticeLabProps) {
  const containerRef = useRef<HTMLElement | null>(null);
  const walkthroughPanelRef = useRef<HTMLElement | null>(null);
  const runtimeRef = useRef<PyodideRuntime | null>(null);
  const loadingRef = useRef(false);
  const isRunningRef = useRef(false);
  const runHandlerRef = useRef<() => void>(() => {});
  const [code, setCode] = useState(() => removeTodoCommentBlocks(problem.starterCode));
  const [output, setOutput] = useState('');
  const [errorOutput, setErrorOutput] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState('Loading Python...');
  const [isRunning, setIsRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [openWalkthroughLine, setOpenWalkthroughLine] = useState<number | null>(null);
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
    () => [...codeEditorExtensions, runShortcutExtension],
    [runShortcutExtension],
  );

  useEffect(() => {
    setCode(removeTodoCommentBlocks(problem.starterCode));
    setOutput('');
    setErrorOutput('');
    setHasRun(false);
    setOpenWalkthroughLine(null);
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
  const isReferenceLoaded = code.includes('# Reference solution');
  const walkthroughAnnotations = useMemo(
    () => getSolutionWalkthroughAnnotations(problem, code),
    [problem, code],
  );
  const walkthroughGutter = useMemo(
    () =>
      createSolutionWalkthroughGutter(
        walkthroughAnnotations,
        openWalkthroughLine,
        (lineNumber) => setOpenWalkthroughLine((current) => (current === lineNumber ? null : lineNumber)),
      ),
    [openWalkthroughLine, walkthroughAnnotations],
  );
  const hasExecutionResult = hasRun || Boolean(errorOutput);
  const activeWalkthrough = walkthroughAnnotations.find(
    (annotation) => annotation.lineNumber === openWalkthroughLine,
  );

  useEffect(() => {
    const panel = walkthroughPanelRef.current;
    if (!panel || typeof window === 'undefined') {
      return;
    }

    const renderMath = () => {
      const renderer = (
        window as Window & {
          renderMathInElement?: (element: HTMLElement, options: Record<string, unknown>) => void;
        }
      ).renderMathInElement;

      renderer?.(panel, {
        delimiters: [{ left: '\\(', right: '\\)', display: false }],
        ignoredTags: ['script', 'style', 'textarea', 'pre', 'code'],
        throwOnError: false,
      });
    };

    renderMath();
    window.addEventListener('load', renderMath, { once: true });
    return () => window.removeEventListener('load', renderMath);
  }, [activeWalkthrough?.lineNumber, activeWalkthrough?.formula]);

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
    setCode(removeTodoCommentBlocks(problem.starterCode));
    setOutput('');
    setErrorOutput('');
    setHasRun(false);
    setOpenWalkthroughLine(null);
  }

  function handleLoadSolution() {
    setCode((currentCode) => augmentCodeWithSolution(problem, currentCode));
    setOutput('');
    setErrorOutput('');
    setHasRun(false);
    setOpenWalkthroughLine(null);
  }

  // The CodeMirror keymap is created once, so route it through a ref to the
  // current controlled editor state instead of rebuilding the editor on each keystroke.
  runHandlerRef.current = () => {
    void handleRun();
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
          <div className="code-practice-lab__workspace-identity">
            <p
              className={`code-practice-lab__status code-practice-lab__status--${status}`}
              aria-live="polite"
            >
              {statusMessage}
            </p>
          </div>
          <div className="code-practice-lab__workspace-controls">
            <button
              className="code-practice-lab__button code-practice-lab__button--solution"
              type="button"
              onClick={handleLoadSolution}
            >
              Solution
            </button>
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
        </header>

        <div className={`code-practice-lab__editor-layout${isReferenceLoaded ? '' : ' code-practice-lab__editor-layout--single'}`}>
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
                extensions={isReferenceLoaded ? [...editorExtensions, walkthroughGutter] : editorExtensions}
                theme={editorThemeExtension}
                height="100%"
                editable
                indentWithTab={false}
                value={code}
                onChange={(value) => setCode(value)}
              />
            </div>
          </div>

          {isReferenceLoaded && <aside
            className="code-practice-lab__annotations"
            aria-labelledby={`${problem.id}-annotations-title`}
          >
            <div className="code-practice-lab__annotations-heading">
              <p className="code-practice-lab__section-label">Walkthrough</p>
              <h2 id={`${problem.id}-annotations-title`}>What this solution line is doing</h2>
            </div>
            {isReferenceLoaded && !activeWalkthrough && (
              <p className="code-practice-lab__annotations-empty">
                Select an annotated line to read its reasoning without leaving the code.
              </p>
            )}
            {isReferenceLoaded && activeWalkthrough && (
              <section
                className="code-practice-lab__walkthrough-panel"
                ref={walkthroughPanelRef}
                aria-label={`Walkthrough for solution line ${activeWalkthrough.lineNumber}`}
              >
                <div className="code-practice-lab__walkthrough-panel-heading">
                  <p className="code-practice-lab__section-label">
                    Line {String(activeWalkthrough.lineNumber).padStart(2, '0')}
                  </p>
                  <p>Why this line matters</p>
                </div>
                <pre className="code-practice-lab__walkthrough-code"><code>{activeWalkthrough.code}</code></pre>
                {activeWalkthrough.formula && (
                  <div className="code-practice-lab__walkthrough-formula">
                    <span>Equation / shape intuition</span>
                    <div className="code-practice-lab__math" aria-label="Equation">
                      {`\\(${activeWalkthrough.formula}\\)`}
                    </div>
                  </div>
                )}
                <p className="code-practice-lab__walkthrough-explanation">
                  {inlineCode(activeWalkthrough.explanation)}
                </p>
                {activeWalkthrough.overview && (
                  <p className="code-practice-lab__walkthrough-overview">
                    <strong>Big picture:</strong> {inlineCode(activeWalkthrough.overview)}
                  </p>
                )}
              </section>
            )}
          </aside>}
        </div>

        {hasExecutionResult && (
          <div className="code-practice-lab__output" aria-live="polite">
            <div>
              <p>{errorOutput ? 'Errors' : 'Output'}</p>
              <pre>{errorOutput || output || 'Program finished with no output.'}</pre>
            </div>
          </div>
        )}
      </article>
    </section>
  );
}
