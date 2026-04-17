import type { PyodideRuntime } from './pyodide-loader';

const EXECUTION_PREFIX = `
import contextlib
import io
import traceback
`;

const PYTHON_PACKAGE_PATTERNS = [
  {
    name: 'numpy',
    pattern: /\b(?:import\s+numpy|from\s+numpy\s+import)\b/,
  },
] as const;

function escapePythonTripleQuotedString(source: string) {
  return source.replace(/\\/g, '\\\\').replace(/"""/g, '\\"""');
}

async function ensurePythonPackages(
  runtime: PyodideRuntime,
  code: string,
  explicitPackages: readonly string[] = [],
) {
  if (typeof runtime.loadPackage !== 'function') {
    return;
  }

  const packageNames = new Set(explicitPackages);
  for (const packageMatcher of PYTHON_PACKAGE_PATTERNS) {
    if (packageMatcher.pattern.test(code)) {
      packageNames.add(packageMatcher.name);
    }
  }

  if (packageNames.size === 0) {
    return;
  }

  await runtime.loadPackage(Array.from(packageNames));
}

export async function runPythonSnippet(
  runtime: PyodideRuntime,
  code: string,
  packages: readonly string[] = [],
) {
  await ensurePythonPackages(runtime, code, packages);

  const escapedCode = escapePythonTripleQuotedString(code);
  const result = await runtime.runPythonAsync(`
${EXECUTION_PREFIX}
_stdout_buffer = io.StringIO()
_stderr_buffer = io.StringIO()
_execution_result = None

with contextlib.redirect_stdout(_stdout_buffer), contextlib.redirect_stderr(_stderr_buffer):
    try:
        exec("""${escapedCode}""", {})
    except Exception:
        traceback.print_exc()

(_stdout_buffer.getvalue(), _stderr_buffer.getvalue())
`);

  const normalizedResult =
    typeof result === 'object' &&
    result !== null &&
    'toJs' in result &&
    typeof result.toJs === 'function'
      ? result.toJs()
      : result;

  if (!Array.isArray(normalizedResult)) {
    return { stdout: '', stderr: 'Unexpected result returned from Python runtime.' };
  }

  return {
    stdout: String(normalizedResult[0] ?? ''),
    stderr: String(normalizedResult[1] ?? ''),
  };
}
