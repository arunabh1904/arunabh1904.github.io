import type { PyodideRuntime } from './pyodide-loader';
import { TORCH_COMPAT_PACKAGE, TORCH_COMPAT_SOURCE } from './torch-compat';

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

const TORCH_IMPORT_PATTERN = /\b(?:import\s+torch|from\s+torch(?:\.|\s+import))\b/;

function escapePythonTripleQuotedString(source: string) {
  return source.replace(/\\/g, '\\\\').replace(/"""/g, '\\"""');
}

async function ensurePythonPackages(
  runtime: PyodideRuntime,
  code: string,
  explicitPackages: readonly string[] = [],
) {
  const usesTorch = explicitPackages.includes(TORCH_COMPAT_PACKAGE) || TORCH_IMPORT_PATTERN.test(code);

  if (typeof runtime.loadPackage !== 'function') {
    return usesTorch;
  }

  // `torch` is a local compatibility module, not a Pyodide package. It uses
  // NumPy underneath, so load that real Pyodide package before injecting it.
  const packageNames = new Set(explicitPackages.filter((packageName) => packageName !== TORCH_COMPAT_PACKAGE));
  if (usesTorch) {
    packageNames.add('numpy');
  }
  for (const packageMatcher of PYTHON_PACKAGE_PATTERNS) {
    if (packageMatcher.pattern.test(code)) {
      packageNames.add(packageMatcher.name);
    }
  }

  if (packageNames.size === 0) {
    return usesTorch;
  }

  await runtime.loadPackage(Array.from(packageNames));
  return usesTorch;
}

export async function runPythonSnippet(
  runtime: PyodideRuntime,
  code: string,
  packages: readonly string[] = [],
) {
  const usesTorch = await ensurePythonPackages(runtime, code, packages);

  const escapedCode = escapePythonTripleQuotedString(code);
  const result = await runtime.runPythonAsync(`
${EXECUTION_PREFIX}
${usesTorch ? TORCH_COMPAT_SOURCE : ''}
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
