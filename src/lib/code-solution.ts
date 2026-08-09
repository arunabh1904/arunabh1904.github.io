import type { CodePracticeProblem } from './code-practice';

const REFERENCE_MARKER = '# Reference solution loaded:';

interface PythonDefinition {
  kind: 'class' | 'function';
  name: string;
  indent: number;
  parentClasses: readonly string[];
  start: number;
  headerEnd: number;
  end: number;
}

function getIndent(line: string) {
  return line.length - line.trimStart().length;
}

function findHeaderEnd(lines: readonly string[], start: number) {
  for (let index = start; index < lines.length; index += 1) {
    if (lines[index].trimEnd().endsWith(':')) {
      return index;
    }
  }

  return start;
}

function findBlockEnd(lines: readonly string[], headerEnd: number, indent: number) {
  for (let index = headerEnd + 1; index < lines.length; index += 1) {
    if (lines[index].trim() && getIndent(lines[index]) <= indent) {
      return index;
    }
  }

  return lines.length;
}

function getPythonDefinitions(code: string): PythonDefinition[] {
  const lines = code.split('\n');
  const definitions: Omit<PythonDefinition, 'parentClasses'>[] = [];

  lines.forEach((line, start) => {
    const match = line.match(/^(\s*)(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)\b/);
    if (!match) {
      return;
    }

    const indent = match[1].length;
    const headerEnd = findHeaderEnd(lines, start);
    definitions.push({
      kind: match[2] === 'class' ? 'class' : 'function',
      name: match[3],
      indent,
      start,
      headerEnd,
      end: findBlockEnd(lines, headerEnd, indent),
    });
  });

  return definitions.map((definition) => ({
    ...definition,
    parentClasses: definitions
      .filter(
        (candidate) =>
          candidate.kind === 'class' &&
          candidate.start < definition.start &&
          candidate.end >= definition.end &&
          candidate.indent < definition.indent,
      )
      .sort((left, right) => left.indent - right.indent)
      .map((candidate) => candidate.name),
  }));
}

function definitionKey(definition: Pick<PythonDefinition, 'kind' | 'name' | 'parentClasses'>) {
  return `${definition.parentClasses.join('.')}/${definition.kind}/${definition.name}`;
}

function findContainingFunction(definitions: readonly PythonDefinition[], lineIndex: number) {
  return definitions
    .filter(
      (definition) =>
        definition.kind === 'function' &&
        definition.start <= lineIndex &&
        lineIndex < definition.end,
    )
    .sort((left, right) => right.indent - left.indent)
    .at(0);
}

function reindentBody(
  lines: readonly string[],
  sourceDefinition: PythonDefinition,
  replacementIndent: string,
) {
  const sourceBodyIndent = sourceDefinition.indent + 4;

  return lines.map((line) => {
    if (!line.trim()) {
      return '';
    }

    return `${replacementIndent}${line.slice(sourceBodyIndent)}`;
  });
}

function getMissingTopLevelImports(solutionCode: string, code: string) {
  const presentLines = new Set(code.split('\n').map((line) => line.trim()));

  return solutionCode
    .split('\n')
    .filter(
      (line) =>
        /^(?:from\s+[\w.]+\s+import\s+.+|import\s+.+)$/.test(line) &&
        !line.startsWith('from __future__') &&
        !presentLines.has(line.trim()),
    );
}

function insertSupportingDefinitions(
  code: string,
  solutionCode: string,
  solutionDefinitions: readonly PythonDefinition[],
) {
  const currentDefinitions = getPythonDefinitions(code);
  const currentTopLevelKeys = new Set(
    currentDefinitions
      .filter((definition) => definition.indent === 0)
      .map((definition) => definitionKey(definition)),
  );
  const helperDefinitions = solutionDefinitions.filter(
    (definition) =>
      definition.kind === 'function' &&
      definition.indent === 0 &&
      !currentTopLevelKeys.has(definitionKey(definition)),
  );
  const missingImports = getMissingTopLevelImports(solutionCode, code);

  if (helperDefinitions.length === 0 && missingImports.length === 0) {
    return code;
  }

  const firstAnnotatedDefinition = currentDefinitions
    .filter(
      (definition) =>
        definition.indent === 0 &&
        code
          .split('\n')
          .slice(definition.start, definition.end)
          .some((line) => line.includes(REFERENCE_MARKER)),
    )
    .sort((left, right) => left.start - right.start)
    .at(0);
  const insertionIndex =
    firstAnnotatedDefinition?.start ??
    currentDefinitions.filter((definition) => definition.indent === 0).at(0)?.start ??
    code.split('\n').length;
  const solutionLines = solutionCode.split('\n');
  const additions = [
    ...missingImports,
    ...(missingImports.length > 0 && helperDefinitions.length > 0 ? [''] : []),
    ...helperDefinitions.flatMap((definition, index) => [
      ...solutionLines.slice(definition.start, definition.end),
      ...(index === helperDefinitions.length - 1 ? [] : ['']),
    ]),
  ];
  const lines = code.split('\n');

  lines.splice(insertionIndex, 0, ...additions, ...(additions.length > 0 ? [''] : []));
  return lines.join('\n');
}

/**
 * Keeps the starter scaffold intact while replacing each explicit exercise
 * placeholder with the matching annotated reference body. The original raise
 * remains as a comment so readers can still see the exact starting point.
 */
export function augmentCodeWithSolution(
  problem: Pick<CodePracticeProblem, 'solutionCode' | 'solutionNotes' | 'starterCode'>,
  currentCode = problem.starterCode,
) {
  if (currentCode.includes(REFERENCE_MARKER)) {
    return currentCode;
  }

  const solutionDefinitions = getPythonDefinitions(problem.solutionCode);
  const solutionByKey = new Map(
    solutionDefinitions.map((definition) => [definitionKey(definition), definition]),
  );
  const starterDefinitions = getPythonDefinitions(currentCode);
  const lines = currentCode.split('\n');
  let didInsertReference = false;

  const annotatedLines = lines.flatMap((line, lineIndex) => {
    const placeholder = line.match(/^(\s*)raise NotImplementedError\("Implement ([A-Za-z_]\w*)"\)\s*$/);
    if (!placeholder) {
      return [line];
    }

    const starterDefinition = findContainingFunction(starterDefinitions, lineIndex);
    if (!starterDefinition || starterDefinition.name !== placeholder[2]) {
      return [line];
    }

    const solutionDefinition = solutionByKey.get(definitionKey(starterDefinition));
    if (!solutionDefinition) {
      return [line];
    }

    const indentation = placeholder[1];
    const solutionLines = problem.solutionCode.split('\n');
    const body = reindentBody(
      solutionLines.slice(solutionDefinition.headerEnd + 1, solutionDefinition.end),
      solutionDefinition,
      indentation,
    );
    const introduction = didInsertReference
      ? []
      : [
          `${indentation}${REFERENCE_MARKER}`,
          ...problem.solutionNotes.map((note) => `${indentation}# ${note}`),
          `${indentation}# The TODO plan above stays in place; the annotated lines below implement it.`,
        ];

    didInsertReference = true;
    return [
      `${indentation}# Original placeholder: ${line.trim()}`,
      ...introduction,
      ...body,
    ];
  });

  if (!didInsertReference) {
    return currentCode;
  }

  return insertSupportingDefinitions(
    annotatedLines.join('\n'),
    problem.solutionCode,
    solutionDefinitions,
  );
}
