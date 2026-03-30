export interface PlaygroundSample {
  label: string;
  code: string;
  description?: string;
}

export interface WalkthroughStep {
  label: string;
  lineHint?: number;
  variables: Record<string, string>;
  output?: string;
}

export interface PythonPlaygroundProps {
  title: string;
  initialCode: string;
  samples: PlaygroundSample[];
  walkthroughSteps?: WalkthroughStep[];
  notes?: string;
}
