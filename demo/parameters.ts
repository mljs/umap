/*
 * The parameters the playground exposes, and how each is edited. Only
 * `numberOfComponents` is fixed, at two, because the plot is a plane.
 */

import type { InitMethod, MetricName } from '../src/index.ts';

/** The editable parameters of a run. */
export interface DemoParameters {
  /** Size of the local neighbourhood. */
  numberOfNeighbors: number;
  /** Minimum separation of two embedded points. */
  minimumDistance: number;
  /** Scale of the embedded points. */
  spread: number;
  /** Epochs to run; `0` picks 200 to 500 from the dataset size. */
  numberOfEpochs: number;
  /** Seed of the default random source. */
  seed: number;
  /** Initial learning rate. */
  learningRate: number;
  /** Negative samples per positive edge. */
  negativeSampleRate: number;
  /** Weight of the negative samples. */
  repulsionStrength: number;
  /** Neighbours assumed to be locally connected. */
  localConnectivity: number;
  /** Blend of the fuzzy union and intersection. */
  setOperationMixRatio: number;
  /** Distance between two points. */
  metric: MetricName;
  /** How the embedding is initialized. */
  init: InitMethod;
}

/** Keys of {@link DemoParameters} that hold a number. */
export type NumericParameter = {
  [K in keyof DemoParameters]: DemoParameters[K] extends number ? K : never;
}[keyof DemoParameters];

/** What one numeric field of the form allows. */
export interface NumberField {
  /** Parameter the field edits. */
  key: NumericParameter;
  /** Smallest value the field accepts. */
  min: number;
  /** Increment of the field. */
  step: number;
  /** One line on what the parameter does. */
  help: string;
}

/** The parameters a fresh page starts from. */
export const DEFAULT_PARAMETERS: DemoParameters = {
  numberOfNeighbors: 15,
  minimumDistance: 0.1,
  spread: 1,
  numberOfEpochs: 0,
  seed: 42,
  learningRate: 1,
  negativeSampleRate: 5,
  repulsionStrength: 1,
  localConnectivity: 1,
  setOperationMixRatio: 1,
  metric: 'euclidean',
  init: 'random',
};

/** The numeric fields of the form, in the order it shows them. */
export const NUMBER_FIELDS: readonly NumberField[] = [
  {
    key: 'numberOfNeighbors',
    min: 2,
    step: 1,
    help: 'Small keeps local detail, large pulls the global shape together.',
  },
  {
    key: 'minimumDistance',
    min: 0,
    step: 0.05,
    help: 'How tightly points are allowed to clump.',
  },
  { key: 'spread', min: 0.1, step: 0.1, help: 'Scale of the embedding.' },
  {
    key: 'numberOfEpochs',
    min: 0,
    step: 50,
    help: '0 picks 200 to 500 from the dataset size.',
  },
  {
    key: 'seed',
    min: 0,
    step: 1,
    help: 'Same seed, same embedding; change it to see the run-to-run spread.',
  },
  { key: 'learningRate', min: 0.05, step: 0.05, help: 'Initial SGD rate.' },
  {
    key: 'negativeSampleRate',
    min: 1,
    step: 1,
    help: 'Repulsive samples drawn per edge.',
  },
  {
    key: 'repulsionStrength',
    min: 0,
    step: 0.5,
    help: 'Weight of those repulsive samples.',
  },
  {
    key: 'localConnectivity',
    min: 1,
    step: 1,
    help: 'Neighbours assumed connected at every point.',
  },
  {
    key: 'setOperationMixRatio',
    min: 0,
    step: 0.1,
    help: '1 is the fuzzy union, 0 the intersection.',
  },
];

/** The metrics the picker offers, as its options. */
export const METRIC_OPTIONS: ReadonlyArray<{
  value: MetricName;
  label: string;
}> = [
  { value: 'euclidean', label: 'euclidean' },
  { value: 'cosine', label: 'cosine' },
];

/** The initializations the picker offers, as its options. */
export const INIT_OPTIONS: ReadonlyArray<{
  value: InitMethod;
  label: string;
}> = [
  { value: 'random', label: 'random' },
  { value: 'spectral', label: 'spectral' },
];
