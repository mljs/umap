/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: the field initializers of the upstream UMAP class
 * gathered into one table of defaults, and split out of ./parameters.ts —
 * which is then types only — to respect the 250 line file limit. Every value
 * is the one upstream initializes its field with, except `seed` and
 * `transformSeed`, which upstream does not have.
 *
 * `random` is not in the table: its default is a generator seeded with `seed`,
 * and a generator is stateful, so one shared instance would make the second
 * model built from the defaults continue the stream of the first instead of
 * repeating it. ./internals.ts builds a fresh one per model.
 */

import { euclidean } from './metrics.ts';
import type { UMAPParameters, UMAPSupervisedParams } from './parameters.ts';
import { DEFAULT_SEED, DEFAULT_TRANSFORM_SEED } from './random.ts';

/**
 * The value every optional parameter of {@link UMAPParameters} falls back to,
 * except `random`.
 */
export const DEFAULT_PARAMETERS: Required<Omit<UMAPParameters, 'random'>> = {
  distanceFunction: euclidean,
  metric: 'euclidean',
  init: 'random',
  learningRate: 1,
  localConnectivity: 1,
  minimumDistance: 0.1,
  numberOfComponents: 2,
  numberOfEpochs: 0,
  numberOfNeighbors: 15,
  negativeSampleRate: 5,
  repulsionStrength: 1,
  seed: DEFAULT_SEED,
  setOperationMixRatio: 1,
  spread: 1,
  transformQueueSize: 4,
  transformSeed: DEFAULT_TRANSFORM_SEED,
};

/**
 * The value every optional parameter of {@link UMAPSupervisedParams} falls
 * back to.
 */
export const DEFAULT_SUPERVISED_PARAMETERS: Required<UMAPSupervisedParams> = {
  targetMetric: 'categorical',
  targetWeight: 0.5,
  targetNumberOfNeighbors: DEFAULT_PARAMETERS.numberOfNeighbors,
};
