/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: model persistence, which upstream does not provide.
 * Only the rebuilt pieces come from the port; the format itself is new.
 *
 * What the format stores, and why:
 *
 * - The nearest neighbor indices and distances, NOT the search graph. That
 *   graph is exactly derivable from them by `makeSearchGraph`, which draws no
 *   random number and costs O(n * k), and it serializes 2.2 to 3.0 times
 *   larger.
 * - NOT the random projection forest: at 50 dimensions it is about twice the
 *   size of the data itself. `getTransformForest` builds one on first use
 *   from the data and `transformSeed` alone, so a reloaded model searches
 *   against exactly the trees the saved model searches against.
 * - `transformSeed`, which is what makes a projection reproducible: a model
 *   that dropped it would project new points differently from the model it
 *   was saved from as soon as the seed was not the default.
 * - `a` and `b`, the only optimization state fields carried from the fit into
 *   the transform, which nothing on the transform path reassigns: a model
 *   omitting them would silently fall back to the defaults for `spread = 1`
 *   and `minimumDistance = 0.1` and project against the wrong curve, with no
 *   error.
 */

import type { UMAPInternals } from './internals.ts';
import { makeSearchFns, makeSearchGraph } from './knn.ts';
import type { MetricName } from './metrics.ts';
import type { UMAPParameters } from './parameters.ts';
import { DEFAULT_TRANSFORM_SEED } from './random.ts';
import type { DistanceFn, RandomFn } from './types.ts';
import { UMAP } from './umap.ts';

/**
 * A fitted UMAP model, as a plain JSON-serializable object.
 */
export interface UMAPModel {
  /** Discriminator of the format. */
  name: 'UMAP';
  /** Version of the format. */
  version: 1;
  /** The data the model was fitted on, one array per point. */
  X: number[][];
  /** The fitted embedding, one array per point. */
  embedding: number[][];
  /** Nearest neighbor index of each fitted point, closest first. */
  knnIndices: number[][];
  /** Distance to each of those nearest neighbors. */
  knnDistances: number[][];
  /**
   * Name of the metric the model was fitted with, or `'custom'` when it was
   * fitted with an unnamed distance function. Loading a `'custom'` model
   * requires that function to be supplied again.
   */
  metric: MetricName | 'custom';
  /** First parameter of the fitted curve `1 / (1 + a * d^(2b))`. */
  a: number;
  /** Second parameter of the fitted curve `1 / (1 + a * d^(2b))`. */
  b: number;
  /** Size of the local neighborhood used to approximate the manifold. */
  numberOfNeighbors: number;
  /** Dimension of the space the data is embedded into. */
  numberOfComponents: number;
  /** Effective minimum distance between two embedded points. */
  minimumDistance: number;
  /** Effective scale of the embedded points. */
  spread: number;
  /** Number of neighbors assumed to be locally connected. */
  localConnectivity: number;
  /** Interpolation between the fuzzy union (1) and intersection (0). */
  setOperationMixRatio: number;
  /** Number of negative samples drawn per positive sample. */
  negativeSampleRate: number;
  /** Weight applied to the negative samples. */
  repulsionStrength: number;
  /** Initial learning rate of the embedding optimization. */
  learningRate: number;
  /** How aggressively `transform` searches for nearest neighbors. */
  transformQueueSize: number;
  /** Number of optimization epochs, `0` meaning "pick from the data size". */
  numberOfEpochs: number;
  /**
   * Seed of the random source a projection of new points draws from. A model
   * written before this field existed carries none and loads with the default.
   * @default 42
   */
  transformSeed?: number;
}

/**
 * The values a serialized model cannot carry.
 */
export interface FromJSONOptions {
  /**
   * The distance function to fit the model with again. It is required when the
   * model was fitted with a `'custom'` metric, and overrides the named metric
   * of the model otherwise.
   * @default the function named by `model.metric`
   */
  distanceFunction?: DistanceFn;
  /**
   * The pseudo-random number generator of the reloaded model. `transform` does
   * not use it — it draws from `transformSeed` instead — so it only matters
   * for a model that is going to be fitted again.
   * @default a generator seeded with the default `seed`
   */
  random?: RandomFn;
  /**
   * Seed of the random source projections draw from, overriding the model's.
   * @default the `transformSeed` of the model, or 42 when it carries none
   */
  transformSeed?: number;
}

/**
 * Serializes a fitted model to a plain object.
 *
 * The returned arrays are the ones the model holds, not copies, so stringify
 * the result before mutating the model any further.
 * @param internals - State of a fitted model.
 * @returns The serialized model.
 * @throws {Error} If the model has not been fitted.
 */
export function toJSON(internals: UMAPInternals): UMAPModel {
  const {
    X,
    embedding,
    knnIndices,
    knnDistances,
    params,
    isInitialized,
    optimizationState,
  } = internals;
  if (
    !isInitialized ||
    knnIndices === undefined ||
    knnDistances === undefined ||
    embedding.length === 0
  ) {
    throw new Error('A UMAP model can only be serialized once it is fitted.');
  }

  const { a, b } = optimizationState;

  return {
    name: 'UMAP',
    version: 1,
    X,
    embedding,
    knnIndices,
    knnDistances,
    metric: params.metric,
    a,
    b,
    numberOfNeighbors: params.numberOfNeighbors,
    numberOfComponents: params.numberOfComponents,
    minimumDistance: params.minimumDistance,
    spread: params.spread,
    localConnectivity: params.localConnectivity,
    setOperationMixRatio: params.setOperationMixRatio,
    negativeSampleRate: params.negativeSampleRate,
    repulsionStrength: params.repulsionStrength,
    learningRate: params.learningRate,
    transformQueueSize: params.transformQueueSize,
    numberOfEpochs: params.numberOfEpochs,
    transformSeed: params.transformSeed,
  };
}

/**
 * Rebuilds a model serialized by {@link toJSON}.
 *
 * The reloaded model projects new points exactly as the model it was saved
 * from does: the search graph is derived from the stored neighbors, and every
 * random draw of a projection, the forest included, comes from
 * `transformSeed`. The data and embedding arrays are adopted, not copied.
 * @param model - The serialized model.
 * @param options - The values a model cannot carry, if it needs any.
 * @returns The reloaded model, ready to `transform`.
 * @throws {RangeError} If `model` is not a UMAP model of a known version, or
 * if it names a metric that does not exist.
 * @throws {Error} If the model was fitted with a custom metric and no
 * `distanceFunction` is supplied to replace it.
 */
export function fromJSON(
  model: UMAPModel,
  options: FromJSONOptions = {},
): UMAP {
  if (model.name !== 'UMAP') {
    throw new RangeError('invalid model: expected a UMAP model');
  }
  const version: number = model.version;
  if (version !== 1) {
    throw new RangeError(`unsupported UMAP model version: ${version}`);
  }

  const {
    distanceFunction,
    random,
    transformSeed = model.transformSeed ?? DEFAULT_TRANSFORM_SEED,
  } = options;
  if (distanceFunction === undefined && model.metric === 'custom') {
    throw new Error(
      'This model was fitted with a custom metric: pass the same distanceFunction to load it. Substituting another metric would project new points to plausible but wrong positions.',
    );
  }

  const parameters: UMAPParameters = {
    numberOfNeighbors: model.numberOfNeighbors,
    numberOfComponents: model.numberOfComponents,
    minimumDistance: model.minimumDistance,
    spread: model.spread,
    localConnectivity: model.localConnectivity,
    setOperationMixRatio: model.setOperationMixRatio,
    negativeSampleRate: model.negativeSampleRate,
    repulsionStrength: model.repulsionStrength,
    learningRate: model.learningRate,
    transformQueueSize: model.transformQueueSize,
    numberOfEpochs: model.numberOfEpochs,
    transformSeed,
  };
  if (random !== undefined) {
    parameters.random = random;
  }
  if (model.metric !== 'custom') {
    parameters.metric = model.metric;
  }
  if (distanceFunction !== undefined) {
    parameters.distanceFunction = distanceFunction;
  }

  const umap = new UMAP(parameters);
  const internals = umap.internals;
  internals.X = model.X;
  internals.embedding = model.embedding;
  internals.knnIndices = model.knnIndices;
  internals.knnDistances = model.knnDistances;

  makeSearchFns(internals);
  internals.searchGraph = makeSearchGraph(
    model.X,
    model.knnIndices,
    model.knnDistances,
  );

  internals.optimizationState.a = model.a;
  internals.optimizationState.b = model.b;
  internals.isInitialized = true;

  return umap;
}
