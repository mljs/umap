/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: the private fields of the upstream UMAP class gathered
 * into one explicit state object, so the transform and the serialization can
 * be plain functions in their own modules instead of methods, and so the class
 * itself stays under the 250 line file limit. Resolving the parameters once at
 * construction replaces upstream's `setParam` loop, which assigned through a
 * string index and could not be typed. No value and no order changed.
 *
 * The random projection forest is held here as the TRANSFORM forest, built on
 * demand from `transformSeed`, where upstream kept the one the fit happened to
 * build. A fitted model and a model reloaded from JSON therefore search
 * against identical trees, which is what makes a projection reproducible
 * across a save and a reload.
 */

import {
  DEFAULT_PARAMETERS,
  DEFAULT_SUPERVISED_PARAMETERS,
} from './defaults.ts';
import type { MetricName } from './metrics.ts';
import { resolveMetric } from './metrics.ts';
import type {
  InitFromRandomFn,
  InitFromTreeFn,
  SearchFn,
} from './nn/search.ts';
import type { FlatTree } from './nn/tree.ts';
import { OptimizationState } from './optimize/state.ts';
import type {
  InitMethod,
  UMAPParameters,
  UMAPSupervisedParams,
} from './parameters.ts';
import { makeSeededRandom } from './random.ts';
import type { SparseMatrix } from './sparse/index.ts';
import type { DistanceFn, RandomFn, Vectors } from './types.ts';

/**
 * Every parameter of a projection, with the defaults already applied.
 */
export interface ResolvedParameters {
  /** Metric between two data points. */
  distanceFunction: DistanceFn;
  /**
   * Name of `distanceFunction`, or `'custom'` when it is a function the caller
   * supplied without naming it. A `'custom'` metric cannot be serialized.
   */
  metric: MetricName | 'custom';
  /** How the initial embedding is built, before the optimization runs. */
  init: InitMethod;
  /** Initial learning rate of the embedding optimization. */
  learningRate: number;
  /** Number of neighbors assumed to be locally connected. */
  localConnectivity: number;
  /** Effective minimum distance between two embedded points. */
  minimumDistance: number;
  /** Dimension of the space the data is embedded into. */
  numberOfComponents: number;
  /** Number of optimization epochs, `0` meaning "pick from the data size". */
  numberOfEpochs: number;
  /** Size of the local neighborhood used to approximate the manifold. */
  numberOfNeighbors: number;
  /** Number of negative samples drawn per positive sample. */
  negativeSampleRate: number;
  /** Random number generator returning a float in [0, 1). */
  random: RandomFn;
  /** Seed `random` was built from, unless the caller supplied `random`. */
  seed: number;
  /** Weight applied to the negative samples. */
  repulsionStrength: number;
  /** Interpolation between the fuzzy union (1) and intersection (0). */
  setOperationMixRatio: number;
  /** Effective scale of the embedded points. */
  spread: number;
  /** How aggressively `transform` searches for nearest neighbors. */
  transformQueueSize: number;
  /** Seed of the random source a projection of new points draws from. */
  transformSeed: number;
}

/**
 * Every supervised projection parameter, with the defaults already applied.
 */
export type ResolvedSupervisedParameters = Required<UMAPSupervisedParams>;

/**
 * The whole mutable state of a `UMAP` instance.
 *
 * It is reachable through `umap.internals` so that the transform and the
 * serialization can live in their own modules; it is not part of the stable
 * public API and its shape may change between minor versions.
 */
export interface UMAPInternals {
  /** Resolved projection parameters. */
  params: ResolvedParameters;
  /** Resolved supervised projection parameters. */
  supervised: ResolvedSupervisedParameters;
  /** The data the model was fitted on. */
  X: Vectors;
  /** Target value of each fitted point, when the projection is supervised. */
  Y: number[] | undefined;
  /** Nearest neighbor index of each fitted point. */
  knnIndices: number[][] | undefined;
  /** Distance to each of those nearest neighbors. */
  knnDistances: number[][] | undefined;
  /** Fuzzy graph of the fitted data. */
  graph: SparseMatrix | undefined;
  /** Symmetrized neighbor graph the transform search walks. */
  searchGraph: SparseMatrix | undefined;
  /**
   * Random projection forest seeding the transform search, built on first use
   * from `params.transformSeed` and cached with the seed it was built from, so
   * a fitted model and a reloaded one seed their searches from the very same
   * trees.
   */
  transformForest: FlatTree[] | undefined;
  /** Seed `transformForest` was built from. */
  transformForestSeed: number | undefined;
  /** Seeds a neighbor heap with randomly drawn data points. */
  initFromRandom: InitFromRandomFn | undefined;
  /** Seeds a neighbor heap with the leaves of a random projection tree. */
  initFromTree: InitFromTreeFn | undefined;
  /** Refines a seeded neighbor heap over the search graph. */
  search: SearchFn | undefined;
  /** The projected embedding, one row per fitted point. */
  embedding: Vectors;
  /** State carried between two steps of the optimization. */
  optimizationState: OptimizationState;
  /** Whether `initializeFit` has run, or a model has been loaded. */
  isInitialized: boolean;
}

/**
 * Builds the state of a new `UMAP` instance, applying the default of
 * every parameter that was left out.
 * @param params - The parameters of the projection.
 * @returns The initial state.
 */
export function createInternals(params: UMAPParameters = {}): UMAPInternals {
  return {
    params: resolveParameters(params),
    supervised: { ...DEFAULT_SUPERVISED_PARAMETERS },
    X: [],
    Y: undefined,
    knnIndices: undefined,
    knnDistances: undefined,
    graph: undefined,
    searchGraph: undefined,
    transformForest: undefined,
    transformForestSeed: undefined,
    initFromRandom: undefined,
    initFromTree: undefined,
    search: undefined,
    embedding: [],
    optimizationState: new OptimizationState(),
    isInitialized: false,
  };
}

/**
 * Applies the parameter defaults and resolves the metric.
 *
 * A `distanceFunction` always wins over a `metric` name for the function
 * actually used; the name is only kept so a fitted model can name its metric
 * when it is serialized, and falls back to `'custom'` for an unnamed function.
 * @param params - The parameters of the projection.
 * @returns The resolved parameters.
 */
export function resolveParameters(params: UMAPParameters): ResolvedParameters {
  const seed = params.seed ?? DEFAULT_PARAMETERS.seed;
  const distanceFunction = params.distanceFunction;
  const metric = params.metric;

  let resolvedDistanceFunction = DEFAULT_PARAMETERS.distanceFunction;
  if (distanceFunction !== undefined) {
    resolvedDistanceFunction = distanceFunction;
  } else if (metric !== undefined) {
    resolvedDistanceFunction = resolveMetric(metric);
  }

  let resolvedMetric: MetricName | 'custom' = DEFAULT_PARAMETERS.metric;
  if (metric !== undefined) {
    resolvedMetric = metric;
  } else if (distanceFunction !== undefined) {
    resolvedMetric = 'custom';
  }

  return {
    distanceFunction: resolvedDistanceFunction,
    metric: resolvedMetric,
    init: params.init ?? DEFAULT_PARAMETERS.init,
    learningRate: params.learningRate ?? DEFAULT_PARAMETERS.learningRate,
    localConnectivity:
      params.localConnectivity ?? DEFAULT_PARAMETERS.localConnectivity,
    minimumDistance:
      params.minimumDistance ?? DEFAULT_PARAMETERS.minimumDistance,
    numberOfComponents:
      params.numberOfComponents ?? DEFAULT_PARAMETERS.numberOfComponents,
    numberOfEpochs: params.numberOfEpochs ?? DEFAULT_PARAMETERS.numberOfEpochs,
    numberOfNeighbors:
      params.numberOfNeighbors ?? DEFAULT_PARAMETERS.numberOfNeighbors,
    negativeSampleRate:
      params.negativeSampleRate ?? DEFAULT_PARAMETERS.negativeSampleRate,
    random: params.random ?? makeSeededRandom(seed),
    seed,
    repulsionStrength:
      params.repulsionStrength ?? DEFAULT_PARAMETERS.repulsionStrength,
    setOperationMixRatio:
      params.setOperationMixRatio ?? DEFAULT_PARAMETERS.setOperationMixRatio,
    spread: params.spread ?? DEFAULT_PARAMETERS.spread,
    transformQueueSize:
      params.transformQueueSize ?? DEFAULT_PARAMETERS.transformQueueSize,
    transformSeed: params.transformSeed ?? DEFAULT_PARAMETERS.transformSeed,
  };
}
