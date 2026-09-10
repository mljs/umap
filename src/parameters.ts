/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: the parameter interfaces moved out of the UMAP module
 * into their own file, every optional property documents the default the
 * reference implementation initializes its field with, `TargetMetric` is a
 * string union instead of a `const enum` so it survives `isolatedModules`, and
 * a `metric` name was added next to `distanceFunction` so a fitted model can be
 * serialized without carrying a function around. An `init` parameter was added
 * for the spectral initialization, which upstream does not implement, a
 * `transformSeed` for the transform-scoped random source, and a `seed` for the
 * random source of a fit, which is seeded here where upstream leaves it to
 * `Math.random`. The table of defaults lives in ./defaults.ts, so this module
 * stays types only and under the 250 line file limit.
 */

import type { MetricName } from './metrics.ts';
import type { DistanceFn, RandomFn } from './types.ts';

/**
 * Metric used to measure the distance between two target values in a
 * supervised projection.
 */
export type TargetMetric = 'categorical' | 'l1' | 'l2';

/**
 * The supported target metrics, keyed by name.
 */
export const TargetMetric = {
  categorical: 'categorical',
  l1: 'l1',
  l2: 'l2',
} as const satisfies Record<TargetMetric, TargetMetric>;

/**
 * How the embedding is initialized before the optimization runs.
 */
export type InitMethod = 'random' | 'spectral';

/**
 * The supported initialization methods, keyed by name.
 */
export const InitMethod = {
  random: 'random',
  spectral: 'spectral',
} as const satisfies Record<InitMethod, InitMethod>;

/**
 * Parameters of a UMAP projection.
 */
export interface UMAPParameters {
  /**
   * The distance function with which to assess nearest neighbors, defaults
   * to euclidean distance.
   *
   * When both `distanceFunction` and `metric` are given, `distanceFunction`
   * is the function actually used and `metric` only names it for
   * serialization.
   * @default euclidean
   */
  distanceFunction?: DistanceFn;
  /**
   * The name of the built-in metric with which to assess nearest neighbors.
   * A model stores this name so that it can be serialized and reloaded without
   * carrying a function around; supplying a `distanceFunction` without a
   * `metric` makes the model unserializable unless the function is supplied
   * again on load.
   * @default 'euclidean'
   */
  metric?: MetricName;
  /**
   * How the initial embedding is built, before the optimization runs.
   *
   * `'random'` draws every coordinate uniformly in [-10, 10). `'spectral'`
   * uses the eigenvectors of the smallest non-zero eigenvalues of the
   * normalized Laplacian of the fuzzy graph, which is what the reference
   * python implementation defaults to.
   *
   * The default here is `'random'`, and the spectral initialization is opt-in:
   * measured on this port it is a 1 to 3% net slowdown at unchanged epoch
   * counts, and its quality has not been measured on a disconnected graph. A
   * spectral initialization that fails for any reason silently falls back to
   * the uniform draw.
   * @default 'random'
   */
  init?: InitMethod;
  /**
   * The initial learning rate for the embedding optimization.
   * @default 1
   */
  learningRate?: number;
  /**
   * The local connectivity required -- i.e. the number of nearest
   * neighbors that should be assumed to be connected at a local level.
   * The higher this value the more connected the manifold becomes
   * locally. In practice this should be not more than the local intrinsic
   * dimension of the manifold.
   * @default 1
   */
  localConnectivity?: number;
  /**
   * The effective minimum distance between embedded points. Smaller values
   * will result in a more clustered/clumped embedding where nearby points
   * on the manifold are drawn closer together, while larger values will
   * result on a more even dispersal of points. The value should be set
   * relative to the ``spread`` value, which determines the scale at which
   * embedded points will be spread out.
   * @default 0.1
   */
  minimumDistance?: number;
  /**
   * The dimension of the space to embed into. This defaults to 2 to
   * provide easy visualization, but can reasonably be set to any
   * integer value in the range 2 to 100.
   * @default 2
   */
  numberOfComponents?: number;
  /**
   * The number of training epochs to be used in optimizing the
   * low dimensional embedding. Larger values result in more accurate
   * embeddings. If None is specified a value will be selected based on
   * the size of the input dataset (200 for large datasets, 500 for small).
   * @default 0
   */
  numberOfEpochs?: number;
  /**
   * The size of local neighborhood (in terms of number of neighboring
   * sample points) used for manifold approximation. Larger values
   * result in more global views of the manifold, while smaller
   * values result in more local data being preserved. In general
   * values should be in the range 2 to 100.
   * @default 15
   */
  numberOfNeighbors?: number;
  /**
   * The number of negative samples to select per positive sample
   * in the optimization process. Increasing this value will result
   * in greater repulsive force being applied, greater optimization
   * cost, but slightly more accuracy.
   * @default 5
   */
  negativeSampleRate?: number;
  /**
   * Weighting applied to negative samples in low dimensional embedding
   * optimization. Values higher than one will result in greater weight
   * being given to negative samples.
   * @default 1
   */
  repulsionStrength?: number;
  /**
   * The pseudo-random number generator used by the stochastic parts of the
   * algorithm.
   *
   * It defaults to a generator seeded with `seed`, built fresh for every
   * model, so a fit repeats by default. Pass `Math.random` — or any other
   * unseeded source — to get a different embedding on every run.
   * @default a generator seeded with `seed`
   */
  random?: RandomFn;
  /**
   * Seed of the default random source of a fit.
   *
   * It is ignored when `random` is given, which is then the source itself.
   * Changing it gives a different, still repeatable, embedding of the same
   * data: umap-learn takes the same seed under the name `random_state`.
   *
   * It must be an integer.
   * @default 42
   */
  seed?: number;
  /**
   * Interpolate between (fuzzy) union and intersection as the set operation
   * used to combine local fuzzy simplicial sets to obtain a global fuzzy
   * simplicial sets. Both fuzzy set operations use the product t-norm.
   * The value of this parameter should be between 0.0 and 1.0; a value of
   * 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
   * intersection.
   * @default 1
   */
  setOperationMixRatio?: number;
  /**
   * The effective scale of embedded points. In combination with ``min_dist``
   * this determines how clustered/clumped the embedded points are.
   * @default 1
   */
  spread?: number;
  /**
   * For transform operations (embedding new points using a trained model)
   * this will control how aggressively to search for nearest neighbors.
   * Larger values will result in slower performance but more accurate
   * nearest neighbor evaluation.
   * @default 4
   */
  transformQueueSize?: number;
  /**
   * Seed of the random source a projection of new points draws from.
   *
   * `transform` never draws from `random`: it builds a fresh generator from
   * this seed on every call, so projecting new points neither advances nor
   * depends on the stream the fit ran on, and the same points always land in
   * the same place. The seed is part of a serialized model, so a reloaded
   * model projects exactly as the model it was saved from. umap-learn keeps
   * the same parameter under the name `transform_seed` (umap_.py:3126).
   *
   * It must be an integer.
   * @default 42
   */
  transformSeed?: number;
}

/**
 * Parameters of a supervised UMAP projection.
 */
export interface UMAPSupervisedParams {
  /**
   * The metric used to measure distance for a target array is using supervised
   * dimension reduction. By default this is 'categorical' which will measure
   * distance in terms of whether categories match or are different. Furthermore,
   * if semi-supervised is required target values of -1 will be treated as
   * unlabelled under the 'categorical' metric. If the target array takes
   * continuous values (e.g. for a regression problem) then metric of 'l1'
   * or 'l2' is probably more appropriate.
   * @default 'categorical'
   */
  targetMetric?: TargetMetric;
  /**
   * Weighting factor between data topology and target topology. A value of
   * 0.0 weights entirely on data, a value of 1.0 weights entirely on target.
   * The default of 0.5 balances the weighting equally between data and target.
   * @default 0.5
   */
  targetWeight?: number;
  /**
   * The number of nearest neighbors to use to construct the target simplcial
   * set. Defaults to the `nearestNeighbors` parameter.
   *
   * As upstream, the fallback is the default of `numberOfNeighbors` and not
   * the value `numberOfNeighbors` was actually given.
   * @default 15
   */
  targetNumberOfNeighbors?: number;
}
