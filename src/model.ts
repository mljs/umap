/*
 * ml-umap: the shape of a serialized model, split out of ./serialize.ts so
 * that module stays under the 250 line limit and holds only the two functions.
 */

import type { MetricName } from './metrics.ts';
import type { DistanceFn, RandomFn } from './types.ts';

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
