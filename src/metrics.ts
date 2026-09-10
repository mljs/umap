/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript, euclidean delegates to the
 * mljs ml-distance package, cosine keeps a local single-pass loop whose squares
 * are written as multiplications instead of `** 2` (bit-identical, faster), a
 * named metric registry was added so a serialized model can rebuild its distance
 * function, vectors that are too short now throw instead of yielding NaN, and
 * that length check is made once per call instead of once per element.
 */

import { distance } from 'ml-distance';

const euclideanDistance = distance.euclidean;

/**
 * A distance function between two vectors of identical length.
 */
export type DistanceFn = (x: number[], y: number[]) => number;

/**
 * Computes the Euclidean distance between two vectors.
 * @param x - First vector.
 * @param y - Second vector, at least as long as `x`.
 * @throws {RangeError} If `y` is shorter than `x`.
 * @returns The Euclidean distance.
 */
export function euclidean(x: number[], y: number[]): number {
  if (y.length < x.length) {
    throw new RangeError('euclidean: vectors must have the same length');
  }
  return euclideanDistance(x, y);
}

/**
 * Computes the cosine distance between two vectors, i.e. one minus their
 * cosine similarity. Two zero vectors are at distance 0 from each other, while
 * a zero vector and a non-zero one are at distance 1.
 * @param x - First vector.
 * @param y - Second vector, at least as long as `x`.
 * @throws {RangeError} If `y` is shorter than `x`.
 * @returns The cosine distance.
 */
export function cosine(x: number[], y: number[]): number {
  if (y.length < x.length) {
    throw new RangeError('cosine: vectors must have the same length');
  }

  // Not delegated to ml-distance's `similarity.cosine`, even though its
  // arithmetic is now identical (the sqrt grouping landed in 5.0.0). umap-learn
  // returns 0 when both norms are zero and 1 when exactly one is
  // (umap/distances.py:616-621) so a NaN can never enter the kNN graph, and
  // those guards test the NORM, not the elements: a vector of values around
  // 1e-300 squares to a zero norm through underflow while no element is zero.
  // `similarity.cosine` returns only the ratio, so applying the guards to its
  // result would mean recomputing both norms here anyway. Delegating would need
  // a guarded `cosineDistance` in ml-distance itself.
  //
  // This is the innermost call of the nearest neighbor descent: the reads are
  // asserted rather than guarded because the length check above already covers
  // the whole loop, and the compiler cannot carry that knowledge into it.
  let result = 0;
  let normX = 0;
  let normY = 0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i] as number;
    const yi = y[i] as number;
    result += xi * yi;
    normX += xi * xi;
    normY += yi * yi;
  }

  if (normX === 0 && normY === 0) {
    return 0;
  } else if (normX === 0 || normY === 0) {
    return 1;
  } else {
    return 1 - result / Math.sqrt(normX * normY);
  }
}

/**
 * Name of a built-in metric. A model stores this name so that it can be
 * serialized and rebuilt without carrying a function around.
 */
export type MetricName = 'euclidean' | 'cosine';

/**
 * The built-in metrics, keyed by name.
 */
export const METRICS: Record<MetricName, DistanceFn> = {
  euclidean,
  cosine,
};

/**
 * Returns the distance function registered under a metric name.
 * @param name - Name of the metric.
 * @throws {RangeError} If no metric is registered under that name.
 * @returns The corresponding distance function.
 */
export function resolveMetric(name: MetricName): DistanceFn {
  const metric: DistanceFn | undefined = METRICS[name];
  if (metric === undefined) {
    throw new RangeError(`unknown metric: ${name}`);
  }
  return metric;
}
