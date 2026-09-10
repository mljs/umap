import type { UMAPParameters } from '../../index.ts';
import { UMAP } from '../../index.ts';
import { makeRandom } from '../random.ts';

import { knnPreservation } from './knnPreservation.ts';
import { continuity, trustworthiness } from './trustworthiness.ts';

/**
 * A series of fits of the same data, one per seed, and how good each of them
 * is.
 */
export interface QualitySeries {
  /** The embedding each seed produced, in the order of the seeds. */
  embeddings: number[][][];
  /** {@link knnPreservation} of each of those embeddings. */
  knn: number[];
  /** {@link trustworthiness} of each of those embeddings. */
  trust: number[];
  /** {@link continuity} of each of those embeddings. */
  continuity: number[];
}

/**
 * What to fit, how, and how to measure it.
 */
export interface SeriesConfig {
  /** The data to project, one array per point. */
  X: number[][];
  /** The seeds to fit from, one embedding per seed. */
  seeds: number[];
  /** Size of the neighborhood the three measures are taken at. */
  k: number;
  /** Parameters of the projection; the generator is set from the seed. */
  params: UMAPParameters;
}

/**
 * Fits one embedding per seed and measures all three neighborhood qualities of
 * each. Results are cached under `key` so that several tests can assert on the
 * same series without paying for the fits twice.
 * @param key - Cache key, unique per configuration.
 * @param config - What to fit and how to measure it.
 * @returns The embeddings and their measures, in the order of the seeds.
 */
export function qualitySeries(
  key: string,
  config: SeriesConfig,
): QualitySeries {
  const cached = cache.get(key);
  if (cached !== undefined) return cached;

  const { X, seeds, k, params } = config;
  const series: QualitySeries = {
    embeddings: [],
    knn: [],
    trust: [],
    continuity: [],
  };
  for (const seed of seeds) {
    const umap = new UMAP({ ...params, random: makeRandom(seed) });
    const embedding = umap.fit(X);
    series.embeddings.push(embedding);
    series.knn.push(knnPreservation(X, embedding, k));
    series.trust.push(trustworthiness(X, embedding, k));
    series.continuity.push(continuity(X, embedding, k));
  }
  cache.set(key, series);
  return series;
}

/**
 * Smallest value of a series.
 * @param values - The values.
 * @returns The minimum, or `Infinity` for an empty series.
 */
export function minimum(values: number[]): number {
  let smallest = Infinity;
  for (const value of values) smallest = Math.min(smallest, value);
  return smallest;
}

/**
 * Largest value of a series.
 * @param values - The values.
 * @returns The maximum, or `-Infinity` for an empty series.
 */
export function maximum(values: number[]): number {
  let largest = -Infinity;
  for (const value of values) largest = Math.max(largest, value);
  return largest;
}

/**
 * Distance between the extremes of a series, i.e. how much the seed alone
 * moves the measure.
 * @param values - The values.
 * @returns The spread.
 */
export function spread(values: number[]): number {
  return maximum(values) - minimum(values);
}

const cache = new Map<string, QualitySeries>();
