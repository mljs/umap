import { makeRandom } from './random.ts';

/**
 * A labelled synthetic dataset, as returned by {@link blobs}.
 */
export interface Blobs {
  /** The points, one array per point. */
  X: number[][];
  /** Index of the blob every point was drawn from. */
  labels: number[];
}

/**
 * Draws points from `k` isotropic Gaussian blobs with centers spread over the
 * `[-10, 10]` cube.
 *
 * Well separated blobs are what makes the fuzzy graph of the data
 * disconnected, which is the case the spectral initialization has to handle
 * component by component.
 * @param n - Number of points.
 * @param dimensions - Dimension of the points.
 * @param k - Number of blobs, which points are assigned to round-robin.
 * @param seed - Seed of the generator.
 * @returns The points and their blob index.
 */
export function blobs(
  n: number,
  dimensions: number,
  k: number,
  seed = 7,
): Blobs {
  const random = makeRandom(seed);
  const centers: number[][] = [];
  for (let c = 0; c < k; c++) {
    const center: number[] = [];
    for (let j = 0; j < dimensions; j++) center.push(random() * 20 - 10);
    centers.push(center);
  }

  const X: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % k;
    const center = centers[label] ?? [];
    const point: number[] = [];
    for (let j = 0; j < dimensions; j++) {
      point.push((center[j] ?? 0) + gaussian(random));
    }
    X.push(point);
    labels.push(label);
  }
  return { X, labels };
}

/**
 * Draws a standard normal deviate by the Box-Muller transform.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns One deviate of mean 0 and variance 1.
 */
function gaussian(random: () => number): number {
  const uniform = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(uniform)) * Math.cos(2 * Math.PI * random());
}
