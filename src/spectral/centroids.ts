import { Matrix } from 'ml-matrix';

import { rowAt } from '../fuzzy/rowAt.ts';
import { atFloat64, atInt32, atNumber } from '../sparse/arrayAt.ts';
import type { Vectors } from '../types.ts';

/**
 * Mean of the data points of every connected component.
 * @param data - The data, one array per point.
 * @param label - Component label of every point.
 * @param count - Number of components.
 * @returns One centroid per component.
 * @throws {RangeError} If the data is empty.
 */
export function componentCentroids(
  data: Vectors,
  label: Int32Array,
  count: number,
): number[][] {
  const first = data[0];
  if (first === undefined) throw new RangeError('centroids: no data');
  const dimensions = first.length;
  const centroids: number[][] = [];
  for (let c = 0; c < count; c++) {
    centroids.push(new Array<number>(dimensions).fill(0));
  }
  const sizes = new Int32Array(count);
  for (let i = 0; i < data.length; i++) {
    const point = rowAt(data, i);
    const component = atInt32(label, i);
    const centroid = rowAt(centroids, component);
    sizes[component] = atInt32(sizes, component) + 1;
    for (let j = 0; j < dimensions; j++) {
      centroid[j] = atNumber(centroid, j) + atNumber(point, j);
    }
  }
  for (let c = 0; c < count; c++) {
    const size = atInt32(sizes, c);
    if (size === 0) continue;
    const centroid = rowAt(centroids, c);
    for (let j = 0; j < dimensions; j++) {
      centroid[j] = atNumber(centroid, j) / size;
    }
  }
  return centroids;
}

/**
 * Affinity between the centroids of the connected components.
 *
 * The reference implementation uses `exp(-d^2)` on raw centroid distances,
 * with no scale normalization. That is a defect rather than a choice: at the
 * centroid distances real data produces, 32 to 43 apart on a 20 dimensional
 * set of blobs, `exp(-d^2)` is `exp(-1018)`, which underflows to exactly 0 in
 * float64, so every off-diagonal entry is 0 and the arrangement it returns is
 * pure noise. Dividing by the median squared distance makes the affinity scale
 * invariant and keeps the ordering the reference implementation meant to
 * express.
 * @param centroids - One centroid per component.
 * @returns The symmetric affinity matrix.
 */
export function centroidAffinity(centroids: number[][]): Matrix {
  const count = centroids.length;
  const squared = Matrix.zeros(count, count);
  const offDiagonal = new Float64Array((count * (count - 1)) / 2);
  let written = 0;
  for (let a = 0; a < count; a++) {
    for (let b = a + 1; b < count; b++) {
      const distance = squaredDistance(
        rowAt(centroids, a),
        rowAt(centroids, b),
      );
      squared.set(a, b, distance);
      squared.set(b, a, distance);
      offDiagonal[written] = distance;
      written++;
    }
  }
  offDiagonal.sort();
  const median = atFloat64(offDiagonal, offDiagonal.length >> 1);
  const scale = median > 0 ? median : 1;
  const affinity = Matrix.zeros(count, count);
  for (let a = 0; a < count; a++) {
    for (let b = 0; b < count; b++) {
      affinity.set(a, b, Math.exp(-squared.get(a, b) / scale));
    }
  }
  return affinity;
}

/**
 * Squared euclidean distance between two positions of equal length.
 * @param a - First position.
 * @param b - Second position.
 * @returns The squared distance.
 */
export function squaredDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const difference = atNumber(a, i) - atNumber(b, i);
    sum += difference * difference;
  }
  return sum;
}
