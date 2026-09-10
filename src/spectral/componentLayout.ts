import { EigenvalueDecomposition, Matrix } from 'ml-matrix';

import { rowAt } from '../fuzzy/rowAt.ts';
import { atFloat64, atNumber } from '../sparse/arrayAt.ts';
import type { Vectors } from '../types.ts';

import {
  centroidAffinity,
  componentCentroids,
  squaredDistance,
} from './centroids.ts';

/**
 * Places the connected components of a graph relative to one another, so that
 * every component can then be embedded inside its own slot.
 *
 * Below `2 * dim + 1` components the reference implementation ignores the data
 * and spreads the components over the vertices of a cross polytope; that is
 * kept as is. Above it, the components are laid out by the affinity of their
 * centroids.
 * @param data - The data the graph was built from, one array per point.
 * @param label - Component label of every point.
 * @param count - Number of components.
 * @param dim - Dimension of the embedding.
 * @returns One position per component.
 */
export function metaLayout(
  data: Vectors,
  label: Int32Array,
  count: number,
  dim: number,
): number[][] {
  if (count > 2 * dim) return centroidLayout(data, label, count, dim);
  return crossPolytopeLayout(count, dim);
}

/**
 * Half the distance from a component to its closest other component, which is
 * the radius of the slot the component may be embedded into.
 * @param meta - One position per component.
 * @param index - The component to measure.
 * @returns The radius, `0.5` when no other position is at a positive distance.
 */
export function componentRadius(meta: number[][], index: number): number {
  const position = rowAt(meta, index);
  let closest = Infinity;
  for (let other = 0; other < meta.length; other++) {
    if (other === index) continue;
    const distance = Math.sqrt(squaredDistance(position, rowAt(meta, other)));
    if (distance > 0 && distance < closest) closest = distance;
  }
  return (Number.isFinite(closest) ? closest : 1) / 2;
}

/**
 * Lays the components out by a spectral embedding of their centroid affinity.
 * @param data - The data the graph was built from, one array per point.
 * @param label - Component label of every point.
 * @param count - Number of components.
 * @param dim - Dimension of the embedding.
 * @returns One position per component.
 */
function centroidLayout(
  data: Vectors,
  label: Int32Array,
  count: number,
  dim: number,
): number[][] {
  const affinity = centroidAffinity(componentCentroids(data, label, count));
  const degree = new Float64Array(count);
  for (let a = 0; a < count; a++) {
    let sum = 0;
    for (let b = 0; b < count; b++) sum += affinity.get(a, b);
    degree[a] = sum;
  }
  const laplacian = Matrix.zeros(count, count);
  for (let a = 0; a < count; a++) {
    for (let b = 0; b < count; b++) {
      const normalizer = Math.sqrt(atFloat64(degree, a) * atFloat64(degree, b));
      const value = normalizer > 0 ? affinity.get(a, b) / normalizer : 0;
      laplacian.set(a, b, (a === b ? 1 : 0) - value);
    }
  }
  const decomposition = new EigenvalueDecomposition(laplacian, {
    assumeSymmetric: true,
  });
  const order = smallestNonTrivial(decomposition.realEigenvalues, dim);
  const vectors = decomposition.eigenvectorMatrix;

  const output: number[][] = [];
  let maximum = 0;
  for (let a = 0; a < count; a++) {
    const row = new Array<number>(dim).fill(0);
    for (let k = 0; k < order.length; k++) {
      const value = vectors.get(a, atNumber(order, k));
      row[k] = value;
      if (value > maximum) maximum = value;
    }
    output.push(row);
  }
  if (maximum !== 0) {
    for (const row of output) {
      for (let k = 0; k < dim; k++) row[k] = atNumber(row, k) / maximum;
    }
  }
  return output;
}

/**
 * Spreads the components over the vertices of a cross polytope.
 * @param count - Number of components.
 * @param dim - Dimension of the embedding.
 * @returns One position per component.
 */
function crossPolytopeLayout(count: number, dim: number): number[][] {
  const half = Math.ceil(count / 2);
  const output: number[][] = [];
  for (let i = 0; i < count; i++) {
    const axis = i % half;
    const row = new Array<number>(dim).fill(0);
    if (axis < dim) row[axis] = i < half ? 1 : -1;
    output.push(row);
  }
  return output;
}

/**
 * Indices of the `dim` smallest eigenvalues, skipping the smallest of all.
 * @param values - The eigenvalues.
 * @param dim - Number of indices wanted.
 * @returns The indices, smallest eigenvalue first.
 */
function smallestNonTrivial(values: number[], dim: number): number[] {
  const taken = new Uint8Array(values.length);
  const order: number[] = [];
  for (let k = 0; k < dim + 1; k++) {
    let best = -1;
    let bestValue = Infinity;
    for (let i = 0; i < values.length; i++) {
      if (taken[i] === 1) continue;
      const value = atNumber(values, i);
      if (value < bestValue) {
        bestValue = value;
        best = i;
      }
    }
    if (best < 0) break;
    taken[best] = 1;
    if (k > 0) order.push(best);
  }
  return order;
}
