import { Matrix } from 'ml-matrix';

import { UMAP } from '../index.ts';
import type { SparseMatrix } from '../sparse/index.ts';
import type { CSRMatrix } from '../spectral/index.ts';

import { blobs } from './blobs.ts';
import { makeRandom } from './random.ts';

/**
 * Fits the graph of a fresh blob dataset.
 * @param n - Number of points.
 * @param dimensions - Dimension of the points.
 * @param k - Number of blobs.
 * @param seed - Seed of the projection.
 * @returns The fuzzy graph of the data.
 */
export function graphOf(
  n: number,
  dimensions: number,
  k: number,
  seed = 3,
): SparseMatrix {
  return graphOfData(blobs(n, dimensions, k).X, seed);
}

/**
 * Fits the graph of a dataset.
 * @param X - The data, one array per point.
 * @param seed - Seed of the projection.
 * @returns The fuzzy graph of the data.
 */
export function graphOfData(X: number[][], seed = 3): SparseMatrix {
  const umap = new UMAP({ random: makeRandom(seed), numberOfComponents: 2 });
  umap.initializeFit(X);
  const { graph } = umap.internals;
  if (graph === undefined) throw new Error('the fit built no graph');
  return graph;
}

/**
 * Densifies the symmetric normalized Laplacian of a graph.
 * @param csr - The graph.
 * @returns `I - D^-1/2 A D^-1/2` as a dense matrix.
 */
export function denseLaplacian(csr: CSRMatrix): Matrix {
  const { n, ptr, idx, val } = csr;
  const invSqrtDegree = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let degree = 0;
    for (let p = ptr[i] ?? 0; p < (ptr[i + 1] ?? 0); p++) degree += val[p] ?? 0;
    invSqrtDegree[i] = degree > 0 ? 1 / Math.sqrt(degree) : 0;
  }
  const laplacian = Matrix.zeros(n, n);
  for (let i = 0; i < n; i++) {
    laplacian.set(i, i, 1);
    for (let p = ptr[i] ?? 0; p < (ptr[i + 1] ?? 0); p++) {
      const j = idx[p] ?? 0;
      const value =
        (invSqrtDegree[i] ?? 0) * (val[p] ?? 0) * (invSqrtDegree[j] ?? 0);
      laplacian.set(i, j, laplacian.get(i, j) - value);
    }
  }
  return laplacian;
}

/**
 * Indices of an array of values, smallest value first.
 * @param values - The values to rank.
 * @returns The indices.
 */
export function ascendingOrder(values: number[]): number[] {
  const order: number[] = [];
  for (let i = 0; i < values.length; i++) order.push(i);
  order.sort((a, b) => (values[a] ?? 0) - (values[b] ?? 0));
  return order;
}
