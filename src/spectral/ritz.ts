import { EigenvalueDecomposition, Matrix } from 'ml-matrix';

import { atNumber } from '../sparse/arrayAt.ts';

/**
 * Eigenpairs of the small symmetric tridiagonal matrix a Lanczos run builds.
 *
 * The `m x m` problem is handed to `ml-matrix`, which agrees with a
 * hand-rolled EISPACK `tql2` to 6e-15 and costs 0.36 ms at `m = 50`, so it is
 * never worth carrying an implementation of our own.
 * @param alpha - Diagonal coefficients.
 * @param beta - Off-diagonal coefficients, one shorter than `alpha`.
 * @returns The eigenvalues and the matrix whose columns are the eigenvectors.
 */
export function ritzPairs(
  alpha: number[],
  beta: number[],
): { values: number[]; vectors: Matrix } {
  const size = alpha.length;
  const tridiagonal = Matrix.zeros(size, size);
  for (let i = 0; i < size; i++) {
    tridiagonal.set(i, i, atNumber(alpha, i));
    if (i + 1 < size) {
      const offDiagonal = atNumber(beta, i);
      tridiagonal.set(i, i + 1, offDiagonal);
      tridiagonal.set(i + 1, i, offDiagonal);
    }
  }
  const decomposition = new EigenvalueDecomposition(tridiagonal, {
    assumeSymmetric: true,
  });
  return {
    values: decomposition.realEigenvalues,
    vectors: decomposition.eigenvectorMatrix,
  };
}

/**
 * Indices of the `count` largest values, largest first.
 * @param values - The values to rank.
 * @param count - Number of indices to return.
 * @returns The indices.
 * @throws {RangeError} If there are fewer than `count` values.
 */
export function largestFirst(values: number[], count: number): number[] {
  const taken = new Uint8Array(values.length);
  const order: number[] = [];
  for (let k = 0; k < count; k++) {
    let best = -1;
    let bestValue = -Infinity;
    for (let i = 0; i < values.length; i++) {
      if (taken[i] === 1) continue;
      const value = atNumber(values, i);
      if (value > bestValue) {
        bestValue = value;
        best = i;
      }
    }
    if (best < 0) throw new RangeError('largestFirst: no value left to take');
    taken[best] = 1;
    order.push(best);
  }
  return order;
}

/**
 * Reads a vector at an index the caller has already bounded.
 * @param vectors - The vectors.
 * @param index - Index of the vector.
 * @returns The vector.
 * @throws {RangeError} If the index is out of bounds.
 */
export function vectorAt(vectors: Float64Array[], index: number): Float64Array {
  const vector = vectors[index];
  if (vector === undefined) throw new RangeError('vector index out of bounds');
  return vector;
}
