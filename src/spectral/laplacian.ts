import { atFloat64, atInt32 } from '../sparse/arrayAt.ts';

import type { CSRMatrix } from './csr.ts';
import { dot, euclideanNorm } from './csr.ts';

/**
 * The symmetric normalized Laplacian `L = I - D^-1/2 A D^-1/2` of a graph,
 * held as the factors the matrix-vector products need rather than as a matrix.
 */
export interface NormalizedLaplacian {
  /** The adjacency matrix `A`. */
  csr: CSRMatrix;
  /** `D^-1/2`, zero on an isolated vertex. */
  invSqrtDegree: Float64Array;
  /**
   * `sqrt(deg) / ||sqrt(deg)||`: the eigenvector of the zero eigenvalue of
   * `L`, known exactly and therefore never worth iterating for.
   */
  trivial: Float64Array;
}

/**
 * Builds the normalized Laplacian factors of a graph.
 * @param csr - The symmetric graph.
 * @returns The factors of `L`.
 * @throws {RangeError} If the graph has no vertex, or if a degree is not a
 * finite non-negative number.
 */
export function normalizedLaplacian(csr: CSRMatrix): NormalizedLaplacian {
  const { n, ptr, val } = csr;
  if (n <= 0) throw new RangeError('normalizedLaplacian: empty graph');
  const invSqrtDegree = new Float64Array(n);
  const trivial = new Float64Array(n);
  let squaredNorm = 0;
  for (let i = 0; i < n; i++) {
    let degree = 0;
    const end = atInt32(ptr, i + 1);
    for (let p = atInt32(ptr, i); p < end; p++) {
      degree += atFloat64(val, p);
    }
    if (!(degree >= 0) || !Number.isFinite(degree)) {
      throw new RangeError('normalizedLaplacian: degree is not finite');
    }
    invSqrtDegree[i] = degree > 0 ? 1 / Math.sqrt(degree) : 0;
    const root = Math.sqrt(degree);
    trivial[i] = root;
    squaredNorm += degree;
  }
  if (!(squaredNorm > 0)) {
    throw new RangeError('normalizedLaplacian: the graph has no edge');
  }
  const norm = Math.sqrt(squaredNorm);
  for (let i = 0; i < n; i++) {
    trivial[i] = atFloat64(trivial, i) / norm;
  }
  return { csr, invSqrtDegree, trivial };
}

/**
 * Computes `y = (2I - L) x = (I + D^-1/2 A D^-1/2) x`.
 *
 * Working with `2I - L` rather than with `L` turns the smallest eigenvalues of
 * `L`, which plain Lanczos converges to slowly, into the largest ones, where
 * it converges fastest. No shift-invert and no linear solve is needed.
 * @param laplacian - The Laplacian factors.
 * @param x - The vector to multiply.
 * @param y - Where the product is written; may not alias `x`.
 */
export function shiftedMultiply(
  laplacian: NormalizedLaplacian,
  x: Float64Array,
  y: Float64Array,
): void {
  normalizedAdjacencyMultiply(laplacian, x, y);
  for (let i = 0; i < y.length; i++) {
    y[i] = atFloat64(x, i) + atFloat64(y, i);
  }
}

/**
 * Computes `y = L x`.
 * @param laplacian - The Laplacian factors.
 * @param x - The vector to multiply.
 * @param y - Where the product is written; may not alias `x`.
 */
export function laplacianMultiply(
  laplacian: NormalizedLaplacian,
  x: Float64Array,
  y: Float64Array,
): void {
  normalizedAdjacencyMultiply(laplacian, x, y);
  for (let i = 0; i < y.length; i++) {
    y[i] = atFloat64(x, i) - atFloat64(y, i);
  }
}

/**
 * Projects the known zero eigenvector out of a vector, in place.
 *
 * Deflating it on every product is what reproduces the "skip the smallest
 * eigenvalue" semantics of the reference implementation, whose first
 * eigenvector is exactly `sqrt(deg)` and carries no layout information.
 * @param laplacian - The Laplacian factors.
 * @param x - The vector to deflate, updated in place.
 */
export function deflate(laplacian: NormalizedLaplacian, x: Float64Array): void {
  const trivial = laplacian.trivial;
  const projection = dot(trivial, x);
  for (let i = 0; i < x.length; i++) {
    x[i] = atFloat64(x, i) - projection * atFloat64(trivial, i);
  }
}

/**
 * Relative Rayleigh residual `||L x - lambda x|| / ||x||` of an eigenpair.
 *
 * It is the convergence check of the Lanczos run: a Ritz pair that has not
 * converged, on a graph that is nearly disconnected for instance, still comes
 * back as a plausible looking vector, and only the residual tells it apart
 * from a converged one.
 * @param laplacian - The Laplacian factors.
 * @param x - The candidate eigenvector.
 * @param eigenvalue - The candidate eigenvalue.
 * @returns The relative residual, `Infinity` for a zero vector.
 */
export function rayleighResidual(
  laplacian: NormalizedLaplacian,
  x: Float64Array,
  eigenvalue: number,
): number {
  const norm = euclideanNorm(x);
  if (!(norm > 0)) return Infinity;
  const product = new Float64Array(x.length);
  laplacianMultiply(laplacian, x, product);
  let squared = 0;
  for (let i = 0; i < x.length; i++) {
    const difference = atFloat64(product, i) - eigenvalue * atFloat64(x, i);
    squared += difference * difference;
  }
  return Math.sqrt(squared) / norm;
}

/**
 * Computes `y = D^-1/2 A D^-1/2 x`.
 * @param laplacian - The Laplacian factors.
 * @param x - The vector to multiply.
 * @param y - Where the product is written; may not alias `x`.
 */
function normalizedAdjacencyMultiply(
  laplacian: NormalizedLaplacian,
  x: Float64Array,
  y: Float64Array,
): void {
  const { csr, invSqrtDegree } = laplacian;
  const { n, ptr, idx, val } = csr;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    const end = atInt32(ptr, i + 1);
    for (let p = atInt32(ptr, i); p < end; p++) {
      const column = atInt32(idx, p);
      sum +=
        atFloat64(val, p) *
        atFloat64(invSqrtDegree, column) *
        atFloat64(x, column);
    }
    y[i] = atFloat64(invSqrtDegree, i) * sum;
  }
}
