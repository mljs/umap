import { atFloat64, atNumber } from '../sparse/arrayAt.ts';
import type { RandomFn } from '../types.ts';

import type { CSRMatrix } from './csr.ts';
import { dot, euclideanNorm } from './csr.ts';
import type { NormalizedLaplacian } from './laplacian.ts';
import {
  deflate,
  normalizedLaplacian,
  rayleighResidual,
  shiftedMultiply,
} from './laplacian.ts';
import { largestFirst, ritzPairs, vectorAt } from './ritz.ts';

/**
 * One eigenpair of the symmetric normalized Laplacian.
 */
export interface SpectralVector {
  /** The eigenvector, of unit euclidean norm. */
  vector: Float64Array;
  /** Its eigenvalue, in `[0, 2]`. */
  eigenvalue: number;
  /** Its relative Rayleigh residual `||L x - lambda x|| / ||x||`. */
  residual: number;
}

/**
 * Options of {@link spectralVectors}.
 */
export interface SpectralVectorsOptions {
  /**
   * Random number generator drawing the Lanczos start vector.
   *
   * It has to be random: on a near regular graph the exactly known eigenvector
   * `sqrt(deg)` is proportional to the all ones vector, so a vector of ones
   * deflates to exactly zero and the iteration breaks down at its first step.
   */
  random: RandomFn;
  /**
   * Number of Lanczos steps, capped at the size of the graph.
   * @default max(4 * (dim + 1) + 10, 40)
   */
  steps?: number;
}

/**
 * Eigenvectors of the `dim` smallest non-zero eigenvalues of the symmetric
 * normalized Laplacian of a graph, by Lanczos with full reorthogonalization.
 *
 * The iteration runs on `2I - L`, whose largest eigenvalues are the smallest
 * of `L`, and deflates the exactly known eigenvector of the zero eigenvalue on
 * every product, so it never has to resolve the trivial eigenpair and never
 * needs a shift-invert or a linear solve.
 * @param csr - The symmetric graph, which must be connected for the result to
 * mean anything.
 * @param dim - Number of eigenvectors to return.
 * @param options - Start vector source and iteration length.
 * @returns The `dim` eigenpairs, smallest eigenvalue first.
 * @throws {RangeError} If the graph is degenerate, or if the Krylov space
 * collapses before `dim` eigenpairs can be extracted.
 */
export function spectralVectors(
  csr: CSRMatrix,
  dim: number,
  options: SpectralVectorsOptions,
): SpectralVector[] {
  const { random, steps } = options;
  const laplacian = normalizedLaplacian(csr);
  const n = csr.n;
  const maximumSteps = Math.min(
    n,
    steps ?? Math.max(8 * dim + 40, MINIMUM_STEPS),
  );
  if (maximumSteps < dim) {
    throw new RangeError('spectralVectors: too few steps for the dimension');
  }

  const { basis, alpha, beta } = iterate(laplacian, n, maximumSteps, random);
  const used = alpha.length;
  if (used < dim) {
    throw new RangeError('spectralVectors: the Krylov space collapsed');
  }

  const { values, vectors } = ritzPairs(alpha, beta);
  const order = largestFirst(values, dim);
  const output: SpectralVector[] = [];
  for (const index of order) {
    const vector = new Float64Array(n);
    for (let t = 0; t < used; t++) {
      const weight = vectors.get(t, index);
      const basisVector = vectorAt(basis, t);
      for (let i = 0; i < n; i++) {
        vector[i] = atFloat64(vector, i) + weight * atFloat64(basisVector, i);
      }
    }
    // eig(2I - L) = 2 - eig(L).
    const eigenvalue = 2 - atNumber(values, index);
    output.push({
      vector,
      eigenvalue,
      residual: rayleighResidual(laplacian, vector, eigenvalue),
    });
  }
  return output;
}

/**
 * Shortest Lanczos run, whatever the embedding dimension.
 */
const MINIMUM_STEPS = 40;

/**
 * Below this the residual vector counts as zero: the Krylov space is
 * exhausted and the run stops.
 */
const BREAKDOWN = 1e-10;

/**
 * Runs the Lanczos recurrence on `2I - L` with the trivial eigenvector
 * deflated, reorthogonalizing every new vector against the whole basis.
 * @param laplacian - The Laplacian factors.
 * @param n - Number of vertices.
 * @param maximumSteps - Upper bound on the number of steps.
 * @param random - Source of the start vector.
 * @returns The orthonormal basis and the tridiagonal coefficients.
 * @throws {RangeError} If the start vector vanishes once deflated.
 */
function iterate(
  laplacian: NormalizedLaplacian,
  n: number,
  maximumSteps: number,
  random: RandomFn,
): { basis: Float64Array[]; alpha: number[]; beta: number[] } {
  const start = new Float64Array(n);
  for (let i = 0; i < n; i++) start[i] = random() * 2 - 1;
  deflate(laplacian, start);
  const startNorm = euclideanNorm(start);
  if (!(startNorm > BREAKDOWN)) {
    throw new RangeError('spectralVectors: the start vector vanished');
  }
  for (let i = 0; i < n; i++) start[i] = atFloat64(start, i) / startNorm;

  const basis: Float64Array[] = [start];
  const alpha: number[] = [];
  const beta: number[] = [];
  const residual = new Float64Array(n);
  for (let j = 0; j < maximumSteps; j++) {
    const current = vectorAt(basis, j);
    shiftedMultiply(laplacian, current, residual);
    deflate(laplacian, residual);
    const diagonal = dot(residual, current);
    alpha.push(diagonal);
    for (let i = 0; i < n; i++) {
      residual[i] = atFloat64(residual, i) - diagonal * atFloat64(current, i);
    }
    if (j > 0) {
      const offDiagonal = atNumber(beta, j - 1);
      const previous = vectorAt(basis, j - 1);
      for (let i = 0; i < n; i++) {
        residual[i] =
          atFloat64(residual, i) - offDiagonal * atFloat64(previous, i);
      }
    }
    reorthogonalize(basis, residual, n);
    const norm = euclideanNorm(residual);
    if (norm < BREAKDOWN || j === maximumSteps - 1) break;
    beta.push(norm);
    const next = new Float64Array(n);
    for (let i = 0; i < n; i++) next[i] = atFloat64(residual, i) / norm;
    basis.push(next);
  }
  return { basis, alpha, beta };
}

/**
 * Projects the whole basis out of the residual vector, twice.
 *
 * The three term recurrence loses orthogonality after a handful of steps
 * otherwise, and ghost eigenvalues then appear wherever the graph has a
 * cluster of close ones. Two passes of Gram-Schmidt are enough.
 * @param basis - The Lanczos basis built so far.
 * @param residual - The vector to orthogonalize, updated in place.
 * @param n - Number of vertices.
 */
function reorthogonalize(
  basis: Float64Array[],
  residual: Float64Array,
  n: number,
): void {
  for (let pass = 0; pass < 2; pass++) {
    for (const basisVector of basis) {
      const projection = dot(basisVector, residual);
      for (let i = 0; i < n; i++) {
        residual[i] =
          atFloat64(residual, i) - projection * atFloat64(basisVector, i);
      }
    }
  }
}
