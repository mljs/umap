import { rowAt } from '../fuzzy/rowAt.ts';
import { atFloat64, atNumber } from '../sparse/arrayAt.ts';
import type { SparseMatrix } from '../sparse/index.ts';
import type { RandomFn, Vectors } from '../types.ts';

import { componentVectors, placeComponents } from './componentEmbedding.ts';
import { connectedComponents } from './components.ts';
import { toCSR } from './csr.ts';
import { vectorAt } from './ritz.ts';

/**
 * Options of {@link spectralEmbedding} and {@link spectralLayout}.
 */
export interface SpectralEmbeddingOptions {
  /** Dimension of the space the data is embedded into. */
  numberOfComponents: number;
  /** Random number generator returning a float in [0, 1). */
  random: RandomFn;
  /**
   * The data the graph was built from, one array per point. It is only read
   * when the graph has more than `2 * numberOfComponents` connected
   * components, to arrange them by the affinity of their centroids.
   */
  data: Vectors;
  /**
   * Largest relative Rayleigh residual an eigenpair may have before it counts
   * as not converged.
   * @default 1e-2
   */
  maxResidual?: number;
}

/**
 * Spectral initialization of an embedding, which never throws.
 *
 * This is the safe entry point: it returns `null` rather than throwing when
 * the graph is degenerate, when the iteration does not converge, or when a
 * coordinate comes back non-finite, so a caller can initialize at random
 * instead and carry on.
 * @param graph - The fuzzy graph of the data.
 * @param options - Embedding dimension, data and random source.
 * @returns The initial embedding, or `null` when it could not be computed.
 */
export function spectralEmbedding(
  graph: SparseMatrix,
  options: SpectralEmbeddingOptions,
): Vectors | null {
  let embedding: Vectors;
  try {
    embedding = spectralLayout(graph, options);
  } catch {
    return null;
  }
  for (const row of embedding) {
    for (const value of row) {
      if (!Number.isFinite(value)) return null;
    }
  }
  return embedding;
}

/**
 * Spectral initialization of an embedding.
 *
 * Every connected component of the graph is embedded on its own, then the
 * components are placed relative to one another; on a connected graph that
 * reduces to a single Lanczos run. The result is scaled into the same
 * `[-10, 10]` box a uniform initialization draws in, and jittered, exactly as
 * the reference implementation does.
 * @param graph - The fuzzy graph of the data.
 * @param options - Embedding dimension, data and random source.
 * @returns The initial embedding, one array per graph row.
 * @throws {RangeError} If the graph is degenerate, or if the eigenvectors of
 * a connected graph do not converge.
 */
export function spectralLayout(
  graph: SparseMatrix,
  options: SpectralEmbeddingOptions,
): Vectors {
  const { numberOfComponents: dim, random, data } = options;
  const maxResidual = options.maxResidual ?? MAX_RESIDUAL;
  const csr = toCSR(graph);
  if (csr.n <= 0) throw new RangeError('spectralLayout: empty graph');
  const { count, label } = connectedComponents(csr);

  const embedding: Vectors = [];
  for (let i = 0; i < csr.n; i++) {
    embedding.push(new Array<number>(dim).fill(0));
  }

  if (count === 1) {
    const vectors = componentVectors(csr, dim, { random, maxResidual });
    if (vectors === null) {
      throw new RangeError('spectralLayout: the eigenvectors did not converge');
    }
    for (let k = 0; k < dim; k++) {
      const vector = vectorAt(vectors, k);
      for (let i = 0; i < csr.n; i++) {
        rowAt(embedding, i)[k] = atFloat64(vector, i);
      }
    }
  } else {
    placeComponents(embedding, csr, label, {
      count,
      dim,
      data,
      random,
      maxResidual,
    });
  }
  return noisyScaleCoords(embedding, random);
}

/**
 * Rescales an embedding into the `[-maxCoord, maxCoord]` box and jitters it,
 * as the `noisy_scale_coords` of the reference implementation does.
 *
 * The jitter matters: a spectral layout can place many points at very nearly
 * the same coordinates, and the optimization has no gradient between two
 * points that coincide exactly.
 * @param coords - The embedding, updated in place.
 * @param random - Random number generator returning a float in [0, 1).
 * @param maxCoord - Largest absolute coordinate after the rescaling.
 * @param noise - Standard deviation of the jitter.
 * @returns The rescaled embedding, which is `coords` itself.
 */
export function noisyScaleCoords(
  coords: Vectors,
  random: RandomFn,
  maxCoord = 10,
  noise = 1e-4,
): Vectors {
  let largest = 0;
  for (const row of coords) {
    for (const value of row) {
      const magnitude = Math.abs(value);
      if (magnitude > largest) largest = magnitude;
    }
  }
  const expansion = largest > 0 ? maxCoord / largest : 1;
  for (const row of coords) {
    for (let k = 0; k < row.length; k++) {
      row[k] = atNumber(row, k) * expansion + gaussian(random) * noise;
    }
  }
  return coords;
}

/**
 * Draws a standard normal deviate by the Box-Muller transform.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns One deviate of mean 0 and variance 1.
 */
export function gaussian(random: RandomFn): number {
  const uniform = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(uniform)) * Math.cos(2 * Math.PI * random());
}

/**
 * Largest relative Rayleigh residual an eigenpair may have before the run
 * counts as not converged.
 */
const MAX_RESIDUAL = 1e-2;
