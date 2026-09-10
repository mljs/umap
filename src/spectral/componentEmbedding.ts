import { rowAt } from '../fuzzy/rowAt.ts';
import { atFloat64, atInt32, atNumber } from '../sparse/arrayAt.ts';
import type { RandomFn, Vectors } from '../types.ts';

import { componentRadius, metaLayout } from './componentLayout.ts';
import { componentSubgraph } from './components.ts';
import type { CSRMatrix } from './csr.ts';
import { spectralVectors } from './lanczos.ts';
import { vectorAt } from './ritz.ts';

/**
 * Options of {@link componentVectors}.
 */
export interface ComponentVectorsOptions {
  /** Random number generator returning a float in [0, 1). */
  random: RandomFn;
  /** Largest relative Rayleigh residual an eigenpair may have. */
  maxResidual: number;
}

/**
 * Options of {@link placeComponents}.
 */
export interface PlaceComponentsOptions extends ComponentVectorsOptions {
  /** Number of connected components. */
  count: number;
  /** Dimension of the space the data is embedded into. */
  dim: number;
  /** The data the graph was built from, one array per point. */
  data: Vectors;
}

/**
 * Eigenvectors of the `dim` smallest non-zero eigenvalues of the normalized
 * Laplacian of a connected graph.
 * @param csr - The connected graph.
 * @param dim - Number of eigenvectors wanted.
 * @param options - Random source and convergence bound.
 * @returns The eigenvectors, or `null` when the run failed or when an
 * eigenpair did not converge to `maxResidual`.
 */
export function componentVectors(
  csr: CSRMatrix,
  dim: number,
  options: ComponentVectorsOptions,
): Float64Array[] | null {
  const { random, maxResidual } = options;
  let pairs;
  try {
    pairs = spectralVectors(csr, dim, { random });
  } catch {
    return null;
  }
  const vectors: Float64Array[] = [];
  for (const pair of pairs) {
    // Negated so a NaN residual counts as a failure.
    if (!(pair.residual <= maxResidual)) return null;
    vectors.push(pair.vector);
  }
  return vectors;
}

/**
 * Embeds every connected component of a graph inside its own slot.
 *
 * A component too small to carry `dim` eigenvectors, and a component whose
 * eigenvectors did not converge, are drawn uniformly inside their slot rather
 * than failing the whole layout.
 * @param embedding - The embedding to fill, updated in place.
 * @param csr - The whole graph.
 * @param label - Component label of every vertex.
 * @param options - Component count, dimension, data, random source and bound.
 */
export function placeComponents(
  embedding: Vectors,
  csr: CSRMatrix,
  label: Int32Array,
  options: PlaceComponentsOptions,
): void {
  const { count, dim, data, random, maxResidual } = options;
  const meta = metaLayout(data, label, count, dim);
  for (let component = 0; component < count; component++) {
    const radius = componentRadius(meta, component);
    const center = rowAt(meta, component);
    const { csr: subgraph, nodes } = componentSubgraph(csr, label, component);
    const vectors = isEmbeddable(subgraph.n, dim)
      ? componentVectors(subgraph, dim, { random, maxResidual })
      : null;
    if (vectors === null) {
      scatter(embedding, nodes, center, dim, radius, random);
      continue;
    }
    place(embedding, nodes, center, vectors, radius);
  }
}

/**
 * Whether a component holds enough vertices for a spectral embedding.
 *
 * The bound is the one of the reference implementation: below it the
 * eigenvectors carry no usable layout.
 * @param size - Number of vertices of the component.
 * @param dim - Dimension of the space the data is embedded into.
 * @returns Whether the component can be embedded spectrally.
 */
function isEmbeddable(size: number, dim: number): boolean {
  return size >= 2 * dim && size > dim + 1;
}

/**
 * Draws the vertices of a component uniformly inside its slot.
 * @param embedding - The embedding to fill, updated in place.
 * @param nodes - Index in the whole graph of every vertex of the component.
 * @param center - Position of the component.
 * @param dim - Dimension of the space the data is embedded into.
 * @param radius - Radius of the slot of the component.
 * @param random - Random number generator returning a float in [0, 1).
 */
function scatter(
  embedding: Vectors,
  nodes: Int32Array,
  center: number[],
  dim: number,
  radius: number,
  random: RandomFn,
): void {
  for (let a = 0; a < nodes.length; a++) {
    const row = rowAt(embedding, atInt32(nodes, a));
    for (let k = 0; k < dim; k++) {
      row[k] = (random() * 2 - 1) * radius + atNumber(center, k);
    }
  }
}

/**
 * Writes the eigenvectors of a component into the embedding, rescaled so the
 * component fills its slot.
 * @param embedding - The embedding to fill, updated in place.
 * @param nodes - Index in the whole graph of every vertex of the component.
 * @param center - Position of the component.
 * @param vectors - Eigenvectors of the component.
 * @param radius - Radius of the slot of the component.
 */
function place(
  embedding: Vectors,
  nodes: Int32Array,
  center: number[],
  vectors: Float64Array[],
  radius: number,
): void {
  let largest = 0;
  for (const vector of vectors) {
    for (let a = 0; a < vector.length; a++) {
      const magnitude = Math.abs(atFloat64(vector, a));
      if (magnitude > largest) largest = magnitude;
    }
  }
  const expansion = largest > 0 ? radius / largest : 1;
  for (let a = 0; a < nodes.length; a++) {
    const row = rowAt(embedding, atInt32(nodes, a));
    for (let k = 0; k < vectors.length; k++) {
      row[k] =
        atFloat64(vectorAt(vectors, k), a) * expansion + atNumber(center, k);
    }
  }
}
