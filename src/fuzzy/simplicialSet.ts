/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript as standalone functions
 * taking their inputs explicitly instead of reading them off the UMAP instance,
 * the trailing positional parameters gathered into an options object, and the
 * number of vectors passed as a count instead of the source data, which is only
 * ever used for its length. The two helpers of the fuzzy set construction live
 * in the sibling modules re-exported below, to respect the 250 line file limit.
 * The arithmetic and the operation order are unchanged.
 */

import {
  NormType,
  SparseMatrix,
  add,
  eliminateZeros,
  multiplyScalar,
  normalize,
  pairwiseMultiply,
  subtract,
  transpose,
} from '../sparse/index.ts';

import { computeMembershipStrengths } from './membership.ts';
import { smoothKNNDistance } from './smoothKnn.ts';

export * from './membership.ts';
export * from './smoothKnn.ts';

/**
 * Options of {@link fuzzySimplicialSet}.
 */
export interface FuzzySimplicialSetOptions {
  /**
   * Number of vectors in the source data. The resulting graph is square of
   * that size.
   */
  nVectors: number;
  /**
   * Size of the local neighborhood used to approximate the geodesic distance.
   */
  numberOfNeighbors: number;
  /**
   * Interpolation between the fuzzy union (1) and the fuzzy intersection (0)
   * of the local simplicial sets.
   * @default 1
   */
  setOperationMixRatio?: number;
  /**
   * Number of nearest neighbors that should be assumed to be locally connected.
   * @default 1
   */
  localConnectivity?: number;
}

/**
 * Given a set of data X, a neighborhood size, and a measure of distance
 * compute the fuzzy simplicial set (here represented as a fuzzy graph in
 * the form of a sparse matrix) associated to the data. This is done by
 * locally approximating geodesic distance at each point, creating a fuzzy
 * simplicial set for each such point, and then combining all the local
 * fuzzy simplicial sets into a global one via a fuzzy union.
 * @param knnIndices - Nearest neighbor index of each point, `-1` when missing.
 * @param knnDistances - Sorted distances to each of those nearest neighbors.
 * @param options - Construction options.
 * @returns The fuzzy graph, as an `nVectors` by `nVectors` sparse matrix.
 */
export function fuzzySimplicialSet(
  knnIndices: number[][],
  knnDistances: number[][],
  options: FuzzySimplicialSetOptions,
): SparseMatrix {
  const {
    nVectors,
    numberOfNeighbors,
    setOperationMixRatio = 1,
    localConnectivity = 1,
  } = options;

  const { sigmas, rhos } = smoothKNNDistance(knnDistances, numberOfNeighbors, {
    localConnectivity,
  });

  const { rows, cols, vals } = computeMembershipStrengths(
    knnIndices,
    knnDistances,
    sigmas,
    rhos,
  );

  const size = [nVectors, nVectors];
  const sparseMatrix = new SparseMatrix(rows, cols, vals, size);

  const transposed = transpose(sparseMatrix);
  const prodMatrix = pairwiseMultiply(sparseMatrix, transposed);

  const a = subtract(add(sparseMatrix, transposed), prodMatrix);
  const b = multiplyScalar(a, setOperationMixRatio);
  const c = multiplyScalar(prodMatrix, 1 - setOperationMixRatio);
  return add(b, c);
}

/**
 * Options of {@link categoricalSimplicialSetIntersection}.
 */
export interface CategoricalSimplicialSetIntersectionOptions {
  /** Distance applied between two points carrying different labels. */
  farDist: number;
  /**
   * Distance applied when either of the two labels is unknown (`-1`).
   * @default 1
   */
  unknownDist?: number;
}

/**
 * Combine a fuzzy simplicial set with another fuzzy simplicial set
 * generated from categorical data using categorical distances. The target
 * data is assumed to be categorical label data (a vector of labels),
 * and this will update the fuzzy simplicial set to respect that label data.
 * @param simplicialSet - The fuzzy graph to update.
 * @param target - Label of each point, `-1` when unknown.
 * @param options - Intersection options.
 * @returns The updated fuzzy graph.
 */
export function categoricalSimplicialSetIntersection(
  simplicialSet: SparseMatrix,
  target: number[],
  options: CategoricalSimplicialSetIntersectionOptions,
): SparseMatrix {
  const { farDist, unknownDist = 1 } = options;
  const intersection = fastIntersection(simplicialSet, target, {
    unknownDist,
    farDist,
  });
  return resetLocalConnectivity(eliminateZeros(intersection));
}

/**
 * Options of {@link fastIntersection}.
 */
export interface FastIntersectionOptions {
  /**
   * Distance applied when either of the two labels is unknown (`-1`).
   * @default 1
   */
  unknownDist?: number;
  /**
   * Distance applied between two points carrying different labels.
   * @default 5
   */
  farDist?: number;
}

/**
 * Under the assumption of categorical distance for the intersecting
 * simplicial set perform a fast intersection.
 * @param graph - The fuzzy graph to intersect.
 * @param target - Label of each point, `-1` when unknown.
 * @param options - Intersection options.
 * @returns The intersected graph, of the same shape as `graph`.
 */
export function fastIntersection(
  graph: SparseMatrix,
  target: number[],
  options: FastIntersectionOptions = {},
): SparseMatrix {
  const { unknownDist = 1, farDist = 5 } = options;
  return graph.map((value, row, col) => {
    if (target[row] === -1 || target[col] === -1) {
      return value * Math.exp(-unknownDist);
    } else if (target[row] !== target[col]) {
      return value * Math.exp(-farDist);
    } else {
      return value;
    }
  });
}

/**
 * Reset the local connectivity requirement -- each data sample should
 * have complete confidence in at least one 1-simplex in the simplicial set.
 * We can enforce this by locally rescaling confidences, and then remerging the
 * different local simplicial sets together.
 * @param simplicialSet - The fuzzy graph to rescale.
 * @returns The rescaled graph, with its zero entries dropped.
 */
export function resetLocalConnectivity(
  simplicialSet: SparseMatrix,
): SparseMatrix {
  const normalized = normalize(simplicialSet, NormType.max);
  const transposed = transpose(normalized);
  const prodMatrix = pairwiseMultiply(transposed, normalized);
  const merged = add(normalized, subtract(transposed, prodMatrix));
  return eliminateZeros(merged);
}
