/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript as a standalone function
 * taking its inputs explicitly instead of reading them off the UMAP instance,
 * and split out of simplicialSet.ts to respect the 250 line file limit. An
 * empty `knnIndices`, or a `sigmas`/`rhos`/`knnDistances` shorter than it, now
 * raises a RangeError where upstream silently produced NaN and `undefined`
 * entries. The arithmetic and the iteration order are unchanged.
 */

import { atNumber } from '../sparse/arrayAt.ts';
import { zeros } from '../utils.ts';

import { rowAt } from './rowAt.ts';

/**
 * Entries of the 1-skeleton of a fuzzy simplicial set, as parallel arrays.
 */
export interface ComputeMembershipStrengthsOptions {
  /**
   * Whether the neighbour set is bipartite, meaning rows and columns are drawn
   * from different index spaces. Set it when projecting new data, where a row
   * is a position in the batch and a column is a training point.
   * @default false
   */
  bipartite?: boolean;
}

export interface MembershipStrengths {
  /** Row index of each entry. */
  rows: number[];
  /** Column index of each entry. */
  cols: number[];
  /** Membership strength of each entry. */
  vals: number[];
}

/**
 * Construct the membership strength data for the 1-skeleton of each local
 * fuzzy simplicial set -- this is formed as a sparse matrix where each row is
 * a local fuzzy simplicial set, with a membership strength for the 1-simplex
 * to each other data point.
 * @param knnIndices - Nearest neighbor index of each point, `-1` when missing.
 * @param knnDistances - Distance to each of those nearest neighbors.
 * @param sigmas - Kernel normalization factor of each point.
 * @param rhos - Nearest locally connected distance of each point.
 * @param options
 * @returns The entries of the 1-skeleton, in row major order.
 */
export function computeMembershipStrengths(
  knnIndices: number[][],
  knnDistances: number[][],
  sigmas: number[],
  rhos: number[],
  options: ComputeMembershipStrengthsOptions = {},
): MembershipStrengths {
  const { bipartite = false } = options;
  const nSamples = knnIndices.length;
  const firstRow = knnIndices[0];
  if (firstRow === undefined) {
    throw new RangeError('knnIndices must hold at least one row');
  }
  const numberOfNeighbors = firstRow.length;

  const rows = zeros(nSamples * numberOfNeighbors);
  const cols = zeros(nSamples * numberOfNeighbors);
  const vals = zeros(nSamples * numberOfNeighbors);

  for (let i = 0; i < nSamples; i++) {
    const indicesRow = rowAt(knnIndices, i);
    const distancesRow = rowAt(knnDistances, i);
    const rhoI = atNumber(rhos, i);
    const sigmaI = atNumber(sigmas, i);
    const offset = i * numberOfNeighbors;

    for (let j = 0; j < numberOfNeighbors; j++) {
      let val = 0;
      const index = atNumber(indicesRow, j);
      if (index === -1) {
        continue; // We didn't get the full knn for i
      }
      if (!bipartite && index === i) {
        // Only meaningful when rows and columns share an index space, where
        // `index === i` means "self". Projecting new data indexes rows by
        // position in the batch and columns by training point, so the test
        // would zero an unrelated pair (umap-learn guards it the same way,
        // umap_.py:438-439).
        val = 0;
      } else {
        const d = atNumber(distancesRow, j) - rhoI;
        if (d <= 0) {
          val = 1;
        } else {
          val = Math.exp(-(d / sigmaI));
        }
      }

      rows[offset + j] = i;
      cols[offset + j] = index;
      vals[offset + j] = val;
    }
  }

  return { rows, cols, vals };
}
