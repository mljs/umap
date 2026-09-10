/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript; index reads are bounds
 * checked and the rows of the candidate heap are hoisted into locals, and the
 * query side (makeInitializations, makeInitializedNNSearch, initializeSearch)
 * lives in ./search.ts, re-exported below, to respect the 250 line file limit.
 * The distance evaluations, the random draws and their order are unchanged.
 */

import type { DistanceFn, RandomFn, Vectors } from '../types.ts';
import { rejectionSample, tauRand } from '../utils.ts';

import { at } from './checkedAt.ts';
import type { Heap } from './heap.ts';
import { heapPush, makeHeap } from './heap.ts';
import type { DeheapSortResult } from './heapOperations.ts';
import { buildCandidates, deheapSort } from './heapOperations.ts';

export * from './search.ts';

/**
 * Creates a nearest neighbor descent function bound to a metric and to a
 * random number generator.
 * @param distanceFunction - Metric between two data points.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns The nearest neighbor descent function.
 */
export function makeNNDescent(
  distanceFunction: DistanceFn,
  random: RandomFn,
): NNDescentFn {
  /**
   * Computes an approximate k nearest neighbor graph by nearest neighbor
   * descent.
   * @param data - Data points to index.
   * @param leafArray - Leaves of a random projection forest, used to seed the graph.
   * @param numberOfNeighbors - Number of neighbors to compute per data point.
   * @param nIters - Maximum number of descent iterations.
   * @param maxCandidates - Number of candidate neighbors considered per iteration.
   * @param delta - Fraction of the graph below which an iteration stops the descent.
   * @param rho - Probability of skipping a candidate in an iteration.
   * @param rpTreeInit - Whether to seed the graph with the forest leaves.
   * @returns The neighbor indices and distances, sorted by increasing distance.
   */
  return function nNDescent(
    data: Vectors,
    leafArray: Vectors,
    numberOfNeighbors: number,
    nIters = 10,
    maxCandidates = 50,
    delta = 0.001,
    rho = 0.5,
    rpTreeInit = true,
  ): DeheapSortResult {
    const nVertices = data.length;
    const currentGraph = makeHeap(data.length, numberOfNeighbors);

    for (let i = 0; i < data.length; i++) {
      const point = at(data, i);
      const indices = rejectionSample(numberOfNeighbors, data.length, random);
      for (let j = 0; j < indices.length; j++) {
        const index = at(indices, j);
        const d = distanceFunction(point, at(data, index));

        heapPush(currentGraph, i, d, index, 1);
        heapPush(currentGraph, index, d, i, 1);
      }
    }
    if (rpTreeInit) {
      seedFromLeaves(currentGraph, data, leafArray, distanceFunction);
    }

    for (let n = 0; n < nIters; n++) {
      const candidateNeighbors = buildCandidates(
        currentGraph,
        nVertices,
        numberOfNeighbors,
        maxCandidates,
        random,
      );
      const candidateIndices = candidateNeighbors[0];
      const candidateFlags = candidateNeighbors[2];

      let c = 0;
      for (let i = 0; i < nVertices; i++) {
        const indicesRow = candidateIndices[i];
        const flagsRow = candidateFlags[i];
        if (indicesRow === undefined || flagsRow === undefined) continue;
        for (let j = 0; j < maxCandidates; j++) {
          const pIndex = indicesRow[j];
          if (pIndex === undefined) continue;
          const p = Math.floor(pIndex);
          if (p < 0 || tauRand(random) < rho) {
            continue;
          }
          const pPoint = at(data, p);
          for (let k = 0; k < maxCandidates; k++) {
            const qIndex = indicesRow[k];
            if (qIndex === undefined) continue;
            const q = Math.floor(qIndex);
            const cj = flagsRow[j];
            const ck = flagsRow[k];
            if (q < 0 || (!cj && !ck)) {
              continue;
            }

            const d = distanceFunction(pPoint, at(data, q));
            c += heapPush(currentGraph, p, d, q, 1);
            c += heapPush(currentGraph, q, d, p, 1);
          }
        }
      }
      if (c <= delta * numberOfNeighbors * data.length) {
        break;
      }
    }
    return deheapSort(currentGraph);
  };
}

/**
 * Approximate nearest neighbor descent, as returned by {@link makeNNDescent}.
 * The optional parameters default to `nIters = 10`, `maxCandidates = 50`,
 * `delta = 0.001`, `rho = 0.5` and `rpTreeInit = true`.
 */
export type NNDescentFn = (
  data: Vectors,
  leafArray: Vectors,
  numberOfNeighbors: number,
  nIters?: number,
  maxCandidates?: number,
  delta?: number,
  rho?: number,
  rpTreeInit?: boolean,
) => DeheapSortResult;

/**
 * Pushes every pair of points sharing a random projection forest leaf onto the
 * neighbor heap.
 * @param currentGraph - Neighbor heap, updated in place.
 * @param data - Data points being indexed.
 * @param leafArray - Leaves of the forest, each right padded with negatives.
 * @param distanceFunction - Metric between two data points.
 */
function seedFromLeaves(
  currentGraph: Heap,
  data: Vectors,
  leafArray: Vectors,
  distanceFunction: DistanceFn,
): void {
  for (let n = 0; n < leafArray.length; n++) {
    const leaf = at(leafArray, n);
    for (let i = 0; i < leaf.length; i++) {
      const left = at(leaf, i);
      if (left < 0) {
        break;
      }
      const leftPoint = at(data, left);
      for (let j = i + 1; j < leaf.length; j++) {
        const right = at(leaf, j);
        if (right < 0) {
          break;
        }
        const d = distanceFunction(leftPoint, at(data, right));
        heapPush(currentGraph, left, d, right, 1);
        heapPush(currentGraph, right, d, left, 1);
      }
    }
  }
}
