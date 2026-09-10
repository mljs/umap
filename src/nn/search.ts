/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript; split out of ./descent.ts
 * to respect the 250 line file limit, index reads are bounds checked, the
 * candidate row of a vertex is scanned in place instead of being sliced out of
 * the csr indices, the `if (forest)` guard of `initializeSearch` is gone
 * because the parameter is not nullable, and `initFromTree` breaks out of the
 * padding of a leaf instead of returning from the whole function, which
 * upstream does and which left it seeding only the first query point.
 */

import type { SparseMatrix } from '../sparse/sparseMatrix.ts';
import { getCSR } from '../sparse/sparseMatrix.ts';
import type { DistanceFn, RandomFn, Vectors } from '../types.ts';
import { rejectionSample } from '../utils.ts';

import { at } from './checkedAt.ts';
import type { Heap } from './heap.ts';
import { heapPush, makeHeap, uncheckedHeapPush } from './heap.ts';
import { smallestFlagged } from './heapOperations.ts';
import type { FlatTree } from './tree.ts';
import { searchFlatTree } from './tree.ts';

/**
 * Creates the two functions seeding the neighbor heap of a set of query
 * points, bound to a metric.
 * @param distanceFunction - Metric between two data points.
 * @returns The random and the tree based initializations.
 */
export function makeInitializations(
  distanceFunction: DistanceFn,
): Initializations {
  /**
   * Seeds the heap of every query point with randomly drawn data points.
   * @param numberOfNeighbors - Number of data points drawn per query point.
   * @param data - Indexed data points.
   * @param queryPoints - Points being searched for.
   * @param heap - Neighbor heap, updated in place.
   * @param random - Random number generator returning a float in [0, 1).
   */
  function initFromRandom(
    numberOfNeighbors: number,
    data: Vectors,
    queryPoints: Vectors,
    heap: Heap,
    random: RandomFn,
  ): void {
    for (let i = 0; i < queryPoints.length; i++) {
      const queryPoint = at(queryPoints, i);
      const indices = rejectionSample(numberOfNeighbors, data.length, random);
      for (let j = 0; j < indices.length; j++) {
        const index = at(indices, j);
        if (index < 0) {
          continue;
        }
        const d = distanceFunction(at(data, index), queryPoint);
        heapPush(heap, i, d, index, 1);
      }
    }
  }

  /**
   * Seeds the heap of every query point with the leaf of a random projection
   * tree the query point falls in.
   * @param flatTree - Flattened random projection tree to descend.
   * @param data - Indexed data points.
   * @param queryPoints - Points being searched for.
   * @param heap - Neighbor heap, updated in place.
   * @param random - Random number generator returning a float in [0, 1).
   */
  function initFromTree(
    flatTree: FlatTree,
    data: Vectors,
    queryPoints: Vectors,
    heap: Heap,
    random: RandomFn,
  ): void {
    for (let i = 0; i < queryPoints.length; i++) {
      const queryPoint = at(queryPoints, i);
      const indices = searchFlatTree(queryPoint, flatTree, random);

      for (let j = 0; j < indices.length; j++) {
        const index = at(indices, j);
        if (index < 0) {
          // A leaf shorter than `leafSize` is right padded with -1, so the
          // padding only ends this leaf.
          break;
        }
        const d = distanceFunction(at(data, index), queryPoint);
        heapPush(heap, i, d, index, 1);
      }
    }
  }

  return { initFromRandom, initFromTree };
}

/**
 * Creates a nearest neighbor search over a prebuilt neighbor graph, bound to a
 * metric.
 * @param distanceFunction - Metric between two data points.
 * @returns The search function.
 */
export function makeInitializedNNSearch(
  distanceFunction: DistanceFn,
): SearchFn {
  /**
   * Refines a seeded neighbor heap by greedily expanding the closest vertex
   * that has not been expanded yet.
   * @param data - Indexed data points.
   * @param graph - Neighbor graph of the indexed data points.
   * @param initialization - Seeded neighbor heap, refined in place.
   * @param queryPoints - Points being searched for.
   * @returns The refined heap.
   */
  return function nnSearchFn(
    data: Vectors,
    graph: SparseMatrix,
    initialization: Heap,
    queryPoints: Vectors,
  ): Heap {
    const { indices, indptr } = getCSR(graph);
    const heapIndices = initialization[0];

    for (let i = 0; i < queryPoints.length; i++) {
      const seeded = heapIndices[i];
      if (seeded === undefined) continue;
      const queryPoint = at(queryPoints, i);
      const tried = new Set(seeded);
      for (;;) {
        // Find smallest flagged vertex
        const vertex = smallestFlagged(initialization, i);

        if (vertex === -1) {
          break;
        }
        // A vertex past the last non-empty row has no offset, and upstream
        // then slices the csr indices from 0 to their end.
        const from = indptr[vertex] ?? 0;
        const to = indptr[vertex + 1] ?? indices.length;
        for (let c = from; c < to; c++) {
          const candidate = at(indices, c);
          if (
            candidate === vertex ||
            candidate === -1 ||
            tried.has(candidate)
          ) {
            continue;
          }
          const d = distanceFunction(at(data, candidate), queryPoint);
          uncheckedHeapPush(initialization, i, d, candidate, 1);
          tried.add(candidate);
        }
      }
    }
    return initialization;
  };
}

/**
 * Builds and seeds the neighbor heap of a set of query points, from random
 * data points and from every tree of a random projection forest.
 * @param forest - Flattened trees of the random projection forest.
 * @param data - Indexed data points.
 * @param queryPoints - Points being searched for.
 * @param numberOfNeighbors - Number of neighbors kept per query point.
 * @param initFromRandom - Random initialization from {@link makeInitializations}.
 * @param initFromTree - Tree initialization from {@link makeInitializations}.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns The seeded heap.
 */
export function initializeSearch(
  forest: FlatTree[],
  data: Vectors,
  queryPoints: Vectors,
  numberOfNeighbors: number,
  initFromRandom: InitFromRandomFn,
  initFromTree: InitFromTreeFn,
  random: RandomFn,
): Heap {
  const results = makeHeap(queryPoints.length, numberOfNeighbors);
  initFromRandom(numberOfNeighbors, data, queryPoints, results, random);
  for (const flatTree of forest) {
    initFromTree(flatTree, data, queryPoints, results, random);
  }
  return results;
}

/**
 * Seeds a neighbor heap with randomly drawn data points.
 */
export type InitFromRandomFn = (
  numberOfNeighbors: number,
  data: Vectors,
  queryPoints: Vectors,
  heap: Heap,
  random: RandomFn,
) => void;

/**
 * Seeds a neighbor heap with the leaves of a random projection tree.
 */
export type InitFromTreeFn = (
  flatTree: FlatTree,
  data: Vectors,
  queryPoints: Vectors,
  heap: Heap,
  random: RandomFn,
) => void;

/**
 * The two heap initializations returned by {@link makeInitializations}.
 */
export interface Initializations {
  /** Seeds a neighbor heap with randomly drawn data points. */
  initFromRandom: InitFromRandomFn;
  /** Seeds a neighbor heap with the leaves of a random projection tree. */
  initFromTree: InitFromTreeFn;
}

/**
 * Refines a seeded neighbor heap over a prebuilt neighbor graph, as returned
 * by {@link makeInitializedNNSearch}.
 */
export type SearchFn = (
  data: Vectors,
  graph: SparseMatrix,
  initialization: Heap,
  queryPoints: Vectors,
) => Heap;
