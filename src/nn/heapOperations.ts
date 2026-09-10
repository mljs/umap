/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript; split out of heap.ts to
 * keep files short; fixed the inverted loop condition in `smallestFlagged`,
 * which upstream wrote as `i > ind.length`, so that its body never ran and the
 * function always returned -1.
 */

import type { RandomFn } from '../types.ts';
import { tauRand } from '../utils.ts';

import type { Heap } from './heap.ts';
import { heapPush, makeHeap } from './heap.ts';

/**
 * Neighbor indices and weights unpacked from a heap by {@link deheapSort}.
 */
export interface DeheapSortResult {
  /** Neighbor indices, one row per point, ordered by increasing weight. */
  indices: number[][];
  /** Neighbor weights, one row per point, in increasing order. */
  weights: number[][];
}

/**
 * Build a heap of candidate neighbors for nearest neighbor descent. For each
 * vertex the candidate neighbors are any current neighbors, and any vertices
 * that have the vertex as one of their nearest neighbors.
 * @param currentGraph - Current neighbor heap, whose flags are cleared in place.
 * @param nVertices - Number of vertices in the graph.
 * @param numberOfNeighbors - Number of neighbors stored per vertex.
 * @param maxCandidates - Number of candidates kept per vertex.
 * @param random - Random number generator.
 * @returns The heap of candidate neighbors.
 */
export function buildCandidates(
  currentGraph: Heap,
  nVertices: number,
  numberOfNeighbors: number,
  maxCandidates: number,
  random: RandomFn,
): Heap {
  const candidateNeighbors = makeHeap(nVertices, maxCandidates);
  const graphIndices = currentGraph[0];
  const graphFlags = currentGraph[2];
  for (let i = 0; i < nVertices; i++) {
    const indicesRow = graphIndices[i];
    const flagsRow = graphFlags[i];
    if (indicesRow === undefined || flagsRow === undefined) continue;
    for (let j = 0; j < numberOfNeighbors; j++) {
      const idx = indicesRow[j];
      const isn = flagsRow[j];
      if (idx === undefined || isn === undefined || idx < 0) {
        continue;
      }
      const d = tauRand(random);
      heapPush(candidateNeighbors, i, d, idx, isn);
      heapPush(candidateNeighbors, idx, d, i, isn);
      flagsRow[j] = 0;
    }
  }
  return candidateNeighbors;
}

/**
 * Given a heap of indices and weights, unpack it into arrays of sorted lists of
 * indices and weights by increasing weight. This is effectively just the second
 * half of heap sort, the first half not being required since the data already
 * is a heap.
 * @param heap - Heap to unpack, sorted in place.
 * @returns The sorted indices and weights.
 */
export function deheapSort(heap: Heap): DeheapSortResult {
  const indices = heap[0];
  const weights = heap[1];

  for (let i = 0; i < indices.length; i++) {
    const indHeap = indices[i];
    const distHeap = weights[i];
    if (indHeap === undefined || distHeap === undefined) continue;

    for (let j = 0; j < indHeap.length - 1; j++) {
      const indHeapIndex = indHeap.length - j - 1;
      const distHeapIndex = distHeap.length - j - 1;

      swapPair(indHeap, 0, indHeapIndex);
      swapPair(distHeap, 0, distHeapIndex);

      siftDown(distHeap, indHeap, distHeapIndex, 0);
    }
  }
  return { indices, weights };
}

/**
 * Search the heap for the smallest element that is still flagged, clearing its
 * flag.
 * @param heap - Heap to search.
 * @param row - Data point being addressed.
 * @returns The index stored at the smallest flagged position, or -1 when the
 * row holds no flagged element.
 */
export function smallestFlagged(heap: Heap, row: number): number {
  const ind = heap[0][row];
  const dist = heap[1][row];
  const flag = heap[2][row];
  if (ind === undefined) return -1;
  if (dist === undefined) return -1;
  if (flag === undefined) return -1;

  let minimumDistance = Infinity;
  let resultIndex = -1;

  // Upstream's condition is `i > ind.length`, which is false on entry, so its
  // body never runs and the function always returns -1.
  for (let i = 0; i < ind.length; i++) {
    const distance = dist[i];
    if (flag[i] === 1 && distance !== undefined && distance < minimumDistance) {
      minimumDistance = distance;
      resultIndex = i;
    }
  }

  if (resultIndex >= 0) {
    flag[resultIndex] = 0;
    const result = ind[resultIndex];
    return result === undefined ? -1 : Math.floor(result);
  }
  return -1;
}

/**
 * Restore the heap property for a heap with an out of place element at position
 * `elt`. This works with a heap pair where heap1 carries the weights and heap2
 * holds the corresponding elements.
 * @param heap1 - Weights, which drive the ordering.
 * @param heap2 - Elements moved alongside the weights.
 * @param ceiling - Exclusive upper bound of the region still in the heap.
 * @param elt - Position of the out of place element.
 */
function siftDown(
  heap1: number[],
  heap2: number[],
  ceiling: number,
  elt: number,
): void {
  let current = elt;
  while (current * 2 + 1 < ceiling) {
    const leftChild = current * 2 + 1;
    const rightChild = leftChild + 1;
    let swap = current;

    const leftValue = heap1[leftChild];
    const currentValue = heap1[swap];
    if (
      currentValue !== undefined &&
      leftValue !== undefined &&
      currentValue < leftValue
    ) {
      swap = leftChild;
    }
    if (rightChild < ceiling) {
      const rightValue = heap1[rightChild];
      const swapValue = heap1[swap];
      if (
        swapValue !== undefined &&
        rightValue !== undefined &&
        swapValue < rightValue
      ) {
        swap = rightChild;
      }
    }

    if (swap === current) {
      break;
    } else {
      swapPair(heap1, current, swap);
      swapPair(heap2, current, swap);
      current = swap;
    }
  }
}

function swapPair(array: number[], a: number, b: number): void {
  const first = array[a];
  const second = array[b];
  if (first === undefined || second === undefined) return;
  array[a] = second;
  array[b] = first;
}
