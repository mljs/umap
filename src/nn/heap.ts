/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript; the algorithms operating
 * on a heap (buildCandidates, deheapSort, smallestFlagged) live in
 * ./heapOperations.ts to keep files short, and `rejectionSample` is the shared
 * one of ../utils.ts.
 */

import { filled } from '../utils.ts';

/**
 * A heap used for approximate nearest neighbor search, maintaining a list of
 * potential neighbors sorted by their distance and flagging whether a potential
 * neighbor was newly added or not. It is stored as three parallel
 * `(nPoints, size)` matrices: the candidate indices, the candidate distances
 * and the "is new" flags.
 */
export type Heap = [number[][], number[][], number[][]];

/**
 * Constructor for the heap objects.
 * @param nPoints - Number of rows, one per data point.
 * @param size - Number of candidate neighbors kept per data point.
 * @returns A heap whose indices are -1, distances `Infinity` and flags 0.
 */
export function makeHeap(nPoints: number, size: number): Heap {
  return [
    makeArrays(nPoints, size, -1),
    makeArrays(nPoints, size, Infinity),
    makeArrays(nPoints, size, 0),
  ];
}

/**
 * Push a new element onto the heap, ignoring it if the row already holds the
 * same index.
 * @param heap - Heap to push onto.
 * @param row - Data point being addressed.
 * @param weight - Distance used for heap sorting.
 * @param index - Element to add.
 * @param flag - Whether this is to be considered a new addition.
 * @returns 1 if the element was pushed, 0 otherwise.
 */
export function heapPush(
  heap: Heap,
  row: number,
  weight: number,
  index: number,
  flag: number,
): number {
  const rowIndex = Math.floor(row);
  const indices = heap[0][rowIndex];
  const weights = heap[1][rowIndex];
  if (indices === undefined) return 0;
  if (weights === undefined) return 0;

  const worst = weights[0];
  if (worst === undefined || weight >= worst) {
    return 0;
  }

  // Break if we already have this element.
  for (const candidate of indices) {
    if (index === candidate) {
      return 0;
    }
  }

  return uncheckedHeapPush(heap, rowIndex, weight, index, flag);
}

/**
 * Push a new element onto the heap without checking whether the row already
 * holds the same index.
 * @param heap - Heap to push onto.
 * @param row - Data point being addressed.
 * @param weight - Distance used for heap sorting.
 * @param index - Element to add.
 * @param flag - Whether this is to be considered a new addition.
 * @returns 1 if the element was pushed, 0 otherwise.
 */
export function uncheckedHeapPush(
  heap: Heap,
  row: number,
  weight: number,
  index: number,
  flag: number,
): number {
  const indices = heap[0][row];
  const weights = heap[1][row];
  const isNew = heap[2][row];
  if (indices === undefined) return 0;
  if (weights === undefined) return 0;
  if (isNew === undefined) return 0;
  const worst = weights[0];
  if (worst === undefined || weight >= worst) {
    return 0;
  }

  // Insert val at position zero
  weights[0] = weight;
  indices[0] = index;
  isNew[0] = flag;

  // The heap width is read from row zero, as upstream does; every row of a heap
  // built by makeHeap has the same width.
  const firstRow = heap[0][0];
  const heapShape2 = firstRow === undefined ? 0 : firstRow.length;

  // Descend the heap, swapping values until the max heap criterion is met
  let i = 0;
  let iSwap = 0;
  for (;;) {
    const ic1 = 2 * i + 1;
    const ic2 = ic1 + 1;

    if (ic1 >= heapShape2) break;
    const weight1 = weights[ic1];
    if (weight1 === undefined) break;

    if (ic2 >= heapShape2) {
      if (weight1 > weight) {
        iSwap = ic1;
      } else {
        break;
      }
    } else {
      const weight2 = weights[ic2];
      if (weight2 === undefined) break;
      if (weight1 >= weight2) {
        if (weight < weight1) {
          iSwap = ic1;
        } else {
          break;
        }
      } else if (weight < weight2) {
        iSwap = ic2;
      } else {
        break;
      }
    }

    const swapWeight = weights[iSwap];
    const swapIndex = indices[iSwap];
    const swapIsNew = isNew[iSwap];
    if (swapWeight === undefined) break;
    if (swapIndex === undefined) break;
    if (swapIsNew === undefined) break;
    weights[i] = swapWeight;
    indices[i] = swapIndex;
    isNew[i] = swapIsNew;

    i = iSwap;
  }

  weights[i] = weight;
  indices[i] = index;
  isNew[i] = flag;
  return 1;
}

function makeArrays(
  nPoints: number,
  size: number,
  fillValue: number,
): number[][] {
  const arrays: number[][] = [];
  for (let i = 0; i < nPoints; i++) {
    arrays.push(filled(size, fillValue));
  }
  return arrays;
}
