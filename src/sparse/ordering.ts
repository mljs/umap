/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript. Upstream materialized and
 * sorted arrays of entry objects and of column indices; the two orderings
 * produced here are identical, only slots are permuted instead of entries and
 * the lexicographic order of a column is precomputed as a number instead of a
 * string. Split out of ./store.ts and ./normalize.ts to respect the 250 line
 * file limit.
 */

import { RANGE_ERROR, atFloat64, atInt32, atNumber } from './arrayAt.ts';
import type { SparseStore } from './store.ts';

/**
 * Slots, that is indices into the triplet arrays, ordered numerically by row
 * then column. A stable two-pass counting sort is used when the dimensions
 * make it cheap, and a comparator sort otherwise.
 * @internal
 */
export function sortedSlots(
  store: SparseStore,
  nRows: number,
  nCols: number,
): Int32Array {
  const { rows, cols, length, irregular } = store;
  const output = new Int32Array(length);
  if (length === 0) return output;
  if (
    !irregular &&
    (nRows | 0) === nRows &&
    (nCols | 0) === nCols &&
    nRows >= 0 &&
    nCols >= 0 &&
    nRows + nCols <= 4 * length + 65536
  ) {
    const byColumn = countingSort(cols, length, nCols);
    return countingSortBy(rows, byColumn, length, nRows, output);
  }
  const order = new Array<number>(length);
  for (let i = 0; i < length; i++) order[i] = i;
  order.sort((x, y) =>
    atInt32(rows, x) === atInt32(rows, y)
      ? atInt32(cols, x) - atInt32(cols, y)
      : atInt32(rows, x) - atInt32(rows, y),
  );
  for (let i = 0; i < length; i++) output[i] = atNumber(order, i);
  return output;
}

/**
 * Sorts slots by the ordering key of their column, with an insertion sort for
 * the short rows that dominate a nearest neighbor graph.
 * @param slots - Slots to sort in place.
 * @param lex - Ordering key of each slot.
 */
export function sortByLexKey(slots: number[], lex: Float64Array): void {
  const size = slots.length;
  if (size > 64) {
    slots.sort((x, y) => atFloat64(lex, x) - atFloat64(lex, y));
    return;
  }
  // The reads below are inlined rather than routed through `atNumber`, whose
  // shared load site turns megamorphic and costs ~11% of `normalize`.
  for (let i = 1; i < size; i++) {
    const slot = slots[i];
    if (slot === undefined) throw new RangeError(RANGE_ERROR);
    const key = atFloat64(lex, slot);
    let j = i - 1;
    while (j >= 0) {
      const previous = slots[j];
      if (previous === undefined) throw new RangeError(RANGE_ERROR);
      if (atFloat64(lex, previous) <= key) break;
      slots[j + 1] = previous;
      j--;
    }
    slots[j + 1] = slot;
  }
}

/**
 * Orders two column indices the way `Array#sort` orders them without a
 * comparator, that is by their decimal representation.
 * @param x - Left column index.
 * @param y - Right column index.
 * @returns A negative number, zero or a positive number.
 */
export function compareAsString(x: number, y: number): number {
  const left = `${x}`;
  const right = `${y}`;
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

/**
 * Maps a non-negative integer to a number whose natural order matches the
 * lexicographic order of its decimal representation. Each decimal digit `d`
 * becomes the base-11 digit `d + 1`, padded on the right with 0, so that a
 * proper prefix always sorts first, exactly like a string comparison.
 * @param value - Integer to map.
 * @returns The ordering key.
 */
export function lexKey(value: number): number {
  let digits = 1;
  let rest = value;
  while (rest >= 10) {
    rest = (rest / 10) | 0;
    digits++;
  }
  let weight = atFloat64(POW11, 10 - digits);
  let key = 0;
  rest = value;
  for (let j = 0; j < digits; j++) {
    key += ((rest % 10) + 1) * weight;
    rest = (rest / 10) | 0;
    weight *= 11;
  }
  return key;
}

const POW11 = new Float64Array(11);
POW11[0] = 1;
for (let i = 1; i < 11; i++) POW11[i] = atFloat64(POW11, i - 1) * 11;

/**
 * Stable counting sort of the slots `0 .. length - 1` by `keys`.
 * @param keys - Key of each slot.
 * @param length - Number of slots.
 * @param range - Exclusive upper bound of the keys.
 * @returns The sorted slots.
 */
function countingSort(
  keys: Int32Array,
  length: number,
  range: number,
): Int32Array {
  const counts = prefixCounts(keys, length, range);
  const output = new Int32Array(length);
  for (let i = 0; i < length; i++) {
    const key = atInt32(keys, i);
    const position = atInt32(counts, key);
    counts[key] = position + 1;
    output[position] = i;
  }
  return output;
}

/**
 * Stable counting sort of already ordered slots by `keys`.
 * @param keys - Key of each slot.
 * @param slots - Slots in the order left by the previous pass.
 * @param length - Number of slots.
 * @param range - Exclusive upper bound of the keys.
 * @param output - Array receiving the sorted slots.
 * @returns `output`.
 */
function countingSortBy(
  keys: Int32Array,
  slots: Int32Array,
  length: number,
  range: number,
  output: Int32Array,
): Int32Array {
  const counts = prefixCounts(keys, length, range);
  for (let i = 0; i < length; i++) {
    const slot = atInt32(slots, i);
    const key = atInt32(keys, slot);
    const position = atInt32(counts, key);
    counts[key] = position + 1;
    output[position] = slot;
  }
  return output;
}

/**
 * Exclusive prefix sums of the key counts, so that `counts[k]` is the position
 * at which the slots of key `k` start.
 * @param keys - Key of each slot.
 * @param length - Number of slots.
 * @param range - Exclusive upper bound of the keys.
 * @returns The prefix sums, one longer than `range`.
 */
function prefixCounts(
  keys: Int32Array,
  length: number,
  range: number,
): Int32Array {
  const counts = new Int32Array(range + 1);
  for (let i = 0; i < length; i++) {
    const key = atInt32(keys, i) + 1;
    counts[key] = atInt32(counts, key) + 1;
  }
  for (let i = 0; i < range; i++) {
    counts[i + 1] = atInt32(counts, i + 1) + atInt32(counts, i);
  }
  return counts;
}
