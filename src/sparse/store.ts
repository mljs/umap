/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript, and the `Map` of entries
 * keyed by `${row}:${col}` replaced by a coordinate (triplet) list in parallel
 * typed arrays with a lazily built key index. Iteration order, duplicate
 * handling and float arithmetic are unchanged. Split out of ./sparseMatrix.ts
 * to respect the 250 line file limit.
 */

import { atFloat64, atInt32, atNumber } from './arrayAt.ts';

/**
 * One entry of a sparse matrix.
 */
export interface SparseEntry {
  /** Stored value. */
  value: number;
  /** Row index. */
  row: number;
  /** Column index. */
  col: number;
}

/**
 * Maps the key of an entry to its slot in the triplet arrays.
 * @internal
 */
export type SparseIndex = Map<number | string, number>;

/**
 * Mutable coordinate (triplet) storage backing a sparse matrix. The arrays may
 * be longer than `length`; only the first `length` slots hold entries.
 * @internal
 */
export interface SparseStore {
  rows: Int32Array;
  cols: Int32Array;
  values: Float64Array;
  length: number;
  index: SparseIndex | null;
  irregular: boolean;
  numericKeys: boolean;
}

/**
 * Copies the first `length` elements of `array` into a plain array.
 * @internal
 */
export function toNumbers(
  array: Int32Array | Float64Array,
  length: number,
): number[] {
  const output = new Array<number>(length);
  if (array instanceof Int32Array) {
    for (let i = 0; i < length; i++) output[i] = atInt32(array, i);
  } else {
    for (let i = 0; i < length; i++) output[i] = atFloat64(array, i);
  }
  return output;
}

/**
 * Builds the triplet storage of a matrix, dropping duplicate coordinates: a
 * repeated `(row, col)` keeps the slot of its first insertion and takes the
 * value of the last one.
 * @internal
 */
export function createStore(
  rows: number[],
  cols: number[],
  values: number[],
  nRows: number,
  nCols: number,
): SparseStore {
  if (rows.length !== cols.length || rows.length !== values.length) {
    throw new Error(LENGTH_ERROR);
  }
  const count = values.length;
  const numericKeys = nRows * nCols <= Number.MAX_SAFE_INTEGER;
  const store: SparseStore = {
    rows: count === 0 ? EMPTY_INT32 : new Int32Array(count),
    cols: count === 0 ? EMPTY_INT32 : new Int32Array(count),
    values: count === 0 ? EMPTY_FLOAT64 : new Float64Array(count),
    length: 0,
    index: null,
    irregular: false,
    numericKeys,
  };
  if (count === 0) return store;

  const index: SparseIndex = new Map();
  let length = 0;
  for (let i = 0; i < count; i++) {
    const row = atNumber(rows, i);
    const col = atNumber(cols, i);
    if (!(row < nRows && col < nCols)) throw new Error(DIM_ERROR);
    let key: number | string;
    if (numericKeys && row >= 0 && col >= 0) {
      key = row * nCols + col;
    } else {
      key = `${row}:${col}`;
      store.irregular = true;
    }
    const slot = index.get(key);
    if (slot === undefined) {
      index.set(key, length);
      store.rows[length] = row;
      store.cols[length] = col;
      store.values[length] = atNumber(values, i);
      length++;
    } else {
      store.values[slot] = atNumber(values, i);
    }
  }
  store.length = length;
  store.index = index;
  return store;
}

/**
 * Builds, and caches on the store, the `key -> slot` index of the entries.
 * @internal
 */
export function ensureIndex(store: SparseStore, nCols: number): SparseIndex {
  const cached = store.index;
  if (cached !== null) return cached;
  const index: SparseIndex = new Map();
  const rows = store.rows;
  const cols = store.cols;
  const length = store.length;
  if (store.numericKeys && !store.irregular) {
    for (let i = 0; i < length; i++) {
      index.set(atInt32(rows, i) * nCols + atInt32(cols, i), i);
    }
  } else {
    for (let i = 0; i < length; i++) {
      index.set(keyOf(store, nCols, atInt32(rows, i), atInt32(cols, i)), i);
    }
  }
  store.index = index;
  return index;
}

/**
 * Key of an entry: `row * nCols + col` when both coordinates are non-negative
 * and the shape allows an exact key, `${row}:${col}` otherwise.
 * @internal
 */
export function keyOf(
  store: SparseStore,
  nCols: number,
  row: number,
  col: number,
): number | string {
  if (store.numericKeys && row >= 0 && col >= 0) return row * nCols + col;
  return `${row}:${col}`;
}

/**
 * Densifies a store into an array of rows, absent entries being zero.
 * @internal
 */
export function toDense(
  store: SparseStore,
  nRows: number,
  nCols: number,
): number[][] {
  const output: number[][] = [];
  for (let i = 0; i < nRows; i++) {
    output.push(nCols > 0 ? new Array<number>(nCols).fill(0) : []);
  }
  const { rows, cols, values, length } = store;
  for (let i = 0; i < length; i++) {
    const target = output[atInt32(rows, i)];
    if (target === undefined) throw new RangeError(DIM_ERROR);
    target[atInt32(cols, i)] = atFloat64(values, i);
  }
  return output;
}

/**
 * Grows the triplet arrays geometrically so that `need` entries fit.
 * @internal
 */
export function growStore(store: SparseStore, need: number): void {
  const capacity = store.rows.length;
  if (need <= capacity) return;
  let next = capacity * 2;
  if (next < 8) next = 8;
  if (next < need) next = need;
  const rows = new Int32Array(next);
  rows.set(store.rows);
  const cols = new Int32Array(next);
  cols.set(store.cols);
  const values = new Float64Array(next);
  values.set(store.values);
  store.rows = rows;
  store.cols = cols;
  store.values = values;
}

/**
 * Message of the error thrown when a coordinate is out of bounds.
 * @internal
 */
export const DIM_ERROR =
  'row and/or col specified outside of matrix dimensions';

/**
 * Message of the error thrown when the triplet arrays have different lengths.
 * @internal
 */
export const LENGTH_ERROR =
  'rows, cols and values arrays must all have the same length';

/**
 * Shared empty row and column array of a matrix without entries.
 * @internal
 */
export const EMPTY_INT32 = new Int32Array(0);

/**
 * Shared empty value array of a matrix without entries.
 * @internal
 */
export const EMPTY_FLOAT64 = new Float64Array(0);
