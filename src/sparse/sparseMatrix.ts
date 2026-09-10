/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript, and entries are kept as a
 * coordinate (triplet) list in parallel typed arrays with a lazily built key
 * index instead of a `Map` keyed by `${row}:${col}` strings. Iteration order,
 * duplicate handling, bounds checks and float arithmetic are unchanged. The
 * helpers and operations live in the sibling modules re-exported below, to
 * respect the 250 line file limit.
 */

import { atFloat64, atInt32 } from './arrayAt.ts';
import { sortedSlots } from './ordering.ts';
import type { SparseEntry, SparseStore } from './store.ts';
import {
  DIM_ERROR,
  createStore,
  ensureIndex,
  growStore,
  keyOf,
  toDense,
  toNumbers,
} from './store.ts';

export * from './elementWise.ts';
export * from './normalize.ts';
export * from './operations.ts';
export * from './store.ts';

/**
 * Internal 2-dimensional sparse matrix. Entries are held in insertion order; a
 * duplicate `(row, col)` keeps the position of its first insertion but takes
 * the value of the last one. Only the upper bound of a coordinate is checked,
 * so negative coordinates are accepted, exactly as upstream accepts them.
 */
export class SparseMatrix {
  /** Number of rows of the matrix. */
  readonly nRows: number;
  /** Number of columns of the matrix. */
  readonly nCols: number;
  /**
   * Backing triplet storage.
   * @internal
   */
  readonly store: SparseStore;

  /**
   * Builds a matrix from parallel entry arrays.
   * @param rows - Row index of each entry.
   * @param cols - Column index of each entry.
   * @param values - Value of each entry.
   * @param dims - `[nRows, nCols]`.
   */
  constructor(
    rows: number[],
    cols: number[],
    values: number[],
    dims: number[],
  ) {
    // A missing dimension behaves like upstream's `undefined`: every bounds
    // check then fails, so no entry can be stored.
    const nRows = dims[0] ?? Number.NaN;
    const nCols = dims[1] ?? Number.NaN;
    this.nRows = nRows;
    this.nCols = nCols;
    this.store = createStore(rows, cols, values, nRows, nCols);
  }

  /**
   * Stores a value, appending an entry when `(row, col)` is not present yet.
   * @param row - Row index.
   * @param col - Column index.
   * @param value - Value to store.
   */
  set(row: number, col: number, value: number): void {
    if (!(row < this.nRows && col < this.nCols)) throw new Error(DIM_ERROR);
    const store = this.store;
    const index = store.index ?? ensureIndex(store, this.nCols);
    const numeric = store.numericKeys && row >= 0 && col >= 0;
    const key = numeric ? row * this.nCols + col : `${row}:${col}`;
    const slot = index.get(key);
    if (slot !== undefined) {
      store.values[slot] = value;
      return;
    }
    const length = store.length;
    if (length >= store.rows.length) growStore(store, length + 1);
    store.rows[length] = row;
    store.cols[length] = col;
    store.values[length] = value;
    index.set(key, length);
    store.length = length + 1;
    if (!numeric) store.irregular = true;
  }

  /**
   * Reads a value.
   * @param row - Row index.
   * @param col - Column index.
   * @param defaultValue - Value returned when the entry is absent.
   * @returns The stored value, or `defaultValue`.
   */
  get(row: number, col: number, defaultValue = 0): number {
    if (!(row < this.nRows && col < this.nCols)) throw new Error(DIM_ERROR);
    const store = this.store;
    const index = store.index ?? ensureIndex(store, this.nCols);
    const slot = index.get(keyOf(store, this.nCols, row, col));
    return slot === undefined ? defaultValue : atFloat64(store.values, slot);
  }

  /**
   * Collects every entry.
   * @param ordered - Sort by row then column instead of insertion order.
   * @returns One object per entry.
   */
  getAll(ordered = true): SparseEntry[] {
    const { rows, cols, values, length } = this.store;
    const output = new Array<SparseEntry>(length);
    if (!ordered) {
      for (let i = 0; i < length; i++) {
        output[i] = {
          value: atFloat64(values, i),
          row: atInt32(rows, i),
          col: atInt32(cols, i),
        };
      }
      return output;
    }
    const slots = sortedSlots(this.store, this.nRows, this.nCols);
    for (let i = 0; i < length; i++) {
      const slot = atInt32(slots, i);
      output[i] = {
        value: atFloat64(values, slot),
        row: atInt32(rows, slot),
        col: atInt32(cols, slot),
      };
    }
    return output;
  }

  /**
   * Reads the shape of the matrix.
   * @returns The `[nRows, nCols]` dimensions.
   */
  getDims(): number[] {
    return [this.nRows, this.nCols];
  }

  /**
   * Collects the row index of every entry.
   * @returns The row indexes, in insertion order.
   */
  getRows(): number[] {
    return toNumbers(this.store.rows, this.store.length);
  }

  /**
   * Collects the column index of every entry.
   * @returns The column indexes, in insertion order.
   */
  getCols(): number[] {
    return toNumbers(this.store.cols, this.store.length);
  }

  /**
   * Collects the value of every entry.
   * @returns The values, in insertion order.
   */
  getValues(): number[] {
    return toNumbers(this.store.values, this.store.length);
  }

  /**
   * Calls `fn` on every entry, in insertion order.
   * @param fn - Callback receiving the value, the row and the column.
   */
  forEach(fn: (value: number, row: number, col: number) => void): void {
    const store = this.store;
    const rows = store.rows;
    const cols = store.cols;
    const values = store.values;
    const length = store.length;
    for (let i = 0; i < length; i++) {
      fn(atFloat64(values, i), atInt32(rows, i), atInt32(cols, i));
    }
    // `Map.prototype.forEach` also visits entries appended during iteration.
    for (let i = length; i < store.length; i++) {
      const values2 = store.values;
      fn(atFloat64(values2, i), atInt32(store.rows, i), atInt32(store.cols, i));
    }
  }

  /**
   * Builds a matrix of the same shape whose values are `fn` of the current ones.
   * @param fn - Callback receiving the value, the row and the column.
   * @returns The mapped matrix.
   */
  map(fn: (value: number, row: number, col: number) => number): SparseMatrix {
    const { rows, cols, values, length, irregular } = this.store;
    const nextValues = new Float64Array(length);
    for (let i = 0; i < length; i++) {
      nextValues[i] = fn(
        atFloat64(values, i),
        atInt32(rows, i),
        atInt32(cols, i),
      );
    }
    const nextRows = rows.slice(0, length);
    const nextCols = cols.slice(0, length);
    const dims = [this.nRows, this.nCols];
    return fromParts(dims, nextRows, nextCols, nextValues, length, irregular);
  }

  /**
   * Densifies the matrix.
   * @returns The matrix as an array of rows, absent entries being zero.
   */
  toArray(): number[][] {
    return toDense(this.store, this.nRows, this.nCols);
  }
}

/**
 * Builds a matrix from deduplicated triplet arrays, skipping the bounds checks
 * and the index construction.
 * @internal
 */
export function fromParts(
  dims: number[],
  rows: Int32Array,
  cols: Int32Array,
  values: Float64Array,
  length: number,
  irregular: boolean,
): SparseMatrix {
  const matrix = new SparseMatrix([], [], [], dims);
  const store = matrix.store;
  // Invariant: `irregular === false` implies that every key is numeric.
  store.irregular = irregular || !store.numericKeys;
  store.rows = rows;
  store.cols = cols;
  store.values = values;
  store.length = length;
  store.index = null;
  return matrix;
}
