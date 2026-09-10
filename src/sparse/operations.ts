/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript, and rewritten against the
 * triplet storage of ./store.ts so that a result is built directly instead of
 * being re-inserted entry by entry into a fresh matrix. Entry order, bounds
 * checks and float arithmetic are unchanged.
 */

import { atFloat64, atInt32 } from './arrayAt.ts';
import { sortedSlots } from './ordering.ts';
import type { SparseMatrix } from './sparseMatrix.ts';
import { fromParts } from './sparseMatrix.ts';

/**
 * Transposes a sparse matrix.
 * @param matrix - Matrix to transpose.
 * @returns The transposed matrix.
 */
export function transpose(matrix: SparseMatrix): SparseMatrix {
  const { rows, cols, values, length, irregular } = matrix.store;
  return fromParts(
    [matrix.nCols, matrix.nRows],
    cols.slice(0, length),
    rows.slice(0, length),
    values.slice(0, length),
    length,
    irregular,
  );
}

/**
 * Scalar multiplication of a matrix.
 * @param a - Matrix to scale.
 * @param scalar - Factor applied to every entry.
 * @returns The scaled matrix.
 */
export function multiplyScalar(a: SparseMatrix, scalar: number): SparseMatrix {
  const { rows, cols, values, length, irregular } = a.store;
  const nextValues = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    nextValues[i] = atFloat64(values, i) * scalar;
  }
  return fromParts(
    [a.nRows, a.nCols],
    rows.slice(0, length),
    cols.slice(0, length),
    nextValues,
    length,
    irregular,
  );
}

/**
 * Removes the zero entries of a matrix.
 * @param m - Matrix to filter.
 * @returns A new matrix without zero entries.
 */
export function eliminateZeros(m: SparseMatrix): SparseMatrix {
  const { rows, cols, values, length, irregular } = m.store;
  let kept = 0;
  for (let i = 0; i < length; i++) {
    if (atFloat64(values, i) !== 0) kept++;
  }
  const nextRows = new Int32Array(kept);
  const nextCols = new Int32Array(kept);
  const nextValues = new Float64Array(kept);
  let k = 0;
  for (let i = 0; i < length; i++) {
    const value = atFloat64(values, i);
    if (value !== 0) {
      nextRows[k] = atInt32(rows, i);
      nextCols[k] = atInt32(cols, i);
      nextValues[k] = value;
      k++;
    }
  }
  return fromParts(
    m.getDims(),
    nextRows,
    nextCols,
    nextValues,
    kept,
    irregular,
  );
}

/**
 * Compressed sparse row representation of a matrix.
 */
export interface CSRMatrix {
  /** Column index of each entry, rows concatenated in increasing order. */
  indices: number[];
  /** Value of each entry, in the same order as `indices`. */
  values: number[];
  /** Offset in `indices` at which each non-empty row starts. */
  indptr: number[];
}

/**
 * Extracts the data, indices and indptr arrays of a matrix, following the csr
 * matrix conventions a lot of the ported python tree search logic depends on.
 *
 * A row without any entry produces no `indptr` element, so `indptr` is only as
 * long as the number of non-empty rows. This is upstream behaviour and the
 * tree search relies on it.
 * @param x - Matrix to convert.
 * @returns The csr arrays.
 */
export function getCSR(x: SparseMatrix): CSRMatrix {
  const store = x.store;
  const rows = store.rows;
  const cols = store.cols;
  const sourceValues = store.values;
  const length = store.length;
  const slots = sortedSlots(store, x.nRows, x.nCols);

  const indices = new Array<number>(length);
  const values = new Array<number>(length);
  const indptr: number[] = [];

  let currentRow = -1;
  for (let i = 0; i < length; i++) {
    const slot = atInt32(slots, i);
    const row = atInt32(rows, slot);
    if (row !== currentRow) {
      currentRow = row;
      indptr.push(i);
    }
    indices[i] = atInt32(cols, slot);
    values[i] = atFloat64(sourceValues, slot);
  }

  return { indices, values, indptr };
}
