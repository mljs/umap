/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript, the operation is selected
 * by a small integer instead of a closure, and the union of the two entry sets
 * is computed from the key indexes rather than from a `Set` of `${row}:${col}`
 * strings. Entry order, bounds checks and float arithmetic are unchanged.
 * Split out of ./operations.ts to respect the 250 line file limit.
 */

import { atFloat64, atInt32, atNumber } from './arrayAt.ts';
import { SparseMatrix, fromParts } from './sparseMatrix.ts';
import { DIM_ERROR, ensureIndex } from './store.ts';

/**
 * Element-wise multiplication of two matrices.
 * @param a - Left operand, which also fixes the dimensions of the result.
 * @param b - Right operand.
 * @returns The product.
 */
export function pairwiseMultiply(
  a: SparseMatrix,
  b: SparseMatrix,
): SparseMatrix {
  return elementWise(a, b, Op.multiply);
}

/**
 * Element-wise addition of two matrices.
 * @param a - Left operand, which also fixes the dimensions of the result.
 * @param b - Right operand.
 * @returns The sum.
 */
export function add(a: SparseMatrix, b: SparseMatrix): SparseMatrix {
  return elementWise(a, b, Op.add);
}

/**
 * Element-wise subtraction of two matrices.
 * @param a - Left operand, which also fixes the dimensions of the result.
 * @param b - Right operand.
 * @returns The difference.
 */
export function subtract(a: SparseMatrix, b: SparseMatrix): SparseMatrix {
  return elementWise(a, b, Op.subtract);
}

/**
 * Element-wise maximum of two matrices.
 * @param a - Left operand, which also fixes the dimensions of the result.
 * @param b - Right operand.
 * @returns The maximum.
 */
export function maximum(a: SparseMatrix, b: SparseMatrix): SparseMatrix {
  return elementWise(a, b, Op.maximum);
}

const Op = {
  multiply: 0,
  add: 1,
  subtract: 2,
  maximum: 3,
} as const;

type Op = (typeof Op)[keyof typeof Op];

/**
 * Applies `op` to the entries of `a` followed by the entries of `b` that `a`
 * does not hold, the missing operand being zero.
 * @param a - Left operand.
 * @param b - Right operand.
 * @param kind - Operation to apply.
 * @returns The result.
 */
function elementWise(a: SparseMatrix, b: SparseMatrix, kind: Op): SparseMatrix {
  const storeA = a.store;
  const storeB = b.store;
  if (
    storeA.irregular ||
    storeB.irregular ||
    !storeA.numericKeys ||
    !storeB.numericKeys
  ) {
    return elementWiseFaithful(a, b, kind);
  }

  const aNRows = a.nRows;
  const aNCols = a.nCols;
  const bNRows = b.nRows;
  const bNCols = b.nCols;
  const indexA = ensureIndex(storeA, aNCols);
  const indexB = ensureIndex(storeB, bNCols);
  const lengthA = storeA.length;
  const lengthB = storeB.length;
  const rowsA = storeA.rows;
  const colsA = storeA.cols;
  const valuesA = storeA.values;
  const rowsB = storeB.rows;
  const colsB = storeB.cols;
  const valuesB = storeB.values;

  const rows = new Int32Array(lengthA + lengthB);
  const cols = new Int32Array(lengthA + lengthB);
  const values = new Float64Array(lengthA + lengthB);
  let length = 0;

  for (let i = 0; i < lengthA; i++) {
    const row = atInt32(rowsA, i);
    const col = atInt32(colsA, i);
    if (!(row < bNRows && col < bNCols)) throw new Error(DIM_ERROR);
    const slot = indexB.get(row * bNCols + col);
    const x = atFloat64(valuesA, i);
    const y = slot === undefined ? 0 : atFloat64(valuesB, slot);
    rows[length] = row;
    cols[length] = col;
    let value = y;
    if (kind === Op.multiply) value = x * y;
    else if (kind === Op.add) value = x + y;
    else if (kind === Op.subtract) value = x - y;
    else if (x > y) value = x;
    values[length] = value;
    length++;
  }

  for (let i = 0; i < lengthB; i++) {
    const row = atInt32(rowsB, i);
    const col = atInt32(colsB, i);
    if (col < aNCols && indexA.get(row * aNCols + col) !== undefined) continue;
    if (!(row < aNRows && col < aNCols)) throw new Error(DIM_ERROR);
    const y = atFloat64(valuesB, i);
    rows[length] = row;
    cols[length] = col;
    // The literal zeros stand for the absent entry of `a`, and the comparison
    // is a statement rather than `0 > y ? 0 : y` so that the signed-zero and
    // NaN results stay those of upstream instead of those of `Math.max`.
    let value = y;
    if (kind === Op.multiply) value = 0 * y;
    else if (kind === Op.add) value = 0 + y;
    else if (kind === Op.subtract) value = 0 - y;
    else if (y < 0) value = 0;
    values[length] = value;
    length++;
  }

  return fromParts([aNRows, aNCols], rows, cols, values, length, false);
}

/**
 * Faithful, slower port of {@link elementWise}, used when either matrix holds
 * coordinates the fast path cannot key numerically.
 * @param a - Left operand.
 * @param b - Right operand.
 * @param kind - Operation to apply.
 * @returns The result.
 */
function elementWiseFaithful(
  a: SparseMatrix,
  b: SparseMatrix,
  kind: Op,
): SparseMatrix {
  const visited = new Set<string>();
  const rows: number[] = [];
  const cols: number[] = [];
  const values: number[] = [];

  const rowsA = a.getRows();
  const colsA = a.getCols();
  for (let i = 0; i < rowsA.length; i++) {
    const row = atNumber(rowsA, i);
    const col = atNumber(colsA, i);
    visited.add(`${row}:${col}`);
    rows.push(row);
    cols.push(col);
    values.push(applyOp(kind, a.get(row, col), b.get(row, col)));
  }

  const rowsB = b.getRows();
  const colsB = b.getCols();
  for (let i = 0; i < rowsB.length; i++) {
    const row = atNumber(rowsB, i);
    const col = atNumber(colsB, i);
    if (visited.has(`${row}:${col}`)) continue;
    rows.push(row);
    cols.push(col);
    values.push(applyOp(kind, a.get(row, col), b.get(row, col)));
  }

  return new SparseMatrix(rows, cols, values, [a.nRows, a.nCols]);
}

/**
 * Applies one of the four supported element-wise operations.
 * @param kind - Operation to apply.
 * @param x - Left value.
 * @param y - Right value.
 * @returns The result.
 */
function applyOp(kind: Op, x: number, y: number): number {
  if (kind === Op.multiply) return x * y;
  if (kind === Op.add) return x + y;
  if (kind === Op.subtract) return x - y;
  // `x > y ? x : y`, kept as a comparison so that `Math.max` semantics for NaN
  // and signed zero are not substituted for upstream's.
  if (x > y) return x;
  return y;
}
