/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript, `NormType` is a union of
 * string literals instead of a const enum, and rows are normalized in place in
 * the triplet storage instead of being re-inserted into a fresh matrix. The
 * order in which the columns of a row are visited, and therefore the summation
 * order of every norm, is unchanged. Split out of ./operations.ts to respect
 * the 250 line file limit.
 */

import { RANGE_ERROR, atFloat64, atInt32, atNumber } from './arrayAt.ts';
import { compareAsString, lexKey, sortByLexKey } from './ordering.ts';
import { SparseMatrix, fromParts } from './sparseMatrix.ts';
import { EMPTY_FLOAT64 } from './store.ts';

/**
 * Name of a vector norm accepted by {@link normalize}.
 */
export type NormType = 'max' | 'l1' | 'l2';

/**
 * The vector norms accepted by {@link normalize}.
 */
export const NormType = {
  max: 'max',
  l1: 'l1',
  l2: 'l2',
} as const satisfies Record<NormType, NormType>;

/**
 * Normalizes every row of a sparse matrix.
 * @param m - Matrix to normalize.
 * @param normType - Norm applied to each row.
 * @returns The normalized matrix.
 */
export function normalize(
  m: SparseMatrix,
  normType: NormType = NormType.l2,
): SparseMatrix {
  // An unknown norm type is only reported once a row is actually normalized,
  // matching upstream, where the norm function is looked up eagerly but called
  // lazily.
  let kind: NormKind = NormKind.unknown;
  if (normType === NormType.l2) kind = NormKind.l2;
  else if (normType === NormType.l1) kind = NormKind.l1;
  else if (normType === NormType.max) kind = NormKind.max;

  const store = m.store;
  if (store.irregular || !store.numericKeys) return normalizeFaithful(m, kind);

  const length = store.length;
  const sourceRows = store.rows;
  const sourceCols = store.cols;
  const sourceValues = store.values;

  const slotsByRow = new Map<number, number[]>();
  for (let i = 0; i < length; i++) {
    const row = atInt32(sourceRows, i);
    const slots = slotsByRow.get(row);
    if (slots === undefined) {
      slotsByRow.set(row, [i]);
    } else {
      slots.push(i);
    }
  }

  // Ordering key of each column index, mapping the decimal representation of a
  // column to a number that compares the same way, so that the lexicographic
  // `cols.sort()` of upstream is reproduced without allocating any string.
  const lex = new Float64Array(length);
  for (let i = 0; i < length; i++) lex[i] = lexKey(atInt32(sourceCols, i));

  const rows = new Int32Array(length);
  const cols = new Int32Array(length);
  const values = new Float64Array(length);
  let count = 0;
  let row = EMPTY_FLOAT64;

  for (const [rowIndex, slots] of slotsByRow) {
    if (kind === NormKind.unknown) {
      throw new TypeError('normFn is not a function');
    }
    sortByLexKey(slots, lex);
    const size = slots.length;
    if (row.length < size) row = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const slot = slots[i];
      if (slot === undefined) throw new RangeError(RANGE_ERROR);
      rows[count + i] = rowIndex;
      cols[count + i] = atInt32(sourceCols, slot);
      row[i] = atFloat64(sourceValues, slot);
    }
    normalizeInto(row, size, kind, values, count);
    count += size;
  }

  return fromParts([m.nRows, m.nCols], rows, cols, values, count, false);
}

const NormKind = {
  unknown: -1,
  max: 0,
  l1: 1,
  l2: 2,
} as const;

type NormKind = (typeof NormKind)[keyof typeof NormKind];

type KnownNormKind = Exclude<NormKind, typeof NormKind.unknown>;

/**
 * Writes the normalized values of the first `size` elements of `row` into
 * `output`, starting at `base`.
 * @param row - Values of the row.
 * @param size - Number of values.
 * @param kind - Norm to apply.
 * @param output - Array receiving the normalized values.
 * @param base - Offset in `output`.
 */
function normalizeInto(
  row: Float64Array,
  size: number,
  kind: KnownNormKind,
  output: Float64Array,
  base: number,
): void {
  if (kind === NormKind.max) {
    let max = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < size; i++) {
      const x = atFloat64(row, i);
      // A comparison rather than `Math.max`, whose NaN handling differs.
      if (x > max) max = x;
    }
    for (let i = 0; i < size; i++) output[base + i] = atFloat64(row, i) / max;
    return;
  }
  if (kind === NormKind.l1) {
    let sum = 0;
    for (let i = 0; i < size; i++) sum += atFloat64(row, i);
    for (let i = 0; i < size; i++) output[base + i] = atFloat64(row, i) / sum;
    return;
  }
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = atFloat64(row, i);
    sum += x * x;
  }
  for (let i = 0; i < size; i++) {
    const x = atFloat64(row, i);
    output[base + i] = Math.sqrt((x * x) / sum);
  }
}

/**
 * Faithful, slower port of {@link normalize}, used when the matrix holds
 * coordinates the fast path cannot key numerically.
 * @param m - Matrix to normalize.
 * @param kind - Norm to apply.
 * @returns The normalized matrix.
 */
function normalizeFaithful(m: SparseMatrix, kind: NormKind): SparseMatrix {
  const store = m.store;
  const sourceRows = store.rows;
  const sourceCols = store.cols;
  const colsByRow = new Map<number, number[]>();
  for (let i = 0; i < store.length; i++) {
    const row = atInt32(sourceRows, i);
    const cols = colsByRow.get(row) ?? [];
    cols.push(atInt32(sourceCols, i));
    colsByRow.set(row, cols);
  }

  const next = new SparseMatrix([], [], [], m.getDims());
  for (const [row, cols] of colsByRow) {
    if (kind === NormKind.unknown) {
      throw new TypeError('normFn is not a function');
    }
    // Deliberate: upstream sorts with `cols.sort()`, without a comparator, so
    // the columns end up in lexicographic order. That order drives the
    // summation order of the norm and is load bearing, not a bug to fix, so
    // the comparator below spells out the default one instead of sorting
    // numerically.
    const sorted = cols.toSorted(compareAsString);
    const size = sorted.length;
    const values = new Float64Array(size);
    for (let i = 0; i < size; i++) values[i] = m.get(row, atNumber(sorted, i));
    const normed = new Float64Array(size);
    normalizeInto(values, size, kind, normed, 0);
    for (let i = 0; i < size; i++) {
      next.set(row, atNumber(sorted, i), atFloat64(normed, i));
    }
  }
  return next;
}
