/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: this reader has no upstream counterpart, it only
 * narrows the `number[] | undefined` that `noUncheckedIndexedAccess` gives
 * every indexed read of a 2d array. It changes no value and no order.
 */

/**
 * Reads `rows[index]` at an index the caller has already bounded.
 *
 * `noUncheckedIndexedAccess` widens every indexed read to `number[] |
 * undefined`, and narrowing it here keeps the numeric loops free of assertions.
 * @internal
 */
export function rowAt(rows: number[][], index: number): number[] {
  const row = rows[index];
  if (row === undefined) throw new RangeError(ROW_RANGE_ERROR);
  return row;
}

/**
 * Message of the error thrown when a row index turns out to be out of bounds.
 * @internal
 */
export const ROW_RANGE_ERROR = 'row index out of bounds';
