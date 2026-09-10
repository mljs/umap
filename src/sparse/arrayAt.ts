/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: these readers have no upstream counterpart, they only
 * narrow the `number | undefined` that `noUncheckedIndexedAccess` gives every
 * indexed read. They change no value and no order.
 */

/**
 * Reads `array[index]` at an index the caller has already bounded.
 *
 * `noUncheckedIndexedAccess` widens every indexed read to `number | undefined`,
 * and narrowing it here keeps the numeric loops free of assertions. There is
 * one reader per array type because a single shared reader makes the load
 * inside it megamorphic, which measured 3.1x slower on `getCSR`.
 * @internal
 */
export function atInt32(array: Int32Array, index: number): number {
  const value = array[index];
  if (value === undefined) throw new RangeError(RANGE_ERROR);
  return value;
}

/**
 * Reads `array[index]` at an index the caller has already bounded.
 * @internal
 */
export function atFloat64(array: Float64Array, index: number): number {
  const value = array[index];
  if (value === undefined) throw new RangeError(RANGE_ERROR);
  return value;
}

/**
 * Reads `array[index]` at an index the caller has already bounded.
 * @internal
 */
export function atNumber(array: number[], index: number): number {
  const value = array[index];
  if (value === undefined) throw new RangeError(RANGE_ERROR);
  return value;
}

/**
 * Message of the error thrown when an index turns out to be out of bounds.
 * @internal
 */
export const RANGE_ERROR = 'index out of bounds';
