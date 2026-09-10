/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript, callback array iteration
 * replaced by loops, the `RandomFn` type moved here, the `vectorsEqual` helper
 * added, and the array constructors split into ./arrays.ts
 * (re-exported here) to respect the 250 line file limit. Arithmetic and
 * iteration order are unchanged.
 */

import { zeros } from './arrays.ts';
import type { RandomFn, Vectors } from './types.ts';

export * from './arrays.ts';

/** Source of uniformly distributed random numbers in the [0, 1) interval. */
export type { RandomFn } from './types.ts';

/**
 * Simple random integer function.
 * @param n - Exclusive upper bound of the generated integer.
 * @param random - Random number generator.
 * @returns An integer in the [0, n) interval.
 */
export function tauRandInt(n: number, random: RandomFn): number {
  return Math.floor(random() * n);
}

/**
 * Simple random float function.
 * @param random - Random number generator.
 * @returns A float in the [0, 1) interval.
 */
export function tauRand(random: RandomFn): number {
  return random();
}

/**
 * Returns the sum of an array.
 * @param input - The array to sum.
 * @returns The sum of every value, accumulated from left to right.
 */
export function sum(input: number[]): number {
  const first = input[0];
  if (first === undefined) {
    throw new TypeError('Reduce of empty array with no initial value');
  }
  let result = first;
  for (let i = 1; i < input.length; i++) {
    const value = input[i];
    if (value === undefined) continue;
    result += value;
  }
  return result;
}

/**
 * Returns the mean of an array.
 * @param input - The array to average.
 * @returns The arithmetic mean of the values.
 */
export function mean(input: number[]): number {
  return sum(input) / input.length;
}

/**
 * Returns the maximum value of an array.
 * @param input - The array to scan.
 * @returns The largest value, or 0 when every value is negative.
 */
export function max(input: number[]): number {
  // Deliberate: the accumulator starts at 0, so the result is never negative.
  // Callers only pass non-negative distances and weights.
  // Deliberate: `>` keeps the accumulator when a value is NaN, where Math.max
  // would propagate it.
  let result = 0;
  for (const value of input) {
    if (value > result) {
      result = value;
    }
  }
  return result;
}

/**
 * Generate nSamples many integers from 0 to poolSize such that no
 * integer is selected twice. The duplication constraint is achieved via
 * rejection sampling.
 * @param nSamples - Number of integers to draw.
 * @param poolSize - Exclusive upper bound of the drawn integers.
 * @param random - Random number generator.
 * @returns The `nSamples` distinct integers, in draw order.
 */
export function rejectionSample(
  nSamples: number,
  poolSize: number,
  random: RandomFn,
): number[] {
  const result = zeros(nSamples);
  for (let i = 0; i < nSamples; i++) {
    let rejectSample = true;
    while (rejectSample) {
      const j = tauRandInt(poolSize, random);
      let broken = false;
      for (let k = 0; k < i; k++) {
        if (j === result[k]) {
          broken = true;
          break;
        }
      }
      if (!broken) {
        rejectSample = false;
      }
      // Deliberate: the sample is stored even when rejected. The slot is only
      // read back once it holds an accepted value, so this is a no-op.
      result[i] = j;
    }
  }
  return result;
}

/**
 * Reshapes a 1d array into a 2d one of given dimensions.
 * @param x - The values to reshape.
 * @param a - Number of rows.
 * @param b - Number of columns.
 * @returns An array of `a` rows of `b` values.
 */
export function reshape2d<T>(x: T[], a: number, b: number): T[][] {
  if (x.length !== a * b) {
    throw new Error('Array dimensions must match input length.');
  }

  const rows: T[][] = [];
  let index = 0;
  for (let i = 0; i < a; i++) {
    rows.push(x.slice(index, index + b));
    index += b;
  }
  return rows;
}

/**
 * Compares two collections of vectors value by value.
 *
 * The scan is O(rows * columns) in the worst case, stops at the first
 * difference, and never runs at all when the two differ in length; identical
 * references are answered without reading anything.
 * @param a - First collection of rows.
 * @param b - Second collection of rows.
 * @returns Whether both hold the same values in the same order.
 */
export function vectorsEqual(a: Vectors, b: Vectors): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;

  for (let row = 0; row < a.length; row++) {
    const left = a[row];
    const right = b[row];
    if (left === undefined || right === undefined) return false;
    if (left !== right) {
      if (left.length !== right.length) return false;
      for (let column = 0; column < left.length; column++) {
        if (left[column] !== right[column]) return false;
      }
    }
  }
  return true;
}
