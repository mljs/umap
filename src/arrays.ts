/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript, callback array iteration
 * replaced by indexed loops, and split out of utils.ts to respect the 250 line
 * file limit. Arithmetic and iteration order are unchanged.
 */

/**
 * Creates an array filled with index values.
 * @param n - Length of the array.
 * @returns An array containing the integers from 0 to `n - 1`.
 */
export function range(n: number): number[] {
  const output: number[] = [];
  for (let i = 0; i < n; i++) {
    output.push(i);
  }
  return output;
}

/**
 * Creates an array filled with a specific value.
 * @param n - Length of the array.
 * @param v - Value to repeat.
 * @returns An array of `n` times `v`.
 */
export function filled(n: number, v: number): number[] {
  const output: number[] = [];
  for (let i = 0; i < n; i++) {
    output.push(v);
  }
  return output;
}

/**
 * Creates an array filled with zeros.
 * @param n - Length of the array.
 * @returns An array of `n` zeros.
 */
export function zeros(n: number): number[] {
  return filled(n, 0);
}

/**
 * Creates an array from a to b, of length len, inclusive.
 * @param a - First value of the array.
 * @param b - Last value of the array.
 * @param len - Number of values.
 * @returns The evenly spaced values from `a` to `b`.
 */
export function linear(a: number, b: number, len: number): number[] {
  const output: number[] = [];
  for (let i = 0; i < len; i++) {
    output.push(a + i * ((b - a) / (len - 1)));
  }
  return output;
}
