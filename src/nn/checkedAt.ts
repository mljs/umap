/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript; this helper replaces the
 * unchecked index reads of the original, which `noUncheckedIndexedAccess`
 * types as possibly `undefined`. No behavioural change for in-range reads.
 */

/**
 * Reads `values[index]`, throwing when the index is out of range.
 *
 * Every call site of the random projection tree code reads an index that is
 * in range by construction; the check exists so the compiler sees a defined
 * value and so a corruption of the tree fails loudly instead of poisoning the
 * arithmetic with `NaN`.
 * @param values - Array to read from.
 * @param index - Index to read.
 * @returns The element at `index`.
 */
export function at<T>(values: readonly T[], index: number): T {
  const value = values[index];
  if (value === undefined) {
    throw new RangeError(`index ${index} is out of range`);
  }
  return value;
}
