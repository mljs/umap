/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: the seeded generator the upstream test suite builds in
 * a `beforeEach` extracted into a helper, so the flat tests can each start from
 * a fresh stream, and the stream itself taken from our own ml-xsadd rather than
 * from the third-party generator upstream seeds with.
 */

import { XSadd } from 'ml-xsadd';

/**
 * Builds the seeded pseudo-random generator the tests fit from.
 * Every call returns a generator positioned at the start of the stream.
 * @param seed - Seed of the stream; it must be an integer.
 * @returns A function returning the next number of the stream, in [0, 1).
 */
export function makeRandom(seed = 42): () => number {
  return new XSadd(seed).random;
}
