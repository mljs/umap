/*
 * ml-umap: the seeded random sources, which umap-js does not have.
 *
 * Upstream defaults `random` to `Math.random` and draws the numbers a
 * projection of new points needs from that same stream, so nothing repeats:
 * two fits of one dataset differ, `transform` advances the stream `fit` ran
 * on, and two loads of one saved model place the same new points in different
 * spots. Both sources are seeded here instead — a fit from `seed`, a
 * projection from `transformSeed`, as umap-learn keeps a `transform_seed`
 * (default 42, umap_.py:3126) — so a run repeats unless the caller asks for a
 * stream that does not, by passing `random: Math.random`.
 */

import { XSadd } from 'ml-xsadd';

import type { RandomFn } from './types.ts';

/**
 * The seed a fit draws from when no `seed` and no `random` are configured.
 */
export const DEFAULT_SEED = 42;

/**
 * The seed a projection of new points uses when none is configured, matching
 * `transform_seed=42` in umap-learn (umap_.py:3126).
 */
export const DEFAULT_TRANSFORM_SEED = 42;

/**
 * Builds a seeded random source.
 *
 * The returned generator is always positioned at the start of its stream, so
 * every call with a given seed yields the same numbers in the same order, and
 * two generators built here never share a draw.
 * @param seed - Seed of the stream; it must be an integer.
 * @param name - Name of the parameter the seed came from, for the error
 * message.
 * @returns A function returning the next number of the stream, in [0, 1).
 * @throws {TypeError} If `seed` is not an integer.
 */
export function makeSeededRandom(seed: number, name = 'seed'): RandomFn {
  if (!Number.isInteger(seed)) {
    throw new TypeError(`${name} must be an integer, but it is ${seed}.`);
  }
  return new XSadd(seed).random;
}
