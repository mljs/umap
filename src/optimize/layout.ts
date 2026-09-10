/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript as standalone functions
 * that take the optimization state and the random source explicitly instead of
 * reading them from a class instance, index reads are bounds checked,
 * `Math.pow(x[i] - y[i], 2)` in rDist is written as a multiplication and the
 * remaining `Math.pow` calls use the `**` operator. Both rewrites are
 * bit-identical; arithmetic and iteration order are unchanged.
 */

import type { RandomFn, Vectors } from '../types.ts';
import { tauRandInt } from '../utils.ts';

import type { OptimizationState } from './state.ts';

const CLIP_VALUE = 4;

const INCONSISTENT_STATE =
  'optimizeLayoutStep: the optimization state arrays have inconsistent lengths';

/**
 * Runs one epoch of the stochastic gradient descent that minimizes the fuzzy
 * set cross entropy between the 1-skeletons of the high dimensional and the
 * low dimensional fuzzy simplicial sets. In practice this is done by sampling
 * edges based on their membership strength, with the (1-p) terms coming from
 * negative sampling similar to word2vec.
 *
 * The head embedding is improved in place; `state.alpha` is decayed and
 * `state.currentEpoch` is incremented before returning.
 * @param state - Optimization state, mutated in place.
 * @param n - Epoch to run, normally `state.currentEpoch`.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns The head embedding of `state`.
 */
export function optimizeLayoutStep(
  state: OptimizationState,
  n: number,
  random: RandomFn,
): Vectors {
  const {
    head,
    tail,
    headEmbedding,
    tailEmbedding,
    epochsPerSample,
    epochOfNextSample,
    epochOfNextNegativeSample,
    epochsPerNegativeSample,
    moveOther,
    initialAlpha,
    alpha,
    gamma,
    a,
    b,
    dim,
    numberOfEpochs,
    nVertices,
  } = state;

  const edgeCount = epochsPerSample.length;
  // Validated once, then indexed directly below. These reads are the innermost
  // of the whole algorithm — this loop is around two thirds of a fit — and a
  // per-element `undefined` check here costs more than the arithmetic it guards.
  if (
    epochOfNextSample.length < edgeCount ||
    epochOfNextNegativeSample.length < edgeCount ||
    epochsPerNegativeSample.length < edgeCount ||
    head.length < edgeCount ||
    tail.length < edgeCount
  ) {
    throw new RangeError(INCONSISTENT_STATE);
  }

  for (let i = 0; i < edgeCount; i++) {
    const nextSample = epochOfNextSample[i] as number;
    if (nextSample > n) {
      continue;
    }

    const j = head[i] as number;
    const k = tail[i] as number;
    const current = headEmbedding[j] as number[];
    const other = tailEmbedding[k] as number[];
    const perSample = epochsPerSample[i] as number;
    const nextNegativeSample = epochOfNextNegativeSample[i] as number;
    const perNegativeSample = epochsPerNegativeSample[i] as number;

    const distSquared = rDist(current, other);

    let gradCoeff = 0;
    if (distSquared > 0) {
      gradCoeff = -2 * a * b * distSquared ** (b - 1);
      gradCoeff /= a * distSquared ** b + 1;
    }

    for (let d = 0; d < dim; d++) {
      const currentD = current[d] as number;
      const otherD = other[d] as number;
      // When `current` and `other` are the same row, `distSquared` and
      // therefore `gradD` are exactly 0, so reading `otherD` before writing
      // `current[d]` always gives the same result as updating both in place.
      const gradD = clip(gradCoeff * (currentD - otherD), CLIP_VALUE);
      current[d] = currentD + gradD * alpha;
      if (moveOther) {
        other[d] = otherD + -gradD * alpha;
      }
    }

    epochOfNextSample[i] = nextSample + perSample;

    const nNegSamples = Math.floor(
      (n - nextNegativeSample) / perNegativeSample,
    );

    for (let p = 0; p < nNegSamples; p++) {
      const negativeK = tauRandInt(nVertices, random);
      const negativeOther = tailEmbedding[negativeK] as number[];

      const negativeDistSquared = rDist(current, negativeOther);

      let negativeGradCoeff = 0;
      if (negativeDistSquared > 0) {
        negativeGradCoeff = 2 * gamma * b;
        negativeGradCoeff /=
          (0.001 + negativeDistSquared) * (a * negativeDistSquared ** b + 1);
      } else if (j === negativeK) {
        continue;
      }

      for (let d = 0; d < dim; d++) {
        const currentD = current[d] as number;
        const otherD = negativeOther[d] as number;
        // Deliberate: a repulsion whose coefficient is 0 still pushes the
        // point by the full clip value, in an arbitrary direction.
        let gradD = 4;
        if (negativeGradCoeff > 0) {
          gradD = clip(negativeGradCoeff * (currentD - otherD), CLIP_VALUE);
        }
        current[d] = currentD + gradD * alpha;
      }
    }

    epochOfNextNegativeSample[i] =
      nextNegativeSample + nNegSamples * perNegativeSample;
  }

  state.alpha = initialAlpha * (1 - n / numberOfEpochs);
  state.currentEpoch += 1;
  return headEmbedding;
}

/**
 * Runs the whole stochastic gradient descent, one epoch at a time, until
 * `state.numberOfEpochs` epochs have run or the callback asks to stop.
 * @param state - Optimization state, mutated in place.
 * @param random - Random number generator returning a float in [0, 1).
 * @param epochCallback - Called with the number of completed epochs after each
 * epoch; returning `false` stops the optimization. Defaults to a callback that
 * never stops it.
 * @returns The optimized head embedding.
 */
export function optimizeLayout(
  state: OptimizationState,
  random: RandomFn,
  epochCallback: (epoch: number) => boolean | void = () => true,
): Vectors {
  let isFinished = false;
  let embedding: Vectors = [];
  while (!isFinished) {
    embedding = optimizeLayoutStep(state, state.currentEpoch, random);
    const epochCompleted = state.currentEpoch;
    const shouldStop = epochCallback(epochCompleted) === false;
    // >= not ===: a non-integer numberOfEpochs would otherwise never satisfy
    // equality against the integer epoch counter and the loop would never
    // terminate.
    isFinished = epochCompleted >= state.numberOfEpochs || shouldStop;
  }
  return embedding;
}

/**
 * Standard clamping of a value into a fixed symmetric range.
 * @param x - Value to clamp.
 * @param clipValue - Bound of the range, applied as `[-clipValue, clipValue]`.
 * @returns The clamped value.
 */
export function clip(x: number, clipValue: number): number {
  if (x > clipValue) return clipValue;
  if (x < -clipValue) return -clipValue;
  return x;
}

/**
 * Reduced Euclidean distance, i.e. the squared Euclidean distance.
 * @param x - First vector.
 * @param y - Second vector, at least as long as `x`.
 * @returns The sum of the squared differences.
 */
export function rDist(x: number[], y: number[]): number {
  if (y.length < x.length) {
    throw new RangeError('rDist: vectors must have the same length');
  }
  // Checked once above, then indexed directly: this runs once per edge plus
  // once per negative sample, so tens of millions of times per fit.
  let result = 0;
  for (let i = 0; i < x.length; i++) {
    const d = (x[i] as number) - (y[i] as number);
    result += d * d;
  }
  return result;
}
