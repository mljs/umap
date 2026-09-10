/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: `UMAP.transform` ported to strict TypeScript as a
 * standalone function taking the model state explicitly, the `map` callbacks
 * truncating the neighbor rows and scanning for the largest graph value
 * replaced by loops and by the shared `max` helper, `getDims()[1]` read as
 * `nCols`, the points to project validated against the dimensionality of the
 * fitted data, `initTransform` moved to ./initTransform.ts and the
 * optimization to ./transformOptimize.ts to respect the 250 line file limit,
 * and — the behavioural divergences below — the fitted embedding frozen, the
 * learning rate quartered, the random source scoped to the projection and the
 * training data short circuited. The arithmetic and the iteration order are
 * otherwise unchanged.
 *
 * DIVERGENCES from umap-js, all four of which follow umap-learn:
 *
 * (a) The optimization runs with `moveOther: false` and against a deep copy of
 *     the fitted embedding, which therefore never moves, whatever is being
 *     projected. umap-js assigns `tailEmbedding = this.embedding` by reference
 *     and derives `moveOther` from
 *     `headEmbedding.length === tailEmbedding.length`, so projecting exactly
 *     as many points as were fitted moves the training points too and corrupts
 *     the model in place. umap-learn passes no `move_other` (layouts.py:257).
 *
 * (b) The learning rate is a quarter of the configured one, as umap-learn
 *     optimizes the layout of new points with `_initial_alpha / 4.0`
 *     (umap_.py:3304). umap-js uses the full rate, so new points take four
 *     times larger steps against a background that never moves back.
 *
 * (c) Every random number is drawn from a generator built from `transformSeed`
 *     for this call alone, and the trees the search is seeded with come from
 *     that seed too. umap-js draws from the model's own `random`, so a
 *     projection both depends on and advances the fit stream and can never be
 *     replayed; a model reloaded from JSON, whose `random` defaults to
 *     `Math.random`, then projects the same points somewhere else on every
 *     load. umap-learn keeps a dedicated `transform_seed` (umap_.py:3126).
 *
 * (d) Projecting the training data returns the fitted embedding itself instead
 *     of searching and optimizing again, as umap-learn does when the input
 *     hashes to `_input_hash` (umap_.py:3095-3110). Re-projecting it lands
 *     nowhere near the embedding it came from — 5.66 away on a 12.0 wide
 *     embedding — because a transform freezes the fitted points and only
 *     relaxes the new ones against them.
 */

import { copyEmbedding } from './embedding.ts';
import { computeMembershipStrengths } from './fuzzy/membership.ts';
import { smoothKNNDistance } from './fuzzy/smoothKnn.ts';
import { initTransform } from './initTransform.ts';
import type { UMAPInternals } from './internals.ts';
import { getTransformForest } from './knn.ts';
import { at } from './nn/checkedAt.ts';
import { deheapSort } from './nn/heapOperations.ts';
import { initializeSearch } from './nn/search.ts';
import { makeSeededRandom } from './random.ts';
import { NormType, SparseMatrix, getCSR, normalize } from './sparse/index.ts';
import { optimizeTransformLayout } from './transformOptimize.ts';
import type { Vectors } from './types.ts';
import { reshape2d, vectorsEqual } from './utils.ts';
import { validateTransformInput } from './validate.ts';

/**
 * Options of one projection of new points.
 */
export interface TransformOptions {
  /**
   * Seed of the random source this projection draws from, overriding the
   * `transformSeed` of the model for this call only. Two calls with the same
   * points and the same seed return the same positions.
   * @default the `transformSeed` the model was built or loaded with
   */
  transformSeed?: number;
}

/**
 * Transforms data to the existing embedding space.
 *
 * The fitted embedding is left untouched, whatever the batch holds, and the
 * projection is reproducible: it draws from a generator built from
 * `transformSeed`, never from the stream the fit ran on, so one batch always
 * lands on the same coordinates. Only the returned positions of the new points
 * are optimized, at a quarter of the configured learning rate.
 *
 * A batch is projected as a whole: one point moves slightly when the rest of
 * the batch changes — by up to 8.5% of the width of the embedding on the
 * fixture of ./__tests__/transformBatchSize.test.ts, which documents the three
 * channels that couple a batch. umap-learn has the same limitation.
 *
 * Projecting the data the model was fitted on returns a copy of the fitted
 * embedding, without searching or optimizing anything, exactly as umap-learn
 * returns `embedding_` for an input that hashes to `_input_hash`
 * (umap_.py:3095-3110). The rows are compared value by value, which costs one
 * pass over the data and stops at the first difference, so a single changed
 * value is enough to project normally again.
 * @param internals - State of a fitted model.
 * @param toTransform - The points to project.
 * @param options - Options of this projection.
 * @returns The position of each of those points in the embedding space.
 * @throws {Error} If the model has not been fitted or loaded.
 * @throws {TypeError} If `toTransform` is empty, or holds rows of unequal
 * length, or holds a value that is not a number, or does not have the
 * dimensionality of the fitted data, or if `transformSeed` is not an integer.
 * @throws {RangeError} If a value of `toTransform` is not finite.
 */
export function transform(
  internals: UMAPInternals,
  toTransform: Vectors,
  options: TransformOptions = {},
): Vectors {
  const {
    // Use the previous rawData
    X: rawData,
    initFromRandom,
    initFromTree,
    search,
    searchGraph,
    embedding: fittedEmbedding,
    params,
  } = internals;

  if (
    rawData.length === 0 ||
    !initFromRandom ||
    !initFromTree ||
    !search ||
    !searchGraph
  ) {
    throw new Error('No data has been fit.');
  }

  validateTransformInput(toTransform, at(rawData, 0).length);

  if (vectorsEqual(rawData, toTransform)) {
    return copyEmbedding(fittedEmbedding);
  }

  const {
    numberOfNeighbors: fittedNumberOfNeighbors,
    transformQueueSize,
    localConnectivity,
    transformSeed: modelTransformSeed,
  } = params;

  const seed = options.transformSeed ?? modelTransformSeed;
  const random = makeSeededRandom(seed, 'transformSeed');
  const rpForest = getTransformForest(internals, seed);

  let numberOfNeighbors = Math.floor(
    fittedNumberOfNeighbors * transformQueueSize,
  );
  numberOfNeighbors = Math.min(rawData.length, numberOfNeighbors);
  const init = initializeSearch(
    rpForest,
    rawData,
    toTransform,
    numberOfNeighbors,
    initFromRandom,
    initFromTree,
    random,
  );

  const result = search(rawData, searchGraph, init, toTransform);

  const { indices: allIndices, weights: allDistances } = deheapSort(result);

  const indices: number[][] = [];
  for (const row of allIndices) {
    indices.push(row.slice(0, fittedNumberOfNeighbors));
  }
  const distances: number[][] = [];
  for (const row of allDistances) {
    distances.push(row.slice(0, fittedNumberOfNeighbors));
  }

  const adjustedLocalConnectivity = Math.max(0, localConnectivity - 1);
  const { sigmas, rhos } = smoothKNNDistance(
    distances,
    fittedNumberOfNeighbors,
    {
      localConnectivity: adjustedLocalConnectivity,
    },
  );

  const { rows, cols, vals } = computeMembershipStrengths(
    indices,
    distances,
    sigmas,
    rhos,
    // Rows index the batch, columns index the training set, so the self test
    // that the fit path relies on would zero an unrelated pair here.
    { bipartite: true },
  );

  const size = [toTransform.length, rawData.length];
  const graph = new SparseMatrix(rows, cols, vals, size);

  // This was a very specially constructed graph with constant degree.
  // That lets us do fancy unpacking by reshaping the csr matrix indices
  // and data. Doing so relies on the constant degree assumption!
  const normed = normalize(graph, NormType.l1);

  const csrMatrix = getCSR(normed);
  const nPoints = toTransform.length;

  const eIndices = reshape2d(
    csrMatrix.indices,
    nPoints,
    fittedNumberOfNeighbors,
  );
  const eWeights = reshape2d(
    csrMatrix.values,
    nPoints,
    fittedNumberOfNeighbors,
  );

  const embedding = initTransform(eIndices, eWeights, fittedEmbedding);

  return optimizeTransformLayout(internals, embedding, graph, random);
}
