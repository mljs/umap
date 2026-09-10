/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: the neighbor search helpers of the UMAP class ported
 * to strict TypeScript as standalone functions taking the state explicitly,
 * `X.length ** 0.5` written as `Math.sqrt(X.length)` (bit-identical), the tree
 * count extracted into `numberOfTrees` so a reloaded model rebuilds its forest
 * with the very same formula, and the index reads bounds checked. The random
 * draws, the arithmetic and their order are unchanged.
 *
 * `getTransformForest` is new: upstream keeps the forest the fit happened to
 * build and lets `transform` search against it, so a model reloaded from JSON
 * — which has no such forest — can only rebuild one from an unrelated stream
 * and projects new points somewhere else. The transform forest is derived
 * instead from `transformSeed`, which every model carries, so a fitted model
 * and a reloaded one hold the same trees.
 */

import type { UMAPInternals } from './internals.ts';
import { at } from './nn/checkedAt.ts';
import { makeNNDescent } from './nn/descent.ts';
import { makeInitializations, makeInitializedNNSearch } from './nn/search.ts';
import type { FlatTree } from './nn/tree.ts';
import { makeForest, makeLeafArray } from './nn/tree.ts';
import { makeSeededRandom } from './random.ts';
import { SparseMatrix, maximum, transpose } from './sparse/index.ts';
import type { Vectors } from './types.ts';

/**
 * The approximate k nearest neighbor graph of a dataset.
 */
export interface NearestNeighborsResult {
  /** Nearest neighbor index of each point, closest first. */
  knnIndices: number[][];
  /** Distance to each of those nearest neighbors. */
  knnDistances: number[][];
}

/**
 * Compute the `numberOfNeighbors` nearest points for each data point in `X`.
 * This may be exact, but more likely is approximated via nearest neighbor
 * descent.
 *
 * The random projection forest built to seed the descent is local to the
 * descent: `transform` searches against {@link getTransformForest} instead.
 * @param internals - State of the model.
 * @returns The neighbor indices and distances.
 */
export function nearestNeighbors(
  internals: UMAPInternals,
): NearestNeighborsResult {
  const X = internals.X;
  const { distanceFunction, numberOfNeighbors, random } = internals.params;
  const metricNNDescent = makeNNDescent(distanceFunction, random);

  const nTrees = numberOfTrees(X.length);
  const nIters = Math.max(5, Math.floor(Math.round(log2(X.length))));

  const rpForest = makeForest(X, numberOfNeighbors, nTrees, random);

  const leafArray = makeLeafArray(rpForest);
  const { indices, weights } = metricNNDescent(
    X,
    leafArray,
    numberOfNeighbors,
    nIters,
  );
  return { knnIndices: indices, knnDistances: weights };
}

/**
 * Number of trees of the random projection forest built for a dataset.
 *
 * A reloaded model rebuilds its forest, so the formula has to be shared
 * between the fit and the deserialization.
 * @param nVectors - Number of data points being indexed.
 * @returns The number of trees.
 */
export function numberOfTrees(nVectors: number): number {
  return 5 + Math.floor(round(Math.sqrt(nVectors) / 20));
}

/**
 * The random projection forest a projection of new points seeds its search
 * with, building it on first use and caching it on the state.
 *
 * It is derived from `params.transformSeed` and from the fitted data alone,
 * never from `params.random`, so a model rebuilt by `fromJSON` holds exactly
 * the trees the model it was saved from holds. The cache is keyed by the seed,
 * so a projection asked for another seed gets the forest of that seed and the
 * whole transform stays a pure function of the model, the points and the seed.
 * @param internals - State of a fitted model, whose cache is filled.
 * @param seed - Seed the forest is built from.
 * @returns The flattened trees of the forest.
 */
export function getTransformForest(
  internals: UMAPInternals,
  seed: number,
): FlatTree[] {
  const { transformForest, transformForestSeed } = internals;
  if (transformForest !== undefined && transformForestSeed === seed) {
    return transformForest;
  }

  const { X, params } = internals;
  const forest = makeForest(
    X,
    params.numberOfNeighbors,
    numberOfTrees(X.length),
    makeSeededRandom(seed, 'transformSeed'),
  );
  internals.transformForest = forest;
  internals.transformForestSeed = seed;
  return forest;
}

/**
 * Builds the three functions the transform search is made of and stores them
 * on the state, bound to the metric of the model.
 * @param internals - State of the model, updated in place.
 */
export function makeSearchFns(internals: UMAPInternals): void {
  const { distanceFunction } = internals.params;
  const { initFromTree, initFromRandom } =
    makeInitializations(distanceFunction);
  internals.initFromTree = initFromTree;
  internals.initFromRandom = initFromRandom;
  internals.search = makeInitializedNNSearch(distanceFunction);
}

/**
 * Builds the symmetrized neighbor graph the transform search walks.
 *
 * It is exactly derivable from the neighbor indices and distances, which is
 * why a serialized model stores those two and rebuilds this graph on load.
 * @param X - The data the model was fitted on.
 * @param knnIndices - Nearest neighbor index of each point.
 * @param knnDistances - Distance to each of those nearest neighbors.
 * @returns The search graph, of size `X.length` by `X.length`.
 */
export function makeSearchGraph(
  X: Vectors,
  knnIndices: number[][],
  knnDistances: number[][],
): SparseMatrix {
  const dims = [X.length, X.length];
  const searchGraph = new SparseMatrix([], [], [], dims);
  for (let i = 0; i < knnIndices.length; i++) {
    const knn = at(knnIndices, i);
    const distances = at(knnDistances, i);
    for (let j = 0; j < knn.length; j++) {
      const neighbor = at(knn, j);
      const distance = at(distances, j);
      if (distance > 0) {
        searchGraph.set(i, neighbor, distance);
      }
    }
  }

  const transposed = transpose(searchGraph);
  return maximum(searchGraph, transposed);
}

function log2(n: number): number {
  return Math.log(n) / Math.log(2);
}

// Handle python3 rounding down from 0.5 discrepancy.
function round(n: number): number {
  return n === 0.5 ? 0 : Math.round(n);
}
