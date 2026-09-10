/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: `UMAP.initializeFit` and `UMAP.optimizeLayoutAsync`
 * ported to strict TypeScript as standalone functions taking the model state
 * explicitly, split out of the class so it respects the 250 line file limit,
 * and the rejection of the async loop narrowed to an `Error`. The data is now
 * validated before anything is computed, and `numberOfNeighbors` clamped to
 * the size of the data, where upstream fitted a non-finite value into a
 * plausible looking embedding without a word. The computation order and the
 * arithmetic are unchanged.
 */

import { initializeSimplicialSetEmbedding } from './embedding.ts';
import {
  assignOptimizationStateParameters,
  getNumberOfEpochs,
  initializeOptimization,
  prepareForOptimizationLoop,
  processGraphForSupervisedProjection,
} from './fitSteps.ts';
import { fuzzySimplicialSet } from './fuzzy/simplicialSet.ts';
import type { UMAPInternals } from './internals.ts';
import { makeSearchFns, makeSearchGraph, nearestNeighbors } from './knn.ts';
import { optimizeLayoutStep } from './optimize/layout.ts';
import type { EpochCallback, Vectors } from './types.ts';
import { validateFitInput } from './validate.ts';

/**
 * Initializes fit by computing KNN and a fuzzy simplicial set, as well as
 * initializing the projected embeddings. Sets the optimization state ahead
 * of optimization steps.
 *
 * `X` is validated first, and `params.numberOfNeighbors` is clamped down to
 * `X.length - 1` when it exceeds the number of points, so the value the model
 * carries afterwards is the one the projection was actually built with.
 * @param internals - State of the model, updated in place.
 * @param X - The data to project, one array per point.
 * @returns The number of epochs to be used for the SGD optimization.
 * @throws {Error} If there are not more points than neighbors.
 * @throws {TypeError} If `X` is empty, or holds rows of unequal length, or
 * holds a value that is not a number.
 * @throws {RangeError} If a value of `X` is not finite, or if
 * `numberOfNeighbors` is below two.
 */
export function initializeFit(internals: UMAPInternals, X: Vectors): number {
  // We don't need to reinitialize if we've already initialized for this data.
  if (internals.X === X && internals.isInitialized) {
    return getNumberOfEpochs(internals);
  }

  internals.params.numberOfNeighbors = validateFitInput(
    X,
    internals.params.numberOfNeighbors,
  );

  const {
    numberOfNeighbors,
    setOperationMixRatio,
    localConnectivity,
    numberOfComponents,
    random,
    init,
  } = internals.params;

  if (X.length <= numberOfNeighbors) {
    throw new Error(
      `Not enough data points (${X.length}) to create numberOfNeighbors: ${numberOfNeighbors}.  Add more data points or adjust the configuration.`,
    );
  }

  internals.X = X;

  if (!internals.knnIndices && !internals.knnDistances) {
    const knnResults = nearestNeighbors(internals);
    internals.knnIndices = knnResults.knnIndices;
    internals.knnDistances = knnResults.knnDistances;
  }
  const knnIndices = internals.knnIndices ?? [];
  const knnDistances = internals.knnDistances ?? [];

  internals.graph = fuzzySimplicialSet(knnIndices, knnDistances, {
    nVectors: X.length,
    numberOfNeighbors,
    setOperationMixRatio,
    localConnectivity,
  });

  // Set up the search graph for subsequent transformation.
  makeSearchFns(internals);
  internals.searchGraph = makeSearchGraph(X, knnIndices, knnDistances);

  // Check if supervised projection, then adjust the graph.
  processGraphForSupervisedProjection(internals);

  const { embedding, head, tail, epochsPerSample } =
    initializeSimplicialSetEmbedding(internals.graph, {
      numberOfEpochs: getNumberOfEpochs(internals),
      numberOfComponents,
      random,
      init,
      data: X,
    });
  internals.embedding = embedding;

  // Set the optimization routine state
  assignOptimizationStateParameters(internals, { head, tail, epochsPerSample });

  // Now, initialize the optimization steps
  initializeOptimization(internals);
  prepareForOptimizationLoop(internals);
  internals.isInitialized = true;

  return getNumberOfEpochs(internals);
}

/**
 * Improve an embedding using stochastic gradient descent to minimize the
 * fuzzy set cross entropy between the 1-skeletons of the high dimensional
 * and low dimensional fuzzy simplicial sets, yielding to the event loop
 * between two epochs so the caller can render the progress.
 * @param internals - State of an initialized model, updated in place.
 * @param epochCallback - Called with the number of completed epochs after each
 * epoch; returning `false` stops the optimization early.
 * @returns Whether every epoch ran.
 */
export function optimizeLayoutAsync(
  internals: UMAPInternals,
  epochCallback: EpochCallback = () => true,
): Promise<boolean> {
  return new Promise((resolve, reject) => {
    const state = internals.optimizationState;
    const step = () => {
      try {
        internals.embedding = optimizeLayoutStep(
          state,
          state.currentEpoch,
          internals.params.random,
        );
        const epochCompleted = state.currentEpoch;
        const shouldStop = epochCallback(epochCompleted) === false;
        const isFinished = epochCompleted === state.numberOfEpochs;
        if (!shouldStop && !isFinished) {
          setTimeout(() => step(), 0);
        } else {
          resolve(isFinished);
        }
      } catch (error) {
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    };
    setTimeout(() => step(), 0);
  });
}
