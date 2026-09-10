/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: the optimization set-up methods of the UMAP class
 * ported to strict TypeScript as standalone functions taking the state
 * explicitly, the `map` callback building the negative sample counts replaced
 * by an indexed loop writing the same values in the same order, the state
 * that upstream leaves implicitly undefined checked before it is read, and
 * `prepareForOptimizationLoop` given an options argument so the learning rate
 * scale and `moveOther` can be set by the caller instead of being derived —
 * umap-learn sets both explicitly and differently on the two paths. The
 * arithmetic and the iteration order of the fit are unchanged.
 */

import { categoricalSimplicialSetIntersection } from './fuzzy/simplicialSet.ts';
import type { UMAPInternals } from './internals.ts';
import { findABParams } from './optimize/abParams.ts';
import type { OptimizationState } from './optimize/state.ts';
import type { SparseMatrix } from './sparse/index.ts';

/**
 * Checks if we're using supervised projection, then process the graph
 * accordingly.
 * @param internals - State of the model, whose `graph` may be replaced.
 * @throws {RangeError} If the target array and the data have different lengths.
 */
export function processGraphForSupervisedProjection(
  internals: UMAPInternals,
): void {
  const { Y, X, supervised } = internals;
  if (!Y) return;

  if (Y.length !== X.length) {
    throw new RangeError('Length of X and y must be equal');
  }

  const { targetMetric, targetWeight } = supervised;
  if (targetMetric === 'categorical') {
    const lt = targetWeight < 1;
    const farDist = lt ? 2.5 * (1 / (1 - targetWeight)) : 1e12;
    internals.graph = categoricalSimplicialSetIntersection(
      requireGraph(internals),
      Y,
      { farDist },
    );
  }
  // Non-categorical supervised embeddings are not implemented, as upstream.
}

/**
 * Assigns optimization state parameters from a partial optimization state.
 * @param internals - State of the model, whose optimization state is updated.
 * @param state - The entries to assign.
 */
export function assignOptimizationStateParameters(
  internals: UMAPInternals,
  state: Partial<OptimizationState>,
): void {
  Object.assign(internals.optimizationState, state);
}

/**
 * Initializes optimization state for stepwise optimization.
 *
 * The head and the tail embedding are deliberately the SAME array here: that
 * aliasing is what makes the derived `moveOther` true, so a fit moves both
 * endpoints of every edge, and what lets a caller watch the embedding converge
 * while it steps through the epochs. The transform path neither aliases them
 * nor derives the flag: it freezes the fitted embedding by passing
 * `moveOther: false` to `prepareForOptimizationLoop`.
 * @param internals - State of the model, whose optimization state is updated.
 */
export function initializeOptimization(internals: UMAPInternals): void {
  const headEmbedding = internals.embedding;
  const tailEmbedding = internals.embedding;

  // Initialized in initializeSimplicialSetEmbedding().
  const { head, tail, epochsPerSample } = internals.optimizationState;

  const numberOfEpochs = getNumberOfEpochs(internals);
  const nVertices = requireGraph(internals).nCols;

  const { spread, minimumDistance } = internals.params;
  const { a, b } = findABParams(spread, minimumDistance);

  assignOptimizationStateParameters(internals, {
    headEmbedding,
    tailEmbedding,
    head,
    tail,
    epochsPerSample,
    a,
    b,
    numberOfEpochs,
    nVertices,
  });
}

/** Options of `prepareForOptimizationLoop`. */
export interface PrepareForOptimizationLoopOptions {
  /**
   * Factor the configured learning rate is multiplied by. umap-learn optimizes
   * a projection of new points with `_initial_alpha / 4.0` (umap_.py:3304), a
   * quarter of the rate a fit runs at, so the transform path passes `0.25`.
   * @default 1
   */
  alphaScale?: number;

  /**
   * Whether the tail endpoint of a sampled edge moves as well. umap-learn
   * leaves `move_other` at its default of `False` when projecting new points
   * (layouts.py:257), which freezes the fitted embedding, so the transform
   * path passes `false`. When it is not given, it is derived as upstream does,
   * from the head and the tail embedding having the same length.
   * @default headEmbedding.length === tailEmbedding.length
   */
  moveOther?: boolean;
}

/**
 * Sets a few optimization state parameters that are necessary before entering
 * the optimization step loop.
 * @param internals - State of the model, whose optimization state is updated.
 * @param options - How the learning rate and `moveOther` are set.
 * @throws {RangeError} If the head embedding is empty.
 */
export function prepareForOptimizationLoop(
  internals: UMAPInternals,
  options: PrepareForOptimizationLoopOptions = {},
): void {
  const { alphaScale = 1, moveOther } = options;

  const { repulsionStrength, learningRate, negativeSampleRate } =
    internals.params;

  const { epochsPerSample, headEmbedding, tailEmbedding } =
    internals.optimizationState;

  const firstRow = headEmbedding[0];
  if (firstRow === undefined) {
    throw new RangeError(
      'prepareForOptimizationLoop: the embedding has no point',
    );
  }
  const dim = firstRow.length;
  const movesTail = moveOther ?? headEmbedding.length === tailEmbedding.length;
  const alpha = learningRate * alphaScale;

  const epochsPerNegativeSample: number[] = [];
  for (const epochs of epochsPerSample) {
    epochsPerNegativeSample.push(epochs / negativeSampleRate);
  }
  const epochOfNextNegativeSample = [...epochsPerNegativeSample];
  const epochOfNextSample = [...epochsPerSample];

  assignOptimizationStateParameters(internals, {
    epochOfNextSample,
    epochOfNextNegativeSample,
    epochsPerNegativeSample,
    moveOther: movesTail,
    initialAlpha: alpha,
    alpha,
    gamma: repulsionStrength,
    dim,
  });
}

/**
 * Gets the number of epochs for optimizing the projection.
 * NOTE: This heuristic differs from the python version.
 * @param internals - State of a fitted model.
 * @returns The configured number of epochs, or the one the data size implies.
 */
export function getNumberOfEpochs(internals: UMAPInternals): number {
  const { numberOfEpochs } = internals.params;
  if (numberOfEpochs > 0) {
    return numberOfEpochs;
  }

  const length = requireGraph(internals).nRows;
  if (length <= 2500) {
    return 500;
  } else if (length <= 5000) {
    return 400;
  } else if (length <= 7500) {
    return 300;
  } else {
    return 200;
  }
}

function requireGraph(internals: UMAPInternals): SparseMatrix {
  const { graph } = internals;
  if (graph === undefined) {
    throw new Error('The fuzzy graph is only available after a fit.');
  }
  return graph;
}
