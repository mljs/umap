/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: the optimization half of `UMAP.transform` split out of
 * ./transform.ts to respect the 250 line file limit, and — the divergences
 * documented there — the fitted embedding copied and frozen with
 * `moveOther: false` and the learning rate quartered, as umap-learn does. The
 * arithmetic and the iteration order are otherwise unchanged.
 */

import { copyEmbedding, makeEpochsPerSample } from './embedding.ts';
import {
  assignOptimizationStateParameters,
  prepareForOptimizationLoop,
} from './fitSteps.ts';
import type { UMAPInternals } from './internals.ts';
import { optimizeLayout } from './optimize/layout.ts';
import type { SparseMatrix } from './sparse/index.ts';
import { eliminateZeros } from './sparse/index.ts';
import type { RandomFn, Vectors } from './types.ts';
import { max } from './utils.ts';

/**
 * Fraction of the configured learning rate the projection of new points runs
 * at, as `_initial_alpha / 4.0` in umap-learn (umap_.py:3304).
 */
const TRANSFORM_ALPHA_SCALE = 1 / 4;

/**
 * Relaxes the new points against the fitted embedding, which never moves.
 *
 * The projection runs for a third of the configured number of epochs, or for
 * the count the size of the graph implies when the model carries none, and the
 * edges too weak to survive that budget are pruned first, as a fit prunes its
 * own.
 * @param internals - State of a fitted model, whose optimization state is
 * overwritten with the state of this projection.
 * @param embedding - Initial position of each new point, optimized in place.
 * @param graph - Fuzzy graph of the new points against the fitted data.
 * @param random - Random source scoped to this projection.
 * @returns The position of each new point in the embedding space.
 */
export function optimizeTransformLayout(
  internals: UMAPInternals,
  embedding: Vectors,
  graph: SparseMatrix,
  random: RandomFn,
): Vectors {
  const { numberOfEpochs: configuredNumberOfEpochs } = internals.params;
  const numberOfEpochs = configuredNumberOfEpochs
    ? Math.floor(configuredNumberOfEpochs / 3)
    : graph.nRows <= 10000
      ? 100
      : 30;

  const graphMax = max(graph.getValues());
  const pruned = eliminateZeros(
    graph.map((value) => (value < graphMax / numberOfEpochs ? 0 : value)),
  );

  const epochsPerSample = makeEpochsPerSample(
    pruned.getValues(),
    numberOfEpochs,
  );

  // Initialize optimization slightly differently than the fit method.
  assignOptimizationStateParameters(internals, {
    headEmbedding: embedding,
    tailEmbedding: copyEmbedding(internals.embedding),
    head: pruned.getRows(),
    tail: pruned.getCols(),
    currentEpoch: 0,
    numberOfEpochs,
    nVertices: pruned.nCols,
    epochsPerSample,
  });
  prepareForOptimizationLoop(internals, {
    alphaScale: TRANSFORM_ALPHA_SCALE,
    moveOther: false,
  });

  return optimizeLayout(internals.optimizationState, random);
}
