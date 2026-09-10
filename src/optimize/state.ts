/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript and moved to its own
 * module so the layout optimization can be called as plain functions taking
 * the state explicitly. Fields, defaults and semantics are unchanged.
 */

import type { Vectors } from '../types.ts';

/**
 * State carried between the steps of the stochastic gradient descent that
 * optimizes the low dimensional embedding.
 *
 * Every field has the value the reference implementation uses before the
 * optimization is initialized, so an instance is usable as-is and callers only
 * assign the entries they know.
 */
export class OptimizationState {
  /**
   * Number of epochs already completed.
   */
  currentEpoch = 0;

  /**
   * Embedding of the points that move, one row per point. This is the array
   * the optimization mutates in place and returns.
   */
  headEmbedding: Vectors = [];

  /**
   * Embedding the head points are attracted to and repelled from. It is the
   * same array as `headEmbedding` when fitting, and the frozen training
   * embedding when transforming new points.
   */
  tailEmbedding: Vectors = [];

  /**
   * Row of `headEmbedding` holding the first endpoint of every graph edge.
   */
  head: number[] = [];

  /**
   * Row of `tailEmbedding` holding the second endpoint of every graph edge.
   */
  tail: number[] = [];

  /**
   * Number of epochs between two positive samples of every edge.
   */
  epochsPerSample: number[] = [];

  /**
   * Epoch at which every edge is next positively sampled.
   */
  epochOfNextSample: number[] = [];

  /**
   * Epoch at which every edge is next negatively sampled.
   */
  epochOfNextNegativeSample: number[] = [];

  /**
   * Number of epochs between two negative samples of every edge.
   */
  epochsPerNegativeSample: number[] = [];

  /**
   * Whether the tail endpoint of an edge moves as well. A fit derives it from
   * the head and the tail embedding being the same array; a transform sets it
   * to `false`, which is what freezes the fitted embedding.
   */
  moveOther = true;

  /**
   * Learning rate at the first epoch; `alpha` decays linearly from it to 0.
   */
  initialAlpha = 1;

  /**
   * Learning rate of the epoch being run.
   */
  alpha = 1;

  /**
   * Weight applied to the repulsive (negative sampling) gradient.
   */
  gamma = 1;

  /**
   * First parameter of the differentiable curve `1 / (1 + a * d^(2b))` fitted
   * by `findABParams` from `spread` and `minimumDistance`.
   *
   * `a` and `b` are the only optimization state fields that carry over from
   * `fit` into `transform`: everything else is rebuilt for the new points. A
   * serialized model MUST therefore store `a` and `b`, otherwise a reloaded
   * model silently falls back to these defaults — which correspond to
   * `spread = 1` and `minimumDistance = 0.1` — and embeds new points against a
   * curve that has nothing to do with the one it was trained with.
   */
  a = 1.5769434603113077;

  /**
   * Second parameter of the differentiable curve `1 / (1 + a * d^(2b))` fitted
   * by `findABParams` from `spread` and `minimumDistance`.
   *
   * Like `a`, it carries over from `fit` into `transform` and must be part of
   * any serialized model; the default corresponds to `spread = 1` and
   * `minimumDistance = 0.1`.
   */
  b = 0.8950608779109733;

  /**
   * Number of components of the embedding, i.e. the length of every row of
   * `headEmbedding`.
   */
  dim = 2;

  /**
   * Total number of epochs the optimization runs for.
   */
  numberOfEpochs = 500;

  /**
   * Number of vertices of the graph, i.e. the exclusive upper bound of the
   * randomly drawn negative sample indices.
   */
  nVertices = 0;
}
