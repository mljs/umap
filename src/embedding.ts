/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript as standalone functions
 * taking the graph, the epoch count and the random source explicitly instead
 * of reading them off the UMAP instance, the `map` callbacks building the
 * embedding and the epoch counts replaced by indexed loops writing the same
 * values in the same order, and the inlined maximum replaced by the shared
 * `max` helper, which scans identically. An opt-in spectral initialization was
 * added next to the uniform one, which upstream does not implement; the
 * uniform path is untouched, and it stays the default. The random draws, the
 * arithmetic and the iteration order are unchanged.
 */

import type { InitMethod } from './parameters.ts';
import type { SparseMatrix } from './sparse/index.ts';
import { spectralEmbedding } from './spectral/index.ts';
import type { RandomFn, Vectors } from './types.ts';
import { filled, max, tauRand, zeros } from './utils.ts';

/**
 * Options of {@link initializeSimplicialSetEmbedding}.
 */
export interface InitializeSimplicialSetEmbeddingOptions {
  /** Number of epochs the optimization will run for. */
  numberOfEpochs: number;
  /** Dimension of the space the data is embedded into. */
  numberOfComponents: number;
  /** Random number generator returning a float in [0, 1). */
  random: RandomFn;
  /**
   * How the embedding is initialized.
   * @default 'random'
   */
  init?: InitMethod;
  /**
   * The data the graph was built from, one array per point. A spectral
   * initialization reads it to arrange the connected components of a
   * disconnected graph; a uniform one never does.
   * @default []
   */
  data?: Vectors;
}

/**
 * The initialized embedding and the edge list the optimization runs over, as
 * returned by {@link initializeSimplicialSetEmbedding}.
 */
export interface SimplicialSetEmbedding {
  /** One initial position per graph row, in the embedding space. */
  embedding: Vectors;
  /** Column index of every retained edge. */
  head: number[];
  /** Row index of every retained edge. */
  tail: number[];
  /** Number of epochs between two positive samples of every edge. */
  epochsPerSample: number[];
}

/**
 * Initialize a fuzzy simplicial set embedding, using a specified
 * initialisation method and then minimizing the fuzzy set cross entropy
 * between the 1-skeletons of the high and low dimensional fuzzy simplicial
 * sets.
 *
 * The embedding is drawn uniformly in [-10, 10) unless `init` asks for the
 * spectral initialization, which is computed on the pruned graph, exactly
 * where the reference python implementation computes it. A spectral
 * initialization that fails falls back to the uniform draw, so it can never
 * make a fit fail.
 * @param graph - Fuzzy graph of the data, as built by `fuzzySimplicialSet`.
 * @param options - Initialization options.
 * @returns The embedding and the edges of the graph, weak edges dropped.
 */
export function initializeSimplicialSetEmbedding(
  graph: SparseMatrix,
  options: InitializeSimplicialSetEmbeddingOptions,
): SimplicialSetEmbedding {
  const {
    numberOfEpochs,
    numberOfComponents,
    random,
    init = 'random',
    data = [],
  } = options;

  const graphMax = max(graph.getValues());
  const pruned = graph.map((value) => {
    if (value < graphMax / numberOfEpochs) {
      return 0;
    } else {
      return value;
    }
  });

  let embedding: Vectors | null = null;
  if (init === 'spectral') {
    embedding = spectralEmbedding(pruned, { numberOfComponents, random, data });
  }
  embedding ??= randomEmbedding(pruned.nRows, numberOfComponents, random);

  // Get graph data in ordered way...
  const weights: number[] = [];
  const head: number[] = [];
  const tail: number[] = [];
  for (const entry of pruned.getAll()) {
    if (entry.value) {
      weights.push(entry.value);
      tail.push(entry.row);
      head.push(entry.col);
    }
  }
  const epochsPerSample = makeEpochsPerSample(weights, numberOfEpochs);

  return { embedding, head, tail, epochsPerSample };
}

/**
 * Draws every point of an embedding uniformly in [-10, 10).
 * @param nRows - Number of points to draw.
 * @param numberOfComponents - Dimension of the space the data is embedded into.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns The embedding, one array per point.
 */
export function randomEmbedding(
  nRows: number,
  numberOfComponents: number,
  random: RandomFn,
): Vectors {
  const embedding: Vectors = [];
  for (let i = 0; i < nRows; i++) {
    const row = zeros(numberOfComponents);
    for (let j = 0; j < numberOfComponents; j++) {
      row[j] = tauRand(random) * 20 + -10; // Random from -10 to 10
    }
    embedding.push(row);
  }
  return embedding;
}

/**
 * Given a set of weights and number of epochs generate the number of
 * epochs per sample for each weight.
 * @param weights - Membership strength of every edge.
 * @param numberOfEpochs - Number of epochs the optimization will run for.
 * @returns The number of epochs between two samples of every edge, `-1` for an
 * edge that is never sampled.
 */
export function makeEpochsPerSample(
  weights: number[],
  numberOfEpochs: number,
): number[] {
  const result = filled(weights.length, -1);
  const maxWeight = max(weights);
  for (let i = 0; i < weights.length; i++) {
    const weight = weights[i];
    if (weight === undefined) {
      throw new RangeError('makeEpochsPerSample: missing weight');
    }
    const nSamples = (weight / maxWeight) * numberOfEpochs;
    if (nSamples > 0) {
      result[i] = numberOfEpochs / nSamples;
    }
  }
  return result;
}

/**
 * Deep copies an embedding.
 *
 * `getEmbedding` and the transform both need a copy: the optimization mutates
 * the rows of the embedding it is given in place, so handing out the live
 * array — or reusing it as the tail embedding of a transform — lets a caller
 * silently corrupt the fitted model.
 * @param embedding - The embedding to copy.
 * @returns A copy holding a new array per point.
 */
export function copyEmbedding(embedding: Vectors): Vectors {
  const result: Vectors = [];
  for (const row of embedding) {
    result.push(row.slice());
  }
  return result;
}
