/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: `initTransform` ported to strict TypeScript, the index
 * reads bounds checked, and split out of ./transform.ts to respect the 250
 * line file limit. The arithmetic and the iteration order are unchanged.
 */

import { at } from './nn/checkedAt.ts';
import type { Vectors } from './types.ts';
import { zeros } from './utils.ts';

/**
 * Given indices and weights and an original embeddings
 * initialize the positions of new points relative to the
 * indices and weights (of their neighbors in the source data).
 * @param indices - Index of the neighbors of each new point, in the fitted data.
 * @param weights - Weight of each of those neighbors, summing to one per point.
 * @param embedding - The fitted embedding.
 * @returns The weighted mean position of the neighbors of each new point.
 * @throws {RangeError} If the fitted embedding is empty.
 */
export function initTransform(
  indices: number[][],
  weights: number[][],
  embedding: Vectors,
): Vectors {
  const firstPoint = embedding[0];
  if (firstPoint === undefined) {
    throw new RangeError('initTransform: the embedding has no point');
  }
  const numberOfComponents = firstPoint.length;
  const firstRow = indices[0];
  if (firstRow === undefined) {
    throw new RangeError('initTransform: there is no point to transform');
  }
  const numberOfNeighbors = firstRow.length;

  const result: Vectors = new Array<number[]>(indices.length);
  for (let i = 0; i < indices.length; i++) {
    result[i] = zeros(numberOfComponents);
  }

  for (let i = 0; i < indices.length; i++) {
    const indicesRow = at(indices, i);
    const weightsRow = at(weights, i);
    const resultRow = at(result, i);
    for (let j = 0; j < numberOfNeighbors; j++) {
      const neighbor = at(indicesRow, j);
      const weight = at(weightsRow, j);
      const neighborPoint = at(embedding, neighbor);
      for (let d = 0; d < numberOfComponents; d++) {
        resultRow[d] = at(resultRow, d) + weight * at(neighborPoint, d);
      }
    }
  }
  return result;
}
