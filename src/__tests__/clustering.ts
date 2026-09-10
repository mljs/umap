/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: the `checkClusters` helper of the upstream test suite
 * extracted into a module that returns the ratio instead of asserting on it,
 * and its `map` / `reduce` chains rewritten as loops.
 */

import { euclidean } from '../index.ts';

/**
 * Ratio between the mean distance separating two points of the same class and
 * the mean distance separating any two points. It is the measure upstream's
 * test suite uses to check that an embedding actually clusters: the lower the
 * ratio, the tighter the classes are relative to the whole cloud.
 * @param embeddings - The embedded points, one array per point.
 * @param labels - Class of each of those points.
 * @returns The clustering ratio, in [0, 1] for any sane embedding.
 */
export function clusterRatio(embeddings: number[][], labels: number[]): number {
  const overallMeanDistance = meanOfMeanDistances(embeddings);

  const groups = new Map<number, number[][]>();
  for (let i = 0; i < labels.length; i++) {
    const label = labels[i];
    const embedding = embeddings[i];
    if (label === undefined || embedding === undefined) continue;
    const group = groups.get(label);
    if (group === undefined) {
      groups.set(label, [embedding]);
    } else {
      group.push(embedding);
    }
  }

  let totalIntraclusterDistance = 0;
  for (const group of groups.values()) {
    totalIntraclusterDistance += meanOfMeanDistances(group) * group.length;
  }

  return totalIntraclusterDistance / embeddings.length / overallMeanDistance;
}

/**
 * Mean over the points of the mean distance from one point to all of them.
 * @param vectors - The points.
 * @returns The mean distance.
 */
function meanOfMeanDistances(vectors: number[][]): number {
  let total = 0;
  for (const vector of vectors) {
    let sum = 0;
    for (const other of vectors) {
      sum += euclidean(vector, other);
    }
    total += sum / vectors.length;
  }
  return total / vectors.length;
}
