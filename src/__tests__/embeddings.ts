/*
 * ml-umap: the measurements the tests take on an embedding.
 */

import type { Vectors } from '../index.ts';

/**
 * Index of the item closest to a point, under the squared euclidean distance,
 * which orders the candidates exactly as the euclidean one does.
 * @param items - The candidate points.
 * @param point - The point to locate.
 * @returns Index of the nearest item.
 * @throws {Error} If `items` holds a hole.
 */
export function nearestNeighborIndex(items: Vectors, point: number[]): number {
  let bestIndex = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item === undefined) throw new Error(`missing item ${i}`);
    let distance = 0;
    for (let dimension = 0; dimension < item.length; dimension++) {
      const difference = (item[dimension] ?? 0) - (point[dimension] ?? 0);
      distance += difference * difference;
    }
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Width of the range an embedding occupies, over every coordinate at once.
 * @param embedding - The embedding to measure.
 * @returns The largest coordinate minus the smallest one.
 */
export function embeddingSpread(embedding: Vectors): number {
  let smallest = Infinity;
  let largest = -Infinity;
  for (const row of embedding) {
    for (const value of row) {
      if (value < smallest) smallest = value;
      if (value > largest) largest = value;
    }
  }
  return largest - smallest;
}

/**
 * Largest absolute difference between two embeddings of the same shape.
 * @param first - First embedding.
 * @param second - Second embedding.
 * @returns The largest per-coordinate distance.
 * @throws {Error} If either embedding holds a hole.
 */
export function maxAbsoluteDifference(first: Vectors, second: Vectors): number {
  let result = 0;
  for (let i = 0; i < first.length; i++) {
    const left = first[i];
    const right = second[i];
    if (left === undefined || right === undefined) {
      throw new Error(`missing row ${i}`);
    }
    for (let j = 0; j < left.length; j++) {
      const difference = Math.abs((left[j] ?? 0) - (right[j] ?? 0));
      if (difference > result) {
        result = difference;
      }
    }
  }
  return result;
}
