import { expect, test } from 'vitest';

import { makeRandom } from '../../__tests__/random.ts';
import { euclidean } from '../../metrics.ts';
import type { RandomFn, Vectors } from '../../types.ts';
import { at } from '../checkedAt.ts';
import { makeHeap } from '../heap.ts';
import { makeInitializations } from '../search.ts';
import { makeForest, searchFlatTree } from '../tree.ts';

const N_NEIGHBORS = 10;
const LEAF_SIZE = Math.max(10, N_NEIGHBORS);

test('initFromTree seeds every query point with its own leaf', () => {
  const data = makePoints(400, makeRandom(1));
  const queryPoints = makePoints(25, makeRandom(2));
  const forest = makeForest(data, N_NEIGHBORS, 1, makeRandom(3));
  const tree = at(forest, 0);

  const heap = makeHeap(queryPoints.length, LEAF_SIZE);
  const { initFromTree } = makeInitializations(euclidean);

  initFromTree(tree, data, queryPoints, heap, makeRandom(4));

  const expectSearch = makeRandom(4);

  for (let i = 0; i < queryPoints.length; i++) {
    const leaf = searchFlatTree(at(queryPoints, i), tree, expectSearch);
    const expected = positiveOnly(leaf).toSorted((a, b) => a - b);
    const seeded = positiveOnly(at(heap[0], i)).toSorted((a, b) => a - b);

    expect(seeded).not.toHaveLength(0);
    expect(seeded).toStrictEqual(expected);
  }
});

test('initFromTree seeds every query point of every tree of a forest', () => {
  const data = makePoints(400, makeRandom(1));
  const queryPoints = makePoints(25, makeRandom(2));
  const forest = makeForest(data, N_NEIGHBORS, 4, makeRandom(3));

  expect(forest).toHaveLength(4);

  const { initFromTree } = makeInitializations(euclidean);
  const random = makeRandom(4);

  for (const tree of forest) {
    const heap = makeHeap(queryPoints.length, LEAF_SIZE);

    initFromTree(tree, data, queryPoints, heap, random);

    for (let i = 0; i < queryPoints.length; i++) {
      expect(positiveOnly(at(heap[0], i))).not.toHaveLength(0);
    }
  }
});

test('initFromTree brings the true nearest neighbor in more often than chance', () => {
  const data = makePoints(400, makeRandom(1));
  const queryPoints = makePoints(25, makeRandom(2));
  const forest = makeForest(data, N_NEIGHBORS, 4, makeRandom(3));
  const { initFromTree } = makeInitializations(euclidean);
  const heap = makeHeap(queryPoints.length, LEAF_SIZE);
  const random = makeRandom(4);

  for (const tree of forest) {
    initFromTree(tree, data, queryPoints, heap, random);
  }

  let found = 0;

  for (let i = 0; i < queryPoints.length; i++) {
    const nearest = nearestIndex(data, at(queryPoints, i));

    if (at(heap[0], i).includes(nearest)) found++;
  }

  // Four leaves of 10 out of 400 points would find it 10% of the time at
  // random; the split is informative, so it is found for almost every query.
  expect(found).toBeGreaterThanOrEqual(23);
});

/**
 * Builds deterministic pseudo-random points in the unit cube.
 * @param count - Number of points.
 * @param random - Seeded generator.
 * @returns The points, of dimension four.
 */
function makePoints(count: number, random: RandomFn): Vectors {
  const points: Vectors = [];

  for (let i = 0; i < count; i++) {
    points.push([random(), random(), random(), random()]);
  }

  return points;
}

/**
 * Drops the `-1` padding of a leaf or of a heap row.
 * @param values - Row to filter.
 * @returns The non-negative entries.
 */
function positiveOnly(values: number[]): number[] {
  const kept: number[] = [];

  for (let i = 0; i < values.length; i++) {
    const value = at(values, i);

    if (value >= 0) kept.push(value);
  }

  return kept;
}

/**
 * Index of the point of `data` closest to `point`.
 * @param data - The candidate points.
 * @param point - The point to locate.
 * @returns Index of the nearest point.
 */
function nearestIndex(data: Vectors, point: number[]): number {
  let bestIndex = 0;
  let bestDistance = Infinity;

  for (let i = 0; i < data.length; i++) {
    const distance = euclidean(at(data, i), point);

    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }

  return bestIndex;
}
