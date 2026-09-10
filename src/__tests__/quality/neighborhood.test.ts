import { expect, test } from 'vitest';

import { knnPreservation } from './knnPreservation.ts';
import { neighborRanks, rankPair } from './ranks.ts';
import { continuity, trustworthiness } from './trustworthiness.ts';

/*
 * Every expected value below is computed by hand from these two 1-D
 * configurations of the same five points. Both are laid out so that no point
 * has two neighbors at the same distance, which makes every rank unambiguous.
 *
 * HIGH: 0, 1, 3, 20, 24            LOW: 0, 30, 1, 3, 7
 *
 * neighbors, nearest first     rank of j from i (1-based, 0 on the diagonal)
 * HIGH  p0: 1 2 3 4            p0: . 1 2 3 4
 *       p1: 0 2 3 4            p1: 1 . 2 3 4
 *       p2: 1 0 3 4            p2: 2 1 . 3 4
 *       p3: 4 2 1 0            p3: 4 3 2 . 1
 *       p4: 3 2 1 0            p4: 4 3 2 1 .
 * LOW   p0: 2 3 4 1            p0: . 4 1 2 3
 *       p1: 4 3 2 0            p1: 4 . 3 2 1
 *       p2: 0 3 4 1            p2: 1 4 . 2 3
 *       p3: 2 0 4 1            p3: 2 4 1 . 3
 *       p4: 3 2 0 1            p4: 3 4 2 1 .
 */
const HIGH = [[0], [1], [3], [20], [24]];
const LOW = [[0], [30], [1], [3], [7]];

test('neighborRanks orders every point against every other', () => {
  const ranks = neighborRanks(HIGH);

  expect(ranks.n).toBe(5);
  expect(ranks.order).toStrictEqual(
    // prettier-ignore
    new Int32Array([
      1, 2, 3, 4,
      0, 2, 3, 4,
      1, 0, 3, 4,
      4, 2, 1, 0,
      3, 2, 1, 0,
    ]),
  );
  expect(ranks.rank).toStrictEqual(
    // prettier-ignore
    new Int32Array([
      0, 1, 2, 3, 4,
      1, 0, 2, 3, 4,
      2, 1, 0, 3, 4,
      4, 3, 2, 0, 1,
      4, 3, 2, 1, 0,
    ]),
  );
});

test('neighbors at the same distance are ordered by index', () => {
  const ranks = neighborRanks([[0], [1], [-1], [2]]);

  // 1 and 2 are both one away from 0, and 0 and 3 are both one away from 1.
  expect(ranks.order.slice(0, 3)).toStrictEqual(new Int32Array([1, 2, 3]));
  expect(ranks.order.slice(3, 6)).toStrictEqual(new Int32Array([0, 3, 2]));
});

test('neighborRanks works on points of any dimension', () => {
  const ranks = neighborRanks([
    [0, 0],
    [1, 0],
    [0, 3],
  ]);

  expect(ranks.order).toStrictEqual(new Int32Array([1, 2, 0, 2, 0, 1]));
});

test('an embedding of itself preserves every neighborhood', () => {
  for (const k of [1, 2]) {
    expect(knnPreservation(HIGH, HIGH, k)).toBe(1);
    expect(trustworthiness(HIGH, HIGH, k)).toBe(1);
    expect(continuity(HIGH, HIGH, k)).toBe(1);
  }
});

test('knnPreservation counts the neighbors the embedding keeps', () => {
  // k = 1: only p4 keeps its nearest neighbor, so 1 of 5.
  expect(knnPreservation(HIGH, LOW, 1)).toBeCloseTo(0.2, 12);
  // k = 2: p0, p2 and p3 keep one of two, p4 keeps both, p1 keeps none.
  expect(knnPreservation(HIGH, LOW, 2)).toBeCloseTo(0.5, 12);
});

test('trustworthiness penalizes the intruders by their rank in the input', () => {
  // k = 1, normalizer n * k * (2n - 3k - 1) / 2 = 15. The nearest neighbor the
  // embedding shows has input rank 2, 4, 2, 2 and 1 for p0..p4, so the summed
  // rank excess is 1 + 3 + 1 + 1 + 0 = 6.
  expect(trustworthiness(HIGH, LOW, 1)).toBeCloseTo(1 - 6 / 15, 12);
  // k = 2, same normalizer: the intruders are 3 from p0 (input rank 3), 4 and
  // 3 from p1 (4 and 3), 3 from p2 (3) and 0 from p3 (4), so 1 + 3 + 1 + 2.
  expect(trustworthiness(HIGH, LOW, 2)).toBeCloseTo(1 - 7 / 15, 12);
});

test('continuity penalizes the lost neighbors by their rank in the embedding', () => {
  // k = 1: the true neighbor the embedding lost sits at embedding rank 4, 4,
  // 4 and 3 for p0..p3, so the summed rank excess is 3 + 3 + 3 + 2 + 0 = 11.
  expect(continuity(HIGH, LOW, 1)).toBeCloseTo(1 - 11 / 15, 12);
  // k = 2: 1 from p0 (embedding rank 4), 0 and 2 from p1 (4 and 3), 1 from p2
  // (4) and 4 from p3 (3), so 2 + 3 + 2 + 1.
  expect(continuity(HIGH, LOW, 2)).toBeCloseTo(1 - 8 / 15, 12);
});

test('continuity is trustworthiness with the two spaces swapped', () => {
  for (const k of [1, 2]) {
    expect(continuity(HIGH, LOW, k)).toBe(trustworthiness(LOW, HIGH, k));
  }
});

test('the measures only depend on the ranks, not on the geometry', () => {
  const scaled = LOW.map(([x]) => [(x ?? 0) * -7.5 + 3]);

  expect(knnPreservation(HIGH, scaled, 2)).toBe(knnPreservation(HIGH, LOW, 2));
  expect(trustworthiness(HIGH, scaled, 2)).toBe(trustworthiness(HIGH, LOW, 2));
  expect(continuity(HIGH, scaled, 2)).toBe(continuity(HIGH, LOW, 2));
});

test('a space with fewer than two points cannot be ranked', () => {
  expect(() => neighborRanks([[1]])).toThrow(
    /at least two points are required/,
  );
});

test('points of different sizes cannot be ranked', () => {
  expect(() => neighborRanks([[1, 2], [3]])).toThrow(
    /every point must have the same size/,
  );
});

test('the two spaces must describe the same points', () => {
  expect(() => rankPair(HIGH, LOW.slice(1))).toThrow(
    /must describe the same points, got 5 and 4/,
  );
  expect(() => knnPreservation(HIGH, LOW.slice(1), 1)).toThrow(
    /must describe the same points/,
  );
});

test('a neighborhood size outside the usable range is rejected', () => {
  expect(() => knnPreservation(HIGH, LOW, 0)).toThrow(
    /k must be an integer in \[1, 4\], got 0/,
  );
  expect(() => knnPreservation(HIGH, LOW, 5)).toThrow(/\[1, 4\], got 5/);
  expect(() => knnPreservation(HIGH, LOW, 1.5)).toThrow(/\[1, 4\], got 1.5/);
  // The Venna & Kaski normalization is only defined below n / 2.
  expect(() => trustworthiness(HIGH, LOW, 3)).toThrow(
    /trustworthiness: k must be an integer in \[1, 2\], got 3/,
  );
  expect(() => continuity(HIGH, LOW, 3)).toThrow(
    /continuity: k must be an integer in \[1, 2\], got 3/,
  );
});
