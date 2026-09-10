import { expect, test } from 'vitest';

import { computeMembershipStrengths } from '../membership.ts';

/*
 * The self test `knnIndices[i][j] === i` means "a point is not similar to
 * itself" only when rows and columns share an index space, which is true when
 * fitting. Projecting new data indexes rows by position in the batch and
 * columns by training point, so the same test zeroes an unrelated pair
 * whenever a neighbour's training index happens to equal a row position.
 * umap-learn guards this with `bipartite` (umap_.py:438-439) and passes
 * `bipartite=True` from transform (umap_.py:3245).
 */

function makeBipartiteCase(batchSize: number, neighbors: number) {
  const knnIndices: number[][] = [];
  const knnDistances: number[][] = [];
  for (let i = 0; i < batchSize; i++) {
    const indices: number[] = [];
    const distances: number[] = [];
    for (let j = 0; j < neighbors; j++) {
      indices.push((i * 3 + j * 7) % 100);
      distances.push(0.1 + j * 0.2);
    }
    knnIndices.push(indices);
    knnDistances.push(distances);
  }
  return {
    knnIndices,
    knnDistances,
    sigmas: Array.from({ length: batchSize }, () => 1),
    rhos: Array.from({ length: batchSize }, () => 0.05),
  };
}

test('the self test zeroes a matching index when the graph is not bipartite', () => {
  const { knnIndices, knnDistances, sigmas, rhos } = makeBipartiteCase(8, 5);

  const strengths = computeMembershipStrengths(
    knnIndices,
    knnDistances,
    sigmas,
    rhos,
  );

  // row 0 has training neighbour 0, which the self test treats as "self"
  const zeroed = strengths.vals.filter(
    (value, index) =>
      strengths.rows[index] === 0 && strengths.cols[index] === 0,
  );

  expect(zeroed).toStrictEqual([0]);
});

test('bipartite keeps a strength whose training index equals the row position', () => {
  const { knnIndices, knnDistances, sigmas, rhos } = makeBipartiteCase(8, 5);

  const strengths = computeMembershipStrengths(
    knnIndices,
    knnDistances,
    sigmas,
    rhos,
    { bipartite: true },
  );

  const kept = strengths.vals.filter(
    (value, index) =>
      strengths.rows[index] === 0 && strengths.cols[index] === 0,
  );

  // Math.exp is implementation-approximated, so compare closely rather than
  // exactly; the point of the test is that the value is kept, not zeroed.
  expect(kept).toHaveLength(1);
  expect(kept[0]).toBeCloseTo(0.951229424500714, 12);
});

test('bipartite changes only the pairs the self test would have zeroed', () => {
  const { knnIndices, knnDistances, sigmas, rhos } = makeBipartiteCase(8, 5);

  const withSelfTest = computeMembershipStrengths(
    knnIndices,
    knnDistances,
    sigmas,
    rhos,
  );
  const bipartite = computeMembershipStrengths(
    knnIndices,
    knnDistances,
    sigmas,
    rhos,
    { bipartite: true },
  );

  const changed: number[] = [];
  for (let i = 0; i < withSelfTest.vals.length; i++) {
    if (!Object.is(withSelfTest.vals[i], bipartite.vals[i])) {
      changed.push(i);
    }
  }

  expect(changed).toHaveLength(1);
  expect(withSelfTest.rows[changed[0] as number]).toBe(
    withSelfTest.cols[changed[0] as number],
  );
});
