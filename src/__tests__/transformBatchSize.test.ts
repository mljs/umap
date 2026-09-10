/*
 * ml-umap: what a projection guarantees about the batch it is passed.
 *
 * GUARANTEED, and asserted below: a given batch always projects to the same
 * coordinates, on the model it was fitted on and on any model fitted the same
 * way; and the fitted embedding never moves, whatever the batch holds —
 * including a batch of exactly as many points as were fitted, the size
 * upstream's `headEmbedding.length === tailEmbedding.length` singles out to
 * rewrite the model in place.
 *
 * NOT GUARANTEED: that one point lands in the same spot when the rest of the
 * batch changes. Projecting a 60 row batch and its first 59 rows moves the 59
 * shared points by up to 2.33 on a 27.46 wide embedding (8.5%), while every
 * one of them still lands in the cluster it belongs to. umap-learn has the
 * same limitation: it too projects a batch as a whole, under one
 * `transform_seed`. Three separate channels carry the coupling, measured on
 * this fixture with the fit held fixed:
 *
 * (1) The optimizer draws its negative samples edge by edge out of ONE stream,
 *     so the 14 extra edges of the 60 row batch shift every draw from the
 *     second epoch on. Everything ahead of the optimizer is bit-identical
 *     between the two batches here — the neighbours found, the sigmas, the
 *     rhos, the edge list, `epochsPerSample`, and the initialized embedding —
 *     so this channel alone accounts for the whole drift.
 * (2) `computeMembershipStrengths` zeroes the entry whose training index
 *     equals the ROW POSITION of the query point, which is a training index
 *     only by coincidence. Shifting the batch by one position changes the
 *     membership strengths of 47 of the 59 points whose neighbour set is
 *     unchanged. Inherited from umap-js, which inherits it from umap-learn.
 * (3) `graphMax` is taken over the whole batch and normalizes both the edge
 *     pruning threshold and `epochsPerSample`. 16 of the 59 prefixes of this
 *     batch have a `graphMax` different from the full batch.
 *
 * Only (1) is a property of the random source, so seeding per query point
 * would buy an invariance that (2) and (3) still break. The batch is the unit
 * of a projection: project the points you want to compare together.
 */

import { expect, test } from 'vitest';

import type { Vectors } from '../index.ts';
import { UMAP } from '../index.ts';

import {
  embeddingSpread,
  maxAbsoluteDifference,
  nearestNeighborIndex,
} from './embeddings.ts';
import { makeRandom } from './random.ts';

// A full fit is 500 epochs and exceeds the 5s vitest default.
const FIT_TIMEOUT = 120_000;

const FITTED_SIZE = 60;
const CLUSTER_COUNT = 3;

const fittedData = makeClouds(FITTED_SIZE, 0, 1234);
// Exactly as many points to project as were fitted, which is the size
// `headEmbedding.length === tailEmbedding.length` singles out.
const wholeBatch = makeClouds(FITTED_SIZE, 0.05, 987);
const shorterBatch = wholeBatch.slice(0, FITTED_SIZE - 1);

test(
  'a batch projects to the same coordinates every time it is projected',
  () => {
    const umap = fit();

    const first = umap.transform(wholeBatch);
    const second = umap.transform(wholeBatch);

    expect(second).toStrictEqual(first);

    // A model fitted the same way projects it to the same coordinates too, so
    // the projection depends on the fitted model and on `transformSeed`, never
    // on how many projections have already run.
    expect(fit().transform(wholeBatch)).toStrictEqual(first);
  },
  FIT_TIMEOUT,
);

test(
  'the fitted embedding never moves, whatever the size of the batch',
  () => {
    const umap = fit();
    const fittedEmbedding = umap.getEmbedding();

    const whole = umap.transform(wholeBatch);

    expect(umap.internals.optimizationState.moveOther).toBe(false);
    expect(whole).toHaveLength(FITTED_SIZE);
    expect(umap.getEmbedding()).toStrictEqual(fittedEmbedding);

    const shorter = umap.transform(shorterBatch);

    expect(umap.internals.optimizationState.moveOther).toBe(false);
    expect(shorter).toHaveLength(FITTED_SIZE - 1);
    expect(umap.getEmbedding()).toStrictEqual(fittedEmbedding);
  },
  FIT_TIMEOUT,
);

test(
  'dropping the last point of a batch nudges the others without moving them to another cluster',
  () => {
    const umap = fit();
    const fittedEmbedding = umap.getEmbedding();

    const whole = umap.transform(wholeBatch);
    const shorter = umap.transform(shorterBatch);
    const shared = whole.slice(0, FITTED_SIZE - 1);

    // The documented limitation: the shared points are NOT bit-identical.
    expect(shared).not.toStrictEqual(shorter);

    // They stay close: the drift is a jitter, not a relocation. Measured at
    // 2.33 on a 27.46 wide embedding.
    const spread = embeddingSpread(fittedEmbedding);
    const drift = maxAbsoluteDifference(shared, shorter);

    expect(drift).toBeGreaterThan(0);
    expect(drift).toBeLessThan(spread / 4);

    // And every point of both batches still lands nearest to a training point
    // of its own cluster, which is what a projection actually promises.
    expect(countInOwnCluster(whole, fittedEmbedding)).toBe(FITTED_SIZE);
    expect(countInOwnCluster(shorter, fittedEmbedding)).toBe(FITTED_SIZE - 1);
  },
  FIT_TIMEOUT,
);

test(
  'a transform runs at a quarter of the learning rate a fit runs at',
  () => {
    const umap = new UMAP({
      random: makeRandom(42),
      numberOfComponents: 2,
      learningRate: 2,
    });

    umap.fit(fittedData);

    expect(umap.internals.optimizationState.initialAlpha).toBe(2);
    expect(umap.internals.optimizationState.moveOther).toBe(true);

    umap.transform(makeClouds(5, 0.05, 987));

    expect(umap.internals.optimizationState.initialAlpha).toBe(0.5);
    expect(umap.internals.optimizationState.moveOther).toBe(false);
  },
  FIT_TIMEOUT,
);

/**
 * Fits a model on {@link fittedData}, from a seeded stream so that two calls
 * return two models holding the very same embedding.
 * @returns The fitted model.
 */
function fit(): UMAP {
  const umap = new UMAP({ random: makeRandom(0), numberOfComponents: 2 });
  umap.fit(fittedData);
  return umap;
}

/**
 * Draws points around three well separated centers, so that the nearest
 * neighbors of a point do not depend on what the search is seeded with. The
 * point at index `i` belongs to cluster `i % 3`.
 * @param count - Number of points to draw.
 * @param offset - Shift applied to every coordinate.
 * @param seed - Seed of the generator drawing the points.
 * @returns The points, five dimensional.
 */
function makeClouds(count: number, offset: number, seed: number): number[][] {
  const random = makeRandom(seed);
  const points: number[][] = [];
  for (let i = 0; i < count; i++) {
    const center = (i % CLUSTER_COUNT) * 4 + offset;
    const point: number[] = [];
    for (let dimension = 0; dimension < 5; dimension++) {
      point.push(center + random());
    }
    points.push(point);
  }
  return points;
}

/**
 * Counts the projected points whose nearest training point belongs to the
 * cluster the projected point was drawn from.
 * @param projected - The projected points, in the order `makeClouds` drew them.
 * @param embedding - The fitted embedding, in that same order.
 * @returns How many points landed in their own cluster.
 */
function countInOwnCluster(projected: Vectors, embedding: Vectors): number {
  let result = 0;
  for (let i = 0; i < projected.length; i++) {
    const point = projected[i];
    if (point === undefined) throw new Error(`missing point ${i}`);
    const nearest = nearestNeighborIndex(embedding, point);
    if (nearest % CLUSTER_COUNT === i % CLUSTER_COUNT) {
      result++;
    }
  }
  return result;
}
