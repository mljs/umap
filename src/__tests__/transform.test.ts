/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: the upstream transform tests rewritten as flat vitest
 * tests, and regression tests added for the embedding aliasing and for the NaN
 * embedding that upstream's transform path returns for a point whose neighbors
 * are all duplicates.
 */

import { expect, test } from 'vitest';

import { smoothKNNDistance } from '../fuzzy/smoothKnn.ts';
import { UMAP, euclidean } from '../index.ts';

import {
  additionalData,
  additionalLabels,
  testData,
  testLabels,
} from './data/upstream-fixtures.ts';
import { makeRandom } from './random.ts';

// A full fit is 500 epochs over 100 points and exceeds the 5s vitest default.
const FIT_TIMEOUT = 120_000;

/*
 * The upstream fixture `transformResult2d` is deliberately NOT asserted here.
 * Our port fixes an inverted loop condition in heap.smallestFlagged, which
 * upstream left always returning -1; that made the graph-search refinement of
 * the initialized NN search dead code. transform() is the only caller of that
 * path, so our transform legitimately lands on different — better refined —
 * neighbors than the vendored fixture recorded. fit() never reaches it, which
 * is why the golden fit fixtures are still asserted exactly in index.test.ts.
 */

test(
  'transform projects additional points into the fitted space',
  () => {
    const umap = new UMAP({ random: makeRandom(), numberOfComponents: 2 });
    umap.fit(testData);

    const transformed = umap.transform(additionalData);

    expect(transformed).toHaveLength(additionalData.length);

    for (const point of transformed) {
      expect(point).toHaveLength(2);

      for (const value of point) {
        expect(Number.isFinite(value)).toBe(true);
      }
    }

    expect(transformed).toMatchSnapshot();
  },
  FIT_TIMEOUT,
);

test(
  'a transformed point lands next to a training point of its own class',
  () => {
    const umap = new UMAP({ random: makeRandom(), numberOfComponents: 2 });
    const embedding = umap.fit(testData);

    const transformed = umap.transform(additionalData);

    for (let i = 0; i < transformed.length; i++) {
      const point = transformed[i];
      if (point === undefined) throw new Error(`missing point ${i}`);
      const nearest = nearestNeighborIndex(embedding, point);

      expect(testLabels[nearest]).toBe(additionalLabels[i]);
    }
  },
  FIT_TIMEOUT,
);

test(
  'transforming exactly as many points as were fitted leaves the model intact',
  () => {
    const umap = new UMAP({ random: makeRandom(), numberOfComponents: 2 });
    umap.fit(testData);
    const before = deepCopy(umap.getEmbedding());

    // moveOther is headEmbedding.length === tailEmbedding.length, so upstream —
    // which aliases the fitted embedding into tailEmbedding — moves the training
    // points too and silently corrupts the model on exactly this input size.
    const transformed = umap.transform(testData);

    expect(transformed).toHaveLength(testData.length);
    expect(umap.getEmbedding()).toStrictEqual(before);
  },
  FIT_TIMEOUT,
);

test(
  'a second transform is unaffected by the first',
  () => {
    const umap = new UMAP({ random: makeRandom(), numberOfComponents: 2 });
    umap.fit(testData);
    const afterFit = deepCopy(umap.getEmbedding());

    umap.transform(additionalData);
    umap.transform(additionalData);

    expect(umap.getEmbedding()).toStrictEqual(afterFit);
  },
  FIT_TIMEOUT,
);

test('transform throws before any data has been fitted', () => {
  const umap = new UMAP({ random: makeRandom(), numberOfComponents: 2 });

  expect(() => umap.transform(additionalData)).toThrow(/No data has been fit/);
});

/**
 * Copies a 2d array, rows included, so later mutation of the source cannot be
 * observed through the copy.
 * @param values - The rows to copy.
 * @returns An independent copy.
 */
function deepCopy(values: number[][]): number[][] {
  const copy: number[][] = new Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const row = values[i];
    if (row === undefined) throw new Error(`missing row ${i}`);
    copy[i] = row.slice();
  }
  return copy;
}

/**
 * Index of the item closest to a point under the euclidean metric.
 * @param items - The candidate points.
 * @param point - The point to locate.
 * @returns Index of the nearest item.
 */
function nearestNeighborIndex(items: number[][], point: number[]): number {
  let bestIndex = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item === undefined) continue;
    const distance = euclidean(item, point);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

test(
  'transform terminates for a numberOfEpochs that is not a multiple of three',
  () => {
    // Regression: transform runs numberOfEpochs/3 epochs, and optimizeLayout
    // stopped on `currentEpoch === numberOfEpochs`. A non-integer target never
    // satisfied that strict equality against the integer epoch counter, so
    // transform looped forever.
    // Inherited from umap-js (umap.ts:1049-1059).
    for (const numberOfEpochs of [200, 137]) {
      const umap = new UMAP({
        random: makeRandom(),
        numberOfComponents: 2,
        numberOfEpochs,
      });

      umap.fit(testData);

      const projected = umap.transform(additionalData);

      expect(projected).toHaveLength(additionalData.length);

      for (const point of projected) {
        expect(point).toHaveLength(2);
        expect(Number.isFinite(point[0])).toBe(true);
        expect(Number.isFinite(point[1])).toBe(true);
      }
    }
  },
  FIT_TIMEOUT,
);

test('smoothKNNDistance gives a rho of zero to a row of zero distances', () => {
  // Regression: transform passes `localConnectivity - 1`, so the default of one
  // reaches smoothKNNDistance as zero. Upstream then reads past the end of the
  // empty array of non-zero distances and gets NaN for rho.
  const { sigmas, rhos } = smoothKNNDistance([[0, 0, 0, 0, 0]], 5, {
    localConnectivity: 0,
  });

  expect(rhos).toStrictEqual([0]);
  expect(Number.isFinite(sigmas[0])).toBe(true);
});

test(
  'transform projects a point whose neighbors are all duplicates',
  () => {
    // Regression: a NaN rho made every membership strength of the row NaN, the
    // l1 normalization divided by NaN, and the initialized position — hence the
    // whole returned embedding — came out NaN, with no error raised.
    const values = [
      [1, 1, 1],
      [2, 2, 2],
      [3, 3, 3],
      [4, 4, 4],
    ];
    const duplicates: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const value = values[i % values.length];
      if (value === undefined) throw new Error(`missing value ${i}`);
      duplicates.push(value.slice());
    }

    const umap = new UMAP({
      random: makeRandom(),
      numberOfComponents: 2,
      numberOfEpochs: 30,
    });
    const embedding = umap.fit(duplicates);

    const projected = umap.transform(values);

    expect(projected).toHaveLength(values.length);

    for (let i = 0; i < projected.length; i++) {
      const point = projected[i];
      if (point === undefined) throw new Error(`missing point ${i}`);

      expect(point).toHaveLength(2);
      expect(Number.isFinite(point[0])).toBe(true);
      expect(Number.isFinite(point[1])).toBe(true);

      const nearest = nearestNeighborIndex(embedding, point);

      expect(duplicates[nearest]).toStrictEqual(values[i]);
    }
  },
  FIT_TIMEOUT,
);
