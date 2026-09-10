import { expect, test } from 'vitest';

import { UMAP } from '../index.ts';
import { findABParams } from '../optimize/abParams.ts';
import type { UMAPModel } from '../serialize.ts';
import type { Vector } from '../types.ts';

import {
  additionalData,
  additionalLabels,
  testData,
  testLabels,
} from './data/upstream-fixtures.ts';
import { nearestNeighborIndex } from './embeddings.ts';
import { makeRandom } from './random.ts';

// A full fit is 500 epochs over 100 points and exceeds the 5s vitest default.
const FIT_TIMEOUT = 120_000;

// Enough epochs to exercise the optimizer, few enough to keep the suite quick,
// for the tests that assert on parameters rather than on embedding values.
const SHORT_FIT = 20;

test(
  'toJSON describes the fitted model and fromJSON restores it exactly',
  () => {
    const umap = new UMAP({ random: makeRandom(), numberOfComponents: 2 });
    umap.fit(testData);
    const model = umap.toJSON();

    expect(model.name).toBe('UMAP');
    expect(model.version).toBe(1);
    expect(model.metric).toBe('euclidean');
    expect(model.numberOfComponents).toBe(2);
    expect(model.numberOfNeighbors).toBe(15);
    expect(model.minimumDistance).toBe(0.1);
    expect(model.spread).toBe(1);
    expect(model.localConnectivity).toBe(1);
    expect(model.setOperationMixRatio).toBe(1);
    expect(model.negativeSampleRate).toBe(5);
    expect(model.repulsionStrength).toBe(1);
    expect(model.learningRate).toBe(1);
    expect(model.transformQueueSize).toBe(4);
    expect(model.numberOfEpochs).toBe(0);
    expect(model.X).toStrictEqual(testData);
    expect(model.embedding).toStrictEqual(umap.getEmbedding());
    expect(model.knnIndices).toHaveLength(testData.length);
    expect(model.knnIndices[0]).toHaveLength(15);

    const reloaded = UMAP.fromJSON(model, { random: makeRandom() });

    expect(reloaded.getEmbedding()).toStrictEqual(umap.getEmbedding());
    expect(reloaded.toJSON()).toStrictEqual(model);
  },
  FIT_TIMEOUT,
);

test(
  'the model survives a JSON string round trip',
  () => {
    const umap = new UMAP({ random: makeRandom(), numberOfEpochs: SHORT_FIT });
    umap.fit(testData);
    const model = umap.toJSON();

    // Through a real JSON string, not a structured clone: what has to survive
    // is what a file or a HTTP body can carry.
    const serialized = JSON.stringify(model);
    const parsed = JSON.parse(serialized) as UMAPModel;
    const reloaded = UMAP.fromJSON(parsed, { random: makeRandom() });

    expect(reloaded.toJSON()).toStrictEqual(model);
  },
  FIT_TIMEOUT,
);

test(
  'the curve parameters a and b are stored, not recomputed from defaults',
  () => {
    const umap = new UMAP({ random: makeRandom(), numberOfEpochs: SHORT_FIT });
    umap.fit(testData);
    const model = umap.toJSON();

    const defaults = findABParams(1, 0.1);

    expect(model.a).toBe(defaults.a);
    expect(model.b).toBe(defaults.b);

    const reloaded = UMAP.fromJSON(model, { random: makeRandom() });

    expect(reloaded.toJSON().a).toBe(model.a);
    expect(reloaded.toJSON().b).toBe(model.b);
  },
  FIT_TIMEOUT,
);

test(
  'a non-default spread and minimumDistance do not fall back to the default curve',
  () => {
    const umap = new UMAP({
      random: makeRandom(),
      numberOfEpochs: SHORT_FIT,
      spread: 2,
      minimumDistance: 0.5,
    });
    umap.fit(testData);
    const model = umap.toJSON();

    const defaults = findABParams(1, 0.1);
    const expected = findABParams(2, 0.5);

    expect(model.spread).toBe(2);
    expect(model.minimumDistance).toBe(0.5);
    expect(model.a).toBe(expected.a);
    expect(model.b).toBe(expected.b);
    expect(model.a).not.toBe(defaults.a);
    expect(model.b).not.toBe(defaults.b);

    const reloaded = UMAP.fromJSON(model, { random: makeRandom() });
    const reloadedModel = reloaded.toJSON();

    expect(reloadedModel.a).toBe(expected.a);
    expect(reloadedModel.b).toBe(expected.b);
    expect(reloadedModel.spread).toBe(2);
    expect(reloadedModel.minimumDistance).toBe(0.5);
  },
  FIT_TIMEOUT,
);

test(
  'a reloaded model projects new points next to their own class',
  () => {
    const umap = new UMAP({ random: makeRandom(), numberOfComponents: 2 });
    const embedding = umap.fit(testData);
    const reloaded = UMAP.fromJSON(umap.toJSON(), { random: makeRandom() });

    const transformed = reloaded.transform(additionalData);

    expect(transformed).toHaveLength(additionalData.length);

    for (let i = 0; i < transformed.length; i++) {
      const point = transformed[i];
      if (point === undefined) throw new Error(`missing point ${i}`);

      expect(point).toHaveLength(2);

      for (const value of point) {
        expect(Number.isFinite(value)).toBe(true);
      }

      expect(testLabels[nearestNeighborIndex(embedding, point)]).toBe(
        additionalLabels[i],
      );
    }
  },
  FIT_TIMEOUT,
);

test(
  'fromJSON refuses a custom metric with no distance function',
  () => {
    const umap = new UMAP({
      random: makeRandom(),
      numberOfEpochs: SHORT_FIT,
      distanceFunction: manhattan,
    });
    umap.fit(testData);
    const model = umap.toJSON();

    expect(model.metric).toBe('custom');
    expect(() => UMAP.fromJSON(model)).toThrow(/fitted with a custom metric/);

    const reloaded = UMAP.fromJSON(model, {
      random: makeRandom(),
      distanceFunction: manhattan,
    });

    expect(reloaded.toJSON()).toStrictEqual(model);
  },
  FIT_TIMEOUT,
);

test(
  'fromJSON throws a RangeError on a foreign model',
  () => {
    const umap = new UMAP({ random: makeRandom(), numberOfEpochs: SHORT_FIT });
    umap.fit(testData);
    const model = umap.toJSON();

    const foreign = { ...model, name: 'PCA' } as unknown as UMAPModel;

    expect(() => UMAP.fromJSON(foreign)).toThrow(RangeError);
  },
  FIT_TIMEOUT,
);

/**
 * Manhattan distance, used as a metric the registry cannot name.
 * @param x - First vector.
 * @param y - Second vector of the same length.
 * @returns The sum of the absolute per-component differences.
 */
function manhattan(x: Vector, y: Vector): number {
  let distance = 0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = y[i];
    if (xi === undefined || yi === undefined) {
      throw new RangeError('manhattan: vectors must have the same length');
    }
    distance += Math.abs(xi - yi);
  }
  return distance;
}
