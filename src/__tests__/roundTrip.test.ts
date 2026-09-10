/*
 * ml-umap: the save / reload / project round trip, end to end through a real
 * JSON string. Every assertion here is exact: a model that comes back from
 * `fromJSON` has to project exactly as the model it was saved from, and
 * projecting the training data has to give the fitted embedding back.
 */

import { expect, test } from 'vitest';

import type { UMAPModel, Vectors } from '../index.ts';
import { UMAP } from '../index.ts';
import { at } from '../nn/checkedAt.ts';

import { blobs } from './blobs.ts';
import { makeRandom } from './random.ts';

// A fit of 150 points over the default 500 epochs exceeds the 5s vitest
// default; 120 epochs is enough to converge on well separated blobs.
const FIT_TIMEOUT = 120_000;
const NUMBER_OF_EPOCHS = 120;

const trainingData = blobs(150, 8, 3, 11).X;
const newPoints = blobs(10, 8, 3, 99).X;

test(
  'the reloaded embedding is the fitted one',
  () => {
    const { fitted, reloaded } = round();

    expect(reloaded.getEmbedding()).toStrictEqual(fitted);
  },
  FIT_TIMEOUT,
);

test(
  'the reloaded model projects the training data onto the fitted embedding',
  () => {
    const { fitted, reloaded } = round();

    const projected = reloaded.transform(trainingData);

    expect(projected).toStrictEqual(fitted);

    // A copy, never the array the model holds.
    const firstRow = at(projected, 0);
    firstRow[0] = Number.NaN;

    expect(reloaded.getEmbedding()).toStrictEqual(fitted);
  },
  FIT_TIMEOUT,
);

test(
  'a freshly fitted model projects its training data onto its embedding',
  () => {
    const { umap, fitted } = round();

    expect(umap.transform(trainingData)).toStrictEqual(fitted);
  },
  FIT_TIMEOUT,
);

test(
  'the reloaded model projects new points exactly as the original does',
  () => {
    const { umap, reloaded } = round();

    expect(reloaded.transform(newPoints)).toStrictEqual(
      umap.transform(newPoints),
    );
  },
  FIT_TIMEOUT,
);

test(
  'two independent loads project new points identically',
  () => {
    const { model } = round();

    const first = UMAP.fromJSON(clone(model));
    const second = UMAP.fromJSON(clone(model));

    expect(first.transform(newPoints)).toStrictEqual(
      second.transform(newPoints),
    );
  },
  FIT_TIMEOUT,
);

test(
  'projecting twice through one model returns the same positions',
  () => {
    const { umap } = round();

    expect(umap.transform(newPoints)).toStrictEqual(umap.transform(newPoints));
  },
  FIT_TIMEOUT,
);

test(
  'one changed training value stops the short circuit',
  () => {
    const { fitted, reloaded } = round();

    const nearMiss = clone(trainingData);
    const row = at(nearMiss, 7);
    row[0] = at(row, 0) + 1e-9;

    const projected = reloaded.transform(nearMiss);

    expect(projected).toHaveLength(fitted.length);
    expect(projected).not.toStrictEqual(fitted);
    // Re-projecting the training data lands far from the embedding it came from,
    // which is the whole reason the short circuit exists.
    expect(maxAbsoluteDifference(projected, fitted)).toBeGreaterThan(1);
  },
  FIT_TIMEOUT,
);

test(
  'a shorter batch of training rows is not mistaken for the training data',
  () => {
    const { fitted, reloaded } = round();

    const head = clone(trainingData).slice(0, 20);

    const projected = reloaded.transform(head);

    expect(projected).toHaveLength(20);
    expect(projected).not.toStrictEqual(fitted.slice(0, 20));
  },
  FIT_TIMEOUT,
);

test(
  'projecting new points does not draw from the fit stream',
  () => {
    let draws = 0;
    const stream = makeRandom(3);
    const umap = new UMAP({
      numberOfEpochs: NUMBER_OF_EPOCHS,
      random: () => {
        draws++;
        return stream();
      },
    });
    umap.fit(trainingData);
    const afterFit = draws;

    umap.transform(newPoints);

    expect(draws).toBe(afterFit);
  },
  FIT_TIMEOUT,
);

test(
  'an explicit transformSeed selects the projection, and survives a reload',
  () => {
    const { model } = round();

    const withSeed = UMAP.fromJSON(clone(model), { transformSeed: 7 });
    const projected = withSeed.transform(newPoints);

    expect(projected).toStrictEqual(
      UMAP.fromJSON(clone(model)).transform(newPoints, { transformSeed: 7 }),
    );
    expect(projected).not.toStrictEqual(
      UMAP.fromJSON(clone(model)).transform(newPoints),
    );

    const resaved = withSeed.toJSON();

    expect(resaved.transformSeed).toBe(7);
    expect(UMAP.fromJSON(clone(resaved)).transform(newPoints)).toStrictEqual(
      projected,
    );
  },
  FIT_TIMEOUT,
);

interface Round {
  /** The model that was fitted. */
  umap: UMAP;
  /** Its embedding, as `fit` returned it. */
  fitted: Vectors;
  /** That model, through `toJSON` and a real JSON string. */
  model: UMAPModel;
  /** A model rebuilt from `model`. */
  reloaded: UMAP;
}

let cached: Round | undefined;

/**
 * Fits one model and takes it through `toJSON`, `JSON.stringify`, `JSON.parse`
 * and `fromJSON`, which is the scenario every test here asserts on.
 *
 * The fit is shared between the tests, so they all read one model rather than
 * paying for a fit each; nothing below mutates it.
 * @returns The fitted model, its embedding, its serialized form and a reload.
 */
function round(): Round {
  if (cached === undefined) {
    const umap = new UMAP({
      random: makeRandom(1),
      numberOfEpochs: NUMBER_OF_EPOCHS,
    });
    const fitted = umap.fit(trainingData);
    const model = clone(umap.toJSON());
    cached = { umap, fitted, model, reloaded: UMAP.fromJSON(clone(model)) };
  }
  return cached;
}

/**
 * Sends a value through a real JSON string, which is what a file or an HTTP
 * body carries, rather than through a structured clone.
 * @param value - The value to serialize and parse back.
 * @returns The parsed copy.
 */
function clone<T>(value: T): T {
  const text = JSON.stringify(value);
  return JSON.parse(text) as T;
}

/**
 * Largest absolute difference between two embeddings of the same shape.
 * @param a - First embedding.
 * @param b - Second embedding.
 * @returns The largest per-coordinate distance.
 */
function maxAbsoluteDifference(a: Vectors, b: Vectors): number {
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    const left = a[i];
    const right = b[i];
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
