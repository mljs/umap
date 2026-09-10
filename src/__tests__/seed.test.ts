/*
 * ml-umap: the fit is seeded by default, where umap-js draws from
 * `Math.random` and no run can be repeated.
 */

import { expect, test } from 'vitest';

import { UMAP } from '../index.ts';

import { blobs } from './blobs.ts';
import { makeRandom } from './random.ts';

const { X } = blobs(80, 4, 3);
const PARAMS = { numberOfComponents: 2, numberOfEpochs: 30 };

test('two models built with the defaults fit the same data identically', () => {
  const first = new UMAP(PARAMS).fit(X);
  const second = new UMAP(PARAMS).fit(X);

  expect(second).toStrictEqual(first);
  // The default source is the stream of seed 42, not a shared generator the
  // second model would continue instead of restarting.
  expect(new UMAP({ ...PARAMS, random: makeRandom(42) }).fit(X)).toStrictEqual(
    first,
  );
  expect(new UMAP({ ...PARAMS, seed: 42 }).fit(X)).toStrictEqual(first);
});

test('another seed gives another, equally repeatable, embedding', () => {
  const byDefault = new UMAP(PARAMS).fit(X);
  const seeded = new UMAP({ ...PARAMS, seed: 7 }).fit(X);

  expect(seeded).not.toStrictEqual(byDefault);
  expect(new UMAP({ ...PARAMS, seed: 7 }).fit(X)).toStrictEqual(seeded);
});

test('an unseeded random source is what makes two fits differ', () => {
  const first = new UMAP({ ...PARAMS, random: Math.random }).fit(X);
  const second = new UMAP({ ...PARAMS, random: Math.random }).fit(X);

  expect(second).not.toStrictEqual(first);
});

test('a supplied random source wins over the seed', () => {
  const seeded = new UMAP({ ...PARAMS, seed: 7 }).fit(X);
  const supplied = new UMAP({
    ...PARAMS,
    seed: 7,
    random: makeRandom(42),
  }).fit(X);

  expect(supplied).not.toStrictEqual(seeded);
  expect(supplied).toStrictEqual(new UMAP(PARAMS).fit(X));
});

test('a seed that is not an integer is rejected', () => {
  expect(() => new UMAP({ seed: 1.5 })).toThrow(
    new TypeError('seed must be an integer, but it is 1.5.'),
  );
});
