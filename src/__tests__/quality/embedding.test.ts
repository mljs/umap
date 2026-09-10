import { expect, test } from 'vitest';

import { UMAP } from '../../index.ts';
import { blobs } from '../blobs.ts';
import { testData } from '../data/upstream-fixtures.ts';
import { makeRandom } from '../random.ts';

import {
  knnPreservation,
  minimum,
  procrustesDisparity,
  qualitySeries,
  spread,
  trustworthiness,
} from './index.ts';

/*
 * HOW THE FLOORS BELOW WERE CHOSEN.
 *
 * Every configuration was fitted from the sixteen seeds 1..16 and measured at
 * k = 10; the tests then run the first eight of them. Observed there:
 *
 *   configuration            knnPreservation  trustworthiness  continuity
 *   digits, 2 components     0.783 - 0.807    0.950 - 0.967    0.941 - 0.955
 *   digits, 3 components     0.782 - 0.806    0.949 - 0.966    0.943 - 0.954
 *   blobs, random init       0.578 - 0.618    0.963 - 0.971    0.966 - 0.970
 *   blobs, spectral init     0.582 - 0.615    0.965 - 0.969    0.967 - 0.972
 *   digits, initialized only 0.081 - 0.118    0.492 - 0.551    0.502 - 0.535
 *
 * The floors sit 0.04 to 0.10 below the smallest value observed in each
 * column, which is between two and ten times the whole spread the seed
 * accounts for, so a failure here is a real regression rather than one
 * unlucky seed.
 *
 * The one thing measured and deliberately NOT asserted is how much two seeds
 * agree geometrically: the Procrustes disparity between two embeddings of the
 * same data from two seeds ranged over 0.03 - 0.91 across these
 * configurations, a spread far too wide to put a bound on. What is stable
 * across seeds is the QUALITY of the fit, which is what the spread test below
 * asserts.
 *
 * The whole file fits 34 embeddings of 200 epochs, which runs in 2.8 seconds
 * here; the four series are each fitted once and shared by the tests that
 * assert on them.
 */

// knnPreservation of a fit that carries no information at all is around 0.1,
// so a floor under ~0.2 would assert nothing.
const DIGITS_KNN_FLOOR = 0.7;
const DIGITS_TRUST_FLOOR = 0.9;
const DIGITS_CONTINUITY_FLOOR = 0.88;

/*
 * The blob floors are much lower than the digit ones, and that is expected
 * rather than a weakness: each blob is 30 points drawn from an isotropic
 * 10-dimensional Gaussian, where the distances between the points of one blob
 * concentrate, so which ten of the other twenty-nine are the nearest is close
 * to arbitrary. What a projection can preserve there is the blob structure,
 * which trustworthiness and continuity do see.
 */
const BLOBS_KNN_FLOOR = 0.48;
const BLOBS_TRUST_FLOOR = 0.92;
const BLOBS_CONTINUITY_FLOOR = 0.92;

const FIRST_SEED = 1;
const SEEDS = [FIRST_SEED, 2, 3, 4, 5, 6, 7, 8];
const K = 10;
const EPOCHS = 200;
const DIGITS_PARAMS = { numberOfNeighbors: 15, numberOfEpochs: EPOCHS };
const BLOBS_PARAMS = { numberOfNeighbors: 10, numberOfEpochs: EPOCHS };

// The labelled 100 x 64 digit sample of the upstream test suite, and four well
// separated Gaussian blobs, whose fuzzy graph is disconnected.
const DIGITS = testData;
const BLOB_DATA = blobs(120, 10, 4).X;

// Eight fits of 200 epochs take about half a second, but a loaded CI runner is
// allowed to be much slower than that.
const FIT_TIMEOUT = 60_000;

test(
  'a two dimensional fit of the digits keeps their neighborhoods',
  () => {
    const series = digits2D();

    expect(minimum(series.knn)).toBeGreaterThan(DIGITS_KNN_FLOOR);
    expect(minimum(series.trust)).toBeGreaterThan(DIGITS_TRUST_FLOOR);
    expect(minimum(series.continuity)).toBeGreaterThan(DIGITS_CONTINUITY_FLOOR);
  },
  FIT_TIMEOUT,
);

test(
  'a three dimensional fit keeps them just as well',
  () => {
    const series = qualitySeries('digits-3d', {
      X: DIGITS,
      seeds: SEEDS,
      k: K,
      params: { ...DIGITS_PARAMS, numberOfComponents: 3 },
    });

    expect(minimum(series.knn)).toBeGreaterThan(DIGITS_KNN_FLOOR);
    expect(minimum(series.trust)).toBeGreaterThan(DIGITS_TRUST_FLOOR);
    expect(minimum(series.continuity)).toBeGreaterThan(DIGITS_CONTINUITY_FLOOR);
  },
  FIT_TIMEOUT,
);

test(
  'both initializations project separated blobs equally well',
  () => {
    const random = qualitySeries('blobs-random', {
      X: BLOB_DATA,
      seeds: SEEDS,
      k: K,
      params: BLOBS_PARAMS,
    });
    const spectral = qualitySeries('blobs-spectral', {
      X: BLOB_DATA,
      seeds: SEEDS,
      k: K,
      params: { ...BLOBS_PARAMS, init: 'spectral' },
    });

    for (const series of [random, spectral]) {
      expect(minimum(series.knn)).toBeGreaterThan(BLOBS_KNN_FLOOR);
      expect(minimum(series.trust)).toBeGreaterThan(BLOBS_TRUST_FLOOR);
      expect(minimum(series.continuity)).toBeGreaterThan(
        BLOBS_CONTINUITY_FLOOR,
      );
    }
  },
  FIT_TIMEOUT,
);

test(
  'the optimization improves hugely on the initialization it starts from',
  () => {
    // Same seed and same parameters as the first fit of the digit series, so
    // this is exactly the embedding that fit started the optimization from.
    const umap = new UMAP({ ...DIGITS_PARAMS, random: makeRandom(FIRST_SEED) });
    umap.initializeFit(DIGITS);
    const initial = umap.getEmbedding();
    const initialKnn = knnPreservation(DIGITS, initial, K);
    const fitted = digits2D();

    expect(initialKnn).toBeLessThan(0.2);
    expect(trustworthiness(DIGITS, initial, K)).toBeLessThan(0.7);
    expect(fitted.knn[0] ?? 0).toBeGreaterThan(4 * initialKnn);
  },
  FIT_TIMEOUT,
);

test(
  'the quality of a fit barely moves from one seed to the next',
  () => {
    const series = digits2D();

    // Observed over the sixteen calibration seeds: 0.024 and 0.016.
    expect(spread(series.knn)).toBeLessThan(0.08);
    expect(spread(series.trust)).toBeLessThan(0.06);
  },
  FIT_TIMEOUT,
);

test(
  'two fits of the same data from the same seed are the same embedding',
  () => {
    const first = new UMAP({ ...DIGITS_PARAMS, random: makeRandom(99) });
    const second = new UMAP({ ...DIGITS_PARAMS, random: makeRandom(99) });
    const embedding = first.fit(DIGITS);
    const repeat = second.fit(DIGITS);

    // The two embeddings are bit identical, so what is left here is the
    // rounding of the decomposition the disparity is computed through.
    expect(repeat).toStrictEqual(embedding);
    expect(procrustesDisparity(embedding, repeat)).toBeCloseTo(0, 12);
  },
  FIT_TIMEOUT,
);

/**
 * The eight two dimensional fits of the digit sample, fitted once and shared
 * by every test that asserts on them.
 * @returns The embeddings and their measures.
 */
function digits2D() {
  return qualitySeries('digits-2d', {
    X: DIGITS,
    seeds: SEEDS,
    k: K,
    params: { ...DIGITS_PARAMS, numberOfComponents: 2 },
  });
}
