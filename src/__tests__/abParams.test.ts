import { expect, test } from 'vitest';

import { findABParams } from '../optimize/abParams.ts';

/*
 * These two numbers are the ONLY source of numerical divergence between this port
 * and PAIR-code/umap-js. Verified by differential testing against the published
 * umap-js package on the same engine and the same seeded PRNG: kNN, the RP-forest,
 * the fuzzy simplicial set, the graph (1666 edges), epochsPerSample, the total RNG
 * draw count (1482395) and the SGD are all bit-identical, and forcing a/b to the
 * values below makes a 100-point fit match upstream 200/200 and 300/300 bit-exactly.
 *
 * a and b come from a Levenberg-Marquardt curve fit, so they depend on the version
 * of ml-levenberg-marquardt: upstream bundles 2.0.0, we depend on 5.x. This test
 * pins our values so any dependency bump that shifts them fails loudly rather than
 * silently changing every embedding this package produces.
 */
const UPSTREAM_LM2 = { a: 1.5694704762346365, b: 0.8941996053733949 };

test('findABParams matches the pinned values for spread=1, minimumDistance=0.1', () => {
  expect(findABParams(1, 0.1)).toStrictEqual({
    a: 1.5681358231455382,
    b: 0.8978140828891751,
  });
});

test('findABParams stays close to the values umap-js computes with LM 2.x', () => {
  const { a, b } = findABParams(1, 0.1);

  expect(a).toBeCloseTo(UPSTREAM_LM2.a, 2);
  expect(b).toBeCloseTo(UPSTREAM_LM2.b, 2);
});

test('findABParams responds to spread and minimumDistance', () => {
  const wide = findABParams(2, 0.1);
  const tight = findABParams(1, 0.5);

  expect(wide).not.toStrictEqual(findABParams(1, 0.1));
  expect(tight).not.toStrictEqual(findABParams(1, 0.1));
});
