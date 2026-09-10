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
  const { a, b } = findABParams(1, 0.1);

  // Compared to 12 decimals rather than exactly. The Levenberg-Marquardt fit
  // runs through Math.pow and Math.exp, which ECMAScript leaves
  // implementation-approximated, and node 22 lands one ULP below node 24 and 26
  // on `b` (0.897814082889175 against 0.8978140828891751, about 1.1e-16).
  // The shift this pin exists to catch is six orders of magnitude larger: LM 2.x
  // against 5.x moves `a` by 1.3e-3 and `b` by 3.6e-3, so a dependency bump
  // still fails here loudly.
  expect(a).toBeCloseTo(1.5681358231455382, 12);
  expect(b).toBeCloseTo(0.8978140828891751, 12);
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
