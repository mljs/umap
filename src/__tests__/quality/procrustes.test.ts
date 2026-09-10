import { expect, test } from 'vitest';

import { procrustesDisparity } from './procrustes.ts';

// A cross in the plane and the same cross in space: both are centered on the
// origin already, which makes the disparities below computable by hand.
const CROSS_2D = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];
const CROSS_3D = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];

// A configuration with no symmetry left to hide a wrong rotation behind.
const RAGGED = [
  [0.4, -1.3],
  [2.1, 0.6],
  [-1.7, 0.2],
  [0.9, 2.8],
  [-2.4, -1.1],
  [1.5, -2.2],
];

test('a configuration is at no disparity from itself', () => {
  expect(procrustesDisparity(CROSS_2D, CROSS_2D)).toBe(0);
  expect(procrustesDisparity(CROSS_3D, CROSS_3D)).toBe(0);
  expect(procrustesDisparity(RAGGED, RAGGED)).toBeCloseTo(0, 12);
});

test('translating, scaling and rotating a configuration changes nothing', () => {
  const moved = similarity2D(RAGGED, 0.723, 4.5, 12, -30);

  expect(procrustesDisparity(RAGGED, moved)).toBeCloseTo(0, 12);
  expect(procrustesDisparity(moved, RAGGED)).toBeCloseTo(0, 12);
});

test('the same holds in three dimensions', () => {
  // A cyclic permutation of the axes is a rotation: its determinant is 1.
  const rotated: number[][] = [];
  for (const point of CROSS_3D) {
    const [x = 0, y = 0, z = 0] = point;
    rotated.push([z * 2.5 - 4, x * 2.5 + 7, y * 2.5 + 0.5]);
  }

  expect(procrustesDisparity(CROSS_3D, rotated)).toBeCloseTo(0, 12);
});

test('a reflection is free, as in scipy', () => {
  const mirrored: number[][] = [];
  for (const point of RAGGED) {
    const [x = 0, y = 0] = point;
    mirrored.push([-x, y]);
  }

  expect(procrustesDisparity(RAGGED, mirrored)).toBeCloseTo(0, 12);
});

test('a configuration stretched along one axis has a known disparity', () => {
  // Doubling one arm of the cross makes the cross-covariance of the two
  // standardized configurations diag(2, 4) / (2 * sqrt(10)), whose singular
  // values sum to 3 / sqrt(10), so the disparity is 1 - 9 / 10.
  const stretched = [
    [1, 0],
    [-1, 0],
    [0, 2],
    [0, -2],
  ];

  expect(procrustesDisparity(CROSS_2D, stretched)).toBeCloseTo(0.1, 12);
  expect(procrustesDisparity(stretched, CROSS_2D)).toBeCloseTo(0.1, 12);
});

test('the disparity of a stretched configuration is right in three dimensions', () => {
  // Same construction one dimension up: the cross-covariance is
  // diag(2, 2, 4) / (6 * sqrt(2)), its singular values sum to 8 / (6 sqrt(2)),
  // and the disparity is 1 - 8 / 9. A rotation formed from only two of the
  // three dimensions would report 1 - 4 / 18 = 0.7778 instead.
  const stretched = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 2],
    [0, 0, -2],
  ];

  expect(procrustesDisparity(CROSS_3D, stretched)).toBeCloseTo(1 / 9, 12);
  expect(procrustesDisparity(stretched, CROSS_3D)).toBeCloseTo(1 / 9, 12);
});

test('two configurations that share no variance are at disparity 1', () => {
  const horizontal = [
    [1, 0],
    [-1, 0],
    [0, 0],
    [0, 0],
  ];
  const vertical = [
    [0, 0],
    [0, 0],
    [0, 1],
    [0, -1],
  ];

  expect(procrustesDisparity(horizontal, vertical)).toBeCloseTo(1, 12);
});

test('two configurations of different shapes cannot be compared', () => {
  expect(() => procrustesDisparity(CROSS_2D, CROSS_2D.slice(1))).toThrow(
    /must describe the same points, got 4 and 3/,
  );
  expect(() => procrustesDisparity(CROSS_2D, CROSS_3D.slice(0, 4))).toThrow(
    /both configurations must have the same dimension/,
  );
  expect(() =>
    procrustesDisparity(
      [
        [1, 2],
        [3, 4, 5],
      ],
      CROSS_2D.slice(0, 2),
    ),
  ).toThrow(/every point of a configuration must have the same dimension/);
});

test('a configuration that carries no shape cannot be compared', () => {
  expect(() => procrustesDisparity([[1, 2]], [[3, 4]])).toThrow(
    /at least two points are required/,
  );
  expect(() => procrustesDisparity([[], []], [[], []])).toThrow(
    /points must have a dimension/,
  );
  expect(() =>
    procrustesDisparity(
      [
        [1, 2],
        [1, 2],
      ],
      CROSS_2D.slice(0, 2),
    ),
  ).toThrow(/collapses to a single point/);
});

/**
 * Rotates, scales and translates a plane configuration.
 * @param points - The configuration, one array per point.
 * @param angle - Rotation angle, in radians.
 * @param scale - Uniform scaling factor.
 * @param shiftX - Translation along the first axis.
 * @param shiftY - Translation along the second axis.
 * @returns The transformed configuration.
 */
function similarity2D(
  points: number[][],
  angle: number,
  scale: number,
  shiftX: number,
  shiftY: number,
): number[][] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const moved: number[][] = [];
  for (const point of points) {
    const [x = 0, y = 0] = point;
    moved.push([
      scale * (x * cos - y * sin) + shiftX,
      scale * (x * sin + y * cos) + shiftY,
    ]);
  }
  return moved;
}
