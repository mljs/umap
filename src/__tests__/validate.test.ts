/*
 * ml-umap: regression tests for the input validation upstream umap-js does not
 * perform. Every message is asserted in full, because naming the offending row,
 * column and value is the whole point of the check.
 */

import { beforeAll, expect, test } from 'vitest';

import { UMAP } from '../index.ts';
import type { Vectors } from '../types.ts';
import {
  validateFitInput,
  validateTransformInput,
  validateVectors,
} from '../validate.ts';

import { makeRandom } from './random.ts';

const SQUARE: Vectors = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
  [2, 2],
  [3, 1],
];

const RAGGED: Vectors = [[1, 2, 3], [4, 5, 6], [7, 8], [9]];
const SECOND_ROW_NAN: Vectors = [
  [1, 2, 3],
  [4, 5, Number.NaN],
];
const SECOND_ROW_INFINITE: Vectors = [
  [1, 2, 3],
  [Number.NEGATIVE_INFINITY, 2, 3],
];

const FITTED = new UMAP({ random: makeRandom(), numberOfEpochs: 20 });

beforeAll(() => {
  FITTED.fit(SQUARE);
});

/** One call that must throw, with the exact error it must throw. */
interface ThrowingCase {
  /** Name of the test. */
  name: string;
  /** The call under test. */
  run: () => unknown;
  /** The class of the expected error. */
  errorClass: ErrorConstructor;
  /** The exact message of the expected error. */
  message: string;
}

const THROWING_CASES: ThrowingCase[] = [
  {
    name: 'validateVectors rejects data holding no row',
    run: () => validateVectors([], 'X'),
    errorClass: TypeError,
    message: 'X must be a non-empty array of rows, but it holds no row.',
  },
  {
    name: 'validateVectors rejects a row holding no column',
    run: () => validateVectors([[]], 'X'),
    errorClass: TypeError,
    message: 'X: row 0 has length 0, every row must have at least one column.',
  },
  {
    name: 'validateVectors names the first row whose length differs',
    run: () => validateVectors(RAGGED, 'X'),
    errorClass: TypeError,
    message:
      'X: row 2 has length 2, but row 0 has length 3; every row must have the same length.',
  },
  {
    name: 'validateVectors names a missing row',
    run: () => validateVectors([[1, 2], undefined, [3, 4]] as Vectors, 'X'),
    errorClass: TypeError,
    message: 'X: row 1 is missing.',
  },
  {
    name: 'validateVectors names the first NaN',
    run: () => validateVectors(SECOND_ROW_NAN, 'X'),
    errorClass: RangeError,
    message: 'X: the value at [1, 2] is NaN, every value must be finite.',
  },
  {
    name: 'validateVectors names the first positive infinity',
    run: () => validateVectors([[1, Number.POSITIVE_INFINITY, 3]], 'X'),
    errorClass: RangeError,
    message: 'X: the value at [0, 1] is Infinity, every value must be finite.',
  },
  {
    name: 'validateVectors names the first negative infinity',
    run: () => validateVectors(SECOND_ROW_INFINITE, 'X'),
    errorClass: RangeError,
    message: 'X: the value at [1, 0] is -Infinity, every value must be finite.',
  },
  {
    name: 'validateVectors names a string where a number was expected',
    run: () => validateVectors([[1, 'a', 3]] as unknown as Vectors, 'X'),
    errorClass: TypeError,
    message: 'X: the value at [0, 1] is not a number, it is the string "a".',
  },
  {
    name: 'validateVectors names a null where a number was expected',
    run: () => validateVectors([[1, null, 3]] as unknown as Vectors, 'X'),
    errorClass: TypeError,
    message: 'X: the value at [0, 1] is not a number, it is null.',
  },
  {
    name: 'validateVectors names an undefined where a number was expected',
    run: () => validateVectors([[1, undefined, 3]] as unknown as Vectors, 'X'),
    errorClass: TypeError,
    message: 'X: the value at [0, 1] is not a number, it is undefined.',
  },
  {
    name: 'validateVectors names a boolean where a number was expected',
    run: () => validateVectors([[1, true, 3]] as unknown as Vectors, 'X'),
    errorClass: TypeError,
    message: 'X: the value at [0, 1] is not a number, it is a boolean.',
  },
  {
    name: 'validateFitInput rejects a numberOfNeighbors below two',
    run: () => validateFitInput(SQUARE, 1),
    errorClass: RangeError,
    message: 'numberOfNeighbors must be at least 2, but it is 1.',
  },
  {
    name: 'validateFitInput rejects data too small for two neighbors',
    run: () => validateFitInput([[1], [2]], 15),
    errorClass: RangeError,
    message:
      'numberOfNeighbors must be at least 2, but a dataset of 2 allows at most 1.',
  },
  {
    name: 'validateTransformInput rejects an unfitted dimensionality',
    run: () => validateTransformInput([[1, 2]], 3),
    errorClass: TypeError,
    message:
      'toTransform: row 0 has length 2, but the model was fitted on 3 columns.',
  },
  {
    name: 'fit rejects a single non-finite value instead of embedding it',
    run: () => new UMAP({ random: makeRandom() }).fit(SECOND_ROW_NAN),
    errorClass: RangeError,
    message: 'X: the value at [1, 2] is NaN, every value must be finite.',
  },
  {
    name: 'fit rejects rows of unequal length',
    run: () => new UMAP({ random: makeRandom() }).fit(RAGGED),
    errorClass: TypeError,
    message:
      'X: row 2 has length 2, but row 0 has length 3; every row must have the same length.',
  },
  {
    name: 'fit rejects empty data',
    run: () => new UMAP({ random: makeRandom() }).fit([]),
    errorClass: TypeError,
    message: 'X must be a non-empty array of rows, but it holds no row.',
  },
  {
    name: 'transform rejects a dimensionality the model was not fitted on',
    run: () => FITTED.transform([[1, 2, 3]]),
    errorClass: TypeError,
    message:
      'toTransform: row 0 has length 3, but the model was fitted on 2 columns.',
  },
  {
    name: 'transform rejects a non-finite value',
    run: () => FITTED.transform([[1, Number.POSITIVE_INFINITY]]),
    errorClass: RangeError,
    message:
      'toTransform: the value at [0, 1] is Infinity, every value must be finite.',
  },
];

test.each(THROWING_CASES)('$name', ({ run, errorClass, message }) => {
  let thrown: unknown;
  try {
    run();
  } catch (error) {
    thrown = error;
  }

  expect(thrown).toBeInstanceOf(errorClass);
  expect(thrown).toHaveProperty('message', message);
});

test('validateVectors returns the number of columns of well formed data', () => {
  expect(validateVectors(SQUARE, 'X')).toBe(2);
});

test('validateFitInput keeps a numberOfNeighbors the data can support', () => {
  expect(validateFitInput(SQUARE, 3)).toBe(3);
  expect(validateFitInput(SQUARE, SQUARE.length)).toBe(SQUARE.length);
});

test('validateFitInput clamps a numberOfNeighbors larger than the data', () => {
  expect(validateFitInput(SQUARE, 15)).toBe(SQUARE.length - 1);
});

test('fit clamps numberOfNeighbors to the data and records the clamped value', () => {
  expect(FITTED.internals.params.numberOfNeighbors).toBe(SQUARE.length - 1);
  expect(FITTED.getEmbedding()).toHaveLength(SQUARE.length);
  expect(FITTED.toJSON().numberOfNeighbors).toBe(SQUARE.length - 1);
});
