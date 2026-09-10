/*
 * ml-umap: validation of the data and of the neighborhood size. Upstream
 * umap-js validates neither, so a single NaN or Infinity anywhere in the data
 * produced a complete, entirely finite and plausible looking embedding with no
 * error and no warning.
 */

import type { Vectors } from './types.ts';

/**
 * Smallest neighborhood a projection can be built on.
 */
export const MINIMUM_NUMBER_OF_NEIGHBORS = 2;

/**
 * Validates the data a model is about to be fitted on, and resolves the
 * neighborhood size the fit can actually use.
 *
 * `numberOfNeighbors` is clamped down to `X.length - 1` when it exceeds the
 * number of points, the way the reference python implementation warns and
 * clamps instead of failing; the returned value is the one the caller has to go
 * on with. A `numberOfNeighbors` equal to the number of points is left alone
 * and rejected by the caller, which is the behaviour this port has always had.
 * @param X - The data to project, one array per point.
 * @param numberOfNeighbors - Requested size of the local neighborhood.
 * @returns The neighborhood size to actually use, clamped to `X.length - 1`.
 * @throws {TypeError} If `X` is empty, or holds rows of unequal length, or
 * holds a value that is not a number.
 * @throws {RangeError} If a value of `X` is not finite, if `numberOfNeighbors`
 * is below two, or if the data is too small for a neighborhood of two points.
 */
export function validateFitInput(
  X: Vectors,
  numberOfNeighbors: number,
): number {
  validateVectors(X, 'X');

  if (
    Number.isNaN(numberOfNeighbors) ||
    numberOfNeighbors < MINIMUM_NUMBER_OF_NEIGHBORS
  ) {
    throw new RangeError(
      `numberOfNeighbors must be at least ${MINIMUM_NUMBER_OF_NEIGHBORS}, but it is ${numberOfNeighbors}.`,
    );
  }

  if (numberOfNeighbors <= X.length) {
    return numberOfNeighbors;
  }

  const clamped = X.length - 1;
  if (clamped < MINIMUM_NUMBER_OF_NEIGHBORS) {
    throw new RangeError(
      `numberOfNeighbors must be at least ${MINIMUM_NUMBER_OF_NEIGHBORS}, but a dataset of ${X.length} allows at most ${clamped}.`,
    );
  }
  return clamped;
}

/**
 * Validates the points a fitted model is about to project.
 * @param toTransform - The points to project, one array per point.
 * @param nFittedColumns - Number of columns of the data the model was fitted on.
 * @throws {TypeError} If `toTransform` is empty, or holds rows of unequal
 * length, or holds a value that is not a number, or does not have the
 * dimensionality of the fitted data.
 * @throws {RangeError} If a value of `toTransform` is not finite.
 */
export function validateTransformInput(
  toTransform: Vectors,
  nFittedColumns: number,
): void {
  const nColumns = validateVectors(toTransform, 'toTransform');

  if (nColumns !== nFittedColumns) {
    throw new TypeError(
      `toTransform: row 0 has length ${nColumns}, but the model was fitted on ${nFittedColumns} columns.`,
    );
  }
}

/**
 * Validates that data is a non-empty collection of equal length rows of finite
 * numbers.
 * @param vectors - The rows to validate.
 * @param name - Name of the parameter, used to build the error messages.
 * @returns The number of columns every row has.
 * @throws {TypeError} If `vectors` is empty, or holds rows of unequal length,
 * or holds a value that is not a number.
 * @throws {RangeError} If a value is `NaN` or infinite.
 */
export function validateVectors(vectors: Vectors, name: string): number {
  const firstRow = vectors[0];
  if (firstRow === undefined) {
    throw new TypeError(
      `${name} must be a non-empty array of rows, but it holds no row.`,
    );
  }

  const nColumns = firstRow.length;
  if (nColumns === 0) {
    throw new TypeError(
      `${name}: row 0 has length 0, every row must have at least one column.`,
    );
  }

  for (let row = 0; row < vectors.length; row++) {
    const values = vectors[row];
    if (values === undefined) {
      throw new TypeError(`${name}: row ${row} is missing.`);
    }
    if (values.length !== nColumns) {
      throw new TypeError(
        `${name}: row ${row} has length ${values.length}, but row 0 has length ${nColumns}; every row must have the same length.`,
      );
    }

    for (let column = 0; column < nColumns; column++) {
      const value = values[column];
      if (typeof value !== 'number') {
        throw new TypeError(
          `${name}: the value at [${row}, ${column}] is not a number, it is ${describeValue(value)}.`,
        );
      }
      if (!Number.isFinite(value)) {
        throw new RangeError(
          `${name}: the value at [${row}, ${column}] is ${value}, every value must be finite.`,
        );
      }
    }
  }

  return nColumns;
}

/**
 * Names a value that should have been a number, for an error message.
 * @param value - The offending value.
 * @returns A short description of it, never throwing.
 */
function describeValue(value: unknown): string {
  if (typeof value === 'string') {
    return `the string "${value}"`;
  }
  if (value === null) {
    return 'null';
  }
  if (value === undefined) {
    return 'undefined';
  }
  return `a ${typeof value}`;
}
