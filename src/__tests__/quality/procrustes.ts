import { Matrix, SingularValueDecomposition } from 'ml-matrix';

/**
 * Normalized Procrustes disparity between two configurations of the same
 * points: the residual sum of squares left once the second configuration has
 * been optimally translated, uniformly scaled and rotated onto the first.
 *
 * Both configurations are first centered on their centroid and divided by
 * their Frobenius norm, which removes the translation and the scale. The
 * remaining orthogonal map is the solution of the orthogonal Procrustes
 * problem: with `M = A' transpose * B'` the cross-covariance of the two
 * standardized configurations and `s` the sum of the singular values of `M`
 * (its nuclear norm), the optimal map has residual
 *
 * ```text
 * min over orthogonal R, scale c of ||A' - c * B' * R||_F^2 = 1 - s^2
 * ```
 *
 * which is the value returned. It is 0 for two configurations that are exactly
 * similar and 1 for two whose cross-covariance vanishes, and it is symmetric
 * in its two arguments. As in `scipy.spatial.procrustes`, `R` ranges over the
 * whole orthogonal group, so a reflection is free: a mirrored configuration
 * scores 0.
 * @param a - First configuration, one array per point.
 * @param b - Second configuration, the same points in the same order.
 * @throws {RangeError} If the two configurations do not have the same shape,
 * hold fewer than two points, or if either collapses to a single point.
 * @returns The disparity, in `[0, 1]`.
 */
export function procrustesDisparity(a: number[][], b: number[][]): number {
  const dimensions = checkShapes(a, b);
  const first = standardize(a, dimensions);
  const second = standardize(b, dimensions);
  const singularValues = new SingularValueDecomposition(
    first.transpose().mmul(second),
  ).diagonal;

  let nuclearNorm = 0;
  for (const value of singularValues) nuclearNorm += value;

  // The nuclear norm of the cross-covariance of two unit-norm configurations
  // is at most 1, so the residual below is only ever negative by rounding.
  return Math.max(0, 1 - nuclearNorm * nuclearNorm);
}

/**
 * Checks that two configurations describe the same points in the same space.
 * @param a - First configuration.
 * @param b - Second configuration.
 * @throws {RangeError} If they do not have the same shape, or hold fewer than
 * two points.
 * @returns The dimension of the configurations.
 */
function checkShapes(a: number[][], b: number[][]): number {
  if (a.length !== b.length) {
    throw new RangeError(
      `procrustesDisparity: both configurations must describe the same points, got ${a.length} and ${b.length}`,
    );
  }
  if (a.length < 2) {
    throw new RangeError(
      'procrustesDisparity: at least two points are required',
    );
  }
  const dimensions = a[0]?.length ?? 0;
  if (dimensions < 1) {
    throw new RangeError('procrustesDisparity: points must have a dimension');
  }
  for (const point of a) {
    if (point.length !== dimensions) {
      throw new RangeError(
        'procrustesDisparity: every point of a configuration must have the same dimension',
      );
    }
  }
  for (const point of b) {
    if (point.length !== dimensions) {
      throw new RangeError(
        'procrustesDisparity: both configurations must have the same dimension',
      );
    }
  }
  return dimensions;
}

/**
 * Centers a configuration on its centroid and scales it to a unit Frobenius
 * norm, so that neither a translation nor a uniform scaling of the input can
 * change the result.
 * @param points - The configuration, one array per point.
 * @param dimensions - Dimension of the configuration.
 * @throws {RangeError} If every point of the configuration is the same, which
 * leaves nothing to scale.
 * @returns The standardized configuration.
 */
function standardize(points: number[][], dimensions: number): Matrix {
  const n = points.length;
  const centered = new Matrix(points);

  for (let column = 0; column < dimensions; column++) {
    let mean = 0;
    for (let row = 0; row < n; row++) mean += centered.get(row, column);
    mean /= n;
    for (let row = 0; row < n; row++) {
      centered.set(row, column, centered.get(row, column) - mean);
    }
  }

  let squaredNorm = 0;
  for (let row = 0; row < n; row++) {
    for (let column = 0; column < dimensions; column++) {
      const value = centered.get(row, column);
      squaredNorm += value * value;
    }
  }
  if (squaredNorm === 0) {
    throw new RangeError(
      'procrustesDisparity: a configuration collapses to a single point',
    );
  }

  return centered.div(Math.sqrt(squaredNorm));
}
