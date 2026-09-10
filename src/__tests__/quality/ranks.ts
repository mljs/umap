/**
 * The distance ordering of a point cloud: for every point, the other points
 * sorted by distance, and the rank each of them holds in that ordering.
 *
 * Ranks are 1-based, so the nearest neighbor of a point has rank 1. Points at
 * exactly equal distance are ordered by index, which keeps every measure
 * derived from this table deterministic.
 */
export interface NeighborRanks {
  /** Number of points. */
  n: number;
  /**
   * Neighbors of each point, nearest first: `order[i * (n - 1) + p]` holds the
   * index of the neighbor of `i` of rank `p + 1`.
   */
  order: Int32Array;
  /**
   * Rank of every point as a neighbor of every other: `rank[i * n + j]` is the
   * 1-based rank of `j` among the neighbors of `i`. A point is not one of its
   * own neighbors, so `rank[i * n + i]` is 0.
   */
  rank: Int32Array;
}

/**
 * The two spaces of a neighborhood quality measure, both ranked.
 */
export interface RankedPair {
  /** Number of points, common to both spaces. */
  n: number;
  /** Ranking of the input space. */
  high: NeighborRanks;
  /** Ranking of the embedding space. */
  low: NeighborRanks;
}

/**
 * Ranks every point of a cloud against every other by Euclidean distance.
 * @param points - The points, one array per point.
 * @throws {RangeError} If there are fewer than two points.
 * @returns The ordering of the neighbors of each point, and the rank table.
 */
export function neighborRanks(points: number[][]): NeighborRanks {
  const n = points.length;
  if (n < 2) {
    throw new RangeError('neighborRanks: at least two points are required');
  }

  const order = new Int32Array(n * (n - 1));
  const rank = new Int32Array(n * n);
  const distances = new Float64Array(n);
  const candidates = new Int32Array(n - 1);

  for (let i = 0; i < n; i++) {
    const point = points[i] ?? [];
    let count = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      // A rank only depends on the ordering, so the square root of the
      // Euclidean distance is never taken.
      distances[j] = squaredDistance(point, points[j] ?? []);
      candidates[count] = j;
      count++;
    }
    candidates.sort((left, right) => {
      const delta = (distances[left] ?? 0) - (distances[right] ?? 0);
      return delta === 0 ? left - right : delta;
    });
    for (let p = 0; p < n - 1; p++) {
      const j = candidates[p] ?? 0;
      order[i * (n - 1) + p] = j;
      rank[i * n + j] = p + 1;
    }
  }

  return { n, order, rank };
}

/**
 * Ranks the two spaces of a quality measure, after checking that they describe
 * the same points.
 * @param highDim - The input points, one array per point.
 * @param embedding - The embedded points, in the same order.
 * @throws {RangeError} If the two spaces hold a different number of points.
 * @returns The number of points and the two rank tables.
 */
export function rankPair(
  highDim: number[][],
  embedding: number[][],
): RankedPair {
  if (highDim.length !== embedding.length) {
    throw new RangeError(
      `rankPair: the two spaces must describe the same points, got ${highDim.length} and ${embedding.length}`,
    );
  }
  return {
    n: highDim.length,
    high: neighborRanks(highDim),
    low: neighborRanks(embedding),
  };
}

/**
 * Squared Euclidean distance between two vectors.
 * @param x - First vector.
 * @param y - Second vector, of the same length.
 * @throws {RangeError} If the two vectors have different lengths.
 * @returns The squared distance.
 */
function squaredDistance(x: number[], y: number[]): number {
  if (x.length !== y.length) {
    throw new RangeError('neighborRanks: every point must have the same size');
  }
  let total = 0;
  for (let i = 0; i < x.length; i++) {
    const delta = (x[i] ?? 0) - (y[i] ?? 0);
    total += delta * delta;
  }
  return total;
}
