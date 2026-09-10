import { rankPair } from './ranks.ts';

/**
 * Fraction of the `k` nearest neighbors a point has in the input space that
 * are still among its `k` nearest neighbors in the embedding, averaged over
 * the points.
 *
 * With `N_k(i)` the k nearest neighbors of point `i`:
 *
 * ```text
 * knn(k) = (1 / (n * k)) * sum_i |N_k_high(i) inter N_k_low(i)|
 * ```
 *
 * It is 1 when every neighborhood is preserved and 0 when none is, and unlike
 * trustworthiness it counts a lost neighbor the same however far the embedding
 * pushed it away.
 * @param highDim - The input points, one array per point.
 * @param embedding - The embedded points, in the same order.
 * @param k - Size of the neighborhood, in `[1, n - 1]`.
 * @throws {RangeError} If the two spaces disagree on the number of points, or
 * if `k` is not a usable neighborhood size.
 * @returns The preserved fraction, in `[0, 1]`.
 */
export function knnPreservation(
  highDim: number[][],
  embedding: number[][],
  k: number,
): number {
  const { n, high, low } = rankPair(highDim, embedding);

  if (!Number.isInteger(k) || k < 1 || k >= n) {
    throw new RangeError(
      `knnPreservation: k must be an integer in [1, ${n - 1}], got ${k}`,
    );
  }

  let kept = 0;
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const j = high.order[i * (n - 1) + p] ?? 0;
      // A rank is 1-based and only a point's own rank is 0, so a rank of at
      // most k means j is one of the k nearest neighbors of i.
      if ((low.rank[i * n + j] ?? 0) <= k) kept++;
    }
  }

  return kept / (n * k);
}
