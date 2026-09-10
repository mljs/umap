import type { NeighborRanks } from './ranks.ts';
import { rankPair } from './ranks.ts';

/**
 * Venna & Kaski trustworthiness: how much the `k` nearest neighbors an
 * embedding shows can be trusted, i.e. how far the points it brought into a
 * neighborhood were in the input space.
 *
 * With `U_k(i)` the points among the `k` nearest neighbors of `i` in the
 * embedding but not in the input space, and `r(i, j)` the 1-based rank of `j`
 * among the neighbors of `i` in the input space:
 *
 * ```text
 * T(k) = 1 - 2 / (n * k * (2n - 3k - 1)) * sum_i sum_{j in U_k(i)} (r(i, j) - k)
 * ```
 *
 * It is 1 when the embedding introduces no false neighbor and decreases as the
 * intruders come from farther away in the input space.
 * @param highDim - The input points, one array per point.
 * @param embedding - The embedded points, in the same order.
 * @param k - Size of the neighborhood; the normalization requires `k < n / 2`.
 * @throws {RangeError} If the two spaces disagree on the number of points, or
 * if `k` is not a usable neighborhood size.
 * @returns The trustworthiness, in `[0, 1]`.
 */
export function trustworthiness(
  highDim: number[][],
  embedding: number[][],
  k: number,
): number {
  const { n, high, low } = rankPair(highDim, embedding);
  checkNeighborhood('trustworthiness', k, n);
  return 1 - penalty(n, k, low, high) / normalizer(n, k);
}

/**
 * Venna & Kaski continuity: the dual of {@link trustworthiness}, measuring how
 * far the embedding pushed the neighbors a point had in the input space.
 *
 * With `V_k(i)` the points among the `k` nearest neighbors of `i` in the input
 * space but not in the embedding, and `s(i, j)` the 1-based rank of `j` among
 * the neighbors of `i` in the embedding:
 *
 * ```text
 * C(k) = 1 - 2 / (n * k * (2n - 3k - 1)) * sum_i sum_{j in V_k(i)} (s(i, j) - k)
 * ```
 *
 * It is 1 when no neighborhood of the input space is torn apart.
 * @param highDim - The input points, one array per point.
 * @param embedding - The embedded points, in the same order.
 * @param k - Size of the neighborhood; the normalization requires `k < n / 2`.
 * @throws {RangeError} If the two spaces disagree on the number of points, or
 * if `k` is not a usable neighborhood size.
 * @returns The continuity, in `[0, 1]`.
 */
export function continuity(
  highDim: number[][],
  embedding: number[][],
  k: number,
): number {
  const { n, high, low } = rankPair(highDim, embedding);
  checkNeighborhood('continuity', k, n);
  return 1 - penalty(n, k, high, low) / normalizer(n, k);
}

/**
 * Sums, over every point, the rank excess of the neighbors one space places in
 * the neighborhood that the other space ranks beyond it.
 * @param n - Number of points.
 * @param k - Size of the neighborhood.
 * @param ordering - The space the neighborhood is taken from.
 * @param ranking - The space the intruders are ranked in.
 * @returns The summed rank excess.
 */
function penalty(
  n: number,
  k: number,
  ordering: NeighborRanks,
  ranking: NeighborRanks,
): number {
  let total = 0;
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const j = ordering.order[i * (n - 1) + p] ?? 0;
      const rank = ranking.rank[i * n + j] ?? 0;
      if (rank > k) total += rank - k;
    }
  }
  return total;
}

/**
 * The `n * k * (2n - 3k - 1) / 2` factor that scales the worst possible
 * penalty to 1.
 * @param n - Number of points.
 * @param k - Size of the neighborhood.
 * @returns The normalization factor.
 */
function normalizer(n: number, k: number): number {
  return (n * k * (2 * n - 3 * k - 1)) / 2;
}

/**
 * Rejects a neighborhood size the Venna & Kaski normalization is not defined
 * for.
 * @param name - Name of the calling measure, for the message.
 * @param k - Size of the neighborhood.
 * @param n - Number of points.
 * @throws {RangeError} If `k` is not an integer in `[1, ceil(n / 2) - 1]`.
 */
function checkNeighborhood(name: string, k: number, n: number): void {
  if (!Number.isInteger(k) || k < 1 || 2 * k >= n) {
    throw new RangeError(
      `${name}: k must be an integer in [1, ${Math.ceil(n / 2) - 1}], got ${k}`,
    );
  }
}
