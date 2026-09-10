/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript as a standalone function
 * taking its inputs explicitly instead of reading them off the UMAP instance,
 * the `distances.filter` / `distances.map` callbacks replaced by loops, the
 * loop invariant mean of the row means computed once instead of once per row,
 * and split out of simplicialSet.ts to respect the 250 line file limit. A row
 * whose neighbor distances are all zero now gets a rho of zero instead of the
 * NaN upstream reads past the end of the empty non-zero distance array for,
 * which `transform` — the only caller passing a local connectivity below one —
 * turned into a NaN embedding. Every constant, the arithmetic and the
 * iteration order are otherwise unchanged.
 */

import { atNumber } from '../sparse/arrayAt.ts';
import { max, mean, zeros } from '../utils.ts';

import { rowAt } from './rowAt.ts';

/**
 * Options of {@link smoothKNNDistance}.
 */
export interface SmoothKNNDistanceOptions {
  /**
   * Number of nearest neighbors that should be assumed to be locally connected,
   * i.e. at a fuzzy distance of zero. A fractional value interpolates between
   * the two surrounding neighbors.
   * @default 1
   */
  localConnectivity?: number;
  /**
   * Number of binary search iterations used to solve for each sigma.
   * @default 64
   */
  nIter?: number;
  /**
   * Bandwidth of the kernel, scaling the cardinality the search targets.
   * @default 1
   */
  bandwidth?: number;
}

/**
 * Per point smooth approximation of the distance to the kth nearest neighbor.
 */
export interface SmoothKNNDistanceResult {
  /** Normalization factor of each point's local kernel. */
  sigmas: number[];
  /** Distance to the nearest locally connected neighbor of each point. */
  rhos: number[];
}

/**
 * Compute a continuous version of the distance to the kth nearest neighbor.
 * That is, this is similar to knn-distance but allows continuous k values
 * rather than requiring an integral k. In essence we are simply computing the
 * distance such that the cardinality of fuzzy set we generate is k.
 * @param distances - Sorted distances to the nearest neighbors of each point.
 * @param k - Cardinality the fuzzy set of each point should reach.
 * @param options - Search options.
 * @returns The sigma and rho of each point.
 */
export function smoothKNNDistance(
  distances: number[][],
  k: number,
  options: SmoothKNNDistanceOptions = {},
): SmoothKNNDistanceResult {
  const { localConnectivity = 1, nIter = 64, bandwidth = 1 } = options;

  const target = (Math.log(k) / Math.log(2)) * bandwidth;
  const rho = zeros(distances.length);
  const result = zeros(distances.length);
  let meanOfMeans = 0;
  let hasMeanOfMeans = false;

  for (let i = 0; i < distances.length; i++) {
    let lo = 0;
    let hi = Infinity;
    let mid = 1;

    const ithDistances = rowAt(distances, i);
    const nonZeroDists: number[] = [];
    for (const distance of ithDistances) {
      if (distance > 0) {
        nonZeroDists.push(distance);
      }
    }

    const rhoI = localRho(nonZeroDists, localConnectivity);
    rho[i] = rhoI;

    for (let n = 0; n < nIter; n++) {
      let psum = 0;
      // Deliberate: the scan starts at 1, skipping each point's own distance.
      for (let j = 1; j < ithDistances.length; j++) {
        const d = atNumber(ithDistances, j) - rhoI;
        if (d > 0) {
          psum += Math.exp(-(d / mid));
        } else {
          psum += 1;
        }
      }

      if (Math.abs(psum - target) < SMOOTH_K_TOLERANCE) {
        break;
      }

      if (psum > target) {
        hi = mid;
        mid = (lo + hi) / 2;
      } else {
        lo = mid;
        if (hi === Infinity) {
          mid *= 2;
        } else {
          mid = (lo + hi) / 2;
        }
      }
    }

    result[i] = mid;

    if (rhoI > 0) {
      const meanIthDistances = mean(ithDistances);
      if (mid < MIN_K_DIST_SCALE * meanIthDistances) {
        result[i] = MIN_K_DIST_SCALE * meanIthDistances;
      }
    } else {
      if (!hasMeanOfMeans) {
        meanOfMeans = meanOfRowMeans(distances);
        hasMeanOfMeans = true;
      }
      if (mid < MIN_K_DIST_SCALE * meanOfMeans) {
        result[i] = MIN_K_DIST_SCALE * meanOfMeans;
      }
    }
  }

  return { sigmas: result, rhos: rho };
}

/**
 * Convergence tolerance of the binary search for sigma, and threshold below
 * which a fractional local connectivity is not interpolated.
 */
export const SMOOTH_K_TOLERANCE = 1e-5;

/**
 * Lower bound of sigma, as a fraction of the mean distance of the point.
 */
export const MIN_K_DIST_SCALE = 1e-3;

/**
 * Distance to the nearest locally connected neighbor of a single point.
 * @param nonZeroDists - Sorted non-zero distances to the point's neighbors.
 * @param localConnectivity - Number of neighbors assumed locally connected.
 * @returns The rho of the point.
 */
function localRho(nonZeroDists: number[], localConnectivity: number): number {
  // Every neighbor of the point sits at distance zero, so the nearest locally
  // connected one does too, whatever the local connectivity is. Upstream only
  // reaches the branches below with a non-empty array when the connectivity is
  // at least one; a lower one falls through to an out of bounds read there,
  // and the NaN it yields poisons every membership strength of the row.
  if (nonZeroDists.length === 0) {
    return 0;
  }
  if (nonZeroDists.length >= localConnectivity) {
    const index = Math.floor(localConnectivity);
    const interpolation = localConnectivity - index;
    if (index > 0) {
      // Both reads are in bounds: `length >= localConnectivity >= index`, and
      // a non-zero interpolation implies `length > index`.
      let rho = atNumber(nonZeroDists, index - 1);
      if (interpolation > SMOOTH_K_TOLERANCE) {
        rho +=
          interpolation *
          (atNumber(nonZeroDists, index) - atNumber(nonZeroDists, index - 1));
      }
      return rho;
    }
    return interpolation * atNumber(nonZeroDists, 0);
  }
  return max(nonZeroDists);
}

/**
 * Mean of the per-row means, which upstream recomputes for every row.
 * @param distances - Distances to the nearest neighbors of each point.
 * @returns The mean of the row means.
 */
function meanOfRowMeans(distances: number[][]): number {
  const rowMeans: number[] = [];
  for (let i = 0; i < distances.length; i++) {
    rowMeans.push(mean(rowAt(distances, i)));
  }
  return mean(rowMeans);
}
