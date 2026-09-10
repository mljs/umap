/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript, the `map` callbacks that
 * build the sample points replaced by indexed loops writing the same values in
 * the same order, and `ml-levenberg-marquardt` imported as the named export it
 * now provides. The fitted curve, the sample points and the solver options are
 * unchanged.
 */

import { levenbergMarquardt } from 'ml-levenberg-marquardt';

import { linear, zeros } from '../utils.ts';

/**
 * Fits the `a` and `b` parameters of the differentiable curve
 * `1 / (1 + a * x^(2b))` used to build the low dimensional fuzzy simplicial
 * complex. It is the curve from that family, which has a simple gradient, that
 * best matches an offset exponential decay of the given spread and minimum
 * distance.
 * @param spread - Effective scale of the embedded points.
 * @param minimumDistance - Minimum distance between embedded points.
 * @returns The two fitted parameters.
 */
export function findABParams(
  spread: number,
  minimumDistance: number,
): { a: number; b: number } {
  const xv = linear(0, spread * 3, 300);
  for (let i = 0; i < xv.length; i++) {
    const value = xv[i];
    if (value === undefined) {
      throw new RangeError(SAMPLE_ERROR);
    }
    if (value < minimumDistance) {
      xv[i] = 1;
    }
  }

  const yv = zeros(xv.length);
  for (let i = 0; i < xv.length; i++) {
    const value = xv[i];
    if (value === undefined) {
      throw new RangeError(SAMPLE_ERROR);
    }
    // Samples below minimumDistance keep the 0 they were initialized with.
    if (value >= minimumDistance) {
      yv[i] = Math.exp(-(value - minimumDistance) / spread);
    }
  }

  const { parameterValues } = levenbergMarquardt({ x: xv, y: yv }, curve, {
    damping: 1.5,
    initialValues: [0.5, 0.5],
    gradientDifference: 10e-2,
    maxIterations: 100,
    errorTolerance: 10e-3,
  });

  const a = parameterValues[0];
  const b = parameterValues[1];
  if (a === undefined || b === undefined) {
    throw new Error('findABParams: the fit did not return two parameters');
  }
  return { a, b };
}

const SAMPLE_ERROR = 'findABParams: could not build the curve sample points';

function curve(params: number[]): (x: number) => number {
  const a = params[0];
  const b = params[1];
  if (a === undefined || b === undefined) {
    throw new RangeError('findABParams: the curve takes exactly 2 parameters');
  }
  // Deliberate: the exponent is not an integer, so this stays an exponentiation.
  return (x: number) => 1 / (1 + a * x ** (2 * b));
}
