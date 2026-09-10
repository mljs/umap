/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: shared type aliases extracted into their own module.
 */

/** A single data point. */
export type Vector = number[];

/** A collection of data points, one array per sample. */
export type Vectors = Vector[];

/** Returns a pseudo-random number in [0, 1). */
export type RandomFn = () => number;

/** Computes the distance between two vectors of equal length. */
export type DistanceFn = (x: Vector, y: Vector) => number;

/** Called after each optimization epoch; return false to stop early. */
export type EpochCallback = (epoch: number) => boolean | void;
