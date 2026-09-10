/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: reduced to the minimum public surface. Everything the
 * package can do is reachable through the UMAP class; the internal wiring it
 * is built from is deliberately not exported.
 */

export { UMAP } from './umap.ts';

// Reference metrics, so a caller can pass or compose a `distanceFunction`.
export { cosine, euclidean } from './metrics.ts';

export type { MetricName } from './metrics.ts';
export type {
  InitMethod,
  TargetMetric,
  UMAPParameters,
  UMAPSupervisedParams,
} from './parameters.ts';
export type { FromJSONOptions, UMAPModel } from './serialize.ts';
// The argument type of UMAP.transform, as FromJSONOptions is of UMAP.fromJSON.
export type { TransformOptions } from './transform.ts';
export type {
  DistanceFn,
  EpochCallback,
  RandomFn,
  Vector,
  Vectors,
} from './types.ts';
