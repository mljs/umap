/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript, the private fields moved
 * into one `UMAPInternals` state object so the fit steps, the transform and
 * the serialization can live in their own modules and the class stays under
 * the 250 line file limit, the `setParam` string-indexed assignment loop
 * replaced by a typed resolution of the parameters, and model persistence
 * (`toJSON` / `fromJSON` / `load`) added.
 *
 * TWO DELIBERATE DIVERGENCES FROM UPSTREAM:
 *
 * (a) `getEmbedding()` returns a COPY of the embedding. Upstream returns the
 *     live internal array, so a caller holding it observes — and can corrupt —
 *     the embedding while it is being fitted. `getEmbeddingRef()` is the
 *     opt-in for the live array, for progressive or animated rendering.
 *
 * (b) `transform()` freezes the fitted embedding, runs at a quarter of the
 *     learning rate, draws from a transform-scoped seed and short circuits the
 *     training data, as umap-learn does. Upstream sets
 *     `tailEmbedding = this.embedding` by reference and derives `moveOther`
 *     from `headEmbedding.length === tailEmbedding.length`, so transforming
 *     exactly as many points as were fitted rewrites the model in place and
 *     makes the answer depend on the size of the batch; it also keeps the full
 *     learning rate. ./transform.ts deep copies the embedding, passes
 *     `moveOther: false` and scales the rate down. The aliasing of the FIT
 *     path, set up in `initializeOptimization`, is intentional upstream
 *     behaviour — it is what makes both endpoints of an edge move, and what
 *     lets a caller watch the embedding converge while stepping — and is
 *     preserved there, at the full learning rate.
 */

import { copyEmbedding } from './embedding.ts';
import { initializeFit, optimizeLayoutAsync } from './fit.ts';
import { getNumberOfEpochs } from './fitSteps.ts';
import type { UMAPInternals } from './internals.ts';
import { createInternals } from './internals.ts';
import { optimizeLayout, optimizeLayoutStep } from './optimize/layout.ts';
import type { UMAPParameters, UMAPSupervisedParams } from './parameters.ts';
import type { FromJSONOptions, UMAPModel } from './serialize.ts';
import { fromJSON, toJSON } from './serialize.ts';
import type { TransformOptions } from './transform.ts';
import { transform } from './transform.ts';
import type { EpochCallback, Vectors } from './types.ts';

/**
 * UMAP projection system, based on the python implementation from McInnes, L,
 * Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension
 * Reduction (https://github.com/lmcinnes/umap).
 *
 * This implementation differs in a few regards:
 * a) The initialization of the embedding for optimization is not computed using
 *    a spectral method, rather it is initialized randomly. This avoids some
 *    computationally intensive matrix eigen computations that aren't easily
 *    ported to JavaScript.
 * b) A lot of "extra" functionality has been omitted from this implementation,
 *    most notably a great deal of alternate distance functions.
 *
 * This implementation provides three methods of reducing dimensionality:
 * 1) fit: fit the data synchronously
 * 2) fitAsync: fit the data asynchronously, with a callback function provided
 *      that is invoked on each optimization step.
 * 3) initializeFit / step: manually initialize the algorithm then explictly
 *      step through each epoch of the SGD optimization
 */
export class UMAP {
  readonly #internals: UMAPInternals;

  /**
   * Builds an unfitted model.
   * @param params - Parameters of the projection; every one is optional.
   */
  constructor(params: UMAPParameters = {}) {
    this.#internals = createInternals(params);
  }

  /**
   * The mutable state of the model.
   *
   * It is the seam the fit, transform and serialization modules work through;
   * it is not part of the stable public API.
   * @returns The internal state.
   */
  get internals(): UMAPInternals {
    return this.#internals;
  }

  /**
   * Fit the data to a projected embedding space synchronously.
   * @param X - The data to project, one array per point.
   * @returns The projected embedding, one array per point.
   */
  fit(X: Vectors): Vectors {
    const internals = this.#internals;
    initializeFit(internals, X);
    optimizeLayout(internals.optimizationState, internals.params.random);
    return this.getEmbedding();
  }

  /**
   * Fit the data to a projected embedding space asynchronously, with a callback
   * function invoked on every epoch of optimization.
   * @param X - The data to project, one array per point.
   * @param callback - Called with the number of completed epochs after each
   * epoch; returning `false` stops the optimization early.
   * @returns The projected embedding, one array per point.
   */
  async fitAsync(
    X: Vectors,
    callback: EpochCallback = () => true,
  ): Promise<Vectors> {
    initializeFit(this.#internals, X);
    await optimizeLayoutAsync(this.#internals, callback);
    return this.getEmbedding();
  }

  /**
   * Initializes fit by computing KNN and a fuzzy simplicial set, as well as
   * initializing the projected embeddings. Sets the optimization state ahead
   * of optimization steps.
   * @param X - The data to project, one array per point.
   * @returns The number of epochs to be used for the SGD optimization.
   */
  initializeFit(X: Vectors): number {
    return initializeFit(this.#internals, X);
  }

  /**
   * Manually step through the optimization process one epoch at a time.
   * @returns The number of epochs completed so far.
   */
  step(): number {
    const internals = this.#internals;
    const state = internals.optimizationState;

    if (state.currentEpoch < getNumberOfEpochs(internals)) {
      optimizeLayoutStep(state, state.currentEpoch, internals.params.random);
    }
    return state.currentEpoch;
  }

  /**
   * Returns the computed projected embedding, as a copy the optimization will
   * not write to. Use {@link UMAP.getEmbeddingRef} to watch the live one.
   * @returns The projected embedding, one array per point.
   */
  getEmbedding(): Vectors {
    return copyEmbedding(this.#internals.embedding);
  }

  /**
   * Returns the live projected embedding, the array the optimization mutates
   * in place. Read it while stepping to render the projection as it converges,
   * and never write to it.
   * @returns The internal embedding, one array per point.
   */
  getEmbeddingRef(): Vectors {
    return this.#internals.embedding;
  }

  /**
   * Initializes parameters needed for supervised projection.
   * @param Y - Target value of each point of the data being fitted.
   * @param params - Parameters of the supervised projection.
   */
  setSupervisedProjection(
    Y: number[],
    params: UMAPSupervisedParams = {},
  ): void {
    const { supervised } = this.#internals;
    this.#internals.Y = Y;
    supervised.targetMetric = params.targetMetric || supervised.targetMetric;
    supervised.targetWeight = params.targetWeight || supervised.targetWeight;
    supervised.targetNumberOfNeighbors =
      params.targetNumberOfNeighbors || supervised.targetNumberOfNeighbors;
  }

  /**
   * Initializes umap with precomputed KNN indices and distances.
   * @param knnIndices - Nearest neighbor index of each point, closest first.
   * @param knnDistances - Distance to each of those nearest neighbors.
   */
  setPrecomputedKNN(knnIndices: number[][], knnDistances: number[][]): void {
    this.#internals.knnIndices = knnIndices;
    this.#internals.knnDistances = knnDistances;
  }

  /**
   * Transforms data to the existing embedding space. The fitted embedding is
   * left untouched.
   *
   * The projection is reproducible: it draws from `transformSeed` rather than
   * from the stream the fit ran on, so one batch always lands in the same
   * place, on this model and on any model reloaded from its JSON. A batch is
   * projected as a whole, so a point moves slightly when the rest of the batch
   * changes; see `transform` in ./transform.ts for what couples them.
   * Projecting the data the model was fitted on returns a copy of the fitted
   * embedding.
   * @param toTransform - The points to project, one array per point.
   * @param options - Options of this projection.
   * @returns The position of each of those points in the embedding space.
   */
  transform(toTransform: Vectors, options: TransformOptions = {}): Vectors {
    return transform(this.#internals, toTransform, options);
  }

  /**
   * Serializes the fitted model to a plain object.
   * @returns The model, ready to be given to `JSON.stringify`.
   */
  toJSON(): UMAPModel {
    return toJSON(this.#internals);
  }

  /**
   * Rebuilds a model serialized by {@link UMAP.toJSON}.
   * @param model - The serialized model.
   * @param options - The values a model cannot carry, if it needs any.
   * @returns The reloaded model, ready to `transform`.
   */
  static fromJSON(model: UMAPModel, options: FromJSONOptions = {}): UMAP {
    return fromJSON(model, options);
  }

  /**
   * Rebuilds a model serialized by {@link UMAP.toJSON}; an alias of
   * {@link UMAP.fromJSON}, named after the other ml-* estimators.
   * @param model - The serialized model.
   * @param options - The values a model cannot carry, if it needs any.
   * @returns The reloaded model, ready to `transform`.
   */
  static load(model: UMAPModel, options: FromJSONOptions = {}): UMAP {
    return fromJSON(model, options);
  }
}
