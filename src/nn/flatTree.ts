/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript; the constructor assigns
 * its fields explicitly because parameter properties are not erasable syntax.
 * The layout of the four arrays is unchanged.
 */

/**
 * A random projection tree flattened into parallel arrays.
 *
 * Node `n` is an internal node when `children[n][0] > 0`: it splits the space
 * on the hyperplane `hyperplanes[n]` / `offsets[n]`, and a search continues at
 * node `children[n][0]` (left) or `children[n][1]` (right). Otherwise the node
 * is a leaf, and `children[n][0]` holds the negated leaf number, an index into
 * `indices`.
 */
export class FlatTree {
  /**
   * Normal vector of the splitting hyperplane of each node. Leaf nodes keep
   * the all zero row that was allocated for them.
   */
  public hyperplanes: number[][];

  /** Offset from the origin of the splitting hyperplane of each node. */
  public offsets: number[];

  /** Left and right child of each node; `-1` where there is none. */
  public children: number[][];

  /** Data point indices held by each leaf, right padded with `-1`. */
  public indices: number[][];

  /**
   * Creates a flat tree from its four parallel arrays.
   * @param hyperplanes - Normal vector of the splitting hyperplane of each node.
   * @param offsets - Offset from the origin of the hyperplane of each node.
   * @param children - Left and right child of each node.
   * @param indices - Data point indices held by each leaf.
   */
  constructor(
    hyperplanes: number[][],
    offsets: number[],
    children: number[][],
    indices: number[][],
  ) {
    this.hyperplanes = hyperplanes;
    this.offsets = offsets;
    this.children = children;
    this.indices = indices;
  }
}
