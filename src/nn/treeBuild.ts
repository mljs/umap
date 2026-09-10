/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript; tree nodes are a
 * discriminated union instead of an interface with optional members, index
 * reads are bounds checked once per vector rather than once per dimension, and
 * the unused tree depth argument of the original was dropped. Arithmetic,
 * iteration order and the order of the random draws are unchanged.
 */

import type { RandomFn, Vectors } from '../types.ts';
import { range, tauRandInt, zeros } from '../utils.ts';

import { at } from './checkedAt.ts';

/** A leaf of a random projection tree, holding the indices it covers. */
export interface LeafNode {
  isLeaf: true;
  indices: number[];
}

/** An internal node of a random projection tree, holding its split. */
export interface InternalNode {
  isLeaf: false;
  leftChild: RandomProjectionTreeNode;
  rightChild: RandomProjectionTreeNode;
  hyperplane: number[];
  offset: number;
}

/** A node of a random projection tree. */
export type RandomProjectionTreeNode = LeafNode | InternalNode;

/**
 * Constructs a random projection tree based on `data`, with leaves of size at
 * most `leafSize`.
 * @param data - Data points to index.
 * @param leafSize - Maximum number of points held by a leaf.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns The root node of the tree.
 */
export function makeTree(
  data: Vectors,
  leafSize: number,
  random: RandomFn,
): RandomProjectionTreeNode {
  const indices = range(data.length);
  return makeEuclideanTree(data, indices, leafSize, random);
}

/**
 * Counts the nodes of a random projection tree, leaves included.
 * @param tree - Root node of the tree.
 * @returns The number of nodes.
 */
export function numNodes(tree: RandomProjectionTreeNode): number {
  if (tree.isLeaf) {
    return 1;
  }
  return 1 + numNodes(tree.leftChild) + numNodes(tree.rightChild);
}

/**
 * Counts the leaves of a random projection tree.
 * @param tree - Root node of the tree.
 * @returns The number of leaves.
 */
export function numLeaves(tree: RandomProjectionTreeNode): number {
  if (tree.isLeaf) {
    return 1;
  }
  return numLeaves(tree.leftChild) + numLeaves(tree.rightChild);
}

interface SplitResult {
  indicesLeft: number[];
  indicesRight: number[];
  hyperplane: number[];
  offset: number;
}

function makeEuclideanTree(
  data: Vectors,
  indices: number[],
  leafSize: number,
  random: RandomFn,
): RandomProjectionTreeNode {
  if (indices.length <= leafSize) {
    return { indices, isLeaf: true };
  }

  const { indicesLeft, indicesRight, hyperplane, offset } =
    euclideanRandomProjectionSplit(data, indices, random);

  const leftChild = makeEuclideanTree(data, indicesLeft, leafSize, random);
  const rightChild = makeEuclideanTree(data, indicesRight, leafSize, random);

  return { leftChild, rightChild, isLeaf: false, hyperplane, offset };
}

/**
 * Given a set of `indices` for data points from `data`, creates a random
 * hyperplane to split the data, returning the two arrays of indices that fall
 * on either side of the hyperplane. This is the basis for a random projection
 * tree, which simply uses this splitting recursively. This particular split
 * uses euclidean distance to determine the hyperplane and which side each data
 * sample falls on.
 * @param data - Data points to split.
 * @param indices - Indices of the data points taking part in the split.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns The indices on each side, and the hyperplane they were split on.
 */
function euclideanRandomProjectionSplit(
  data: Vectors,
  indices: number[],
  random: RandomFn,
): SplitResult {
  const dim = at(data, 0).length;

  // Select two random points, set the hyperplane between them.
  const leftIndex = tauRandInt(indices.length, random);
  let rightIndex = tauRandInt(indices.length, random);
  rightIndex += leftIndex === rightIndex ? 1 : 0;
  rightIndex = rightIndex % indices.length;
  const left = at(data, at(indices, leftIndex));
  const right = at(data, at(indices, rightIndex));
  checkDimension(left, dim);
  checkDimension(right, dim);

  // Compute the normal vector to the hyperplane (the vector between the two
  // points) and the offset from the origin.
  let hyperplaneOffset = 0;
  const hyperplaneVector = zeros(dim);

  // The reads of the two loops below are asserted rather than bounds checked:
  // `checkDimension` covers a whole vector at once, and these are the loops the
  // cost of building a forest is concentrated in.
  for (let i = 0; i < dim; i++) {
    const leftValue = left[i] as number;
    const rightValue = right[i] as number;
    const difference = leftValue - rightValue;
    hyperplaneVector[i] = difference;
    hyperplaneOffset -= (difference * (leftValue + rightValue)) / 2;
  }

  // For each point compute the margin (project into normal vector). If we are
  // on the lower side of the hyperplane put it in one pile, otherwise put it in
  // the other pile (if we hit the hyperplane on the nose, flip a coin).
  let nLeft = 0;
  let nRight = 0;
  const side = zeros(indices.length);
  for (let i = 0; i < indices.length; i++) {
    const point = at(data, at(indices, i));
    checkDimension(point, dim);
    let margin = hyperplaneOffset;
    for (let d = 0; d < dim; d++) {
      margin += (hyperplaneVector[d] as number) * (point[d] as number);
    }
    if (margin === 0) {
      const coin = tauRandInt(2, random);
      side[i] = coin;
      if (coin === 0) {
        nLeft += 1;
      } else {
        nRight += 1;
      }
    } else if (margin > 0) {
      side[i] = 0;
      nLeft += 1;
    } else {
      side[i] = 1;
      nRight += 1;
    }
  }

  // Now that we have the counts, allocate the arrays and populate them with the
  // indices according to which side they fell on.
  const indicesLeft = zeros(nLeft);
  const indicesRight = zeros(nRight);
  let leftCount = 0;
  let rightCount = 0;
  for (let i = 0; i < side.length; i++) {
    const index = at(indices, i);
    if (at(side, i) === 0) {
      indicesLeft[leftCount] = index;
      leftCount += 1;
    } else {
      indicesRight[rightCount] = index;
      rightCount += 1;
    }
  }

  return {
    indicesLeft,
    indicesRight,
    hyperplane: hyperplaneVector,
    offset: hyperplaneOffset,
  };
}

function checkDimension(point: number[], dim: number): void {
  if (point.length < dim) {
    throw new RangeError(
      `makeTree: every point must have at least ${dim} dimensions`,
    );
  }
}
