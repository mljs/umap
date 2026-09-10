/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript and split across
 * ./flatTree.ts, ./treeBuild.ts and ./treeSearch.ts to respect the file size
 * limit; index reads are bounds checked, and a leaf writes its indices with an
 * indexed loop instead of `splice` with a spread. The trees are built in the
 * same order and consume the same random draws as the reference.
 */

import type { RandomFn, Vectors } from '../types.ts';
import { filled, zeros } from '../utils.ts';

import { at } from './checkedAt.ts';
import { FlatTree } from './flatTree.ts';
import type { RandomProjectionTreeNode } from './treeBuild.ts';
import { makeTree, numLeaves, numNodes } from './treeBuild.ts';

export { FlatTree } from './flatTree.ts';
export { searchFlatTree } from './treeSearch.ts';

/**
 * Builds a random projection forest with `nTrees` trees.
 * @param data - Data points to index.
 * @param numberOfNeighbors - Number of neighbors being searched; with a floor
 * of 10 it sets the leaf size.
 * @param nTrees - Number of trees in the forest.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns The flattened trees of the forest.
 */
export function makeForest(
  data: Vectors,
  numberOfNeighbors: number,
  nTrees: number,
  random: RandomFn,
): FlatTree[] {
  const leafSize = Math.max(10, numberOfNeighbors);

  const trees: RandomProjectionTreeNode[] = [];
  for (let i = 0; i < nTrees; i++) {
    trees.push(makeTree(data, leafSize, random));
  }

  const forest: FlatTree[] = [];
  for (const tree of trees) {
    forest.push(flattenTree(tree, leafSize));
  }
  return forest;
}

/**
 * Generates an array of sets of candidate nearest neighbors by taking the
 * leaves of all the trees of a random projection forest. Any given tree has
 * leaves that are a set of potential nearest neighbors. Given enough trees,
 * the set of all such leaves gives a good likelihood of getting a good set of
 * nearest neighbors in composite. Since such a random projection forest is
 * inexpensive to compute, this can be a useful means of seeding other nearest
 * neighbor algorithms.
 * @param rpForest - Flattened trees of a random projection forest.
 * @returns The leaves of every tree, or `[[-1]]` when the forest is empty.
 */
export function makeLeafArray(rpForest: FlatTree[]): number[][] {
  if (rpForest.length === 0) {
    return [[-1]];
  }
  const output: number[][] = [];
  for (const tree of rpForest) {
    for (const leaf of tree.indices) {
      output.push(leaf);
    }
  }
  return output;
}

interface FlattenState {
  nodeNum: number;
  leafNum: number;
}

function flattenTree(
  tree: RandomProjectionTreeNode,
  leafSize: number,
): FlatTree {
  const nNodes = numNodes(tree);
  const nLeaves = numLeaves(tree);

  // Every node gets a hyperplane row as wide as the root's hyperplane, so the
  // rows of the leaves stay zero. A root that is itself a leaf gives width 0.
  const hyperplaneLength = tree.isLeaf ? 0 : tree.hyperplane.length;
  const hyperplanes: number[][] = [];
  const children: number[][] = [];
  for (let i = 0; i < nNodes; i++) {
    hyperplanes.push(zeros(hyperplaneLength));
    children.push([-1, -1]);
  }

  const offsets = zeros(nNodes);
  const indices: number[][] = [];
  for (let i = 0; i < nLeaves; i++) {
    indices.push(filled(leafSize, -1));
  }

  recursiveFlatten(tree, hyperplanes, offsets, children, indices, 0, 0);
  return new FlatTree(hyperplanes, offsets, children, indices);
}

function recursiveFlatten(
  tree: RandomProjectionTreeNode,
  hyperplanes: number[][],
  offsets: number[],
  children: number[][],
  indices: number[][],
  nodeNum: number,
  leafNum: number,
): FlattenState {
  if (tree.isLeaf) {
    // Leaf 0 is stored as `-0`, which the search negates back to index 0.
    at(children, nodeNum)[0] = -leafNum;

    const leaf = at(indices, leafNum);
    const treeIndices = tree.indices;
    for (let i = 0; i < treeIndices.length; i++) {
      leaf[i] = at(treeIndices, i);
    }
    return { nodeNum, leafNum: leafNum + 1 };
  }

  hyperplanes[nodeNum] = tree.hyperplane;
  offsets[nodeNum] = tree.offset;
  at(children, nodeNum)[0] = nodeNum + 1;

  const leftResult = recursiveFlatten(
    tree.leftChild,
    hyperplanes,
    offsets,
    children,
    indices,
    nodeNum + 1,
    leafNum,
  );

  at(children, nodeNum)[1] = leftResult.nodeNum + 1;

  return recursiveFlatten(
    tree.rightChild,
    hyperplanes,
    offsets,
    children,
    indices,
    leftResult.nodeNum + 1,
    leftResult.leafNum,
  );
}
