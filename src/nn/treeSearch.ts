/*
 * Derived from umap-js (https://github.com/PAIR-code/umap-js)
 * Copyright 2019 Google LLC. Licensed under the Apache License, Version 2.0.
 * Modified for ml-umap: ported to strict TypeScript; index reads are bounds
 * checked and the children row of the current node is kept in a local instead
 * of being re-read. The descent, the margin arithmetic and the coin flip on a
 * zero margin are unchanged.
 */

import type { RandomFn, Vector } from '../types.ts';
import { tauRandInt } from '../utils.ts';

import { at } from './checkedAt.ts';
import type { FlatTree } from './flatTree.ts';

/**
 * Searches a flattened random projection tree for the leaf a point falls in.
 * @param point - Point to locate.
 * @param tree - Flattened tree to descend.
 * @param random - Random number generator returning a float in [0, 1), used to break ties on the hyperplane.
 * @returns The data point indices of the leaf, right padded with `-1`.
 */
export function searchFlatTree(
  point: Vector,
  tree: FlatTree,
  random: RandomFn,
): number[] {
  let node = 0;
  let children = at(tree.children, node);
  let leftChild = at(children, 0);

  while (leftChild > 0) {
    const side = selectSide(
      at(tree.hyperplanes, node),
      at(tree.offsets, node),
      point,
      random,
    );
    node = side === 0 ? leftChild : at(children, 1);
    children = at(tree.children, node);
    leftChild = at(children, 0);
  }

  // A leaf stores the negated leaf number, so `-0` maps back to leaf 0.
  return at(tree.indices, -1 * leftChild);
}

/**
 * Selects the side of the tree to search during flat tree search.
 * @param hyperplane - Normal vector of the splitting hyperplane of the node.
 * @param offset - Offset from the origin of that hyperplane.
 * @param point - Point being located.
 * @param random - Random number generator returning a float in [0, 1).
 * @returns 0 for the left side, 1 for the right side.
 */
function selectSide(
  hyperplane: number[],
  offset: number,
  point: Vector,
  random: RandomFn,
): number {
  let margin = offset;
  for (let d = 0; d < point.length; d++) {
    margin += at(hyperplane, d) * at(point, d);
  }

  if (margin === 0) {
    return tauRandInt(2, random);
  } else if (margin > 0) {
    return 0;
  } else {
    return 1;
  }
}
