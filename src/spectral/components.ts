import { atFloat64, atInt32 } from '../sparse/arrayAt.ts';

import type { CSRMatrix } from './csr.ts';

/**
 * The connected components of a graph, as returned by
 * {@link connectedComponents}.
 */
export interface GraphComponents {
  /** Number of connected components. */
  count: number;
  /** Component index of every vertex, in `[0, count)`. */
  label: Int32Array;
}

/**
 * A connected component of a graph, as returned by {@link componentSubgraph}.
 */
export interface ComponentSubgraph {
  /** The induced subgraph, whose vertices are numbered from zero. */
  csr: CSRMatrix;
  /** Index in the whole graph of every vertex of the subgraph. */
  nodes: Int32Array;
}

/**
 * Labels the connected components of a graph by breadth-first search.
 *
 * A disconnected graph is not an edge case here: five well separated Gaussian
 * blobs produce a fuzzy graph with exactly five components. The Laplacian of a
 * graph with `k` components has `k` zero eigenvalues, so its smallest
 * non-trivial eigenvectors span an arbitrary basis of the component indicator
 * space and every vertex of a component lands on the same point. Each
 * component therefore has to be embedded on its own.
 * @param csr - The symmetric graph.
 * @returns The number of components and the label of every vertex.
 */
export function connectedComponents(csr: CSRMatrix): GraphComponents {
  const { n, ptr, idx } = csr;
  const label = new Int32Array(n).fill(-1);
  const queue = new Int32Array(n);
  let count = 0;
  for (let start = 0; start < n; start++) {
    if (atInt32(label, start) !== -1) continue;
    label[start] = count;
    queue[0] = start;
    let head = 0;
    let tail = 1;
    while (head < tail) {
      const vertex = atInt32(queue, head);
      head++;
      const end = atInt32(ptr, vertex + 1);
      for (let p = atInt32(ptr, vertex); p < end; p++) {
        const neighbor = atInt32(idx, p);
        if (atInt32(label, neighbor) === -1) {
          label[neighbor] = count;
          queue[tail] = neighbor;
          tail++;
        }
      }
    }
    count++;
  }
  return { count, label };
}

/**
 * Extracts the subgraph induced by one connected component.
 * @param csr - The whole graph.
 * @param label - Component label of every vertex.
 * @param target - The component to extract.
 * @returns The induced subgraph and the vertices it was built from.
 */
export function componentSubgraph(
  csr: CSRMatrix,
  label: Int32Array,
  target: number,
): ComponentSubgraph {
  const local = new Int32Array(csr.n).fill(-1);
  let size = 0;
  for (let i = 0; i < csr.n; i++) {
    if (atInt32(label, i) === target) {
      local[i] = size;
      size++;
    }
  }
  const nodes = new Int32Array(size);
  for (let i = 0; i < csr.n; i++) {
    const position = atInt32(local, i);
    if (position >= 0) nodes[position] = i;
  }

  const ptr = new Int32Array(size + 1);
  for (let a = 0; a < size; a++) {
    const vertex = atInt32(nodes, a);
    let degree = 0;
    const end = atInt32(csr.ptr, vertex + 1);
    for (let p = atInt32(csr.ptr, vertex); p < end; p++) {
      if (atInt32(local, atInt32(csr.idx, p)) >= 0) degree++;
    }
    ptr[a + 1] = atInt32(ptr, a) + degree;
  }
  const nnz = atInt32(ptr, size);
  const idx = new Int32Array(nnz);
  const val = new Float64Array(nnz);
  let slot = 0;
  for (let a = 0; a < size; a++) {
    const vertex = atInt32(nodes, a);
    const end = atInt32(csr.ptr, vertex + 1);
    for (let p = atInt32(csr.ptr, vertex); p < end; p++) {
      const column = atInt32(local, atInt32(csr.idx, p));
      if (column < 0) continue;
      idx[slot] = column;
      val[slot] = atFloat64(csr.val, p);
      slot++;
    }
  }
  return { csr: { n: size, ptr, idx, val }, nodes };
}
