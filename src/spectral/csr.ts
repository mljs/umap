import { atFloat64, atInt32 } from '../sparse/arrayAt.ts';
import type { SparseMatrix } from '../sparse/index.ts';

/**
 * A symmetric graph in compressed sparse row form.
 *
 * Unlike the `getCSR` of `../sparse/operations.ts`, `ptr` holds one offset per
 * row plus the trailing total, so an empty row is a valid, addressable row.
 * The tree search relies on the other convention; the matrix-vector products
 * below rely on this one.
 */
export interface CSRMatrix {
  /** Number of rows, which is also the number of vertices. */
  n: number;
  /** Offset of the first entry of every row, plus the total at index `n`. */
  ptr: Int32Array;
  /** Column index of every entry. */
  idx: Int32Array;
  /** Value of every entry. */
  val: Float64Array;
}

/**
 * Builds the CSR form of a graph, dropping the entries that are exactly zero.
 * @param graph - The symmetric graph to convert.
 * @returns The graph in CSR form.
 * @throws {RangeError} If an entry has a negative coordinate, which no fuzzy
 * graph has and which the row offsets could not address.
 */
export function toCSR(graph: SparseMatrix): CSRMatrix {
  const n = graph.nRows;
  const { rows, cols, values, length } = graph.store;
  const ptr = new Int32Array(n + 1);
  for (let i = 0; i < length; i++) {
    if (atFloat64(values, i) === 0) continue;
    const row = atInt32(rows, i);
    if (row < 0 || atInt32(cols, i) < 0) {
      throw new RangeError('toCSR: negative coordinate');
    }
    ptr[row + 1] = atInt32(ptr, row + 1) + 1;
  }
  for (let i = 0; i < n; i++) {
    ptr[i + 1] = atInt32(ptr, i + 1) + atInt32(ptr, i);
  }
  const nnz = atInt32(ptr, n);
  const idx = new Int32Array(nnz);
  const val = new Float64Array(nnz);
  const cursor = ptr.slice(0, n);
  for (let i = 0; i < length; i++) {
    const value = atFloat64(values, i);
    if (value === 0) continue;
    const row = atInt32(rows, i);
    const slot = atInt32(cursor, row);
    cursor[row] = slot + 1;
    idx[slot] = atInt32(cols, i);
    val[slot] = value;
  }
  return { n, ptr, idx, val };
}

/**
 * Dot product of two vectors of the same length.
 * @param a - First vector.
 * @param b - Second vector.
 * @returns The sum of the elementwise products.
 */
export function dot(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += atFloat64(a, i) * atFloat64(b, i);
  }
  return sum;
}

/**
 * Euclidean norm of a vector.
 * @param x - The vector.
 * @returns Its length.
 */
export function euclideanNorm(x: Float64Array): number {
  return Math.sqrt(dot(x, x));
}
