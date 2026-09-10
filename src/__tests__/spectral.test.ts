import { EigenvalueDecomposition } from 'ml-matrix';
import { expect, test } from 'vitest';

import { DEFAULT_PARAMETERS } from '../defaults.ts';
import { initializeSimplicialSetEmbedding } from '../embedding.ts';
import { UMAP } from '../index.ts';
import { SparseMatrix } from '../sparse/index.ts';
import {
  connectedComponents,
  spectralEmbedding,
  spectralVectors,
  toCSR,
} from '../spectral/index.ts';

import { blobs } from './blobs.ts';
import { clusterRatio } from './clustering.ts';
import {
  ascendingOrder,
  denseLaplacian,
  graphOf,
  graphOfData,
} from './graphs.ts';
import { makeRandom } from './random.ts';

// Largest relative Rayleigh residual observed over the seeds tried below is
// 4.1e-6; the guard the library itself applies is the much looser 1e-2.
const RESIDUAL_BOUND = 1e-4;

test('the eigenpairs of a connected graph have a tiny Rayleigh residual', () => {
  const csr = toCSR(graphOf(150, 4, 1));

  expect(connectedComponents(csr).count).toBe(1);

  for (const seed of [11, 12, 13]) {
    const pairs = spectralVectors(csr, 3, { random: makeRandom(seed) });

    expect(pairs).toHaveLength(3);

    for (const pair of pairs) {
      expect(pair.residual).toBeLessThan(RESIDUAL_BOUND);
      expect(pair.eigenvalue).toBeGreaterThan(0);
      expect(pair.eigenvalue).toBeLessThan(2);
    }
  }
});

test('the eigenpairs match a dense decomposition of the same Laplacian', () => {
  const csr = toCSR(graphOf(150, 4, 1));
  const decomposition = new EigenvalueDecomposition(denseLaplacian(csr), {
    assumeSymmetric: true,
  });
  const ascending = ascendingOrder(decomposition.realEigenvalues);
  const dense = decomposition.eigenvectorMatrix;
  const pairs = spectralVectors(csr, 3, { random: makeRandom(11) });

  // The smallest eigenvalue of a connected graph is 0 and its eigenvector is
  // exactly sqrt(deg), which the Lanczos run deflates away, so the pairs line
  // up with the dense ones from index 1 on.
  for (let k = 0; k < 3; k++) {
    const column = ascending[k + 1] ?? 0;
    const expected = decomposition.realEigenvalues[column] ?? 0;
    const pair = pairs[k];

    expect(pair).toBeDefined();
    expect(pair?.eigenvalue).toBeCloseTo(expected, 9);

    // Eigenvectors are only defined up to a sign, so compare the magnitude of
    // the cosine rather than the coordinates.
    let cosine = 0;
    for (let i = 0; i < csr.n; i++) {
      cosine += dense.get(i, column) * (pair?.vector[i] ?? 0);
    }

    expect(Math.abs(cosine)).toBeGreaterThan(0.999999);
  }
});

test('the graph of five separated blobs has exactly five components', () => {
  const { X, labels } = blobs(500, 20, 5);
  const csr = toCSR(graphOfData(X));
  const { count, label } = connectedComponents(csr);

  expect(count).toBe(5);

  // Every component holds exactly one blob, and every blob exactly one
  // component: a component is a relabelling of the blob index.
  const blobsOfComponent = new Map<number, Set<number>>();
  for (let i = 0; i < csr.n; i++) {
    const component = label[i] ?? -1;
    const known = blobsOfComponent.get(component) ?? new Set<number>();
    known.add(labels[i] ?? -1);
    blobsOfComponent.set(component, known);
  }
  const blobsPerComponent: number[] = [];
  for (const set of blobsOfComponent.values()) blobsPerComponent.push(set.size);

  expect(blobsPerComponent).toStrictEqual([1, 1, 1, 1, 1]);
});

test('a disconnected graph is embedded component by component', () => {
  const { X, labels } = blobs(300, 20, 5);
  const embedding = spectralEmbedding(graphOfData(X), {
    numberOfComponents: 2,
    random: makeRandom(11),
    data: X,
  });

  expect(embedding).not.toBeNull();
  expect(embedding).toHaveLength(X.length);

  const rows = embedding ?? [];
  for (const row of rows) {
    expect(row).toHaveLength(2);

    for (const value of row) {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value)).toBeLessThanOrEqual(10.001);
    }
  }

  // Each blob has to stay together: the components are placed apart from one
  // another rather than collapsed onto a single point.
  expect(clusterRatio(rows, labels)).toBeLessThan(0.3);
});

test('a graph with non-finite weights falls back instead of throwing', () => {
  const size = 20;
  const rows: number[] = [];
  const cols: number[] = [];
  const values: number[] = [];
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (i === j) continue;
      rows.push(i);
      cols.push(j);
      values.push(Number.NaN);
    }
  }
  const graph = new SparseMatrix(rows, cols, values, [size, size]);
  const data = blobs(size, 3, 1).X;

  expect(
    spectralEmbedding(graph, {
      numberOfComponents: 2,
      random: makeRandom(1),
      data,
    }),
  ).toBeNull();

  const { embedding } = initializeSimplicialSetEmbedding(graph, {
    numberOfEpochs: 100,
    numberOfComponents: 2,
    random: makeRandom(5),
    init: 'spectral',
    data,
  });

  expect(embedding).toHaveLength(size);

  for (const row of embedding) {
    for (const value of row) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(-10);
      expect(value).toBeLessThan(10);
    }
  }
});

test('a graph without any vertex falls back instead of throwing', () => {
  const graph = new SparseMatrix([], [], [], [0, 0]);

  expect(
    spectralEmbedding(graph, {
      numberOfComponents: 2,
      random: makeRandom(1),
      data: [],
    }),
  ).toBeNull();
});

test('a uniform initialization draws exactly what it drew before', () => {
  const graph = graphOf(120, 5, 3, 42);
  const { embedding, head, tail, epochsPerSample } =
    initializeSimplicialSetEmbedding(graph, {
      numberOfEpochs: 100,
      numberOfComponents: 2,
      random: makeRandom(11),
      init: 'random',
    });

  expect(embedding).toHaveLength(120);
  expect(head).toHaveLength(2178);
  expect(embedding[0]).toStrictEqual([6.053599119186401, -3.0161333084106445]);
  expect(embedding[1]).toStrictEqual([-2.0337915420532227, -4.950821399688721]);
  expect(embedding[119]).toStrictEqual([6.194940805435181, 9.693316221237183]);
  expect(epochsPerSample[0]).toBe(2.644436410158809);
  expect(head[0]).toBe(3);
  expect(tail[0]).toBe(0);
});

test('the uniform initialization is the default', () => {
  expect(DEFAULT_PARAMETERS.init).toBe('random');

  // The default and an explicit `init: 'random'` draw from the stream in the
  // very same order, so the two fits are bit identical and land on the exact
  // coordinates recorded below.
  const { X } = blobs(120, 5, 3);
  const byDefault = new UMAP({
    random: makeRandom(3),
    numberOfComponents: 2,
    numberOfEpochs: 30,
  }).fit(X);
  const explicit = new UMAP({
    random: makeRandom(3),
    numberOfComponents: 2,
    numberOfEpochs: 30,
    init: 'random',
  }).fit(X);

  expect(byDefault).toStrictEqual(explicit);
  expect(byDefault[0]).toStrictEqual([1.39966973579613, -0.5085180897830234]);
  expect(byDefault[1]).toStrictEqual([
    -6.004144574880468, 0.003439386985661404,
  ]);
});

test('a spectral fit clusters the blobs it was given', () => {
  const { X, labels } = blobs(300, 20, 5);
  const embedding = new UMAP({
    random: makeRandom(42),
    numberOfComponents: 2,
    numberOfEpochs: 100,
    init: 'spectral',
  }).fit(X);

  expect(embedding).toHaveLength(X.length);

  for (const row of embedding) {
    for (const value of row) {
      expect(Number.isFinite(value)).toBe(true);
    }
  }

  expect(clusterRatio(embedding, labels)).toBeLessThan(0.15);
});
