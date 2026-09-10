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
  // The graph weights this is derived from pass through Math.exp and Math.log
  // in smoothKNNDistance, which ECMAScript leaves implementation-approximated,
  // so the last bits are a property of the engine build rather than of us. The
  // coordinates above are safe by contrast: they are the uniform draw, integer
  // PRNG arithmetic with no transcendental in the path.
  expect(epochsPerSample[0]).toBeCloseTo(2.644436410158809, 10);
  expect(head[0]).toBe(3);
  expect(tail[0]).toBe(0);
});

test('the uniform initialization is the default', () => {
  expect(DEFAULT_PARAMETERS.init).toBe('random');

  // The default and an explicit `init: 'random'` draw from the stream in the
  // very same order, so the two fits are bit identical to each other.
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

  // The claim is that the default and an explicit 'random' draw the same
  // initialization, which is platform independent because both fits consume the
  // identical stream. The coordinates themselves are not: the 500-epoch SGD is
  // chaotic in its inputs and Math.pow/Math.exp are implementation-approximated,
  // so a recorded point is a valid embedding of this data on one machine rather
  // than a reachable target on another (see the header of index.test.ts).
  expect(byDefault).toStrictEqual(explicit);
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

/*
 * Regression: the Lanczos step budget used to be `4 * (dim + 1) + 10`, which is
 * too short to converge the dim-th interior eigenpair once dim passes about 5.
 * The residual gate then rejected the run and the fit fell back to the uniform
 * random initialization with no signal to the caller, so `init: 'spectral'`
 * silently stopped being spectral for roughly numberOfComponents 5 to 14 —
 * exactly the range the README recommends for preprocessing. Measured before
 * the fix on a connected 300-point graph: 1.5e-5 at dim 2, but 1.5e-2 to 5.1e-2
 * at dims 6 to 10, against a gate of 1e-2. It self-healed at dim >= 15 only
 * because the budget grew with dim, which is what identified the budget rather
 * than the algorithm as the cause.
 */
test('eigenpairs converge at every embedding dimension a user might ask for', () => {
  const csr = toCSR(graphOf(300, 8, 1));

  expect(connectedComponents(csr).count).toBe(1);

  // 6, 10 and 14 are inside the range the short budget used to fail on; 2 and
  // 20 bracket it. One seed each keeps the run inside the coverage timeout.
  for (const dim of [2, 6, 10, 14, 20]) {
    const pairs = spectralVectors(csr, dim, { random: makeRandom(11) });

    expect(pairs).toHaveLength(dim);

    for (const pair of pairs) {
      expect(pair.residual).toBeLessThan(RESIDUAL_BOUND);
    }
  }
}, 30_000);

test('spectral initialization is used, not silently skipped, at ten dimensions', () => {
  const { X } = blobs(300, 8, 1);
  const graph = graphOfData(X, 1);

  const embedding = spectralEmbedding(graph, {
    numberOfComponents: 10,
    random: makeRandom(11),
    data: X,
  });

  expect(embedding).not.toBeNull();
  expect(embedding).toHaveLength(300);
  expect(embedding?.[0]).toHaveLength(10);
});
