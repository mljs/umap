// A/B benchmark for the metrics migration: does delegating euclidean to
// ml-distance cost anything versus the hand-written loop it replaces, and what
// would delegating cosine cost on top of the zero-norm guards umap-learn needs?
// Run manually: node benchmark/metrics.js
// Every variant runs in ONE process on the SAME data. Each one is its own
// top-level function literal, passed to benchmark.js directly rather than
// through a shared wrapper, so the cases cannot pollute each other's inline
// caches — routing them through one `(pairs) => sweep(fn, pairs)` arrow made two
// copies of the same loop read 46% apart.
import BenchmarkModule from 'benchmark';
import { distance, similarity } from 'ml-distance';
import { XSadd } from 'ml-xsadd';

import { cosine, euclidean } from '../src/metrics.ts';

const Benchmark = BenchmarkModule.default ?? BenchmarkModule;
const options = {
  minSamples: 30,
  maxTime: Number(process.env.MAXTIME ?? 5),
};
const WARMUP_SWEEPS = Number(process.env.WARMUP ?? 3000);

const mlEuclidean = distance.euclidean;
const mlCosineSimilarity = similarity.cosine;

const PAIRS = Number(process.env.PAIRS ?? 512);
const DIMS = (process.env.DIMS ?? '10,50,200').split(',').map(Number);

let pairs = [];
let sink = 0;

function makePairs(dim, seed) {
  const random = new XSadd(seed).random;
  const built = [];
  for (let i = 0; i < PAIRS; i++) {
    const x = [];
    const y = [];
    for (let j = 0; j < dim; j++) {
      x.push(random() * 2 - 1);
      y.push(random() * 2 - 1);
    }
    built.push([x, y]);
  }
  return built;
}

// --- variants ---------------------------------------------------------------

// The pre-migration ml-umap euclidean: one length check per call.
function euclideanHandWritten(x, y) {
  if (y.length < x.length) {
    throw new RangeError('euclidean: vectors must have the same length');
  }
  let result = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - y[i];
    result += d * d;
  }
  return Math.sqrt(result);
}

// The migrated ml-umap euclidean: the same single length check, then a
// delegated call. Copied here so it is compared against the hand-written loop
// inside one process, with its own inline caches.
function euclideanWrapped(x, y) {
  if (y.length < x.length) {
    throw new RangeError('euclidean: vectors must have the same length');
  }
  return mlEuclidean(x, y);
}

// What umap-js shipped: the length check inside the loop, and `** 2`.
function euclideanPerElementGuard(x, y) {
  let result = 0;
  for (let i = 0; i < x.length; i++) {
    if (x.length !== y.length) {
      throw new RangeError('euclidean: vectors must have the same length');
    }
    result += (x[i] - y[i]) ** 2;
  }
  return Math.sqrt(result);
}

// The shipped ml-umap cosine, copied so it keeps its own inline caches.
function cosineHandWritten(x, y) {
  if (y.length < x.length) {
    throw new RangeError('cosine: vectors must have the same length');
  }
  let result = 0;
  let normX = 0;
  let normY = 0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = y[i];
    result += xi * yi;
    normX += xi * xi;
    normY += yi * yi;
  }
  if (normX === 0 && normY === 0) return 0;
  if (normX === 0 || normY === 0) return 1;
  return 1 - result / Math.sqrt(normX * normY);
}

// Delegating cosine forces extra passes: the zero-norm guards need the norms,
// which ml-distance's similarity does not return.
function isZeroVector(v) {
  for (let i = 0; i < v.length; i++) {
    if (v[i] !== 0) return false;
  }
  return true;
}

function cosineDelegated(x, y) {
  if (y.length < x.length) {
    throw new RangeError('cosine: vectors must have the same length');
  }
  const zeroX = isZeroVector(x);
  const zeroY = isZeroVector(y);
  if (zeroX && zeroY) return 0;
  if (zeroX || zeroY) return 1;
  return 1 - mlCosineSimilarity(x, y);
}

// --- one sweep literal per variant, called by benchmark.js directly ---------

function sweepEuclideanHandWritten() {
  let total = 0;
  for (let i = 0; i < pairs.length; i++) {
    total += euclideanHandWritten(pairs[i][0], pairs[i][1]);
  }
  sink = total;
}

function sweepEuclideanMlDistance() {
  let total = 0;
  for (let i = 0; i < pairs.length; i++) {
    total += mlEuclidean(pairs[i][0], pairs[i][1]);
  }
  sink = total;
}

function sweepEuclideanWrapped() {
  let total = 0;
  for (let i = 0; i < pairs.length; i++) {
    total += euclideanWrapped(pairs[i][0], pairs[i][1]);
  }
  sink = total;
}

function sweepEuclideanPerElementGuard() {
  let total = 0;
  for (let i = 0; i < pairs.length; i++) {
    total += euclideanPerElementGuard(pairs[i][0], pairs[i][1]);
  }
  sink = total;
}

function sweepEuclideanExport() {
  let total = 0;
  for (let i = 0; i < pairs.length; i++) {
    total += euclidean(pairs[i][0], pairs[i][1]);
  }
  sink = total;
}

function sweepCosineHandWritten() {
  let total = 0;
  for (let i = 0; i < pairs.length; i++) {
    total += cosineHandWritten(pairs[i][0], pairs[i][1]);
  }
  sink = total;
}

function sweepCosineDelegated() {
  let total = 0;
  for (let i = 0; i < pairs.length; i++) {
    total += cosineDelegated(pairs[i][0], pairs[i][1]);
  }
  sink = total;
}

function sweepCosineExport() {
  let total = 0;
  for (let i = 0; i < pairs.length; i++) {
    total += cosine(pairs[i][0], pairs[i][1]);
  }
  sink = total;
}

function sweepCalibration() {
  let total = 0;
  for (let i = 0; i < pairs.length; i++) {
    total += euclideanHandWritten(pairs[i][0], pairs[i][1]);
  }
  sink = total;
}

// benchmark.js calibrates its clock on the first case of a suite, which lands a
// warm-up penalty on whatever is listed first; this throwaway absorbs it.
const CASES = [
  ['-- calibration, ignore --       ', sweepCalibration],
  ['euclidean  ml-umap hand-written ', sweepEuclideanHandWritten],
  ['euclidean  ml-distance raw      ', sweepEuclideanMlDistance],
  ['euclidean  check + ml-distance  ', sweepEuclideanWrapped],
  ['euclidean  src/metrics.ts export', sweepEuclideanExport],
  ['euclidean  per-element guard    ', sweepEuclideanPerElementGuard],
  ['cosine     ml-umap grouped      ', sweepCosineHandWritten],
  ['cosine     ml-distance + guards ', sweepCosineDelegated],
  ['cosine     src/metrics.ts export', sweepCosineExport],
];

console.log(
  `# metrics A/B — ${PAIRS} vector pairs per sweep, node ${process.version}\n`,
);

for (const dim of DIMS) {
  pairs = makePairs(dim, dim * 31 + 7);
  console.log(`## d=${dim}`);
  for (const [name, sweep] of CASES) {
    sweep();
    console.log(`   ${name} sum=${sink.toFixed(12)}`);
  }

  // Warm every case to its optimized tier before any of them is timed,
  // otherwise whichever case benchmark.js runs first pays for the tier-up.
  for (let round = 0; round < WARMUP_SWEEPS; round++) {
    for (const [, sweep] of CASES) sweep();
  }

  const suite = new Benchmark.Suite();
  for (const [name, sweep] of CASES) {
    suite.add(name, sweep, options);
  }
  suite
    .on('cycle', (event) => {
      const bench = event.target;
      const nsPerCall = (bench.stats.mean / PAIRS) * 1e9;
      console.log(
        `   ${bench.name} ${nsPerCall.toFixed(2)} ns/call  ±${bench.stats.rme.toFixed(2)}%  (${bench.stats.sample.length} samples)`,
      );
    })
    .run();
  console.log('');
}
