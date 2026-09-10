// Paired, interleaved A/B of fit() — ml-umap versus the published umap-js.
// benchmark.js runs one case to completion before the next, so drift between the two
// blocks biases the ratio. Here each round times both implementations back to back and
// alternates which goes first, so drift cancels. Run: node benchmark/paired-fit.js
// The umap-js arm is optional (see ./upstream.js): without it installed the rounds
// still run and report the ml-umap timings, with no ratio to report.
import { XSadd } from 'ml-xsadd';

import { UMAP } from '../src/index.ts';

import { loadUpstreamUmap } from './upstream.js';

const N = Number(process.env.N ?? 750);
const D = Number(process.env.D ?? 20);
const ROUNDS = Number(process.env.ROUNDS ?? 15);
const CLUSTERS = 5;

const UpstreamUMAP = await loadUpstreamUmap();

function makeData(count, seed) {
  const random = new XSadd(seed).random;
  const data = [];
  for (let i = 0; i < count; i++) {
    const cluster = i % CLUSTERS;
    const row = [];
    for (let j = 0; j < D; j++) row.push(random() * 2 + cluster * 5);
    data.push(row);
  }
  return data;
}

const data = makeData(N, 1);

// umap-js still spells the option `nComponents`; ml-umap spells it out.
function timeFit(Ctor) {
  const random = new XSadd(42).random;
  const umap =
    Ctor === UMAP
      ? new Ctor({ random, numberOfComponents: 2 })
      : new Ctor({ random, nComponents: 2 });
  const start = performance.now();
  umap.fit(data);
  return performance.now() - start;
}

// warm both up so JIT state is comparable
for (let i = 0; i < 3; i++) {
  timeFit(UMAP);
  if (UpstreamUMAP) timeFit(UpstreamUMAP);
}

const ours = [];
const theirs = [];
for (let round = 0; round < ROUNDS; round++) {
  if (!UpstreamUMAP) {
    ours.push(timeFit(UMAP));
  } else if (round % 2 === 0) {
    ours.push(timeFit(UMAP));
    theirs.push(timeFit(UpstreamUMAP));
  } else {
    theirs.push(timeFit(UpstreamUMAP));
    ours.push(timeFit(UMAP));
  }
}

function median(values) {
  const sorted = [...values].toSorted((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

console.log(
  `# paired fit A/B — N=${N}, D=${D}, ${ROUNDS} interleaved rounds, node ${process.version}\n`,
);
console.log(
  `  ml-umap  median ${median(ours).toFixed(1)} ms   min ${Math.min(...ours).toFixed(1)}   max ${Math.max(...ours).toFixed(1)}`,
);

if (!UpstreamUMAP) {
  console.log('\n  no upstream arm: install umap-js to get the paired ratio.');
} else {
  const ratios = ours.map((value, index) => theirs[index] / value);

  console.log(
    `  umap-js  median ${median(theirs).toFixed(1)} ms   min ${Math.min(...theirs).toFixed(1)}   max ${Math.max(...theirs).toFixed(1)}`,
  );
  console.log(`\n  per-round ratio (umap-js / ml-umap):`);
  console.log(
    `    median ${median(ratios).toFixed(3)}x   min ${Math.min(...ratios).toFixed(3)}x   max ${Math.max(...ratios).toFixed(3)}x`,
  );
  console.log(`    all: ${ratios.map((r) => r.toFixed(2)).join(' ')}`);
}
