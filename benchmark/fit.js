// A/B benchmark: ml-umap versus the published umap-js it is ported from.
// Run manually: node benchmark/fit.js
// Both variants run in ONE process, on the SAME data, with the SAME seeded PRNG.
// Imports the TypeScript sources so the benchmark can never measure a stale lib/.
// The umap-js arm is optional (see ./upstream.js): without it installed only the
// ml-umap arm runs, and the equivalence check is skipped.
import BenchmarkModule from 'benchmark';
import { XSadd } from 'ml-xsadd';

import { UMAP } from '../src/index.ts';

import { loadUpstreamUmap } from './upstream.js';

const Benchmark = BenchmarkModule.default ?? BenchmarkModule;
const options = { minSamples: 30, maxTime: 30 };

const N = Number(process.env.N ?? 750);
const D = Number(process.env.D ?? 20);
const CLUSTERS = 5;
const N_NEW = Number(process.env.N_NEW ?? 100);

const UpstreamUMAP = await loadUpstreamUmap();

function makeData(count, seed) {
  const random = new XSadd(seed).random;
  const data = [];
  for (let i = 0; i < count; i++) {
    const cluster = i % CLUSTERS;
    const row = [];
    for (let j = 0; j < D; j++) {
      row.push(random() * 2 + cluster * 5);
    }
    data.push(row);
  }
  return data;
}

const data = makeData(N, 1);
const fresh = makeData(N_NEW, 99);

// umap-js still spells the option `nComponents`; ml-umap spells it out.
function makeUmap(Ctor) {
  const random = new XSadd(42).random;
  return Ctor === UMAP
    ? new Ctor({ random, numberOfComponents: 2 })
    : new Ctor({ random, nComponents: 2 });
}

function fitWith(Ctor) {
  return makeUmap(Ctor).fit(data);
}

function fittedModel(Ctor) {
  const umap = makeUmap(Ctor);
  umap.fit(data);
  return umap;
}

function checksum(embedding) {
  let total = 0;
  for (const row of embedding) {
    for (const value of row) total += value;
  }
  return total;
}

const ours = fitWith(UMAP);

console.log(
  `# ml-umap${UpstreamUMAP ? ' vs umap-js' : ''} — N=${N}, D=${D}, ${CLUSTERS} clusters, transform ${N_NEW} new points`,
);
console.log(`# node ${process.version}\n`);
console.log('## equivalence');
console.log(`  ours   checksum = ${checksum(ours).toFixed(9)}`);

if (UpstreamUMAP) {
  const theirs = fitWith(UpstreamUMAP);
  let exact = 0;
  let maxAbs = 0;
  for (let i = 0; i < theirs.length; i++) {
    const a = ours[i];
    const b = theirs[i];
    for (let j = 0; j < b.length; j++) {
      if (Object.is(a[j], b[j])) exact++;
      const d = Math.abs(a[j] - b[j]);
      if (d > maxAbs) maxAbs = d;
    }
  }

  console.log(`  theirs checksum = ${checksum(theirs).toFixed(9)}`);
  console.log(
    `  ${exact}/${theirs.length * 2} bit-exact, maxAbs = ${maxAbs.toExponential(3)}\n`,
  );
} else {
  console.log('  no upstream arm to compare against\n');
}

const ourModel = fittedModel(UMAP);

const suite = new Benchmark.Suite();
suite.add('fit::ml-umap', () => fitWith(UMAP), options);
suite.add('transform::ml-umap', () => ourModel.transform(fresh), options);

if (UpstreamUMAP) {
  const theirModel = fittedModel(UpstreamUMAP);
  suite.add('fit::umap-js', () => fitWith(UpstreamUMAP), options);
  suite.add('transform::umap-js', () => theirModel.transform(fresh), options);
}

const results = [];
suite.on('cycle', (event) => {
  const bench = event.target;
  results.push({
    name: String(bench.name),
    ms: (1 / bench.hz) * 1000,
    rme: bench.stats.rme,
    samples: bench.stats.sample.length,
  });
});
suite.run({ async: false });

console.log('## timings');
console.log('   case                    ms/op      RME     n   ratio');
for (const group of ['fit', 'transform']) {
  const rows = results.filter((r) => r.name.startsWith(`${group}::`));
  const reference = rows.find((r) => r.name.endsWith('umap-js'));
  for (const row of rows) {
    const ratio = reference
      ? `${(reference.ms / row.ms).toFixed(2)}x`
      : 'n/a (no upstream)';
    console.log(
      `   ${row.name.padEnd(22)} ${row.ms.toFixed(2).padStart(8)}  ±${row.rme.toFixed(1).padStart(4)}%  ${String(row.samples).padStart(3)}  ${ratio}`,
    );
  }
}
