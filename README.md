# ml-umap

[![NPM version](https://img.shields.io/npm/v/ml-umap.svg)](https://www.npmjs.com/package/ml-umap)
[![npm download](https://img.shields.io/npm/dm/ml-umap.svg)](https://www.npmjs.com/package/ml-umap)
[![test coverage](https://img.shields.io/codecov/c/github/mljs/umap.svg)](https://codecov.io/gh/mljs/umap)
[![license](https://img.shields.io/npm/l/ml-umap.svg)](https://github.com/mljs/umap/blob/main/LICENSE)

UMAP dimensionality reduction, in TypeScript.

Derived from [umap-js](https://github.com/PAIR-code/umap-js), with model
persistence, spectral initialization and several upstream bug fixes. See
[Differences from umap-js](#differences-from-umap-js).

## Installation

```console
npm install ml-umap
```

## Usage

```js
import { UMAP } from 'ml-umap';

// Twelve samples of three measurements each, in two well separated groups.
const trainingData = [
  [1.0, 2.0, 3.0],
  [1.1, 2.2, 3.1],
  [0.9, 1.8, 2.9],
  [1.2, 2.1, 3.2],
  [0.8, 2.0, 3.1],
  [1.05, 1.95, 2.95],
  [8.0, 9.0, 7.0],
  [8.2, 9.1, 7.2],
  [7.9, 8.8, 6.9],
  [8.1, 9.3, 7.1],
  [7.8, 9.0, 7.3],
  [8.05, 8.95, 7.05],
];

const umap = new UMAP({ numberOfComponents: 2, numberOfNeighbors: 5 });
const embedding = umap.fit(trainingData);
// embedding is one [x, y] pair per input row, in the same order
```

A dataset has to hold more points than `numberOfNeighbors`, which defaults to
`15`; the twelve points above are fitted with a neighbourhood of five.

`fit` runs the whole optimization. To drive it yourself — to render progress, or
to yield to the event loop — use the stepwise API:

```js
const umap = new UMAP({ numberOfNeighbors: 5 });
const totalEpochs = umap.initializeFit(trainingData);

for (let epoch = 0; epoch < totalEpochs; epoch++) {
  umap.step();
}

const embedding = umap.getEmbedding();
```

`getEmbedding()` returns a copy. Use `getEmbeddingRef()` for the live array the
optimizer writes into, which is what you want when animating a running fit.

There is also `fitAsync(data, callback)`, which yields between epochs and stops
early if the callback returns `false`.

## Projecting new data

A fitted model can place points it has never seen into the same space:

```js
const newPoints = [
  [1.05, 2.05, 3.0],
  [8.1, 9.1, 7.2],
];

const umap = new UMAP({ numberOfNeighbors: 5 });
umap.fit(trainingData);

const projected = umap.transform(newPoints);
// one [x, y] pair per new point, next to the group it belongs to
```

`transform` is not the same as fitting the combined data. It assumes the new
points are drawn from the same distribution as the training set, and it never
moves the fitted embedding.

Projections are reproducible: the same batch, on the same fitted model, always
lands on the same coordinates — including after `toJSON` / `fromJSON`, because
the projection draws from a dedicated `transformSeed` (default `42`) instead of
from the stream the fit ran on. Pass a different one per call to get a
different projection:

```js
const projected = umap.transform(newPoints, { transformSeed: 7 });
```

### The batch is the unit of a projection

A batch is projected as a whole, so where one point lands depends on the rest of
the batch it was passed in. Projecting a 60-point batch and then its first 59
points moves those 59 by up to 8.5% of the width of the embedding — enough to
see, far too little to change the cluster any of them belongs to. Three separate
things couple the points of a batch: the optimizer draws its negative samples
for the whole batch out of one stream, the membership strengths of a row are
zeroed against the training point whose index matches the row's position, and
the edge-pruning threshold is normalised by the largest membership strength in
the batch. umap-learn has the same limitation, for the same reasons.

So project together the points you intend to compare, and do not expect a point
projected alone to match the same point projected inside a batch. The fitted
embedding itself never moves, whatever the batch holds.

## Saving and loading a model

```js
import { UMAP } from 'ml-umap';

const umap = new UMAP({ numberOfNeighbors: 5 });
umap.fit(trainingData);

const model = umap.toJSON(); // a plain object, safe to JSON.stringify
const restored = UMAP.fromJSON(model); // UMAP.load is an alias

const projected = restored.transform(newPoints);
```

The model stores the training data, the embedding, the k-nearest-neighbour
arrays, the fitted `a`/`b` curve parameters and every scalar option. The
random-projection forest and the search graph are rebuilt on load rather than
stored, which keeps the model roughly a third of the size.

A model fitted with a custom `distanceFunction` records `metric: 'custom'` and
`fromJSON` throws unless you supply the same function again, since substituting
a different one would silently produce wrong projections:

```js
const restored = UMAP.fromJSON(model, { distanceFunction: myDistance });
```

A loaded model can `transform`, but cannot be re-fitted or stepped — the fuzzy
graph a fit needs is not part of the format.

## Options

| Option                 | Default       | Meaning                                                       |
| ---------------------- | ------------- | ------------------------------------------------------------- |
| `numberOfComponents`   | `2`           | Dimensions of the output embedding                            |
| `numberOfNeighbors`    | `15`          | Local neighbourhood size                                      |
| `minimumDistance`      | `0.1`         | Minimum separation of embedded points                         |
| `spread`               | `1`           | Scale of the embedded points                                  |
| `numberOfEpochs`       | `0`           | Optimization epochs; `0` picks 200-500 from the dataset size  |
| `metric`               | `'euclidean'` | `'euclidean'` or `'cosine'`                                   |
| `distanceFunction`     | `euclidean`   | A custom distance; wins over `metric`                         |
| `init`                 | `'random'`    | `'random'` or `'spectral'`                                    |
| `random`               | seeded        | Source of randomness; pass `Math.random` for runs that differ |
| `seed`                 | `42`          | Seed of the default random source of a fit                    |
| `learningRate`         | `1`           | Initial SGD learning rate                                     |
| `negativeSampleRate`   | `5`           | Negative samples per positive edge                            |
| `repulsionStrength`    | `1`           | Weight of negative samples                                    |
| `localConnectivity`    | `1`           | Locally connected neighbours assumed per point                |
| `setOperationMixRatio` | `1`           | Blend of union and intersection for the fuzzy set             |
| `transformQueueSize`   | `4`           | Search breadth multiplier used by `transform`                 |
| `transformSeed`        | `42`          | Seed the projection of new points draws from                  |

### Choosing numberOfComponents

Two or three components is what a plot needs, and that is what the default is
for. When the embedding feeds something else — a clustering, a classifier, a
nearest-neighbour index — more room helps, but not without limit: neighbourhood
preservation, trustworthiness and continuity rise steeply and then go flat, and
the knee sits between eight and ten components on every dataset measured,
including one built with an intrinsic dimension of 20. Downstream accuracy
behaves the same way. On 20 overlapping Gaussians in 50 dimensions, leave-one-out
10-nearest-neighbour accuracy went 0.353 at two components, 0.488 at five, 0.514
at eight and 0.517 at ten, then stopped moving — against 0.542 measured on the
input data itself, with a seed-to-seed spread of 0.005 to 0.01.

| The embedding is for                        | numberOfComponents |
| ------------------------------------------- | ------------------ |
| looking at                                  | `2` or `3`         |
| clustering, a classifier, a neighbour index | `10`               |

Do not derive the value from an estimate of the intrinsic dimension of the data.
Levina-Bickel MLE, TwoNN, correlation dimension and the PCA participation ratio
all overshoot the useful number of components by three to twenty times as soon as
the data has cluster structure or noise — 25 and 29 where the quality curves had
flattened by 10 — and they follow the noise rather than the structure: on a
5-dimensional subspace of a 100-dimensional space, per-coordinate Gaussian noise
of standard deviation 0.05 takes the MLE from 4.7 to 10.1 and TwoNN from 5.5 to
17.5. A constant near ten is the better estimator. Neither umap-learn nor the
UMAP paper offers a selection rule of its own; the paper calls the effect of the
embedding dimension "largely self-evident".

### Reproducibility

Runs repeat by default. `random` is a generator seeded with `seed` (default
`42`), built fresh for each model, so fitting the same data twice gives the same
embedding. Change `seed` for a different embedding that is just as repeatable:

```js
const umap = new UMAP({ seed: 7 });
```

Pass `random` to draw from a source of your own; an unseeded one is what makes
every run differ:

```js
const umap = new UMAP({ random: Math.random });
```

`random` wins over `seed`, and `transform` uses neither — it draws from
`transformSeed`.

### Spectral initialization

`init: 'spectral'` starts from the leading eigenvectors of the normalized graph
Laplacian instead of a uniform random cloud, which preserves global structure
better and varies less between seeds. It costs roughly 1% of the fit time. It is
opt-in while its quality is evaluated on more datasets, and falls back to the
random initialization if the eigenvectors do not converge.

## Playground

`npm run dev` serves an interactive page — Vite and React, in [demo/](demo) —
that fits a dataset one animation frame at a time, so what a parameter does is
visible while it happens. It is a development tool and is not part of the
published package.

Every parameter is a field on the left, and the fit restarts a quarter of a
second after the last edit. Hovering a point reads the row behind it: the
infrared spectrum of that pill, or the four measurements of that flower, drawn
against the range of the whole dataset so two hovers can be compared.

`held out for transform` keeps one point in five or ten out of the fit
entirely. Once the fit is over, `Project N held out` runs `transform` on them
and draws each as a ring: a ring landing on the cluster of its own colour is
the projection path working, and the fitted embedding never moves while it
happens.

Two datasets are wired in:

- **Iris**, the 150 flowers of [`ml-dataset-iris`](https://github.com/mljs/dataset-iris),
  four measurements each.
- **Ecstasy pills**, 486 pills seized in north-east Switzerland, fitted as
  their mid-infrared spectra over 1024 wavenumbers and coloured by seizure.
  The spectra are published by Patiny, Zasso, Esseiva and Wist,
  [10.5281/zenodo.4120340](https://doi.org/10.5281/zenodo.4120340), CC-BY-4.0.
  What is committed in `demo/datasets/data/ecstasy.json` is the principal
  component model of them that [react-cheminfo](https://github.com/cheminfo/react-cheminfo)
  demonstrates — scores, loadings, mean and scales — and the page rebuilds the
  spectra from it on load, which keeps a 4 MB matrix out of the repository. The
  rebuilt spectra carry the standard normal variate the published processing
  applied: every one has a median within 0.06 of zero and a standard deviation
  between 0.95 and 1.06.

## Migrating from umap-js

The abbreviated option names are spelled out here, so code ported from umap-js
has to rename them. A serialized model uses the same names as the options.

| umap-js            | ml-umap                   |
| ------------------ | ------------------------- |
| `nComponents`      | `numberOfComponents`      |
| `nNeighbors`       | `numberOfNeighbors`       |
| `nEpochs`          | `numberOfEpochs`          |
| `minDist`          | `minimumDistance`         |
| `setOpMixRatio`    | `setOperationMixRatio`    |
| `distanceFn`       | `distanceFunction`        |
| `targetNNeighbors` | `targetNumberOfNeighbors` |

Every other option keeps the name it has upstream.

## Differences from umap-js

This port fixes several defects in umap-js, all of which affect `transform`:

- `heap.smallestFlagged` had an inverted loop condition, so it always returned
  `-1` and the graph-search refinement never ran.
- `initFromTree` returned instead of breaking on a leaf's padding, so only the
  first query point was ever seeded from the random-projection forest. Fixing
  both raised transform recall from 43.9% to 58.1% on uniform 32-dimensional
  data.
- `optimizeLayout` stopped on `currentEpoch === nEpochs`. Since `transform`
  runs a third of the configured epochs, any epoch count not divisible by three
  never satisfied that equality and `transform` looped forever.
- `transform` ran the optimizer at four times the reference learning rate, and
  let the fitted embedding move whenever the batch happened to be the same size
  as the training set, silently corrupting the model in place.
- `transform` drew from the model's own `random`, so it both depended on and
  advanced the stream the fit ran on: a projection could never be replayed, and
  a model reloaded from JSON — whose `random` falls back to `Math.random` —
  placed the same points somewhere else on every load. It draws from
  `transformSeed` here, as umap-learn draws from `transform_seed`.
- `computeMembershipStrengths` applied its self test ("a point is not similar
  to itself") when projecting new data, where rows index the batch and columns
  index the training set. Those are unrelated index spaces, so it zeroed the
  strength of any pair whose training index happened to equal a batch row
  position. umap-learn guards this with a `bipartite` flag.

Two behaviours also differ without a defect being involved: `random` defaults
to a generator seeded with `seed` here, so a fit repeats unless `Math.random` is
asked for, where upstream defaults to `Math.random` and no run can be replayed;
and `getEmbedding()` returns a copy, where upstream hands out the live array.

## Validated against umap-js

The port is checked against the published `umap-js@1.4.0` it derives from, not
merely eyeballed. On the current code the fit path is structurally identical to
upstream: the same edge count, the same `head` and `tail` sums, the same
`epochsPerSample` sum, the same resolved `numberOfEpochs`, and the same total
number of RNG draws. With the `a`/`b` curve parameters forced equal, the two
embeddings then agree bit for bit — 200/200 and 300/300 coordinates at n = 100,
and 1500/1500 and 2250/2250 at n = 750.

`a` and `b` are the sole divergence, and they differ from a dependency version
rather than from a change of behaviour: both implementations fit the same UMAP
curve by non-linear least squares, ml-umap through `ml-levenberg-marquardt`
5.1.0 and upstream through the 2.0.0 it bundles. The two solvers stop at
slightly different points, which shifts every coordinate downstream.

Because that comparison has been made, `umap-js` is not a devDependency here.
The A/B benchmarks import it lazily and skip the upstream arm when it is
absent, so re-checking ml-umap against a future upstream release is one install
away:

```console
npm i -D umap-js
node benchmark/fit.js
node benchmark/paired-fit.js
```

## License

[MIT](./LICENSE).

This package contains code derived from
[umap-js](https://github.com/PAIR-code/umap-js), Copyright 2019 Google LLC,
licensed under the Apache License 2.0. See [NOTICE](./NOTICE).
