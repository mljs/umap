/*
 * The playground: pick a dataset, move a parameter, watch the fit converge.
 */

import { useEffect, useState } from 'react';

import { Controls } from './Controls.tsx';
import { Profile } from './Profile.tsx';
import type { PointOrigin } from './Scatter.tsx';
import { Scatter } from './Scatter.tsx';
import { DATASETS } from './datasets/index.ts';
import type { DemoParameters } from './parameters.ts';
import { DEFAULT_PARAMETERS } from './parameters.ts';
import { useUmapRun } from './useUmapRun.ts';

const FIRST_DATASET = DATASETS[0] ?? null;

/** How long an edit is left alone before the fit restarts on it. */
const SETTLE_MILLISECONDS = 250;

/**
 * The whole page.
 * @returns The playground.
 */
export function App() {
  const [datasetKey, setDatasetKey] = useState(FIRST_DATASET?.key ?? '');
  const [parameters, setParameters] = useState<DemoParameters>(
    withDatasetDefaults(FIRST_DATASET?.numberOfNeighbors),
  );
  const [holdOutEvery, setHoldOutEvery] = useState(0);
  const [hovered, setHovered] = useState<{
    row: number;
    origin: PointOrigin;
  } | null>(null);
  const { state, run, project, stop } = useUmapRun();

  const dataset =
    DATASETS.find((candidate) => candidate.key === datasetKey) ?? FIRST_DATASET;

  // The fit follows the form: an edit restarts it, once the edit has settled.
  useEffect(() => {
    if (dataset === null) return undefined;
    const timer = setTimeout(() => {
      run(dataset, parameters, holdOutEvery);
    }, SETTLE_MILLISECONDS);
    return () => clearTimeout(timer);
  }, [dataset, parameters, holdOutEvery, run]);

  if (dataset === null) return <p>No dataset.</p>;

  return (
    <main className="page">
      <header className="header">
        <h1>ml-umap</h1>
        <p>
          Fit a dataset one animation frame at a time, and see what every
          parameter does to it.
        </p>
      </header>

      <section className="sidebar">
        <Controls
          datasets={DATASETS}
          dataset={dataset}
          onDataset={(key) => {
            const next = DATASETS.find((candidate) => candidate.key === key);
            setDatasetKey(key);
            setParameters(withDatasetDefaults(next?.numberOfNeighbors));
            // A row index means nothing in another dataset.
            setHovered(null);
          }}
          parameters={parameters}
          onParameters={setParameters}
          holdOutEvery={holdOutEvery}
          onHoldOutEvery={setHoldOutEvery}
          isFitting={state.status === 'fitting'}
          heldOutCount={state.heldRows.length}
          projectedCount={state.projected.length}
          canProject={state.status === 'fitted' && state.heldRows.length > 0}
          onProject={project}
          onRun={() => run(dataset, parameters, holdOutEvery)}
          onStop={stop}
          onReset={() =>
            setParameters(withDatasetDefaults(dataset.numberOfNeighbors))
          }
        />
      </section>

      <section className="plot">
        <Scatter
          dataset={dataset}
          state={state}
          hoveredRow={hovered?.row ?? null}
          onHover={(row, origin) =>
            setHovered(row === null || origin === null ? null : { row, origin })
          }
        />
        <Profile
          dataset={dataset}
          row={hovered?.row ?? null}
          origin={hovered?.origin ?? null}
        />
      </section>

      <footer className="status">
        <span>
          epoch {state.epoch} / {state.totalEpochs}
        </span>
        <span>fit {Math.round(state.fitMilliseconds)} ms</span>
        <span>
          {state.heldRows.length} held out, {state.projected.length} projected
          {state.projected.length > 0
            ? ` in ${Math.round(state.transformMilliseconds)} ms`
            : ''}
        </span>
        <span>
          {dataset.X.length} points, {dataset.X[0]?.length ?? 0} dimensions
        </span>
        {dataset.credit === null ? null : (
          <a href={dataset.credit.url} target="_blank" rel="noreferrer">
            {dataset.credit.text}
          </a>
        )}
        {state.message === null ? null : (
          <span className="error">{state.message}</span>
        )}
      </footer>
    </main>
  );
}

/**
 * The default parameters, with the neighbourhood the dataset is worth starting
 * from.
 * @param numberOfNeighbors - That neighbourhood, when a dataset is picked.
 * @returns The parameters to start from.
 */
function withDatasetDefaults(
  numberOfNeighbors: number | undefined,
): DemoParameters {
  return {
    ...DEFAULT_PARAMETERS,
    numberOfNeighbors:
      numberOfNeighbors ?? DEFAULT_PARAMETERS.numberOfNeighbors,
  };
}
