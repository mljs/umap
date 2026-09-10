/*
 * The playground drives the stepwise API rather than `fit`, so the embedding
 * can be watched while it converges, and it projects the held out rows only
 * when asked: adding data the model never saw is the thing being demonstrated,
 * so it is a button rather than something that happens on its own.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import type { UMAPParameters, Vectors } from '../src/index.ts';
import { UMAP } from '../src/index.ts';

import type { DemoDataset } from './datasets/index.ts';

/** Everything the page shows about one run. */
export interface RunState {
  /** Where the run is. */
  status: 'idle' | 'fitting' | 'fitted' | 'error';
  /** Epochs completed so far. */
  epoch: number;
  /** Epochs the fit will run, as `initializeFit` resolved them. */
  totalEpochs: number;
  /** Time spent in `initializeFit` and `step`, without the frame waits. */
  fitMilliseconds: number;
  /** Time the last `transform` took. */
  transformMilliseconds: number;
  /** The embedding of the fitted rows. */
  fitted: Vectors;
  /** Index in the dataset of each fitted row. */
  fittedRows: number[];
  /** Index of every row kept out of the fit. */
  heldRows: number[];
  /** Where the held out rows were projected, once they have been. */
  projected: Vectors;
  /** Index in the dataset of each projected row. */
  projectedRows: number[];
  /** What went wrong, when something did. */
  message: string | null;
}

const IDLE: RunState = {
  status: 'idle',
  epoch: 0,
  totalEpochs: 0,
  fitMilliseconds: 0,
  transformMilliseconds: 0,
  fitted: [],
  fittedRows: [],
  heldRows: [],
  projected: [],
  projectedRows: [],
  message: null,
};

/** How many frames an animated fit is spread over. */
const FRAMES_PER_FIT = 120;

/** The fitted model, kept so the held out rows can be added to it later. */
interface FittedModel {
  umap: UMAP;
  held: number[][];
  heldRows: number[];
}

/**
 * Runs a fit one animation frame at a time, and projects the held out rows on
 * demand.
 * @returns The state of the current run, and the three controls of it.
 */
export function useUmapRun(): {
  state: RunState;
  run: (
    dataset: DemoDataset,
    parameters: UMAPParameters,
    holdOutEvery: number,
  ) => void;
  project: () => void;
  stop: () => void;
} {
  const [state, setState] = useState<RunState>(IDLE);
  const frame = useRef<number | null>(null);
  const model = useRef<FittedModel | null>(null);

  const stop = useCallback(() => {
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = null;
  }, []);

  useEffect(() => stop, [stop]);

  const run = useCallback(
    (
      dataset: DemoDataset,
      parameters: UMAPParameters,
      holdOutEvery: number,
    ) => {
      stop();
      model.current = null;

      const split = splitRows(dataset.X, holdOutEvery);
      let umap: UMAP;
      let totalEpochs: number;
      let elapsed = performance.now();
      try {
        umap = new UMAP({ ...parameters, numberOfComponents: 2 });
        totalEpochs = umap.initializeFit(split.train);
      } catch (error) {
        setState({ ...IDLE, status: 'error', message: describe(error) });
        return;
      }
      elapsed = performance.now() - elapsed;

      const perFrame = Math.max(1, Math.ceil(totalEpochs / FRAMES_PER_FIT));
      setState({
        ...IDLE,
        status: 'fitting',
        totalEpochs,
        fitMilliseconds: elapsed,
        fittedRows: split.trainRows,
        heldRows: split.heldRows,
      });

      const tick = () => {
        let epoch = 0;
        let fitted: Vectors;
        try {
          const started = performance.now();
          for (let i = 0; i < perFrame; i++) epoch = umap.step();
          elapsed += performance.now() - started;
          fitted = umap.getEmbedding();
        } catch (error) {
          frame.current = null;
          setState((previous) => ({
            ...previous,
            status: 'error',
            message: describe(error),
          }));
          return;
        }

        const done = epoch >= totalEpochs;
        if (done) {
          frame.current = null;
          model.current = { umap, held: split.held, heldRows: split.heldRows };
        } else {
          frame.current = requestAnimationFrame(tick);
        }

        setState((previous) => ({
          ...previous,
          status: done ? 'fitted' : 'fitting',
          epoch,
          fitted,
          fitMilliseconds: elapsed,
        }));
      };

      frame.current = requestAnimationFrame(tick);
    },
    [stop],
  );

  const project = useCallback(() => {
    const current = model.current;
    if (current === null || current.held.length === 0) return;

    const started = performance.now();
    let projected: Vectors;
    try {
      projected = current.umap.transform(current.held);
    } catch (error) {
      setState((previous) => ({
        ...previous,
        status: 'error',
        message: describe(error),
      }));
      return;
    }

    setState((previous) => ({
      ...previous,
      projected,
      projectedRows: current.heldRows,
      transformMilliseconds: performance.now() - started,
      message: null,
    }));
  }, []);

  return { state, run, project, stop };
}

/**
 * Splits the rows into the ones the model is fitted on and the ones kept back
 * for `transform`.
 * @param X - The whole dataset, one array per row.
 * @param holdOutEvery - Hold out one row every this many; `0` holds out none.
 * @returns The two sets of rows, and where each came from.
 */
function splitRows(
  X: number[][],
  holdOutEvery: number,
): {
  train: number[][];
  trainRows: number[];
  held: number[][];
  heldRows: number[];
} {
  const train: number[][] = [];
  const trainRows: number[] = [];
  const held: number[][] = [];
  const heldRows: number[] = [];

  for (let row = 0; row < X.length; row++) {
    const values = X[row];
    if (values === undefined) continue;
    if (holdOutEvery > 0 && row % holdOutEvery === 0) {
      held.push(values);
      heldRows.push(row);
    } else {
      train.push(values);
      trainRows.push(row);
    }
  }

  return { train, trainRows, held, heldRows };
}

/**
 * Reads the message of whatever was thrown.
 * @param error - The thrown value.
 * @returns Its message.
 */
function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
