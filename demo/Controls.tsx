/*
 * The parameter form. It edits a plain object and hands it back whole, so a
 * run is always started from one consistent set of parameters.
 */

import { Field, SelectField } from './Field.tsx';
import type { DemoDataset } from './datasets/index.ts';
import type { DemoParameters, NumericParameter } from './parameters.ts';
import { INIT_OPTIONS, METRIC_OPTIONS, NUMBER_FIELDS } from './parameters.ts';

/** What the form needs to draw itself and report a change. */
export interface ControlsProps {
  /** The datasets the picker lists. */
  datasets: readonly DemoDataset[];
  /** The dataset currently picked. */
  dataset: DemoDataset;
  /** Called with the key of the dataset the user picked. */
  onDataset: (key: string) => void;
  /** The parameters currently set. */
  parameters: DemoParameters;
  /** Called with the parameters after an edit. */
  onParameters: (parameters: DemoParameters) => void;
  /** One row in this many is held out and projected; `0` holds out none. */
  holdOutEvery: number;
  /** Called with the hold-out rate the user picked. */
  onHoldOutEvery: (holdOutEvery: number) => void;
  /** Whether a fit is running. */
  isFitting: boolean;
  /** How many rows were kept out of the fit. */
  heldOutCount: number;
  /** How many of those have been projected so far. */
  projectedCount: number;
  /** Whether the held out rows can be projected right now. */
  canProject: boolean;
  /** Projects the held out rows into the fitted embedding. */
  onProject: () => void;
  /** Starts a run. */
  onRun: () => void;
  /** Stops the running fit. */
  onStop: () => void;
  /** Puts every parameter back to its default. */
  onReset: () => void;
}

/**
 * The left hand form: the dataset, every parameter, and the run buttons.
 * @param props - What to show and where to report to.
 * @returns The form.
 */
export function Controls(props: ControlsProps) {
  const {
    datasets,
    dataset,
    onDataset,
    parameters,
    onParameters,
    holdOutEvery,
    onHoldOutEvery,
    isFitting,
    heldOutCount,
    projectedCount,
    canProject,
    onProject,
    onRun,
    onStop,
    onReset,
  } = props;

  function setNumber(key: NumericParameter, value: number) {
    if (Number.isNaN(value)) return;
    onParameters({ ...parameters, [key]: value });
  }

  return (
    <form className="controls" onSubmit={(event) => event.preventDefault()}>
      <SelectField
        label="dataset"
        value={dataset.key}
        options={datasets.map((candidate) => ({
          value: candidate.key,
          label: candidate.name,
        }))}
        onChange={onDataset}
        help={dataset.description}
      />

      <div className="buttons">
        <button type="button" className="primary" onClick={onRun}>
          {isFitting ? 'Restart' : 'Fit'}
        </button>
        <button type="button" onClick={onStop} disabled={!isFitting}>
          Stop
        </button>
        <button type="button" onClick={onReset}>
          Defaults
        </button>
      </div>

      <SelectField
        label="metric"
        value={parameters.metric}
        options={METRIC_OPTIONS}
        onChange={(value) =>
          onParameters({
            ...parameters,
            metric: value as DemoParameters['metric'],
          })
        }
      />

      <SelectField
        label="init"
        value={parameters.init}
        options={INIT_OPTIONS}
        onChange={(value) =>
          onParameters({ ...parameters, init: value as DemoParameters['init'] })
        }
      />

      {NUMBER_FIELDS.map((field) => (
        <Field key={field.key} label={field.key} help={field.help}>
          <input
            type="number"
            min={field.min}
            step={field.step}
            value={parameters[field.key]}
            onChange={(event) =>
              setNumber(field.key, event.target.valueAsNumber)
            }
          />
        </Field>
      ))}

      <SelectField
        label="held out for transform"
        value={holdOutEvery}
        options={[
          { value: 0, label: 'none' },
          { value: 10, label: 'every 10th point' },
          { value: 5, label: 'every 5th point' },
        ]}
        onChange={(value) => onHoldOutEvery(Number(value))}
        help="Held out rows take no part in the fit. Add them afterwards with the button below; they are drawn as rings."
      />

      <button type="button" onClick={onProject} disabled={!canProject}>
        {heldOutCount === 0
          ? 'Nothing held out'
          : `Project ${heldOutCount} held out`}
      </button>
      <span className="field-help">
        {projectedCount > 0
          ? `${projectedCount} points added by transform, the fitted embedding untouched.`
          : 'Fit first, then add the held out points to the embedding.'}
      </span>
    </form>
  );
}
