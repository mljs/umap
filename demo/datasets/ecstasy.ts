/*
 * The seized ecstasy pills. What is committed is the published principal
 * component model — scores, loadings, the mean and the scales — and the
 * spectra are rebuilt from it here, so the page fits and draws the same 1024
 * point spectra without a four megabyte matrix in the repository.
 */

import ecstasy from './data/ecstasy.json' with { type: 'json' };
import type { DemoDataset } from './types.ts';

/**
 * Builds the ecstasy dataset, spectra included.
 * @returns The 486 pills, one reconstructed spectrum each.
 */
export function makeEcstasy(): DemoDataset {
  const { samples, categories, source, groupLabel, variables } = ecstasy;
  const spectra = reconstruct();

  const labels: string[] = new Array(samples.length);
  const names: string[] = new Array(samples.length);
  const colors: Record<string, string> = {};
  for (let row = 0; row < samples.length; row++) {
    const sample = samples[row];
    if (sample === undefined) continue;
    labels[row] = sample.category;
    names[row] = `${sample.id} — ${sample.category}`;
    colors[sample.category] = sample.color;
  }

  return {
    key: 'ecstasy',
    name: 'Ecstasy pills',
    description: `${spectra.length} seized pills, as their mid-infrared spectra over ${variables.numberOfPoints} wavenumbers, in ${categories.length} seizures.`,
    credit: { text: source.title, url: source.url },
    X: spectra,
    names,
    labels,
    categories,
    colors,
    groupLabel,
    numberOfNeighbors: 10,
    profile: {
      kind: 'line',
      x: wavenumbers(),
      columns: [],
      rows: spectra,
      xLabel: `${variables.label} (${variables.unit})`,
      yLabel: variables.valueLabel,
      reversed: variables.direction === 'descending',
    },
  };
}

/**
 * Rebuilds every spectrum from the principal component model, which is the
 * inverse of what `ml-pca` did to them: the scores back through the loadings,
 * then the scaling and the centring undone.
 * @returns One spectrum per pill.
 */
function reconstruct(): number[][] {
  const { scores, loadings, mean, scales } = ecstasy;
  const columns = mean.length;
  const spectra: number[][] = new Array(scores.length);

  for (let row = 0; row < scores.length; row++) {
    const score = scores[row] ?? [];
    const spectrum: number[] = new Array(columns);

    for (let column = 0; column < columns; column++) {
      let value = 0;
      for (let component = 0; component < score.length; component++) {
        value += (score[component] ?? 0) * (loadings[component]?.[column] ?? 0);
      }
      spectrum[column] = value * (scales[column] ?? 1) + (mean[column] ?? 0);
    }

    spectra[row] = spectrum;
  }

  return spectra;
}

/**
 * The wavenumbers the spectra were resampled onto. They are evenly spaced, to
 * within 5e-5 cm⁻¹ of the published axis, so the range is stored instead of
 * the thousand values.
 * @returns The axis, ascending.
 */
function wavenumbers(): number[] {
  const { from, to, numberOfPoints } = ecstasy.variables;
  const step = (to - from) / (numberOfPoints - 1);
  const values: number[] = new Array(numberOfPoints);
  for (let i = 0; i < numberOfPoints; i++) values[i] = from + i * step;
  return values;
}
