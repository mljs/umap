/*
 * The iris flowers: four measurements, small enough to fit instantly, and the
 * one dataset every reader already knows the answer to.
 */

import { getClasses, getNumbers } from 'ml-dataset-iris';

import type { DemoDataset } from './types.ts';

const COLORS: Record<string, string> = {
  setosa: '#4c72b0',
  versicolor: '#dd8452',
  virginica: '#55a868',
};

const COLUMNS = ['sepal length', 'sepal width', 'petal length', 'petal width'];

/**
 * Builds the iris dataset.
 * @returns The 150 flowers, four measurements each.
 */
export function makeIris(): DemoDataset {
  const X = getNumbers();
  const labels = getClasses();
  const names: string[] = new Array(X.length);
  for (let row = 0; row < X.length; row++) {
    names[row] = `flower ${row + 1} (${labels[row] ?? '?'})`;
  }

  return {
    key: 'iris',
    name: 'Iris',
    description:
      '150 flowers, four measurements each (sepal and petal, length and width).',
    credit: {
      text: 'ml-dataset-iris',
      url: 'https://github.com/mljs/dataset-iris',
    },
    X,
    names,
    labels,
    categories: ['setosa', 'versicolor', 'virginica'],
    colors: COLORS,
    groupLabel: 'Species',
    numberOfNeighbors: 15,
    profile: {
      kind: 'bars',
      x: [0, 1, 2, 3],
      columns: COLUMNS,
      rows: X,
      xLabel: 'Measurement',
      yLabel: 'cm',
      reversed: false,
    },
  };
}
