/*
 * What the playground needs of a dataset: the matrix it fits, and the raw
 * measurement behind every row, so a point can be read as the thing it stands
 * for rather than as a dot.
 */

/** The measurement behind the rows of a dataset, as the hover draws it. */
export interface DemoProfile {
  /** A spectrum is a line, a handful of features are bars. */
  kind: 'line' | 'bars';
  /** Position of each column on the x axis, for a line. */
  x: number[];
  /** Name of each column, for bars. */
  columns: string[];
  /** The values, one row per sample, in the row order of the dataset. */
  rows: number[][];
  /** What the x axis is. */
  xLabel: string;
  /** What the y axis is. */
  yLabel: string;
  /** Whether x is drawn from high to low, as an infrared spectrum is. */
  reversed: boolean;
}

/** A dataset the playground can fit, with what it takes to draw it. */
export interface DemoDataset {
  /** Identifier of the dataset in the picker. */
  key: string;
  /** Name shown in the picker. */
  name: string;
  /** One line on what the points are. */
  description: string;
  /** Where the data comes from, when it is published. */
  credit: { text: string; url: string } | null;
  /** The points, one array per sample: what is actually fitted. */
  X: number[][];
  /** Name of each sample, in the row order of `X`. */
  names: string[];
  /** Class of each point, in the row order of `X`. */
  labels: string[];
  /** The classes, in legend order. */
  categories: string[];
  /** Colour of every class. */
  colors: Record<string, string>;
  /** What one class is, for the legend heading. */
  groupLabel: string;
  /** Neighbourhood size this dataset is worth starting from. */
  numberOfNeighbors: number;
  /** The measurement behind each row. */
  profile: DemoProfile;
}
