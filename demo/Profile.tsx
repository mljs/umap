/*
 * What a point actually is: the spectrum, or the measurements, behind the row
 * under the pointer. The y range is the whole dataset's, not the row's, so two
 * hovers can be compared.
 */

import { useMemo } from 'react';

import type { DemoDataset } from './datasets/index.ts';

/** What the hover panel draws. */
export interface ProfileProps {
  /** The dataset the row belongs to. */
  dataset: DemoDataset;
  /** Row under the pointer, or `null` when the pointer is off the plot. */
  row: number | null;
  /** Whether that row was fitted or projected. */
  origin: 'fitted' | 'projected' | null;
}

/** The range every row of a dataset fits in. */
interface Bounds {
  /** Smallest value the dataset holds. */
  minimum: number;
  /** Largest value it holds. */
  maximum: number;
}

/** Width and height of the drawing, in view box units. */
const WIDTH = 100;
const HEIGHT = 34;
/** Margin left around it. */
const MARGIN = 2;

/**
 * Draws the measurement behind one row.
 * @param props - The dataset, the row, and where the row came from.
 * @returns The panel.
 */
export function Profile(props: ProfileProps) {
  const { dataset, row, origin } = props;
  const { profile, names } = dataset;
  const bounds = useMemo(
    () => getBounds(profile.rows, profile.kind === 'bars'),
    [profile.rows, profile.kind],
  );

  const values = row === null ? undefined : profile.rows[row];

  return (
    <div className="profile">
      <div className="profile-header">
        <span className="profile-name">
          {row === null || values === undefined
            ? 'Hover a point to read its data'
            : names[row]}
        </span>
        {origin === null ? null : (
          <span className={`tag tag-${origin}`}>{origin}</span>
        )}
        <span className="profile-axis">
          {profile.yLabel} vs {profile.xLabel}
        </span>
      </div>

      <svg
        className="profile-chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="The data behind the hovered point"
      >
        <rect
          x={0}
          y={0}
          width={WIDTH}
          height={HEIGHT}
          className="profile-ground"
        />
        {values === undefined ? null : profile.kind === 'line' ? (
          <polyline
            className="profile-line"
            points={linePoints(profile.x, values, bounds, profile.reversed)}
          />
        ) : (
          values.map((value, column) => (
            <rect
              key={profile.columns[column] ?? column}
              className="profile-bar"
              x={MARGIN + (column * (WIDTH - 2 * MARGIN)) / values.length + 1}
              y={y(value, bounds)}
              width={(WIDTH - 2 * MARGIN) / values.length - 2}
              height={Math.max(
                0.2,
                y(bounds.minimum, bounds) - y(value, bounds),
              )}
            />
          ))
        )}
      </svg>

      <div className="profile-ticks">
        {profile.kind === 'line' ? (
          <>
            <span>
              {format(profile.reversed ? lastX(profile.x) : profile.x[0])}
            </span>
            <span>
              {format(profile.reversed ? profile.x[0] : lastX(profile.x))}
            </span>
          </>
        ) : (
          profile.columns.map((column) => <span key={column}>{column}</span>)
        )}
      </div>
    </div>
  );
}

/**
 * The polyline of one row.
 * @param x - Position of every column.
 * @param values - The row.
 * @param bounds - The y range of the whole dataset.
 * @param reversed - Whether x runs from high to low.
 * @returns The `points` attribute of the polyline.
 */
function linePoints(
  x: number[],
  values: number[],
  bounds: Bounds,
  reversed: boolean,
): string {
  const first = x[0] ?? 0;
  const last = lastX(x);
  const span = last - first || 1;
  const points: string[] = new Array(values.length);

  for (let i = 0; i < values.length; i++) {
    const position = ((x[i] ?? 0) - first) / span;
    const across = reversed ? 1 - position : position;
    const left = MARGIN + across * (WIDTH - 2 * MARGIN);
    points[i] = `${left.toFixed(2)},${y(values[i] ?? 0, bounds).toFixed(2)}`;
  }

  return points.join(' ');
}

/**
 * Places one value on the vertical axis.
 * @param value - The value.
 * @param bounds - The y range of the whole dataset.
 * @returns Its position, in view box units.
 */
function y(value: number, bounds: Bounds): number {
  const span = bounds.maximum - bounds.minimum || 1;
  const share = (value - bounds.minimum) / span;
  return HEIGHT - MARGIN - share * (HEIGHT - 2 * MARGIN);
}

/**
 * The y range every row of the dataset fits in.
 *
 * Bars are measured from zero, not from the smallest value in the dataset:
 * a petal width of 0.1 cm against a floor of 0.1 would draw as nothing and
 * read as a hundred times smaller than it is.
 * @param rows - Every row of the dataset.
 * @param fromZero - Whether the range has to reach zero.
 * @returns The smallest and largest value it holds.
 */
function getBounds(rows: number[][], fromZero: boolean): Bounds {
  let minimum = fromZero ? 0 : Infinity;
  let maximum = fromZero ? 0 : -Infinity;

  for (const row of rows) {
    for (const value of row) {
      if (value < minimum) minimum = value;
      if (value > maximum) maximum = value;
    }
  }

  return { minimum, maximum };
}

/**
 * The last position of an axis.
 * @param x - The axis.
 * @returns Its last value.
 */
function lastX(x: number[]): number {
  return x.at(-1) ?? 1;
}

/**
 * Rounds an axis label.
 * @param value - The position.
 * @returns It, without decimals.
 */
function format(value: number | undefined): string {
  return value === undefined ? '' : Math.round(value).toString();
}
