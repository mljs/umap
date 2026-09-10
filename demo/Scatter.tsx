/*
 * The plot. It is a plain SVG rather than a charting library: the point of the
 * page is to watch the embedding move and to read the row under the pointer,
 * and a dependency would not draw either any better.
 */

import type { Vectors } from '../src/index.ts';

import type { DemoDataset } from './datasets/index.ts';
import type { RunState } from './useUmapRun.ts';

/** Where a drawn point came from. */
export type PointOrigin = 'fitted' | 'projected';

/** What the plot draws. */
export interface ScatterProps {
  /** The dataset the run was started from, for the colours and the legend. */
  dataset: DemoDataset;
  /** The run to draw. */
  state: RunState;
  /** Row under the pointer, drawn larger and outlined. */
  hoveredRow: number | null;
  /** Called when the pointer enters or leaves a point. */
  onHover: (row: number | null, origin: PointOrigin | null) => void;
}

/** Side of the square the embedding is scaled into. */
const SIDE = 100;
/** Margin left around it, in the same units. */
const MARGIN = 3;
/** Above this many classes the legend drops the names and keeps the swatches. */
const NAMED_LEGEND_LIMIT = 12;
/** Colour of a point whose class has none. */
const NO_COLOR = '#888888';

/**
 * Draws the fitted embedding, and the projected points over it as rings.
 * @param props - The dataset, the run, and the hover.
 * @returns The plot and its legend.
 */
export function Scatter(props: ScatterProps) {
  const { dataset, state, hoveredRow, onHover } = props;
  const { fitted, fittedRows, projected, projectedRows } = state;
  if (fitted.length === 0) {
    return <p className="placeholder">Press Fit to embed {dataset.name}.</p>;
  }

  const bounds = getBounds(fitted, projected);
  const radius = fitted.length > 300 ? 0.7 : 1;

  return (
    <>
      <svg
        className="scatter"
        viewBox={`0 0 ${SIDE} ${SIDE}`}
        role="img"
        aria-label={`${dataset.name} embedded in two dimensions`}
        onPointerLeave={() => onHover(null, null)}
      >
        {fitted.map((point, index) => {
          const row = fittedRows[index];
          return (
            <circle
              key={row ?? index}
              cx={scale(point[0] ?? 0, bounds.minX, bounds.span)}
              cy={scale(point[1] ?? 0, bounds.minY, bounds.span)}
              r={row === hoveredRow ? radius * 2.4 : radius}
              fill={colorOf(dataset, row)}
              fillOpacity={0.85}
              stroke={row === hoveredRow ? 'currentColor' : 'none'}
              strokeWidth={0.4}
              onPointerEnter={() => onHover(row ?? null, 'fitted')}
            />
          );
        })}
        {projected.map((point, index) => {
          const row = projectedRows[index];
          const cx = scale(point[0] ?? 0, bounds.minX, bounds.span);
          const cy = scale(point[1] ?? 0, bounds.minY, bounds.span);
          return (
            <g key={`t${row ?? index}`}>
              <circle
                cx={cx}
                cy={cy}
                r={radius * (row === hoveredRow ? 2.6 : 1.8)}
                fill="none"
                stroke={
                  row === hoveredRow ? 'currentColor' : colorOf(dataset, row)
                }
                strokeWidth={row === hoveredRow ? 0.7 : 0.45}
              />
              {/* A hairline ring is hard to aim at, and its middle falls
                  through to the fitted point below it: this widens what can
                  be hit without painting anything. */}
              <circle
                cx={cx}
                cy={cy}
                r={radius * 1.8}
                fill="none"
                stroke="transparent"
                strokeWidth={radius * 1.4}
                onPointerEnter={() => onHover(row ?? null, 'projected')}
              />
            </g>
          );
        })}
      </svg>

      <div className="legend">
        <span className="legend-title">{dataset.groupLabel}</span>
        {dataset.categories.map((category) => (
          <span key={category} className="legend-entry" title={category}>
            <span
              className="swatch"
              style={{ background: dataset.colors[category] ?? NO_COLOR }}
            />
            {dataset.categories.length <= NAMED_LEGEND_LIMIT ? category : null}
          </span>
        ))}
      </div>
    </>
  );
}

/**
 * Reads the colour of one row of the dataset.
 * @param dataset - The dataset the row belongs to.
 * @param row - Index of the row, when it is known.
 * @returns The colour of the class of that row.
 */
function colorOf(dataset: DemoDataset, row: number | undefined): string {
  if (row === undefined) return NO_COLOR;
  const label = dataset.labels[row];
  if (label === undefined) return NO_COLOR;
  return dataset.colors[label] ?? NO_COLOR;
}

/**
 * Bounds the two sets of points share, as one square so the aspect ratio of
 * the embedding is kept.
 * @param fitted - The fitted embedding.
 * @param projected - The projected points, possibly none.
 * @returns The lower corner of the square and its side.
 */
function getBounds(
  fitted: Vectors,
  projected: Vectors,
): { minX: number; minY: number; span: number } {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const points of [fitted, projected]) {
    for (const point of points) {
      const x = point[0] ?? 0;
      const y = point[1] ?? 0;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  const span = Math.max(maxX - minX, maxY - minY, 1e-6);
  return {
    minX: minX - (span - (maxX - minX)) / 2,
    minY: minY - (span - (maxY - minY)) / 2,
    span,
  };
}

/**
 * Maps one coordinate into the drawing square.
 * @param value - The coordinate.
 * @param minimum - Lower bound of the embedding on that axis.
 * @param span - Side of the square the embedding occupies.
 * @returns The coordinate in the units of the view box.
 */
function scale(value: number, minimum: number, span: number): number {
  return MARGIN + ((value - minimum) / span) * (SIDE - 2 * MARGIN);
}
