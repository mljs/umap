/*
 * The form is one shape repeated: a label, a control, and a line of help. It
 * is here once rather than four times in ./Controls.tsx.
 */

import type { ReactNode } from 'react';

/** One labelled row of the form. */
export interface FieldProps {
  /** Name of the parameter, shown above the control. */
  label: string;
  /** One line on what it does, shown under the control. */
  help?: ReactNode;
  /** The control itself. */
  children: ReactNode;
}

/** A labelled row whose control is a picker over a fixed list. */
export interface SelectFieldProps<T extends string | number> {
  /** Name of the parameter. */
  label: string;
  /** The value currently picked. */
  value: T;
  /** What can be picked, in the order the list shows them. */
  options: ReadonlyArray<{ value: T; label: string }>;
  /** Called with the value the user picked. */
  onChange: (value: string) => void;
  /** One line on what it does. */
  help?: ReactNode;
}

/**
 * Wraps one control in its label and its help line.
 * @param props - The label, the help, and the control.
 * @returns The row.
 */
export function Field(props: FieldProps) {
  const { label, help, children } = props;
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {help === undefined ? null : <span className="field-help">{help}</span>}
    </label>
  );
}

/**
 * A labelled picker over a fixed list of values.
 * @param props - The label, the value, the options and the handler.
 * @returns The row.
 */
export function SelectField<T extends string | number>(
  props: SelectFieldProps<T>,
) {
  const { label, value, options, onChange, help } = props;
  return (
    <Field label={label} help={help}>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </Field>
  );
}
