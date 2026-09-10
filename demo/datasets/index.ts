/*
 * The datasets the picker offers.
 */

import { makeEcstasy } from './ecstasy.ts';
import { makeIris } from './iris.ts';

export type { DemoDataset, DemoProfile } from './types.ts';

/** The datasets of the picker, in the order it lists them. */
export const DATASETS = [makeIris(), makeEcstasy()];
