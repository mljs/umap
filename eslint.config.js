import { defineConfig, globalIgnores } from 'eslint/config';
import reactBase from 'eslint-config-cheminfo-react/base';
import ts from 'eslint-config-cheminfo-typescript';

export default defineConfig(
  globalIgnores([
    'coverage',
    'demo/dist',
    'lib',
    // Vendored from PAIR-code/umap-js; its Apache-2.0 header must stay verbatim.
    'src/__tests__/data/upstream-fixtures.ts',
  ]),
  ts,
  {
    // Benchmarks are run manually from a terminal and report through stdout.
    files: ['benchmark/**'],
    languageOptions: {
      globals: { console: 'readonly', process: 'readonly' },
    },
    rules: { 'no-console': 'off' },
  },
  {
    // The playground is a Vite React app, run from `npm run dev`.
    files: ['demo/**'],
    extends: [reactBase],
  },
);
