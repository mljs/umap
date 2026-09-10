// Optional upstream arm for the A/B benchmarks: the published umap-js this port
// derives from. It is deliberately NOT a devDependency — the port has already
// been validated against umap-js@1.4.0 (see "Validated against umap-js" in
// README.md) — so it is loaded lazily and only when someone installs it to
// re-check ml-umap against a newer upstream release.

/**
 * Load the published umap-js `UMAP` constructor, if it happens to be installed.
 * @returns {Promise<(new (options: object) => object) | null>} The upstream
 * constructor, or `null` when umap-js is absent — the caller then runs its
 * ml-umap arm alone.
 */
export async function loadUpstreamUmap() {
  try {
    const upstream = await import('umap-js');
    return upstream.UMAP;
  } catch (error) {
    if (error?.code !== 'ERR_MODULE_NOT_FOUND') throw error;
    console.log(
      '# umap-js is not installed — upstream comparison skipped; enable it with: npm i -D umap-js\n',
    );
    return null;
  }
}
