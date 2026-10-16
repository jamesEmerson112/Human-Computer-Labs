/**
 * Builds the embedded lamp as one classic script, embed-dist/lamp-embed.js, plus a hash list of its sources.
 *
 * APIs:
 * - default: the Vite configuration for npm run build:embed
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';

/**
 * Description: The viewer folder, which every path in the hash list is relative to.
 */
const viewerRoot = fileURLToPath(new URL('.', import.meta.url));

/**
 * Description: The output folder of the embed build.
 */
const outDir = resolve(viewerRoot, 'embed-dist');

/**
 * Description: Files outside the bundle whose change also makes the bundle stale. The protocol types
 * and tsconfig.json leave no module in the bundle but still decide whether and how it builds.
 */
const extraSources = ['package-lock.json', 'src/protocol.ts', 'tsconfig.json', 'vite.embed.config.ts'];

/**
 * Description: Hashes a file's bytes.
 * Inputs:
 * - path: the absolute file path.
 * Outputs: the sha256 of the file as lowercase hex.
 */
function sha256(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

/**
 * Description: Writes embed-dist/lamp-embed.sources.json, a sha256 for every source module in the
 * bundle outside node_modules, plus the files in extraSources.
 * Inputs: none.
 * Outputs: the Vite plugin.
 * Constraints: paths are relative to the viewer folder and use forward slashes.
 */
function sourceHashes(): Plugin {
  const modules = new Set<string>();
  return {
    name: 'lamp-embed-source-hashes',
    generateBundle(_options, bundle) {
      for (const output of Object.values(bundle)) {
        if (output.type !== 'chunk') continue;
        for (const id of [...Object.keys(output.modules), ...(output.moduleIds ?? [])]) {
          if (id.startsWith('\0')) continue;
          const path = id.split('?')[0];
          if (!isAbsolute(path) || /[\\/]node_modules[\\/]/.test(path) || !existsSync(path)) continue;
          modules.add(path);
        }
      }
    },
    closeBundle() {
      const paths = [...modules, ...extraSources.map((name) => resolve(viewerRoot, name))];
      const entries = paths.map((path) => [relative(viewerRoot, path).split('\\').join('/'), sha256(path)] as const);
      for (const [name] of entries) if (name.startsWith('../')) throw new Error(`Bundle source ${name} is outside the viewer folder.`);
      entries.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
      writeFileSync(resolve(outDir, 'lamp-embed.sources.json'), `${JSON.stringify({ files: Object.fromEntries(entries) }, null, 2)}\n`);
    },
  };
}

export default defineConfig({
  publicDir: false,
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  plugins: [sourceHashes()],
  build: {
    outDir,
    emptyOutDir: true,
    copyPublicDir: false,
    minify: true,
    lib: {
      entry: resolve(viewerRoot, 'src/embed/index.ts'),
      name: 'LampEmbed',
      formats: ['iife'],
      fileName: () => 'lamp-embed.js',
    },
  },
});
