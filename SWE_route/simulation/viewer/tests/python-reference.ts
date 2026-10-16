/**
 * Runs a Python reference script from the simulation's locked environment and streams its JSON lines.
 *
 * APIs:
 * - SIMULATION: the SWE_route/simulation folder
 * - pythonPath: the Python interpreter the reference tests use
 * - streamJsonLines: run a script and hand each parsed stdout line to a callback
 */
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

/**
 * Description: The SWE_route/simulation folder, the parent of the viewer.
 */
export const SIMULATION = resolve(fileURLToPath(new URL('.', import.meta.url)), '../..');

/**
 * Description: Finds the Python interpreter for reference scripts.
 * Inputs: none; LAMP_PYTHON overrides the default ../.venv interpreter.
 * Outputs: the interpreter path.
 * Constraints: throws with the uv sync --locked hint when the interpreter does not exist.
 */
export function pythonPath(): string {
  const python = process.env.LAMP_PYTHON
    ?? resolve(SIMULATION, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
  if (!existsSync(python)) {
    throw new Error(`No Python at ${python}. Run "uv sync --locked" in SWE_route/simulation, or set LAMP_PYTHON.`);
  }
  return python;
}

/**
 * Description: Runs a Python script with the simulation folder as its working directory and parses
 * stdout one line at a time, so traces of tens of MB never sit in memory whole.
 * Inputs:
 * - script: the script path, absolute or relative to the simulation folder.
 * - onLine: called with each parsed line in order.
 * Outputs: a promise of the number of lines read.
 * Constraints: rejects when the script fails, quoting its stderr, or when onLine throws.
 */
export async function streamJsonLines(script: string, onLine: (data: any) => void): Promise<number> {
  const child = spawn(pythonPath(), [resolve(SIMULATION, script)], { cwd: SIMULATION, stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk: string) => { stderr += chunk; });
  const exited = new Promise<number | null>((done, fail) => {
    child.on('error', fail);
    child.on('close', (code) => done(code));
  });
  let count = 0;
  try {
    for await (const line of createInterface({ input: child.stdout, crlfDelay: Infinity })) {
      if (line.trim() === '') continue;
      onLine(JSON.parse(line));
      count += 1;
    }
  } catch (error) {
    child.kill();
    await exited.catch(() => undefined);
    throw error;
  }
  const code = await exited;
  if (code !== 0) throw new Error(`${script} exited with ${code}: ${stderr}`);
  return count;
}
