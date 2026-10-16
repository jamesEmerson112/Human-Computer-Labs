/**
 * Loads the real URDF, scene and gesture library from disk for the embed tests.
 *
 * APIs:
 * - URDF_TEXT: the lamp URDF text
 * - SCENE: models/derived/scene.json, parsed
 * - gestureFiles: the parsed gestures/*.json in file name order, archive excluded
 */
import { readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { SceneDescription } from '../src/protocol.ts';
import { SIMULATION } from './python-reference.ts';

/**
 * Description: The lamp URDF text, from the starter pack.
 */
export const URDF_TEXT = readFileSync(resolve(SIMULATION, '../starter_pack/robot/dummy_lamp_5dof.urdf'), 'utf8');

/**
 * Description: models/derived/scene.json, parsed.
 */
export const SCENE = JSON.parse(readFileSync(resolve(SIMULATION, 'models/derived/scene.json'), 'utf8')) as SceneDescription;

/**
 * Description: Reads every gestures/*.json, not recursively, as the player page inlines them.
 * Inputs: none.
 * Outputs: the parsed objects in file name order.
 * Constraints: strips a UTF-8 byte order mark, as Python's utf-8-sig does.
 */
export function gestureFiles(): unknown[] {
  const directory = resolve(SIMULATION, 'gestures');
  return readdirSync(directory)
    .filter((name) => name.endsWith('.json'))
    .sort()
    .map((name) => JSON.parse(readFileSync(resolve(directory, name), 'utf8').replace(/^\uFEFF/, '')));
}
