/**
 * Checks that the embed's gesture validation accepts the whole library and rejects what lamp_sim and
 * controller/tests/motion_parity.rs reject.
 *
 * APIs: none; node:test cases only.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { parseGesture, parseLibrary } from '../src/embed/gestures.ts';
import { parseUrdf } from '../src/embed/urdf.ts';
import { gestureFiles, URDF_TEXT } from './library.ts';

/**
 * Description: The joint names the gestures may use.
 */
const JOINTS = Object.keys(parseUrdf(URDF_TEXT).limits);

/**
 * Description: A minimal valid gesture with the given steps, as definition() in motion_parity.rs.
 * Inputs:
 * - steps: the raw step list.
 * Outputs: the raw gesture object.
 */
function definition(steps: unknown): Record<string, unknown> {
  return { schema_version: 1, name: 'test', label: 'Test', units: 'radians', steps };
}

test('the whole library parses with its families and the Python defaults', () => {
  const library = parseLibrary(gestureFiles(), JOINTS);
  assert.equal(library.size, 126);
  const counts: Record<string, number> = {};
  for (const gesture of library.values()) counts[gesture.group || '(none)'] = (counts[gesture.group || '(none)'] ?? 0) + 1;
  assert.deepEqual(counts, { '(none)': 9, Angry: 25, Disgusted: 13, Fearful: 19, Happy: 28, Sad: 19, Surprised: 13 });
  const nod = library.get('nod')!;
  assert.equal(nod.label, 'Nod');
  assert.deepEqual(nod.steps.map((step) => [step.kind, step.seconds, step.onLimit]), [
    ['offsets', 0.25, 'reverse'], ['hold_seconds', 0.25, 'clamp'], ['return_to_start', 0.25, 'clamp'],
  ]);
  assert.deepEqual(nod.steps[0].angles, [['head_pitch_joint', 0.3]]);
  assert.equal(library.get('emotion_sad_lonely_isolated')!.label, 'Sad / Lonely / Isolated');
});

test('invalid steps and metadata are rejected, as in motion_parity.rs', () => {
  for (const steps of [
    [],
    [{ targets: {} }],
    [{ targets: { typo_joint: 0.1 } }],
    [{ targets: { head_pitch_joint: true } }],
    [{ targets: { head_pitch_joint: '0.3' } }],
    [{ targets: { head_pitch_joint: 0.1 }, hold_seconds: 1 }],
    [{ hold_seconds: 0 }],
    [{ hold_seconds: -1 }],
    [{ hold_seconds: 61 }],
    [{ hold_seconds: true }],
    [{ return_to_start: false }],
    [{ return_to_start: 1 }],
    [{ offsets: { head_pitch_joint: 0.3 }, on_limit: 'ignore' }],
    [{ targets: { head_pitch_joint: 0.3 }, on_limit: 'reverse' }],
    [{ target: { head_pitch_joint: 0.3 } }],
    Array.from({ length: 129 }, () => ({ hold_seconds: 0.1 })),
  ]) {
    assert.throws(() => parseGesture(definition(steps), JOINTS), undefined, `accepted ${JSON.stringify(steps)}`);
  }
  for (const [key, value] of [
    ['schema_version', 2],
    ['schema_version', true],
    ['units', 'degrees'],
    ['name', '../nod'],
    ['name', 'a'.repeat(65)],
    ['label', ' '],
    ['unexpected', 1],
    ['group', ' '],
    ['group', 7],
    ['group', 'a'.repeat(61)],
    ['description', 'a'.repeat(501)],
  ] as const) {
    const data = { ...definition([{ hold_seconds: 1 }]), [key]: value };
    assert.throws(() => parseGesture(data, JOINTS), undefined, `accepted ${JSON.stringify(data)}`);
  }
});

test('non-finite numbers, null fields and non-objects are rejected', () => {
  // JSON.parse turns 1e999 into Infinity, so the finite checks matter.
  assert.equal(JSON.parse('1e999'), Infinity);
  assert.throws(() => parseGesture(definition([{ hold_seconds: Infinity }]), JOINTS));
  assert.throws(() => parseGesture(definition([{ offsets: { head_pitch_joint: Infinity } }]), JOINTS));
  assert.throws(() => parseGesture(definition([{ offsets: { head_pitch_joint: 0.1 }, on_limit: null }]), JOINTS));
  assert.throws(() => parseGesture(definition([{ targets: { head_pitch_joint: 0.1 }, duration_seconds: null }]), JOINTS));
  assert.throws(() => parseGesture(definition([7]), JOINTS));
  assert.throws(() => parseGesture(null, JOINTS));
  assert.throws(() => parseGesture([], JOINTS));
});

test('edge cases follow the Python schema rules', () => {
  // Python's `$` lets one final newline through the name pattern.
  assert.equal(parseGesture({ ...definition([{ hold_seconds: 1 }]), name: 'nod\n' }, JOINTS).name, 'nod\n');
  // Python's \s includes the information separators, so a label made of them is blank.
  assert.throws(() => parseGesture({ ...definition([{ hold_seconds: 1 }]), label: '\x1c\x1f' }, JOINTS));
  // Lengths count code points, not UTF-16 units.
  assert.equal(parseGesture({ ...definition([{ hold_seconds: 1 }]), label: '\u{1F642}'.repeat(60) }, JOINTS).label.length, 120);
  assert.throws(() => parseGesture({ ...definition([{ hold_seconds: 1 }]), label: '\u{1F642}'.repeat(61) }, JOINTS));
  const defaults = parseGesture({ ...definition([{ hold_seconds: 1 }]), $schema: '../gesture.schema.json' }, JOINTS);
  assert.deepEqual([defaults.description, defaults.group], ['', '']);
  assert.throws(() => parseGesture({ ...definition([{ hold_seconds: 1 }]), $schema: 1 }, JOINTS));
});

test('a library rejects duplicate names and labels and an empty list', () => {
  const first = definition([{ targets: { head_pitch_joint: 0.1 } }]);
  assert.throws(() => parseLibrary([], JOINTS), /No gestures/);
  assert.throws(() => parseLibrary([first, first], JOINTS), /duplicate gesture name/);
  assert.throws(() => parseLibrary([first, { ...first, name: 'another' }], JOINTS), /duplicate gesture label/);
  assert.throws(() => parseLibrary([first, { bad: true }], JOINTS), /gesture 1:/);
  const library = parseLibrary([first, { ...first, name: 'another', label: 'Another' }], JOINTS);
  assert.deepEqual([...library.keys()], ['test', 'another']);
});
