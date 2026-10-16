/**
 * Checks the MELD label to lamp clip mapping: pool sizes, families, the calm pool and no immediate repeats.
 *
 * APIs: none; node:test cases only.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { CALM_POOL, Chooser, MELD_FAMILIES } from '../src/embed/emotions.ts';
import { parseGesture, parseLibrary } from '../src/embed/gestures.ts';
import { parseUrdf } from '../src/embed/urdf.ts';
import { gestureFiles, URDF_TEXT } from './library.ts';

/**
 * Description: The real library, validated.
 */
const LIBRARY = parseLibrary(gestureFiles(), Object.keys(parseUrdf(URDF_TEXT).limits));

/**
 * Description: A small seeded generator (mulberry32), so the random picks repeat between runs.
 * Inputs:
 * - seed: any 32-bit integer.
 * Outputs: a function returning numbers in [0, 1).
 */
function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

test('each MELD label has the agreed pool, core clip included', () => {
  const chooser = new Chooser(LIBRARY.values(), seeded(1));
  const sizes = Object.fromEntries(['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral'].map((label) => [label, chooser.pool(label).length]));
  assert.deepEqual(sizes, { joy: 28, sadness: 19, anger: 25, fear: 19, surprise: 13, disgust: 13, neutral: 4 });
  assert.deepEqual(MELD_FAMILIES, { joy: 'Happy', sadness: 'Sad', anger: 'Angry', fear: 'Fearful', surprise: 'Surprised', disgust: 'Disgusted' });
  for (const [label, family] of Object.entries(MELD_FAMILIES)) {
    const pool = chooser.pool(label);
    assert.ok(pool.includes(`emotion_${family.toLowerCase()}`), `${label} pool includes the ${family} core clip`);
    for (const name of pool) assert.equal(LIBRARY.get(name)!.group, family);
  }
  assert.deepEqual(chooser.pool('neutral'), ['idle_hold', 'nod', 'curious', 'scan']);
});

test('the calm clips exist and have no group', () => {
  for (const name of CALM_POOL) {
    const gesture = LIBRARY.get(name);
    assert.ok(gesture, `${name} exists`);
    assert.equal(gesture.group, '');
  }
});

test('every MELD label resolves to a clip of its family, and unknown labels throw', () => {
  const chooser = new Chooser(LIBRARY.values(), seeded(2));
  for (const [label, family] of Object.entries(MELD_FAMILIES)) {
    const clip = chooser.choose(label)!;
    assert.equal(clip.family, family);
    assert.equal(clip.calm, false);
    assert.equal(clip.label, LIBRARY.get(clip.name)!.label);
    assert.equal(LIBRARY.get(clip.name)!.group, family);
  }
  const calm = chooser.choose('neutral')!;
  assert.deepEqual([calm.family, calm.calm, CALM_POOL.includes(calm.name)], ['Calm', true, true]);
  assert.equal(chooser.choose(' Joy ')!.family, 'Happy');
  for (const label of ['happy', 'Bad', '', 'toString', 'constructor', '__proto__']) {
    assert.throws(() => chooser.choose(label), /Unknown MELD emotion/, label);
  }
});

test('picks never repeat back to back within a family and cover the whole pool', () => {
  const chooser = new Chooser(LIBRARY.values(), seeded(3));
  for (const label of [...Object.keys(MELD_FAMILIES), 'neutral']) {
    const pool = new Set(chooser.pool(label));
    const seen = new Set<string>();
    let previous: string | undefined;
    for (let index = 0; index < 2000; index++) {
      const clip = chooser.choose(label)!;
      assert.notEqual(clip.name, previous, `${label} repeated ${clip.name}`);
      assert.ok(pool.has(clip.name));
      seen.add(clip.name);
      previous = clip.name;
      // Another family's pick in between does not reset this family's memory.
      if (index % 7 === 0) chooser.choose(label === 'joy' ? 'anger' : 'joy');
    }
    assert.equal(seen.size, pool.size, `${label} reached every clip`);
  }
});

test('the injected generator decides the pick, and edge values stay in range', () => {
  const first = new Chooser(LIBRARY.values(), seeded(9));
  const second = new Chooser(LIBRARY.values(), seeded(9));
  for (let index = 0; index < 50; index++) assert.deepEqual(first.choose('fear'), second.choose('fear'));
  const low = new Chooser(LIBRARY.values(), () => 0);
  assert.equal(low.choose('surprise')!.name, low.pool('surprise')[0]);
  assert.equal(low.choose('surprise')!.name, low.pool('surprise')[1]);
  const high = new Chooser(LIBRARY.values(), () => 0.9999999999999999);
  assert.equal(high.choose('disgust')!.name, high.pool('disgust').at(-1));
});

test('a one-clip pool may repeat and an empty pool gives null', () => {
  const joints = Object.keys(parseUrdf(URDF_TEXT).limits);
  const only = parseGesture({ schema_version: 1, name: 'only', label: 'Only', group: 'Sad', units: 'radians', steps: [{ hold_seconds: 1 }] }, joints);
  const chooser = new Chooser([only], seeded(4));
  assert.equal(chooser.choose('sadness')!.name, 'only');
  assert.equal(chooser.choose('sadness')!.name, 'only');
  assert.equal(chooser.choose('joy'), null);
  assert.equal(chooser.choose('neutral'), null);
});
