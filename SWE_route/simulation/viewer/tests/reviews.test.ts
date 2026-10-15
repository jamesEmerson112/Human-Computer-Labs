import assert from 'node:assert/strict';
import test from 'node:test';
import type { Gesture } from '../src/protocol.ts';
import { createReport, getReview, MAX_REVIEW_ANIMATIONS, MAX_REVIEW_NOTES, parseReport, REVIEW_STORAGE_KEY, updateReview } from '../src/reviews.ts';

const now = '2026-10-14T20:00:00.000Z';
const later = '2026-10-14T20:01:00.000Z';
const gestures: Gesture[] = [
  { name: 'nod', label: 'Nod', description: 'Nod once', group: '' },
  { name: 'emotion_happy', label: 'Happy', description: 'Happy movement', group: 'Happy' },
  { name: 'shake', label: 'Shake head', description: 'Shake once', group: '' },
];

test('every unchecked animation exports as fail, including missing and filtered-out entries', () => {
  assert.equal(REVIEW_STORAGE_KEY, 'lamp-animation-review-v1');
  assert.deepEqual(getReview({}, 'nod'), { passed: false, notes: '', updated_at: null });
  const report = createReport(gestures, updateReview({}, 'nod', { passed: true }, now), later);
  assert.deepEqual(report.summary, { total: 3, passed: 1, failed: 2 });
  assert.equal(report.rule, 'checked_pass_unchecked_fail');
  assert.deepEqual(report.animations.map((animation) => [animation.name, animation.status]), [['nod', 'pass'], ['emotion_happy', 'fail'], ['shake', 'fail']]);
  assert.equal(report.animations[1].updated_at, null);
});

test('editing pass and notes is immutable, preserves the other field, and never implies a pass from notes', () => {
  const original = updateReview({}, 'nod', { notes: 'Too abrupt at the beginning.' }, now);
  assert.equal(getReview(original, 'nod').passed, false);
  const checked = updateReview(original, 'nod', { passed: true }, later);
  assert.deepEqual(getReview(checked, 'nod'), { passed: true, notes: 'Too abrupt at the beginning.', updated_at: later });
  assert.deepEqual(getReview(original, 'nod'), { passed: false, notes: 'Too abrupt at the beginning.', updated_at: now });
  const unchecked = updateReview(checked, 'nod', { passed: false }, later);
  assert.equal(createReport(gestures, unchecked, later).summary.passed, 0);
  const copy = getReview(checked, 'nod');
  copy.passed = false;
  assert.equal(getReview(checked, 'nod').passed, true);
});

test('JSON storage/export round trip preserves notes, statuses and timestamps using stable names', () => {
  const reviews = updateReview(updateReview({}, 'nod', { passed: true, notes: 'Return is smooth.\nLooks good.' }, now), 'shake', { notes: 'Neck turn is too wide.' }, later);
  const stored = JSON.stringify(createReport(gestures, reviews, later));
  const reordered = [{ ...gestures[2], label: 'Renamed shake' }, gestures[0], gestures[1]];
  const parsed = parseReport(JSON.parse(stored), reordered);
  assert.equal(parsed.matched, 3);
  assert.deepEqual(parsed.unknownNames, []);
  assert.deepEqual(getReview(parsed.reviews, 'nod'), getReview(reviews, 'nod'));
  assert.equal(createReport(reordered, parsed.reviews, later).animations[0].label, 'Renamed shake');
  assert.deepEqual(createReport(gestures, parsed.reviews, later), JSON.parse(stored));
});

test('partial imports default missing current animations to fail and report valid unknown names', () => {
  const foreign = { name: 'retired_clip', label: 'Retired', description: '', group: '' };
  const imported = createReport([gestures[0], foreign], updateReview({}, 'nod', { passed: true }, now), now);
  const parsed = parseReport(imported, gestures);
  assert.equal(parsed.matched, 1);
  assert.deepEqual(parsed.unknownNames, ['retired_clip']);
  assert.deepEqual(getReview(parsed.reviews, 'emotion_happy'), { passed: false, notes: '', updated_at: null });
  assert.deepEqual(createReport(gestures, parsed.reviews, now).summary, { total: 3, passed: 1, failed: 2 });
  assert.throws(() => parseReport(createReport([foreign], {}, now), gestures), /no animations from the current catalog/);
});

test('malformed or duplicate entries anywhere in a report are rejected before replacing existing reviews', () => {
  const existing = updateReview({}, 'nod', { passed: true, notes: 'Keep this review' }, now);
  const before = JSON.stringify(existing);
  const invalidEntries = [
    { name: '../nod' }, { name: 'nod\n' }, { name: '' }, { status: 'unreviewed' }, { status: true },
    { notes: 1 }, { notes: 'x'.repeat(MAX_REVIEW_NOTES + 1) }, { updated_at: '2026-02-30T00:00:00.000Z' },
    { label: ' ' }, { group: false }, { extra: true },
  ];
  for (const patch of invalidEntries) {
    const report = createReport(gestures, {}, now);
    Object.assign(report.animations[2], patch);
    assert.throws(() => parseReport(report, gestures), JSON.stringify(patch));
    assert.equal(JSON.stringify(existing), before);
  }
  const duplicate = createReport(gestures, {}, now);
  duplicate.animations[2].name = 'nod';
  assert.throws(() => parseReport(duplicate, gestures), /duplicate name/);
  const invalidUnknown = createReport(gestures, {}, now);
  Object.assign(invalidUnknown.animations[2], { name: 'retired_clip', notes: 7 });
  assert.throws(() => parseReport(invalidUnknown, gestures), /notes/);
  assert.equal(JSON.stringify(existing), before);
});

test('format/version/rule, complete envelope, count consistency and file-entry bounds are enforced', () => {
  for (const patch of [
    { schema_version: 2 }, { type: 'another-report' }, { rule: 'unchecked_unreviewed' },
    { exported_at: 'yesterday' }, { summary: { total: 3, passed: 3, failed: 0 } },
    { animations: [] }, { animations: Array(MAX_REVIEW_ANIMATIONS + 1).fill(null) }, { extra: true },
  ]) assert.throws(() => parseReport({ ...createReport(gestures, {}, now), ...patch }, gestures));
  assert.throws(() => parseReport(null, gestures));
  assert.throws(() => parseReport([], gestures));
  assert.throws(() => parseReport({ schema_version: 1 }, gestures));
  assert.throws(() => createReport([], {}, now), /loaded gesture catalog/);
});

test('notes allow the boundary and ordinary markup without turning it into behavior; object-key names stay safe', () => {
  const text = '<script>not executed</script>\n' + '🙂'.repeat(MAX_REVIEW_NOTES - 30);
  const reviews = updateReview({}, 'nod', { notes: text }, now);
  assert.equal(getReview(parseReport(createReport(gestures, reviews, now), gestures).reviews, 'nod').notes, text);
  assert.doesNotThrow(() => updateReview({}, 'nod', { notes: '🙂'.repeat(MAX_REVIEW_NOTES) }, now));
  assert.throws(() => updateReview({}, 'nod', { notes: '🙂'.repeat(MAX_REVIEW_NOTES + 1) }, now));
  assert.deepEqual(getReview({}, 'constructor'), { passed: false, notes: '', updated_at: null });
  const named = { name: 'constructor', label: 'Constructor', description: '', group: '' };
  const parsed = parseReport(createReport([named], updateReview({}, named.name, { passed: true }, now), now), [named]);
  assert.equal(getReview(parsed.reviews, 'constructor').passed, true);
  assert.throws(() => updateReview({}, '__proto__', { passed: true }, now), /name/);
});
