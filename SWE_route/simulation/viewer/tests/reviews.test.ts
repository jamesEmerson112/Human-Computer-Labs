import assert from 'node:assert/strict';
import test from 'node:test';
import type { Gesture } from '../src/protocol.ts';
import { admitCatalog, createReport, getReview, LEGACY_REVIEW_STORAGE_KEY, MAX_REVIEW_ANIMATIONS, MAX_REVIEW_FILE_BYTES, MAX_REVIEW_NOTES, parseMigration, parseReport, reconcileStorage, restoreReview, REVIEW_STORAGE_KEY, reviewRing, serializeReport, updateReview, type ReviewMigration, type ReviewState } from '../src/reviews.ts';

const now = '2026-10-14T20:00:00.000Z';
const later = '2026-10-14T20:01:00.000Z';
const revision = 'a'.repeat(64);
const nextRevision = 'b'.repeat(64);
const gestures: Gesture[] = [
  { name: 'nod', label: 'Nod', description: 'Nod once', group: '', revision },
  { name: 'emotion_happy', label: 'Happy', description: 'Happy movement', group: 'Happy', revision },
  { name: 'shake', label: 'Shake head', description: 'Shake once', group: '', revision },
];

test('every unchecked animation exports as fail, including missing and filtered-out entries', () => {
  assert.equal(REVIEW_STORAGE_KEY, 'lamp-animation-review-v2');
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
  const reordered = [gestures[2], gestures[0], gestures[1]];
  const parsed = parseReport(JSON.parse(stored), reordered);
  assert.equal(parsed.matched, 3);
  assert.deepEqual(parsed.unknownNames, []);
  assert.deepEqual(getReview(parsed.reviews, 'nod'), getReview(reviews, 'nod'));
  assert.equal(createReport(reordered, parsed.reviews, later).animations[0].label, 'Shake head');
  assert.deepEqual(createReport(gestures, parsed.reviews, later), JSON.parse(stored));
});

test('partial imports default missing current animations to fail and report valid unknown names', () => {
  const foreign = { name: 'retired_clip', label: 'Retired', description: '', group: '', revision };
  const imported = createReport([gestures[0], foreign], updateReview({}, 'nod', { passed: true }, now), now);
  const parsed = parseReport(imported, gestures);
  assert.equal(parsed.matched, 1);
  assert.deepEqual(parsed.unknownNames, ['retired_clip']);
  assert.deepEqual(getReview(parsed.reviews, 'emotion_happy'), { passed: false, notes: '', updated_at: null });
  assert.deepEqual(createReport(gestures, parsed.reviews, now).summary, { total: 3, passed: 1, failed: 2 });
  assert.throws(() => parseReport(createReport([foreign], {}, now), gestures), /no animations from the current catalog/);
  const retiredMapping = mapping();
  const { description: _description, ...retiredIdentity } = foreign;
  retiredMapping.retired.push({ ...retiredIdentity, reason: 'Removed from selected wheel' });
  const retiredOnly = parseReport(createReport([foreign], {}, now), gestures, retiredMapping);
  assert.equal(retiredOnly.matched, 0);
  assert.equal(retiredOnly.history[0].animation.name, 'retired_clip');
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
    { schema_version: 3 }, { type: 'another-report' }, { rule: 'unchecked_unreviewed' },
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
  const named = { name: 'constructor', label: 'Constructor', description: '', group: '', revision };
  const parsed = parseReport(createReport([named], updateReview({}, named.name, { passed: true }, now), now), [named]);
  assert.equal(getReview(parsed.reviews, 'constructor').passed, true);
  assert.throws(() => updateReview({}, '__proto__', { passed: true }, now), /name/);
});

function oldReport(items: Gesture[], reviews = updateReview({}, items[0].name, { passed: true, notes: 'Human review' }, now)) {
  const { history: _history, ...report } = createReport(items, reviews, now);
  return { ...report, schema_version: 1, animations: report.animations.map(({ revision: _revision, ...animation }) => animation) };
}
const source: Gesture = { name: 'emotion_anger_aggressive_hostile', label: 'Anger / Aggressive / Hostile', group: 'Anger', revision, description: '' };
const target: Gesture = { name: 'emotion_angry_aggressive_hostile', label: 'Angry / Aggressive / Hostile', group: 'Angry', revision: nextRevision, description: '' };
function mapping(preserve_review = true): ReviewMigration {
  const { description: _oldDescription, ...oldIdentity } = source;
  const { description: _newDescription, ...newIdentity } = target;
  return { schema_version: 1, source_catalog_id: 'six-core-wheel-2026-10-13', target_catalog_id: 'emotion-wheel-without-bad-2026-10-14',
    entries: [{ source: oldIdentity, target: newIdentity, preserve_review, reason: 'Equivalent root rename; identical movement.' }], retired: [] };
}

test('legacy equivalent rename carries approval only under the declared source metadata and current target revision', () => {
  const old = oldReport([source]);
  const migrated = parseReport(old, [target], mapping());
  assert.equal(getReview(migrated.reviews, target.name).passed, true);
  assert.deepEqual(migrated.migratedNames, [`${source.name} → ${target.name}`]);
  assert.equal(migrated.history[0].animation.revision, null);
  assert.match(migrated.history[0].reason, /baseline assumption/);
  const stale = parseReport(old, [{ ...target, revision: 'c'.repeat(64) }], mapping());
  assert.equal(getReview(stale.reviews, target.name).passed, false);
  assert.equal(getReview(stale.reviews, target.name).notes, 'Human review');
  assert.deepEqual(stale.resetNames, [target.name]);
  const v2 = createReport([source], updateReview({}, source.name, { passed: true }, now), now);
  assert.equal(getReview(parseReport(v2, [target], mapping()).reviews, target.name).passed, true);
  v2.animations[0].revision = 'c'.repeat(64);
  assert.throws(() => parseReport(v2, [target], mapping()), /no animations/);
});

test('changed meaning never carries approval and repeated leaf labels cannot collide', () => {
  const moved = parseReport(oldReport([source]), [target], mapping(false));
  assert.equal(getReview(moved.reviews, target.name).passed, false);
  assert.equal(getReview(moved.reviews, target.name).notes, 'Human review');
  const incorrect = oldReport([source]);
  incorrect.animations[0].label = 'Other / Hostile';
  assert.throws(() => parseReport(incorrect, [target], mapping()), /no animations/);
  const sameLeaf = { ...target, name: 'emotion_sad_hurt_embarrassed', label: 'Sad / Hurt / Embarrassed', group: 'Sad' };
  const foreignLeaf = { ...source, name: 'emotion_disgusted_disapproving_embarrassed', label: 'Disgusted / Disapproving / Embarrassed', group: 'Disgusted' };
  assert.throws(() => parseReport(oldReport([foreignLeaf]), [sameLeaf], mapping()), /no animations/);
});

test('live catalog reload and reconnect clear changed revisions or metadata while retaining notes and history', () => {
  const reviews = updateReview({}, 'nod', { passed: true, notes: 'Approved old movement' }, now);
  for (const replacement of [{ ...gestures[0], revision: nextRevision }, { ...gestures[0], label: 'Different meaning' }]) {
    const snapshot = createReport(gestures, reviews, now);
    const changed = parseReport(snapshot, [replacement, ...gestures.slice(1)]);
    assert.equal(getReview(changed.reviews, 'nod').passed, false);
    assert.equal(getReview(changed.reviews, 'nod').notes, 'Approved old movement');
    assert.equal(changed.history[0].animation.status, 'pass');
    const roundTrip = createReport([replacement, ...gestures.slice(1)], changed.reviews, later, changed.history);
    const reconnected = parseReport(roundTrip, [replacement, ...gestures.slice(1)]);
    assert.equal(getReview(reconnected.reviews, 'nod').passed, false);
    assert.deepEqual(reconnected.history, changed.history);
  }
});

test('v1 storage remains untouched, v2 takes precedence, and unavailable migration fails closed', () => {
  const original = JSON.stringify(oldReport([source]));
  const data = new Map([[LEGACY_REVIEW_STORAGE_KEY, original]]);
  const storage = { getItem: (key: string) => data.get(key) ?? null };
  const migrated = restoreReview(storage, [target], mapping())!;
  assert.equal(getReview(migrated.reviews, target.name).passed, true);
  assert.equal(data.get(LEGACY_REVIEW_STORAGE_KEY), original);
  assert.equal(data.size, 1);
  const unchecked = restoreReview(storage, [source])!;
  assert.equal(getReview(unchecked.reviews, source.name).passed, false);
  assert.equal(getReview(unchecked.reviews, source.name).notes, 'Human review');
  assert.equal(unchecked.history[0].animation.status, 'pass');
  data.set(REVIEW_STORAGE_KEY, JSON.stringify(createReport([target], {}, later, migrated.history)));
  assert.equal(getReview(restoreReview(storage, [target], mapping())!.reviews, target.name).passed, false);
  assert.equal(data.get(LEGACY_REVIEW_STORAGE_KEY), original);
});

test('history is strictly validated and malformed history never changes existing state', () => {
  const migrated = parseReport(oldReport([source]), [target], mapping());
  const report = createReport([target], migrated.reviews, later, migrated.history);
  assert.deepEqual(createReport([target], parseReport(report, [target]).reviews, later, parseReport(report, [target]).history), report);
  const before = JSON.stringify(migrated);
  for (const patch of [{ reason: '' }, { recorded_at: 'yesterday' }, { extra: true }, { animation: { ...report.history[0].animation, status: 'unreviewed' } }]) {
    const malformed = structuredClone(report);
    Object.assign(malformed.history[0], patch);
    assert.throws(() => parseReport(malformed, [target]));
    assert.equal(JSON.stringify(migrated), before);
  }
});

test('migration rejects duplicate identities and malformed revisions, and ring filtering follows full paths', () => {
  const duplicate = mapping(); duplicate.entries.push(duplicate.entries[0]);
  assert.throws(() => parseMigration(duplicate), /Duplicate/);
  const malformed = mapping(); malformed.entries[0].target.revision = 'unknown';
  assert.throws(() => parseMigration(malformed), /revision/);
  assert.equal(reviewRing(gestures[0]), 'utility');
  assert.equal(reviewRing(gestures[1]), 'core');
  assert.equal(reviewRing({ ...target, label: 'Angry / Aggressive' }), 'middle');
  assert.equal(reviewRing(target), 'outer');
});

test('legitimate history larger than the old 1 MiB cap exports and imports with the same UTF-8 size bound', () => {
  const animation = createReport(gestures, updateReview({}, 'nod', { notes: '🙂'.repeat(MAX_REVIEW_NOTES) }, now), now).animations[0];
  const history = Array.from({ length: 200 }, (_, index) => ({ animation, reason: `Prior review ${index}`, recorded_at: now }));
  const report = createReport(gestures, {}, now, history);
  const encoded = serializeReport(report);
  assert.ok(new TextEncoder().encode(encoded).byteLength > 1024 * 1024);
  assert.ok(new TextEncoder().encode(encoded).byteLength <= MAX_REVIEW_FILE_BYTES);
  const imported = parseReport(JSON.parse(encoded), gestures);
  assert.deepEqual(createReport(gestures, imported.reviews, now, imported.history), report);
  assert.throws(() => createReport(gestures, {}, now, Array(1400).fill(history[0])), /10 MiB/);
});

test('catalog admission failure leaves approvals bound to the old revision, including after retry', () => {
  const original = gestures[0];
  const reviews = updateReview({}, original.name, { passed: true, notes: '🙂'.repeat(MAX_REVIEW_NOTES) }, now);
  const base = createReport([original], reviews, now);
  const candidates = Array.from({ length: 1400 }, (_, index) => ({
    animation: base.animations[0], reason: `Archived review ${String(index).padStart(4, '0')}`, recorded_at: now,
  }));
  // Fill a valid existing export to its byte bound. Admitting a changed motion
  // needs one more historical record, which must fail before any rebinding.
  let low = 0, high = candidates.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    const size = new TextEncoder().encode(JSON.stringify({ ...base, history: candidates.slice(0, middle) }, null, 2) + '\n').byteLength;
    if (size <= MAX_REVIEW_FILE_BYTES) low = middle;
    else high = middle - 1;
  }
  const previous: ReviewState = { gestures: [original], reviews, history: candidates.slice(0, low) };
  const before = serializeReport(createReport(previous.gestures, previous.reviews, now, previous.history));
  const changed = [{ ...original, revision: nextRevision }];
  assert.throws(() => admitCatalog(previous, changed), /10 MiB/);
  assert.equal(serializeReport(createReport(previous.gestures, previous.reviews, now, previous.history)), before);
  assert.equal(previous.gestures[0].revision, revision);
  assert.equal(getReview(previous.reviews, original.name).passed, true);
  // A later valid attempt must compare the original hash, never the rejected one.
  const recovered = admitCatalog({ ...previous, history: [] }, changed);
  assert.equal(recovered.report.animations[0].revision, nextRevision);
  assert.equal(recovered.report.animations[0].status, 'fail');
  assert.equal(recovered.history[0].animation.revision, revision);
});

test('cross-tab history union survives immediate reload and converges without storage event loops', () => {
  const first = updateReview({}, 'nod', { passed: true, notes: 'First tab review' }, now);
  const second = updateReview({}, 'nod', { passed: false, notes: 'Second tab correction' }, later);
  const originalAnimation = createReport(gestures, first, now).animations[0];
  const firstTab: ReviewState = { gestures, reviews: first, history: [{ animation: originalAnimation, reason: 'Existing history', recorded_at: now }] };
  const secondTab: ReviewState = { gestures, reviews: second, history: [] };
  const incoming = createReport(gestures, second, later);
  const merged = reconcileStorage(firstTab, incoming);
  assert.equal(merged.persist, true);
  const saved = serializeReport(merged.report);
  const reloaded = restoreReview({ getItem: (key) => key === REVIEW_STORAGE_KEY ? saved : null }, gestures)!;
  assert.equal(getReview(reloaded.reviews, 'nod').notes, 'Second tab correction');
  assert.ok(reloaded.history.some((entry) => entry.animation.notes === 'First tab review'));
  assert.ok(reloaded.history.some((entry) => entry.reason === 'Existing history'));
  const peer = reconcileStorage(secondTab, JSON.parse(saved));
  assert.equal(peer.persist, false);
  const reordered = structuredClone(peer.report);
  reordered.exported_at = later;
  reordered.history.reverse().forEach((entry) => { entry.recorded_at = later; });
  const echoed = reconcileStorage({ gestures, reviews: merged.reviews, history: merged.history }, reordered);
  assert.equal(echoed.persist, false, 'Timestamp/order differences must not produce another storage write');
  const before = JSON.stringify(firstTab);
  const malformed = structuredClone(incoming);
  malformed.history.push({ animation: originalAnimation, reason: '', recorded_at: now });
  assert.throws(() => reconcileStorage(firstTab, malformed), /reason/);
  assert.equal(JSON.stringify(firstTab), before);
});

test('tabs holding different catalog revisions exchange history without repeatedly rewriting each other', () => {
  const oldCatalog = [gestures[0]], newCatalog = [{ ...gestures[0], revision: nextRevision }];
  const tabs: ReviewState[] = [
    { gestures: oldCatalog, reviews: updateReview({}, 'nod', { passed: true, notes: 'Old review' }, now), history: [] },
    { gestures: newCatalog, reviews: updateReview({}, 'nod', { passed: true, notes: 'New review' }, later), history: [] },
  ];
  let saved = createReport(tabs[1].gestures, tabs[1].reviews, later);
  let writes = 0;
  for (let event = 0; event < 10; event += 1) {
    const index = event % 2;
    const result = reconcileStorage(tabs[index], saved);
    tabs[index] = { ...tabs[index], reviews: result.reviews, history: result.history };
    if (!result.persist) break;
    writes += 1;
    saved = result.report;
  }
  assert.ok(writes > 0 && writes < 6, `Expected bounded history exchange, got ${writes} writes`);
  assert.equal(getReview(tabs[0].reviews, 'nod').passed, false);
  assert.equal(getReview(tabs[1].reviews, 'nod').passed, false);
  assert.ok(saved.history.some((entry) => entry.animation.notes === 'Old review'));
  assert.ok(saved.history.some((entry) => entry.animation.notes === 'New review'));
});
