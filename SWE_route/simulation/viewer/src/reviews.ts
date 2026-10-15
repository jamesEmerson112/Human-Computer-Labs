import type { Gesture } from './protocol';

export const LEGACY_REVIEW_STORAGE_KEY = 'lamp-animation-review-v1';
export const REVIEW_STORAGE_KEY = 'lamp-animation-review-v2';
export const MAX_REVIEW_NOTES = 2000;
export const MAX_REVIEW_ANIMATIONS = 10000;
export const MAX_REVIEW_HISTORY = 50000;
export const MAX_REVIEW_FILE_BYTES = 10 * 1024 * 1024;

export interface ReviewEntry { passed: boolean; notes: string; updated_at: string | null }
export type Reviews = Record<string, ReviewEntry>;
export interface AnimationIdentity { name: string; label: string; group: string; revision: string }
export interface ReviewAnimation extends Omit<AnimationIdentity, 'revision'> {
  revision: string | null;
  status: 'pass' | 'fail'; notes: string; updated_at: string | null;
}
export interface ReviewHistory { animation: ReviewAnimation; reason: string; recorded_at: string }
export interface ReviewReport {
  schema_version: 2;
  type: 'lamp-animation-review';
  exported_at: string;
  rule: 'checked_pass_unchecked_fail';
  summary: { total: number; passed: number; failed: number };
  animations: ReviewAnimation[];
  history: ReviewHistory[];
}
export interface ReviewMigration {
  schema_version: 1;
  source_catalog_id: 'six-core-wheel-2026-10-13';
  target_catalog_id: 'emotion-wheel-without-bad-2026-10-14';
  entries: Array<{ source: AnimationIdentity; target: AnimationIdentity; preserve_review: boolean; reason: string }>;
  retired: Array<AnimationIdentity & { reason: string }>;
}
export interface ParsedReviewReport {
  reviews: Reviews; history: ReviewHistory[]; matched: number;
  unknownNames: string[]; migratedNames: string[]; resetNames: string[]; legacy: boolean;
}
export interface ReviewState {
  gestures: readonly Gesture[];
  reviews: Reviews;
  history: ReviewHistory[];
}
export interface ReviewTransaction extends ParsedReviewReport { report: ReviewReport }

/** Use the same encoding and byte bound for browser saves and downloaded files. */
export function serializeReport(report: ReviewReport): string {
  const encoded = JSON.stringify(report, null, 2) + '\n';
  require(new TextEncoder().encode(encoded).byteLength <= MAX_REVIEW_FILE_BYTES, 'Review exceeds the supported 10 MiB file size. Existing saved data is unchanged.');
  return encoded;
}

function require(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
function record(value: unknown, where: string): Record<string, unknown> {
  require(value !== null && typeof value === 'object' && !Array.isArray(value), `${where} must be an object.`);
  const prototype = Object.getPrototypeOf(value);
  require(prototype === Object.prototype || prototype === null, `${where} must be a plain object.`);
  return value as Record<string, unknown>;
}
function exactKeys(value: Record<string, unknown>, keys: string[], where: string): void {
  require(Object.keys(value).length === keys.length && keys.every((key) => Object.hasOwn(value, key)), `${where} has missing or unexpected fields.`);
}
function validName(value: unknown): value is string {
  return typeof value === 'string' && value.length <= 64 && /^[a-z]/.test(value) && !/[^a-z0-9_]/.test(value);
}
function timestamp(value: unknown, where: string, nullable = false): void {
  if (nullable && value === null) return;
  require(typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value)
    && Number.isFinite(Date.parse(value)) && new Date(value).toISOString() === value,
  `${where} must be an ISO UTC timestamp${nullable ? ' or null' : ''}.`);
}
function notes(value: unknown, where: string): asserts value is string {
  require(typeof value === 'string' && [...value].length <= MAX_REVIEW_NOTES, `${where} must be text of at most ${MAX_REVIEW_NOTES} characters.`);
}
function identity(value: Record<string, unknown>, where: string, nullableRevision = false): void {
  require(validName(value.name), `${where} has an invalid name.`);
  require(typeof value.label === 'string' && value.label.trim().length > 0 && [...value.label].length <= 60, `${where} label is invalid.`);
  require(typeof value.group === 'string' && [...value.group].length <= 60 && (value.group === '' || value.group.trim().length > 0), `${where} group is invalid.`);
  require((nullableRevision && value.revision === null) || (typeof value.revision === 'string' && /^[a-f0-9]{64}$/.test(value.revision)), `${where} revision must be a lowercase SHA256 hash.`);
}
function catalog(gestures: readonly Gesture[]): Map<string, Gesture> {
  require(gestures.length > 0 && gestures.length <= MAX_REVIEW_ANIMATIONS, 'A review needs a loaded gesture catalog.');
  const result = new Map<string, Gesture>();
  for (const gesture of gestures) {
    identity({ ...gesture }, 'Catalog animation');
    require(!result.has(gesture.name), 'The gesture catalog has a duplicate name.');
    result.set(gesture.name, gesture);
  }
  return result;
}
function reason(value: unknown, where: string): asserts value is string {
  require(typeof value === 'string' && value.trim().length > 0 && value.length <= 2000, `${where} reason is invalid.`);
}

/** The migration is trusted only as an explicit baseline mapping, never by leaf name. */
export function parseMigration(value: unknown): ReviewMigration {
  const migration = record(value, 'Review migration');
  require(migration.schema_version === 1 && migration.source_catalog_id === 'six-core-wheel-2026-10-13'
    && migration.target_catalog_id === 'emotion-wheel-without-bad-2026-10-14', 'Unsupported review migration.');
  require(Array.isArray(migration.entries) && migration.entries.length <= MAX_REVIEW_ANIMATIONS
    && Array.isArray(migration.retired) && migration.retired.length <= MAX_REVIEW_ANIMATIONS, 'Invalid migration entries.');
  const sources = new Set<string>(), targets = new Set<string>();
  for (const value of migration.entries) {
    const entry = record(value, 'Migration entry');
    exactKeys(entry, ['source', 'target', 'preserve_review', 'reason'], 'Migration entry');
    for (const side of ['source', 'target']) {
      const item = record(entry[side], `Migration ${side}`);
      exactKeys(item, ['name', 'label', 'group', 'revision'], `Migration ${side}`);
      identity(item, `Migration ${side}`);
      const seen = side === 'source' ? sources : targets;
      require(!seen.has(item.name as string), `Duplicate migration ${side}.`);
      seen.add(item.name as string);
    }
    require(typeof entry.preserve_review === 'boolean', 'Migration preserve_review must be boolean.');
    reason(entry.reason, 'Migration');
  }
  for (const value of migration.retired) {
    const entry = record(value, 'Retired animation');
    exactKeys(entry, ['name', 'label', 'group', 'revision', 'reason'], 'Retired animation');
    identity(entry, 'Retired animation');
    reason(entry.reason, 'Retired animation');
    require(!sources.has(entry.name as string), 'Duplicate retired or mapped source.');
    sources.add(entry.name as string);
  }
  return structuredClone(migration) as unknown as ReviewMigration;
}

/** Unchecked means fail, including an animation with no saved entry. */
export function getReview(reviews: Reviews, name: string): ReviewEntry {
  return Object.hasOwn(reviews, name) ? { ...reviews[name] } : { passed: false, notes: '', updated_at: null };
}
export function updateReview(
  reviews: Reviews, name: string, patch: { passed?: boolean; notes?: string }, now = new Date().toISOString(),
): Reviews {
  require(validName(name), 'Animation name is invalid.');
  timestamp(now, 'Update time');
  const entry = getReview(reviews, name);
  if (patch.passed !== undefined) {
    require(typeof patch.passed === 'boolean', 'Pass must be checked or unchecked.');
    entry.passed = patch.passed;
  }
  if (patch.notes !== undefined) { notes(patch.notes, 'Notes'); entry.notes = patch.notes; }
  entry.updated_at = now;
  return { ...reviews, [name]: entry };
}
function parseAnimation(value: unknown, where: string, legacy = false, historical = false): ReviewAnimation {
  const animation = record(value, where);
  exactKeys(animation, ['name', 'label', 'group', 'status', 'notes', 'updated_at', ...(legacy ? [] : ['revision'])], where);
  const normalized = { ...animation, ...(legacy ? { revision: null } : {}) };
  identity(normalized, where, legacy || historical);
  require(animation.status === 'pass' || animation.status === 'fail', `${where} status must be pass or fail.`);
  notes(animation.notes, `${where} notes`);
  timestamp(animation.updated_at, `${where} update time`, true);
  return normalized as unknown as ReviewAnimation;
}
function parseHistory(value: unknown): ReviewHistory[] {
  require(Array.isArray(value) && value.length <= MAX_REVIEW_HISTORY, 'Review history is invalid or too large.');
  return value.map((value, index) => {
    const where = `History ${index + 1}`, entry = record(value, where);
    exactKeys(entry, ['animation', 'reason', 'recorded_at'], where);
    reason(entry.reason, where);
    timestamp(entry.recorded_at, `${where} time`);
    return { animation: parseAnimation(entry.animation, where, false, true), reason: entry.reason, recorded_at: entry.recorded_at as string };
  });
}
/** Deduplicate repeated reload/import evidence without dropping prior review versions. */
export function mergeHistory(...groups: readonly ReviewHistory[][]): ReviewHistory[] {
  const found = new Set<string>();
  const result = groups.flat().filter((entry) => {
    const key = historyKey(entry);
    if (found.has(key)) return false;
    found.add(key); return true;
  });
  require(result.length <= MAX_REVIEW_HISTORY, 'Review history is full. Export your review before continuing.');
  return result;
}
function animationKey(animation: ReviewAnimation): string {
  return JSON.stringify([animation.name, animation.label, animation.group, animation.revision,
    animation.status, animation.notes, animation.updated_at]);
}
function historyKey(entry: ReviewHistory): string {
  // A timestamp records when a tab observed an old review; different tabs can
  // observe the same review at different times without creating new evidence.
  return JSON.stringify([animationKey(entry.animation), entry.reason]);
}
function historyContentKey(report: ReviewReport): string {
  return JSON.stringify([...new Set(report.history.map(historyKey))].sort());
}
/** Export all current animations, independent of UI filters, with recoverable history. */
export function createReport(
  gestures: readonly Gesture[], reviews: Reviews, now = new Date().toISOString(), history: readonly ReviewHistory[] = [],
): ReviewReport {
  catalog(gestures);
  timestamp(now, 'Export time');
  const animations = gestures.map((gesture): ReviewAnimation => {
    const entry = getReview(reviews, gesture.name);
    require(typeof entry.passed === 'boolean', 'Pass must be checked or unchecked.');
    notes(entry.notes, 'Notes'); timestamp(entry.updated_at, 'Update time', true);
    return { name: gesture.name, label: gesture.label, group: gesture.group, revision: gesture.revision,
      status: entry.passed ? 'pass' : 'fail', notes: entry.notes, updated_at: entry.updated_at };
  });
  const passed = animations.filter((animation) => animation.status === 'pass').length;
  const report: ReviewReport = { schema_version: 2, type: 'lamp-animation-review', exported_at: now,
    rule: 'checked_pass_unchecked_fail', summary: { total: animations.length, passed, failed: animations.length - passed },
    animations, history: parseHistory(history) };
  serializeReport(report);
  return report;
}
function sameIdentity(left: ReviewAnimation | AnimationIdentity, right: AnimationIdentity, legacy = false): boolean {
  return left.name === right.name && left.label === right.label && left.group === right.group && (legacy || left.revision === right.revision);
}

/** Validate the complete envelope before producing replacement state. Legacy files
 * have no hashes: approvals require an explicit baseline mapping and an exact
 * current target hash. Absent mappings fail closed, preserving notes/history. */
export function parseReport(value: unknown, gestures: readonly Gesture[], migration?: ReviewMigration): ParsedReviewReport {
  const current = catalog(gestures);
  const report = record(value, 'Review report');
  const legacy = report.schema_version === 1;
  exactKeys(report, ['schema_version', 'type', 'exported_at', 'rule', 'summary', 'animations', ...(legacy ? [] : ['history'])], 'Review report');
  require((legacy || report.schema_version === 2) && report.type === 'lamp-animation-review' && report.rule === 'checked_pass_unchecked_fail', 'Unsupported review format, version, or pass/fail rule.');
  timestamp(report.exported_at, 'Export time');
  require(Array.isArray(report.animations) && report.animations.length > 0 && report.animations.length <= MAX_REVIEW_ANIMATIONS,
    `Review animations must contain between 1 and ${MAX_REVIEW_ANIMATIONS} entries.`);
  const seen = new Set<string>();
  const animations = report.animations.map((value, index) => {
    const animation = parseAnimation(value, `Animation ${index + 1}`, legacy);
    require(!seen.has(animation.name), 'Review has a duplicate name.'); seen.add(animation.name);
    return animation;
  });
  const history = legacy ? [] : parseHistory(report.history);
  const summary = record(report.summary, 'Summary');
  exactKeys(summary, ['total', 'passed', 'failed'], 'Summary');
  const passed = animations.filter((animation) => animation.status === 'pass').length;
  require(summary.total === animations.length && summary.passed === passed && summary.failed === animations.length - passed,
    'Summary counts do not match the animation statuses.');
  const validatedMigration = migration ? parseMigration(migration) : undefined;
  const mappings = new Map((validatedMigration?.entries ?? []).map((entry) => [entry.source.name, entry]));
  const retired = new Map((validatedMigration?.retired ?? []).map((entry) => [entry.name, entry]));
  const entries = new Map<string, ReviewEntry>();
  const unknownNames: string[] = [], migratedNames: string[] = [], resetNames: string[] = [];
  let knownRetired = 0;
  // Direct current identities win if a file also contains an obsolete source.
  const directNames = new Set(animations.filter((a) => !legacy && current.has(a.name) && sameIdentity(a, current.get(a.name)!)).map((a) => a.name));
  for (const animation of animations) {
    const mapping = mappings.get(animation.name);
    const sourceMatches = mapping && sameIdentity(animation, mapping.source, legacy);
    const mappedTarget = sourceMatches ? current.get(mapping.target.name) : undefined;
    const direct = current.get(animation.name);
    const target = directNames.has(animation.name) ? direct : mappedTarget ?? direct;
    const exact = !legacy && direct && sameIdentity(animation, direct);
    const equivalent = sourceMatches && mappedTarget && mapping.preserve_review && sameIdentity(mapping.target, mappedTarget);
    const preserve = Boolean(exact || equivalent);
    const conflict = target && target.name !== animation.name && directNames.has(target.name);
    let why = '';
    if (!target) {
      unknownNames.push(animation.name); why = 'Retired or unknown animation; retained for recovery.';
      const retiredIdentity = retired.get(animation.name);
      if (retiredIdentity && sameIdentity(animation, retiredIdentity, legacy)) knownRetired += 1;
    }
    else if (conflict) why = 'An exact current review takes precedence over this older mapped review.';
    else {
      require(!entries.has(target.name), 'Multiple imported reviews map to the same animation.');
      entries.set(target.name, { passed: preserve && animation.status === 'pass', notes: animation.notes, updated_at: animation.updated_at });
      if (target.name !== animation.name) migratedNames.push(`${animation.name} → ${target.name}`);
      if (!preserve) {
        resetNames.push(target.name);
        why = legacy ? 'Legacy baseline or target could not be verified, or meaning/motion changed; approval cleared.'
          : 'Animation revision or meaning changed; approval cleared.';
      } else if (legacy || target.name !== animation.name) why = legacy
        ? 'Migrated under the explicit legacy baseline assumption; the old report had no motion hash.'
        : 'Migrated to an explicitly equivalent animation.';
    }
    if (why) history.push({ animation, reason: why, recorded_at: report.exported_at as string });
  }
  require(entries.size > 0 || knownRetired > 0, 'This report has no animations from the current catalog or recorded retired baseline.');
  return { reviews: Object.fromEntries(entries), history: mergeHistory(history), matched: entries.size,
    unknownNames, migratedNames, resetNames, legacy };
}

/** No caller state is mutated or rebound until the complete new catalog and
 * reconciled export pass validation, including the accumulated history size. */
export function admitCatalog(previous: ReviewState, gestures: readonly Gesture[], migration?: ReviewMigration): ReviewTransaction {
  const snapshot = createReport(previous.gestures, previous.reviews, undefined, previous.history);
  const reconciled = parseReport(snapshot, gestures, migration);
  const report = createReport(gestures, reconciled.reviews, undefined, reconciled.history);
  return { ...reconciled, report };
}

/** Merge evidence before admitting a cross-tab replacement. Persist only when
 * logical content changes, ignoring export/observation timestamps and ordering,
 * so peer tabs converge without repeatedly writing each other's history. */
export function reconcileStorage(
  previous: ReviewState, value: unknown, migration?: ReviewMigration,
): ReviewTransaction & { persist: boolean } {
  const snapshot = createReport(previous.gestures, previous.reviews, undefined, previous.history);
  const incomingValue = value === null ? createReport(previous.gestures, {}) : value;
  const incoming = parseReport(incomingValue, previous.gestures, migration);
  const nextReviews = incoming.matched ? incoming.reviews : previous.reviews;
  const nextAnimations = new Map(createReport(previous.gestures, nextReviews).animations.map((entry) => [entry.name, entry]));
  const conflicts = snapshot.animations.filter((animation) =>
    (animation.updated_at !== null || animation.status === 'pass' || animation.notes)
    && animationKey(animation) !== animationKey(nextAnimations.get(animation.name)!));
  const history = mergeHistory(previous.history, incoming.history, conflicts.map((animation) => ({
    animation, reason: 'Review before another tab replaced saved data.', recorded_at: snapshot.exported_at,
  })));
  const report = createReport(previous.gestures, nextReviews, undefined, history);
  // parseReport has fully validated incomingValue before this structural cast.
  // Different tabs may still have different catalog revisions loaded. Rewriting
  // merely because their active hashes differ would create an endless echo.
  // Reconciled approvals stay local; only newly recovered evidence needs a write.
  const persist = value === null || incoming.legacy || historyContentKey(report) !== historyContentKey(incomingValue as ReviewReport);
  return { ...incoming, reviews: nextReviews, history, report, persist };
}

/** A storage read is non-mutating: the v1 key remains a permanent backup. */
export function restoreReview(
  storage: Pick<Storage, 'getItem'>, gestures: readonly Gesture[], migration?: ReviewMigration,
): ParsedReviewReport | undefined {
  const saved = storage.getItem(REVIEW_STORAGE_KEY) ?? storage.getItem(LEGACY_REVIEW_STORAGE_KEY);
  return saved === null ? undefined : parseReport(JSON.parse(saved), gestures, migration);
}

export function reviewRing(gesture: Gesture): 'core' | 'middle' | 'outer' | 'utility' {
  if (!gesture.group) return 'utility';
  return gesture.label.split(' / ').length === 1 ? 'core' : gesture.label.split(' / ').length === 2 ? 'middle' : 'outer';
}
