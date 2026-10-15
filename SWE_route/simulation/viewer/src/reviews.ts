import type { Gesture } from './protocol';

export const REVIEW_STORAGE_KEY = 'lamp-animation-review-v1';
export const MAX_REVIEW_NOTES = 2000;
export const MAX_REVIEW_ANIMATIONS = 10000;

export interface ReviewEntry { passed: boolean; notes: string; updated_at: string | null }
export type Reviews = Record<string, ReviewEntry>;
export interface ReviewReport {
  schema_version: 1;
  type: 'lamp-animation-review';
  exported_at: string;
  rule: 'checked_pass_unchecked_fail';
  summary: { total: number; passed: number; failed: number };
  animations: Array<{
    name: string; label: string; group: string;
    status: 'pass' | 'fail'; notes: string; updated_at: string | null;
  }>;
}
export interface ParsedReviewReport { reviews: Reviews; matched: number; unknownNames: string[] }

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

function catalogNames(gestures: readonly Gesture[]): Set<string> {
  require(gestures.length > 0 && gestures.length <= MAX_REVIEW_ANIMATIONS, 'A review needs a loaded gesture catalog.');
  const names = new Set<string>();
  for (const gesture of gestures) {
    require(validName(gesture.name) && !names.has(gesture.name), 'The gesture catalog has an invalid or duplicate name.');
    names.add(gesture.name);
  }
  return names;
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
  if (patch.notes !== undefined) {
    notes(patch.notes, 'Notes');
    entry.notes = patch.notes;
  }
  entry.updated_at = now;
  return { ...reviews, [name]: entry };
}

/** Export the complete current catalog; filtering the UI must not omit failures. */
export function createReport(gestures: readonly Gesture[], reviews: Reviews, now = new Date().toISOString()): ReviewReport {
  catalogNames(gestures);
  timestamp(now, 'Export time');
  const animations: ReviewReport['animations'] = gestures.map((gesture) => {
    const entry = getReview(reviews, gesture.name);
    require(typeof entry.passed === 'boolean', 'Pass must be checked or unchecked.');
    notes(entry.notes, 'Notes');
    timestamp(entry.updated_at, 'Update time', true);
    return { name: gesture.name, label: gesture.label, group: gesture.group,
      status: entry.passed ? 'pass' : 'fail', notes: entry.notes, updated_at: entry.updated_at };
  });
  const passed = animations.filter((animation) => animation.status === 'pass').length;
  return { schema_version: 1, type: 'lamp-animation-review', exported_at: now,
    rule: 'checked_pass_unchecked_fail', summary: { total: animations.length, passed, failed: animations.length - passed }, animations };
}

/** Validate before returning any replacement state. Valid unknown names are reported,
 * and missing current animations remain unchecked/fail through getReview(). */
export function parseReport(value: unknown, gestures: readonly Gesture[]): ParsedReviewReport {
  const currentNames = catalogNames(gestures);
  const report = record(value, 'Review report');
  exactKeys(report, ['schema_version', 'type', 'exported_at', 'rule', 'summary', 'animations'], 'Review report');
  require(report.schema_version === 1 && report.type === 'lamp-animation-review' && report.rule === 'checked_pass_unchecked_fail', 'Unsupported review format, version, or pass/fail rule.');
  timestamp(report.exported_at, 'Export time');
  require(Array.isArray(report.animations) && report.animations.length > 0 && report.animations.length <= MAX_REVIEW_ANIMATIONS,
    `Review animations must contain between 1 and ${MAX_REVIEW_ANIMATIONS} entries.`);
  const seen = new Set<string>();
  const entries: Array<[string, ReviewEntry]> = [];
  const unknownNames: string[] = [];
  let passed = 0;
  for (const [index, value] of report.animations.entries()) {
    const where = `Animation ${index + 1}`;
    const animation = record(value, where);
    exactKeys(animation, ['name', 'label', 'group', 'status', 'notes', 'updated_at'], where);
    require(validName(animation.name) && !seen.has(animation.name), `${where} has an invalid or duplicate name.`);
    seen.add(animation.name);
    require(typeof animation.label === 'string' && animation.label.trim().length > 0 && [...animation.label].length <= 60, `${where} label is invalid.`);
    require(typeof animation.group === 'string' && [...animation.group].length <= 60 && (animation.group === '' || animation.group.trim().length > 0), `${where} group is invalid.`);
    require(animation.status === 'pass' || animation.status === 'fail', `${where} status must be pass or fail.`);
    notes(animation.notes, `${where} notes`);
    timestamp(animation.updated_at, `${where} update time`, true);
    if (animation.status === 'pass') passed += 1;
    if (currentNames.has(animation.name)) {
      entries.push([animation.name, { passed: animation.status === 'pass', notes: animation.notes, updated_at: animation.updated_at as string | null }]);
    } else unknownNames.push(animation.name);
  }
  const summary = record(report.summary, 'Summary');
  exactKeys(summary, ['total', 'passed', 'failed'], 'Summary');
  require(summary.total === report.animations.length && summary.passed === passed && summary.failed === report.animations.length - passed,
    'Summary counts do not match the animation statuses.');
  require(entries.length > 0, 'This report has no animations from the current catalog.');
  return { reviews: Object.fromEntries(entries), matched: entries.length, unknownNames };
}
