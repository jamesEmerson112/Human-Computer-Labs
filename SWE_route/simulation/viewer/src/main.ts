import './style.css';
import { Connection } from './connection';
import { Stage } from './stage';
import { PerformancePlayer } from './performance';
import type { Command, Gesture, Hello, SceneDescription, State } from './protocol';
import { MAX_REVIEW_FILE_BYTES, REVIEW_STORAGE_KEY, admitCatalog, createReport, getReview, mergeHistory, parseMigration, parseReport, reconcileStorage, restoreReview, reviewRing, serializeReport, updateReview, type ParsedReviewReport, type ReviewHistory, type ReviewMigration, type Reviews } from './reviews';

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <header class="app-header">
    <a class="brand" href="/" aria-label="Lamp playground home"><span class="brand-mark" aria-hidden="true">↗</span><span>Lamp <strong>playground</strong></span></a>
    <span class="header-caption">A little motion. A lot of character.</span>
    <div class="connection"><span id="connection-dot" class="dot"></span><span id="connection-status" role="status">Loading scene</span></div>
  </header>
  <main class="workspace">
    <section class="stage-panel" aria-label="Lamp stage">
      <div id="stage"></div>
      <div class="stage-heading"><span class="eyebrow">LIVE STAGE</span><h1>Meet your lamp.</h1><p>Pick a feeling. See it move.</p></div>
      <div class="stage-tools"><button id="reset-view" class="quiet-button">Reset view</button><button id="light-view" class="quiet-button">Light view</button><label class="axes-toggle"><input id="axes" type="checkbox"> Axes</label></div>
      <div id="error" class="error" role="alert" hidden></div>
      <div class="stage-footer"><span class="orbit-hint"><span aria-hidden="true">⤢</span> Drag to orbit · scroll to zoom</span><span id="render-rate">— fps</span></div>
      <div class="transport">
        <div class="activity-block"><span class="activity-dot"></span><div><span class="eyebrow">NOW PLAYING</span><strong id="activity">Waiting for the controller</strong></div></div>
        <div class="transport-actions"><button id="reset" class="secondary" data-command disabled>Neutral / Reset pose</button><button id="stop" class="stop-button" data-command disabled><span aria-hidden="true">■</span> Stop</button></div>
      </div>
    </section>
    <aside class="control-panel" aria-label="Lamp controls">
      <section id="performance-panel" class="performance-section">
        <div class="section-heading"><h2>A speaking moment</h2><span id="performance-phase" class="count" role="status">Ready</span></div>
        <p class="section-note">The same words, a different feeling.</p>
        <label class="sr-only" for="performance-preset">Speaking moment</label><select id="performance-preset"></select>
        <blockquote id="performance-line">Loading the speaking moments…</blockquote>
        <p id="performance-voice" class="fine-print">Fixed local demo voice</p>
        <div class="performance-actions"><button id="performance-play" class="play-button" disabled>Perform this moment</button><button id="performance-stop" class="stop-button" data-command disabled>Stop</button></div>
        <p id="performance-phrase" aria-live="polite"></p>
        <p id="performance-message" class="fine-print" role="status">Choose a moment, then listen and watch.</p>
      </section>
      <section class="library">
        <div class="section-heading"><h2>Gesture library</h2><span id="gesture-count" class="count">— clips</span></div>
        <p class="section-note">Check animations that pass. Every unchecked animation counts as failed.</p>
        <div class="library-filters"><label class="sr-only" for="group-filter">Emotion family</label><select id="group-filter"><option value="All">All families</option></select><label class="sr-only" for="gesture-search">Search gestures</label><input id="gesture-search" type="search" placeholder="Find a feeling…" autocomplete="off"></div>
        <label class="sr-only" for="ring-filter">Review round</label><select id="ring-filter"><option value="all">All review rounds</option><option value="core">Round 1 · Core emotions</option><option value="middle">Round 2 · Middle emotions</option><option value="outer">Round 3 · Outer emotions</option><option value="utility">Utility gestures</option></select>
        <div class="review-toolbar"><span id="review-summary" role="status">0 passed · 0 failed</span><label class="sr-only" for="review-filter">Review results</label><select id="review-filter"><option value="all">All results</option><option value="failed">Failed</option><option value="passed">Passed</option></select></div>
        <ul id="gesture-select" aria-label="Animations and pass checkboxes" aria-describedby="gesture-description"></ul>
        <div class="gesture-detail"><div><span id="gesture-family" class="eyebrow">SELECT A GESTURE</span><h3 id="gesture-label">A moment of expression</h3></div><p id="gesture-description">The gesture library appears when the controller connects.</p></div>
        <div class="play-actions"><button id="play" class="play-button" data-command disabled><span aria-hidden="true">▶</span> <span id="play-label">Play gesture</span></button><button id="next-gesture" class="secondary" disabled>Next →</button></div>
        <details class="review-notes-section"><summary>Notes for this animation <span class="detail-hint">optional</span></summary><label id="review-notes-label" class="sr-only" for="review-notes">Review notes</label><textarea id="review-notes" rows="3" maxlength="2000" placeholder="What should change?" disabled></textarea></details>
        <div class="review-files"><button id="export-review" class="secondary" disabled>Export review</button><button id="import-review" class="secondary" disabled>Import review</button><input id="review-file" type="file" accept=".json,application/json" hidden></div>
        <p id="review-message" class="fine-print" role="status">Checks save in this browser. Export when you're done.</p>
        <div class="pace-row"><label for="pace">Pace <output id="pace-value">1.00×</output></label><input id="pace" data-command disabled type="range" min="0.1" max="3" step="0.05" value="1"><div class="range-captions"><span>Gentle</span><span>3.0×</span></div><p class="section-note">Joint speed limits still apply.</p></div>
      </section>
      <section class="light-section">
        <div class="section-heading"><h2>Light</h2><label class="switch"><input id="light-enabled" data-command disabled type="checkbox" checked><span class="switch-track"></span><span class="sr-only">Lamp light enabled</span></label></div>
        <label class="slider-label" for="brightness">Brightness <output id="brightness-value">100%</output></label><input id="brightness" data-command disabled type="range" min="0" max="10" step="0.1" value="5">
      </section>
      <details class="manual-section"><summary><span>Manual pose</span><span class="detail-hint">5 joints</span></summary><p class="section-note">Moving a joint interrupts the current gesture.</p><div id="joints"></div></details>
      <details class="diagnostics"><summary><span>Session details</span><span class="detail-hint" id="physics-rate">— Hz</span></summary><label class="slider-label mode-label" for="backend-mode">Simulation mode</label><select id="backend-mode" data-command disabled><option value="mujoco">Physics · corrected contact</option><option value="mujoco-source">Physics · original collision</option><option value="mujoco-kinematic">Kinematic playback</option></select><dl id="diagnostics-list"></dl><p class="fine-print">View interpolates actual poses with a 33 ms buffer. Ack time measures command receipt, not visible motion.</p><button id="reload" class="secondary full-width" data-command disabled>Reload gesture files</button></details>
      <p class="panel-footer">Five joints. Endless personality.</p>
    </aside>
  </main>`;

function element<T extends HTMLElement = HTMLElement>(id: string): T { return document.getElementById(id) as T; }
function showError(message: string): void {
  const error = element('error');
  error.textContent = message;
  error.hidden = false;
  window.clearTimeout(errorTimeout);
  errorTimeout = window.setTimeout(() => { error.hidden = true; }, 9000);
}

let errorTimeout = 0;
let performancePlayer: PerformancePlayer | undefined;
let ready = false;
let sceneReady = false;
let gestures: Gesture[] = [];
let latestState: State | undefined;
let currentHello: Hello | undefined;
let lastUiUpdate = -Infinity;
let ackMs: number | undefined;
let stateRate = 0;
let stateRateStart = performance.now();
let stateCount = 0;
const jointInputs = new Map<string, HTMLInputElement>();
const jointOutputs = new Map<string, HTMLOutputElement>();
const gestureList = element<HTMLUListElement>('gesture-select');
let selectedName = '';
let filteredGestures: Gesture[] = [];
let reviews: Reviews = {};
let reviewHistory: ReviewHistory[] = [];
let reviewMigration: ReviewMigration | undefined;
let migrationWarning = '';
let storageRestoreFailed = false;
let reviewsLoaded = false;
let catalogReady = false;
let importPending = false;
const reviewFilter = element<HTMLSelectElement>('review-filter');
const ringFilter = element<HTMLSelectElement>('ring-filter');
const reviewNotes = element<HTMLTextAreaElement>('review-notes');
const groupFilter = element<HTMLSelectElement>('group-filter');
const search = element<HTMLInputElement>('gesture-search');
const pace = element<HTMLInputElement>('pace');
const brightness = element<HTMLInputElement>('brightness');
const lightEnabled = element<HTMLInputElement>('light-enabled');

let stage: Stage;
try {
  stage = new Stage(element('stage'), (fps) => { element('render-rate').textContent = `${Math.round(fps)} fps`; });
} catch (error) {
  showError(`The 3D stage could not start: ${String(error)}`);
  throw error;
}

function updateEnabled(): void {
  for (const input of document.querySelectorAll<HTMLButtonElement | HTMLInputElement | HTMLSelectElement>('[data-command]')) input.disabled = !ready || !sceneReady;
  element<HTMLButtonElement>('play').disabled = !ready || !sceneReady || !catalogReady || !selectedName;
  element<HTMLButtonElement>('import-review').disabled = !catalogReady || !gestures.length || importPending;
  performancePlayer?.setConnected(ready && sceneReady);
  pace.disabled = !ready || !sceneReady || Boolean(performancePlayer?.active);
}

function sendManual(command: Command): void {
  performancePlayer?.cancel('Stopped by a manual control.');
  connection.send(command);
}

function selectionChanged(): void {
  const selected = gestures.find((gesture) => gesture.name === selectedName);
  element('gesture-family').textContent = selected?.group || 'Original gestures';
  element('gesture-label').textContent = selected?.label ?? 'No matching gestures';
  element('gesture-description').textContent = selected?.description ?? 'Try another family or search term.';
  element('play-label').textContent = selected ? `Play ${selected.label}` : 'Play gesture';
  reviewNotes.disabled = !selected || !catalogReady;
  reviewNotes.value = selected ? getReview(reviews, selected.name).notes : '';
  element('review-notes-label').textContent = selected ? `Notes for ${selected.label}` : 'Review notes';
  for (const button of gestureList.querySelectorAll<HTMLButtonElement>('[data-gesture]')) {
    const active = button.dataset.gesture === selectedName;
    button.setAttribute('aria-pressed', String(active));
    button.closest('li')!.classList.toggle('selected', active);
  }
  const index = filteredGestures.findIndex((gesture) => gesture.name === selectedName);
  element<HTMLButtonElement>('next-gesture').disabled = index < 0 || index === filteredGestures.length - 1;
  updateEnabled();
}

function filterGestures(): void {
  const focused = document.activeElement instanceof HTMLElement && gestureList.contains(document.activeElement) ? document.activeElement : null;
  const focusName = focused?.dataset.reviewName || focused?.dataset.gesture;
  const focusCheckbox = focused instanceof HTMLInputElement;
  const scrollTop = gestureList.scrollTop;
  const term = search.value.trim().toLowerCase();
  filteredGestures = gestures.filter((gesture) =>
    (groupFilter.value === 'All' || (gesture.group || 'Original') === groupFilter.value)
    && (ringFilter.value === 'all' || reviewRing(gesture) === ringFilter.value)
    && `${gesture.label} ${gesture.description} ${gesture.group}`.toLowerCase().includes(term)
    && (reviewFilter.value === 'all' || getReview(reviews, gesture.name).passed === (reviewFilter.value === 'passed')));
  gestureList.replaceChildren(...filteredGestures.map((gesture) => {
    const row = document.createElement('li');
    row.className = 'gesture-row';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.dataset.reviewName = gesture.name;
    checkbox.checked = getReview(reviews, gesture.name).passed;
    checkbox.disabled = !catalogReady;
    checkbox.setAttribute('aria-label', `Pass ${gesture.label}`);
    checkbox.title = `Mark ${gesture.label} as passed`;
    checkbox.addEventListener('change', () => {
      reviews = updateReview(reviews, gesture.name, { passed: checkbox.checked });
      saveReview();
      filterGestures();
    });
    const button = document.createElement('button');
    button.className = 'gesture-option';
    button.dataset.gesture = gesture.name;
    button.textContent = gesture.label;
    button.addEventListener('click', () => { selectedName = gesture.name; selectionChanged(); });
    button.addEventListener('dblclick', () => { if (ready && sceneReady && catalogReady) sendManual({ type: 'play', name: gesture.name }); });
    row.append(checkbox, button);
    return row;
  }));
  if (!filteredGestures.some((gesture) => gesture.name === selectedName)) selectedName = filteredGestures[0]?.name ?? '';
  if (!filteredGestures.length) {
    const empty = document.createElement('li');
    empty.className = 'empty-gestures'; empty.textContent = 'No animations match these filters.';
    gestureList.append(empty);
  }
  const passed = gestures.filter((gesture) => getReview(reviews, gesture.name).passed).length;
  element('review-summary').textContent = `${passed} passed · ${gestures.length - passed} failed`;
  selectionChanged();
  gestureList.scrollTop = scrollTop;
  if (focused) {
    const candidates = gestureList.querySelectorAll<HTMLElement>(focusCheckbox ? '[data-review-name]' : '[data-gesture]');
    const nextFocus = [...candidates].find((node) => (node.dataset.reviewName || node.dataset.gesture) === focusName)
      ?? gestureList.querySelector<HTMLElement>('[aria-pressed="true"]');
    nextFocus?.focus({ preventScroll: true });
  }
}

function reviewMessage(message: string): void {
  if (element('review-message').textContent !== message) element('review-message').textContent = message;
}

function saveReview(message = 'Saved in this browser. Export when you’re done.'): void {
  try {
    if (storageRestoreFailed) throw new Error('Preserve unreadable saved data.');
    localStorage.setItem(REVIEW_STORAGE_KEY, serializeReport(createReport(gestures, reviews, undefined, reviewHistory)));
    reviewMessage(message);
  } catch {
    reviewMessage('Changes are only saved in this tab. Export your review to keep them.');
  }
}

function describeReview(result: ParsedReviewReport): string {
  return `${result.migratedNames.length} renamed; ${result.resetNames.length} need another check; ${result.unknownNames.length} retired or unknown retained in history.`
    + (result.legacy ? ' Legacy reviews use the recorded baseline; the original browser backup is kept.' : '')
    + (result.legacy && migrationWarning ? ` ${migrationWarning}` : '');
}

function loadReview(nextGestures: Gesture[]): { reviews: Reviews; history: ReviewHistory[]; message?: string; failed: boolean } {
  // Validate the catalog even when there is no browser data to restore.
  createReport(nextGestures, {});
  try {
    const saved = restoreReview(localStorage, nextGestures, reviewMigration);
    if (saved) {
      createReport(nextGestures, saved.reviews, undefined, saved.history);
      return { reviews: saved.reviews, history: saved.history, failed: false,
        message: `Restored saved checks and notes. ${describeReview(saved)}` };
    }
  } catch {
    return { reviews: {}, history: [], failed: true,
      message: 'Could not restore saved checks. Saved data is untouched; export this tab or import a valid review file to recover.' };
  }
  return { reviews: {}, history: [], failed: false };
}

const jointLabels: Record<string, string> = {
  base_yaw_joint: 'Base turn', shoulder_pitch_joint: 'Shoulder', elbow_pitch_joint: 'Elbow', neck_yaw_joint: 'Neck turn', head_pitch_joint: 'Head tilt',
};

function receiveHello(hello: Hello): void {
  try {
    const next = reviewsLoaded && gestures.length
      ? (() => {
        const admitted = admitCatalog({ gestures, reviews, history: reviewHistory }, hello.gestures, reviewMigration);
        return { reviews: admitted.reviews, history: admitted.history, failed: storageRestoreFailed,
          message: `Catalog refreshed. ${describeReview(admitted)}` };
      })()
      : loadReview(hello.gestures);
    // Commit identities and approvals together only after complete admission.
    currentHello = hello;
    gestures = hello.gestures;
    reviews = next.reviews;
    reviewHistory = next.history;
    storageRestoreFailed = next.failed;
    reviewsLoaded = true;
    catalogReady = true;
    if (next.message) {
      if (next.failed) reviewMessage(next.message);
      else saveReview(next.message);
    }
  } catch (error) {
    catalogReady = false;
    filterGestures();
    const message = `Could not apply the updated catalog; earlier checks remain bound to their original animations. Playback and review editing are disabled. ${error instanceof Error ? error.message : String(error)}`;
    reviewMessage(message);
    showError(message);
    return;
  }
  element<HTMLButtonElement>('export-review').disabled = !gestures.length;
  element<HTMLButtonElement>('import-review').disabled = !catalogReady || !gestures.length || importPending;
  element('gesture-count').textContent = `${gestures.length} clips`;
  element('physics-rate').textContent = `${hello.physics_hz} Hz physics`;
  const previousGroup = groupFilter.value;
  const groups = ['All', 'Original', ...new Set(gestures.map((gesture) => gesture.group).filter(Boolean))];
  groupFilter.replaceChildren(...groups.map((group) => {
    const option = document.createElement('option');
    option.value = group;
    const count = gestures.filter((gesture) => (gesture.group || 'Original') === group).length;
    option.textContent = group === 'All' ? 'All families' : `${group} · ${count}`;
    return option;
  }));
  groupFilter.value = groups.includes(previousGroup) ? previousGroup : 'All';
  filterGestures();
  const joints = element('joints');
  joints.replaceChildren();
  jointInputs.clear();
  jointOutputs.clear();
  for (const [name, limit] of Object.entries(hello.limits)) {
    const row = document.createElement('div');
    row.className = 'joint-row';
    const label = document.createElement('label');
    label.htmlFor = name;
    label.textContent = jointLabels[name] ?? name;
    const output = document.createElement('output');
    output.textContent = '0.0°';
    output.title = 'Actual angle';
    label.append(output);
    const input = document.createElement('input');
    input.type = 'range'; input.id = name;
    input.min = String(limit.lower); input.max = String(limit.upper); input.step = '0.005'; input.value = '0';
    input.setAttribute('data-command', '');
    input.disabled = !ready;
    input.addEventListener('change', () => sendManual({ type: 'targets', targets: { [name]: Number(input.value) } }));
    input.addEventListener('input', () => { output.textContent = `${(Number(input.value) * 180 / Math.PI).toFixed(1)}° target`; });
    jointInputs.set(name, input); jointOutputs.set(name, output);
    row.append(label, input); joints.append(row);
  }
  updateEnabled();
}

function fmt(value: unknown, decimals = 2): string { return typeof value === 'number' ? value.toFixed(decimals) : '—'; }
function updateDiagnostics(state: State): void {
  const values = [
    ['Controller', `${currentHello?.control_hz ?? '—'} Hz`],
    ['State stream', `${stateRate.toFixed(1)} Hz`],
    ['Simulation time', `${state.simulation_time_s.toFixed(1)} s`],
    ['Tick compute', `${fmt(state.timing.tick_compute_ms)} ms`],
    ['Loop lag', `${fmt(Number(state.timing.loop_lag_s ?? 0) * 1000)} ms`],
    ['Discarded time', `${fmt(state.timing.dropped_time_s, 3)} s`],
    ['Command ack', ackMs === undefined ? '—' : `${ackMs.toFixed(1)} ms`],
    ['Contacts', fmt(state.diagnostics.contacts, 0)],
    ['Penetration', `${fmt(Number(state.diagnostics.penetration_m ?? 0) * 1000, 3)} mm`],
    ['Tracking error', `${fmt(Number(state.diagnostics.tracking_error_rad ?? 0) * 180 / Math.PI)}°`],
    ['Warnings', fmt(state.diagnostics.warnings, 0)],
  ];
  element('diagnostics-list').replaceChildren(...values.flatMap(([label, value]) => {
    const dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = label; dd.textContent = value;
    return [dt, dd];
  }));
}

function receiveState(state: State, atMs: number): void {
  if (latestState && latestState.backend !== state.backend) stage.buffer.clear();
  latestState = state;
  performancePlayer?.receiveState(state);
  stage.receive(state, atMs);
  stateCount += 1;
  if (atMs - stateRateStart > 1000) { stateRate = stateCount * 1000 / (atMs - stateRateStart); stateRateStart = atMs; stateCount = 0; }
  if (atMs - lastUiUpdate < 100) return;
  lastUiUpdate = atMs;
  element('activity').textContent = state.activity;
  if (document.activeElement !== pace) pace.value = String(state.speed);
  element('pace-value').textContent = `${Number(pace.value).toFixed(2)}×`;
  if (document.activeElement !== brightness) brightness.value = String(state.light.brightness);
  element('brightness-value').textContent = `${Math.round(Number(brightness.value) * 20)}%`;
  if (state.backend) element<HTMLSelectElement>('backend-mode').value = state.backend;
  lightEnabled.checked = state.light.enabled;
  for (const [name, input] of jointInputs) {
    if (document.activeElement !== input) {
      input.value = String(state.requested[name] ?? 0);
      jointOutputs.get(name)!.textContent = `${((state.actual[name] ?? 0) * 180 / Math.PI).toFixed(1)}°`;
    }
  }
  if (element<HTMLDetailsElement>('diagnostics-list').closest('details')?.open) updateDiagnostics(state);
}

// Resolve the baseline mapping before the first hello can restore v1 storage.
// Failure is safe: legacy passes clear while notes and history remain available.
try {
  const response = await fetch('/api/review-migration', { cache: 'no-store', signal: AbortSignal.timeout(5000) });
  if (!response.ok) throw new Error(`Migration unavailable (${response.status}).`);
  reviewMigration = parseMigration(await response.json());
} catch {
  migrationWarning = 'The baseline mapping is unavailable; legacy approvals were cleared for review.';
}

const connection = new Connection({
  hello: receiveHello,
  state: receiveState,
  reset: () => { stage.buffer.clear(); lastUiUpdate = -Infinity; performancePlayer?.cancel('Connection refreshed. Press Perform to start again.', false); },
  status: (status, isReady) => {
    if (ready !== isReady) { ready = isReady; updateEnabled(); }
    element('connection-status').textContent = status;
    element('connection-dot').classList.toggle('connected', isReady);
    if (isReady && sceneReady) element('error').hidden = true;
  },
  ack: (ack, latencyMs) => { ackMs = latencyMs; performancePlayer?.receiveAck(ack, latencyMs); },
  error: showError,
});

performancePlayer = new PerformancePlayer((command) => connection.send(command), () => {
  pace.disabled = !ready || !sceneReady || Boolean(performancePlayer?.active);
});

groupFilter.addEventListener('change', filterGestures);
search.addEventListener('input', filterGestures);
reviewFilter.addEventListener('change', filterGestures);
ringFilter.addEventListener('change', filterGestures);
reviewNotes.addEventListener('input', () => {
  if (!selectedName) return;
  reviews = updateReview(reviews, selectedName, { notes: reviewNotes.value });
  saveReview();
});
element('next-gesture').addEventListener('click', () => {
  const index = filteredGestures.findIndex((gesture) => gesture.name === selectedName);
  if (index < 0 || index >= filteredGestures.length - 1) return;
  selectedName = filteredGestures[index + 1].name;
  selectionChanged();
  gestureList.querySelector<HTMLElement>('[aria-pressed="true"]')?.scrollIntoView({ block: 'nearest' });
});
gestureList.addEventListener('keydown', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLButtonElement) || !target.dataset.gesture) return;
  const index = filteredGestures.findIndex((gesture) => gesture.name === target.dataset.gesture);
  const offset = { ArrowDown: 1, ArrowUp: -1, Home: -index, End: filteredGestures.length - index - 1 }[event.key];
  if (offset === undefined) return;
  event.preventDefault();
  selectedName = filteredGestures[Math.max(0, Math.min(filteredGestures.length - 1, index + offset))].name;
  selectionChanged();
  gestureList.querySelector<HTMLElement>('[aria-pressed="true"]')?.focus();
});
element('export-review').addEventListener('click', () => {
  if (!gestures.length) return;
  const report = createReport(gestures, reviews, undefined, reviewHistory);
  const url = URL.createObjectURL(new Blob([serializeReport(report)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `lamp-animation-review-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  document.body.append(link); link.click(); link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  reviewMessage('Review exported. Share this file when you’re ready to revise the failed animations.');
});
const reviewFile = element<HTMLInputElement>('review-file');
element('import-review').addEventListener('click', () => reviewFile.click());
reviewFile.addEventListener('change', async () => {
  const file = reviewFile.files?.[0];
  if (!file || importPending) return;
  importPending = true;
  element<HTMLButtonElement>('import-review').disabled = true;
  try {
    if (file.size > MAX_REVIEW_FILE_BYTES) throw new Error('Choose a review file no larger than 10 MiB.');
    const imported = parseReport(JSON.parse(await file.text()), gestures, reviewMigration);
    const previous = createReport(gestures, reviews, undefined, reviewHistory);
    const history = mergeHistory(reviewHistory, imported.history, previous.animations
      .filter((animation) => animation.updated_at !== null || animation.status === 'pass' || animation.notes)
      .map((animation) => ({ animation, reason: 'Review before an imported replacement.', recorded_at: previous.exported_at })));
    // A retired-only recovery adds history without replacing active checks.
    const nextReviews = imported.matched ? imported.reviews : reviews;
    createReport(gestures, nextReviews, undefined, history);
    reviews = nextReviews;
    reviewHistory = history;
    storageRestoreFailed = false;
    saveReview(`Imported ${imported.matched} animations. ${describeReview(imported)} Saved in this browser.`);
    filterGestures();
  } catch (error) {
    reviewMessage(`Import failed; your checks are unchanged. ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    importPending = false;
    reviewFile.value = '';
    element<HTMLButtonElement>('import-review').disabled = !catalogReady || !gestures.length;
  }
});
window.addEventListener('storage', (event) => {
  if (event.key !== REVIEW_STORAGE_KEY || !gestures.length) return;
  try {
    const incoming = reconcileStorage({ gestures, reviews, history: reviewHistory }, event.newValue ? JSON.parse(event.newValue) : null, reviewMigration);
    // Persist recovered evidence before committing the replacement. A failed
    // write keeps this tab's prior review intact and is reported below.
    if (incoming.persist) localStorage.setItem(REVIEW_STORAGE_KEY, serializeReport(incoming.report));
    reviews = incoming.reviews;
    reviewHistory = incoming.history;
    filterGestures();
    reviewMessage(`Review updated from another tab. ${describeReview(incoming)}`);
  } catch { reviewMessage('Could not load the review from another tab; this tab’s checks are unchanged.'); }
});
element('play').addEventListener('click', () => sendManual({ type: 'play', name: selectedName }));
element('stop').addEventListener('click', () => sendManual({ type: 'stop' }));
element('performance-stop').addEventListener('click', () => sendManual({ type: 'stop' }));
element('reset').addEventListener('click', () => sendManual({ type: 'reset' }));
element('reload').addEventListener('click', () => sendManual({ type: 'reload' }));
pace.addEventListener('input', () => { element('pace-value').textContent = `${Number(pace.value).toFixed(2)}×`; });
pace.addEventListener('change', () => sendManual({ type: 'pace', speed: Number(pace.value) }));
brightness.addEventListener('input', () => { element('brightness-value').textContent = `${Math.round(Number(brightness.value) * 20)}%`; });
brightness.addEventListener('change', () => connection.send({ type: 'light', enabled: lightEnabled.checked, brightness: Number(brightness.value) }));
lightEnabled.addEventListener('change', () => connection.send({ type: 'light', enabled: lightEnabled.checked, brightness: Number(brightness.value) }));
element<HTMLSelectElement>('backend-mode').addEventListener('change', (event) => sendManual({ type: 'mode', mode: (event.target as HTMLSelectElement).value as NonNullable<State['backend']> }));
element('axes').addEventListener('change', (event) => stage.setAxes((event.target as HTMLInputElement).checked));
element('reset-view').addEventListener('click', () => stage.resetView());
element('light-view').addEventListener('click', () => stage.lightView());
element('diagnostics-list').closest('details')!.addEventListener('toggle', () => { if (latestState) updateDiagnostics(latestState); });
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && connection.ready) { event.preventDefault(); sendManual({ type: 'stop' }); }
});

fetch('/api/scene').then(async (response) => {
  if (!response.ok) throw new Error(`Scene request failed (${response.status}).`);
  await stage.load(await response.json() as SceneDescription);
  sceneReady = true;
  updateEnabled();
}).catch((error: unknown) => showError(`Could not load the lamp: ${String(error)}`));
