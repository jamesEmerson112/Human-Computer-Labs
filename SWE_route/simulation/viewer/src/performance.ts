import type { Ack, Command, PerformanceState, State } from './protocol';

export const PHRASE_GAP_S = 0.18;
export const EMOTIONS = ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised', 'Neutral'] as const;
export const PRESETS = [
  ...EMOTIONS.map((emotion) => ({ id: `greeting-${emotion.toLowerCase()}`, scene: 'greeting', emotion, label: `${emotion} · Welcome back` })),
  { id: 'question-happy', scene: 'question', emotion: 'Happy', label: 'Happy · A question' },
  { id: 'reassurance-happy', scene: 'reassurance', emotion: 'Happy', label: 'Happy · Reassurance' },
];
export interface Phrase { text: string; audio: string; duration_s: number }
export interface PerformanceScene { id: string; label: string; text: string; phrases: Phrase[] }
export interface Demo { schema_version: 1; voice: { label: string }; scenes: PerformanceScene[] }

export function audioPath(path: string): string {
  if (!/^(?:\/performances\/)?audio\/[a-zA-Z0-9_-]+\.wav$/.test(path)) throw new Error('The demo contains an invalid audio path.');
  return path.startsWith('/') ? path : `/performances/${path}`;
}

/** Bound and validate the local fixture before any motion command is possible. */
export function parseDemo(value: unknown): Demo {
  const raw = value as Partial<Demo> | null;
  if (!raw || raw.schema_version !== 1 || !Array.isArray(raw.scenes) || raw.scenes.length !== 3) throw new Error('The speaking demo is unavailable.');
  const ids = new Set<string>();
  for (const scene of raw.scenes) {
    if (!scene || !['greeting', 'question', 'reassurance'].includes(scene.id) || ids.has(scene.id)
      || typeof scene.label !== 'string' || typeof scene.text !== 'string' || !scene.text.trim() || scene.text.length > 1000
      || !Array.isArray(scene.phrases) || scene.phrases.length < 1 || scene.phrases.length > 12) throw new Error('The demo contains an invalid scene.');
    ids.add(scene.id);
    for (const phrase of scene.phrases) {
      if (!phrase || typeof phrase.text !== 'string' || !phrase.text.trim() || typeof phrase.audio !== 'string'
        || !Number.isFinite(phrase.duration_s) || phrase.duration_s <= 0 || phrase.duration_s > 30) throw new Error('The demo contains an invalid phrase.');
      audioPath(phrase.audio);
    }
    if (scene.phrases.map((phrase) => phrase.text).join(' ') !== scene.text) throw new Error('The demo speech does not match its displayed text.');
  }
  const label = typeof raw.voice?.label === 'string' ? raw.voice.label : 'Fixed local demo voice';
  return { schema_version: 1, scenes: raw.scenes, voice: { label } };
}

export function sceneDuration(scene: PerformanceScene): number {
  return scene.phrases.reduce((total, phrase) => total + phrase.duration_s, 0) + PHRASE_GAP_S * (scene.phrases.length - 1);
}

/** Map the estimated server onset to the audio output clock, not a JS timer. */
export function audioStartTime(input: {
  serverOnsetS: number; ackServerS: number; receivedMs: number; roundTripMs: number;
  contextTime: number; output?: { contextTime?: number; performanceTime?: number };
}): number {
  const { serverOnsetS, ackServerS, receivedMs, roundTripMs, contextTime, output } = input;
  if (![serverOnsetS, ackServerS, receivedMs, roundTripMs, contextTime].every(Number.isFinite) || roundTripMs < 0) throw new Error('Speech timing is unavailable.');
  const targetMs = receivedMs + (serverOnsetS - ackServerS) * 1000 - roundTripMs / 2;
  const outputContext = output?.contextTime, outputPerformance = output?.performanceTime;
  const useOutput = typeof outputContext === 'number' && typeof outputPerformance === 'number'
    && Number.isFinite(outputContext) && Number.isFinite(outputPerformance)
    && outputContext > 0 && Math.abs(receivedMs - outputPerformance) < 1000;
  const when = useOutput ? outputContext + (targetMs - outputPerformance) / 1000 : contextTime + (targetMs - receivedMs) / 1000;
  if (when < contextTime + 0.025 || when > contextTime + 5) throw new Error('The speech cue arrived too late. Press Perform to try again.');
  return when;
}

export function combineAudio(context: Pick<AudioContext, 'createBuffer' | 'sampleRate'>, buffers: AudioBuffer[], scene: PerformanceScene): AudioBuffer {
  if (buffers.length !== scene.phrases.length || !buffers.length) throw new Error('Some demo audio is missing.');
  for (let i = 0; i < buffers.length; i++) {
    const buffer = buffers[i];
    if (buffer.sampleRate !== context.sampleRate || buffer.numberOfChannels < 1 || buffer.numberOfChannels > 2
      || Math.abs(buffer.duration - scene.phrases[i].duration_s) > 0.02) throw new Error('The demo audio timing does not match its motion cues.');
  }
  const gap = Math.round(PHRASE_GAP_S * context.sampleRate);
  const frames = buffers.reduce((sum, buffer) => sum + buffer.length, 0) + gap * (buffers.length - 1);
  const channels = Math.max(...buffers.map((buffer) => buffer.numberOfChannels));
  const joined = context.createBuffer(channels, frames, context.sampleRate);
  for (let channel = 0; channel < channels; channel++) {
    const destination = joined.getChannelData(channel);
    let offset = 0;
    for (const buffer of buffers) {
      destination.set(buffer.getChannelData(Math.min(channel, buffer.numberOfChannels - 1)), offset);
      offset += buffer.length + gap;
    }
  }
  return joined;
}

interface Pending {
  id: string; commandId: string; generation: number; scene: PerformanceScene; buffer: AudioBuffer;
  emotion: string; observed: boolean;
}

export class PerformancePlayer {
  private demo?: Demo;
  private context?: AudioContext;
  private source?: AudioBufferSourceNode;
  private ownedId?: string;
  private pending?: Pending;
  private generation = 0;
  private preparing = false;
  private connected = false;
  private fetchAbort?: AbortController;
  private ackTimeout = 0;
  private latest?: State;
  private acceptedAtS = -Infinity;
  private lastPreset = '';
  private scheduledAudioTime?: number;
  private readonly panel = document.getElementById('performance-panel')!;
  private readonly select = document.getElementById('performance-preset') as HTMLSelectElement;
  private readonly play = document.getElementById('performance-play') as HTMLButtonElement;

  constructor(private send: (command: Command) => string | undefined, private changed: () => void) {
    for (const preset of PRESETS) this.select.add(new Option(preset.label, preset.id));
    this.select.addEventListener('change', () => { this.cancelPreparation(); this.render(); });
    this.play.addEventListener('click', () => { void this.perform(); });
    window.addEventListener('pagehide', () => this.cancel('Stopped', true));
    this.render();
    void this.loadDemo();
  }

  get active(): boolean { return this.preparing || Boolean(this.latest?.performance); }

  setConnected(connected: boolean): void {
    if (!connected && this.connected) this.cancel('Disconnected. Press Perform after reconnecting.', false);
    this.connected = connected;
    this.render();
  }

  private message(message: string): void { document.getElementById('performance-message')!.textContent = message; }

  private async loadDemo(): Promise<void> {
    try {
      const response = await fetch('/performances/demo.json', { cache: 'no-store', signal: AbortSignal.timeout(5000) });
      if (!response.ok) throw new Error('Speaking demo files are unavailable. Gesture playback is still available.');
      this.demo = parseDemo(await response.json());
      document.getElementById('performance-voice')!.textContent = `${this.demo.voice.label} · fixed demo voice`;
      this.message('Choose a moment, then listen and watch.');
    } catch (error) { this.message(error instanceof Error ? error.message : 'The speaking demo is unavailable.'); }
    this.render();
  }

  private cancelPreparation(): void {
    this.generation += 1;
    this.fetchAbort?.abort(); this.fetchAbort = undefined;
    window.clearTimeout(this.ackTimeout);
    if (this.pending && this.connected) this.send({ type: 'cancel_performance', performance_id: this.pending.id });
    this.pending = undefined;
    this.preparing = false;
  }

  private stopAudio(): void {
    if (this.source) { this.source.onended = null; this.source.stop(); this.source.disconnect(); this.source = undefined; }
    this.scheduledAudioTime = undefined;
  }

  cancel(reason = 'Stopped', notifyServer = true): void {
    const id = this.ownedId;
    // Clear ownership first: an old audio event cannot cancel a newer request.
    this.ownedId = undefined;
    this.stopAudio();
    if (!notifyServer) {
      this.pending = undefined;
      this.latest = undefined;
      this.acceptedAtS = -Infinity;
    }
    this.cancelPreparation();
    if (notifyServer && id && this.connected) this.send({ type: 'cancel_performance', performance_id: id });
    this.message(reason);
    this.render();
  }

  private async perform(): Promise<void> {
    if (!this.demo || !this.connected) return;
    this.cancelPreparation();
    const generation = this.generation;
    const preset = PRESETS.find((entry) => entry.id === this.select.value)!;
    const scene = this.demo.scenes.find((entry) => entry.id === preset.scene)!;
    const abort = new AbortController();
    this.fetchAbort = abort;
    this.preparing = true;
    this.message('Preparing this moment…'); this.render();
    try {
      // resume() runs directly from the click, before fetching any audio.
      if (!this.context) {
        this.context = new AudioContext();
        this.context.addEventListener('statechange', () => {
          if (this.context?.state !== 'running' && (this.ownedId || this.pending)) this.cancel('Audio paused. Press Perform to start again.');
        });
      }
      await this.context.resume();
      const context = this.context;
      const buffers = await Promise.all(scene.phrases.map(async (phrase) => {
        const response = await fetch(audioPath(phrase.audio), { signal: abort.signal });
        if (!response.ok) throw new Error('This moment could not load its voice. No new performance was started.');
        const bytes = await response.arrayBuffer();
        if (bytes.byteLength > 8 * 1024 * 1024) throw new Error('The demo audio file is too large.');
        return context.decodeAudioData(bytes);
      }));
      if (generation !== this.generation || !this.connected) return;
      if (context.state !== 'running') throw new Error('Audio is paused. Press Perform to try again.');
      const buffer = combineAudio(context, buffers, scene);
      const id = crypto.randomUUID();
      const commandId = this.send({ type: 'perform', performance_id: id, scene: scene.id, emotion: preset.emotion, intensity: 0.7 });
      if (!commandId) throw new Error('The connection changed. Press Perform after reconnecting.');
      this.pending = { id, commandId, generation, scene, buffer, emotion: preset.emotion, observed: false };
      this.ackTimeout = window.setTimeout(() => {
        if (this.pending?.id === id) this.cancel('The performance was not acknowledged. Press Perform to try again.');
      }, 5000);
      this.message('Ready for the speaking cue…');
    } catch (error) {
      if (generation !== this.generation) return;
      this.preparing = false;
      this.message(error instanceof Error ? error.message : 'The speaking demo could not start.');
    }
    this.render();
  }

  receiveAck(ack: Ack, roundTripMs: number, receivedMs = performance.now()): void {
    const pending = this.pending;
    if (!pending || pending.commandId !== ack.id || pending.generation !== this.generation) return;
    window.clearTimeout(this.ackTimeout);
    this.pending = undefined;
    this.preparing = false;
    if (!ack.ok) { this.message(ack.error ?? 'The controller could not start this moment.'); this.render(); return; }
    try {
      const state = ack.performance;
      if (!state || state.id !== pending.id || state.scene !== pending.scene.id || state.emotion !== pending.emotion
        || state.text !== pending.scene.text || Math.abs(state.audio_duration_s - sceneDuration(pending.scene)) > 0.02
        || !Number.isFinite(state.audio_duration_s) || !this.context || this.context.state !== 'running') throw new Error('The speech cue does not match this moment.');
      if (this.latest && this.latest.server_time_s >= ack.server_time_s && this.latest.performance?.id !== state.id) throw new Error('This moment was replaced before speech started.');
      const when = audioStartTime({ serverOnsetS: state.start_at_s, ackServerS: ack.server_time_s, receivedMs, roundTripMs,
        contextTime: this.context.currentTime, output: this.context.getOutputTimestamp?.() });
      this.stopAudio();
      this.ownedId = state.id;
      this.acceptedAtS = ack.server_time_s;
      const source = this.context.createBufferSource();
      source.buffer = pending.buffer;
      source.connect(this.context.destination);
      this.source = source;
      this.scheduledAudioTime = when;
      source.onended = () => {
        if (this.source === source && this.ownedId === state.id) { source.disconnect(); this.source = undefined; }
      };
      source.start(when);
      this.lastPreset = this.select.value;
      this.message('Voice plays in this tab. Stop ends voice and movement.');
    } catch (error) {
      this.send({ type: 'cancel_performance', performance_id: pending.id });
      this.message(error instanceof Error ? error.message : 'The speaking cue could not start.');
    }
    this.render();
  }

  receiveState(state: State): void {
    // A watch-channel snapshot may predate the command but arrive after its ack.
    // Only a state at least as recent as that ack can replace its ownership.
    if (state.server_time_s < this.acceptedAtS || (this.latest && state.server_time_s < this.latest.server_time_s)) return;
    this.latest = state;
    const id = state.performance?.id;
    if (this.ownedId && id !== this.ownedId) { this.stopAudio(); this.ownedId = undefined; }
    if (this.pending) {
      if (id === this.pending.id) this.pending.observed = true;
      else if (this.pending.observed) { this.cancelPreparation(); this.message('This moment was replaced.'); }
    }
    this.render();
  }

  private render(): void {
    const preset = PRESETS.find((entry) => entry.id === this.select.value) ?? PRESETS[0];
    const selected = this.demo?.scenes.find((scene) => scene.id === preset.scene);
    const active = this.latest?.performance;
    const phase = this.preparing ? 'preparing' : active?.phase ?? 'ready';
    const phaseLabels: Record<string, string> = { preparing: 'Preparing', anticipating: 'Getting ready', speaking: 'Speaking', settling: 'Settling', idle: 'Attentive', ready: 'Ready' };
    this.panel.dataset.phase = phase;
    this.panel.dataset.ownedId = this.ownedId ?? '';
    this.panel.dataset.activeId = active?.id ?? '';
    this.panel.dataset.preparation = String(this.preparing);
    this.panel.dataset.scheduledAudioTime = this.scheduledAudioTime === undefined ? '' : String(this.scheduledAudioTime);
    document.getElementById('performance-phase')!.textContent = this.connected ? phaseLabels[phase] : 'Offline';
    document.getElementById('performance-line')!.textContent = selected?.text ?? 'Loading the speaking moments…';
    const phraseScene = this.demo?.scenes.find((scene) => scene.id === active?.scene);
    document.getElementById('performance-phrase')!.textContent = active?.phase === 'speaking' && active.phrase_index !== null
      ? phraseScene?.phrases[active.phrase_index]?.text ?? '' : active && active.id !== this.ownedId ? 'A moment is playing in another tab.' : '';
    this.play.disabled = !this.connected || !this.demo || this.preparing;
    this.play.textContent = this.lastPreset === preset.id ? 'Repeat this moment' : 'Perform this moment';
    this.changed();
  }
}
