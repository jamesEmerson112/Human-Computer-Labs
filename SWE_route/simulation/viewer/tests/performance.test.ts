import assert from 'node:assert/strict';
import test from 'node:test';
import { audioPath, audioStartTime, combineAudio, parseDemo, PerformancePlayer, PRESETS, sceneDuration, type PerformanceScene } from '../src/performance.ts';
import type { Command, PerformanceState, State } from '../src/protocol.ts';

const scene: PerformanceScene = { id: 'greeting', label: 'Welcome back', text: 'Oh! You are back.', phrases: [
  { text: 'Oh!', audio: 'audio/hello.wav', duration_s: 0.5 },
  { text: 'You are back.', audio: 'audio/back.wav', duration_s: 1.5 },
] };
const demo = { schema_version: 1, voice: { label: 'Local voice' }, scenes: [scene,
  { ...scene, id: 'question' }, { ...scene, id: 'reassurance' }] };

test('the fixed input comparisons cover seven emotions and three Happy text intents', () => {
  assert.equal(PRESETS.length, 9);
  assert.equal(new Set(PRESETS.map((preset) => preset.id)).size, 9);
  assert.equal(PRESETS.filter((preset) => preset.scene === 'greeting').length, 7);
  assert.deepEqual(PRESETS.filter((preset) => preset.emotion === 'Happy').map((preset) => preset.scene), ['greeting', 'question', 'reassurance']);
});

test('fixture admission requires exact spoken text, complete distinct scenes, and valid phrase timing', () => {
  assert.deepEqual(parseDemo(demo).scenes[0], scene);
  assert.throws(() => parseDemo({ ...demo, scenes: [scene, scene, scene] }), /invalid scene/);
  assert.throws(() => parseDemo({ ...demo, scenes: [{ ...scene, text: 'Different words' }, ...demo.scenes.slice(1)] }), /does not match/);
  assert.throws(() => parseDemo({ ...demo, scenes: [{ ...scene, phrases: [{ ...scene.phrases[0], duration_s: NaN }] }, ...demo.scenes.slice(1)] }), /invalid phrase/);
  assert.equal(sceneDuration(scene), 2.18);
});

test('audio paths remain within the local fixture directory', () => {
  assert.equal(audioPath('audio/hello.wav'), '/performances/audio/hello.wav');
  assert.equal(audioPath('/performances/audio/hello.wav'), '/performances/audio/hello.wav');
  for (const path of ['../hello.wav', 'https://example.com/audio/a.wav', 'audio/../a.wav', '/audio/a.wav', 'audio/a.wav?x=1']) {
    assert.throws(() => audioPath(path), /invalid audio path/);
  }
});

test('onset estimation compensates for acknowledgement transit and output latency', () => {
  const when = audioStartTime({ serverOnsetS: 50.8, ackServerS: 50, receivedMs: 10000, roundTripMs: 100,
    contextTime: 20, output: { contextTime: 19.95, performanceTime: 10000 } });
  assert.ok(Math.abs(when - 20.7) < 1e-9);
});

test('unsupported or uninitialized output clocks use the current audio clock', () => {
  const args = { serverOnsetS: 50.8, ackServerS: 50, receivedMs: 10000, roundTripMs: 100, contextTime: 20 };
  assert.ok(Math.abs(audioStartTime(args) - 20.75) < 1e-9);
  assert.equal(audioStartTime({ ...args, output: { contextTime: 0, performanceTime: 0 } }), audioStartTime(args));
  assert.equal(audioStartTime({ ...args, output: { contextTime: 10, performanceTime: 1000 } }), audioStartTime(args));
});

test('late or invalid acknowledgements never start speech out of sync', () => {
  const args = { serverOnsetS: 50.8, ackServerS: 50, receivedMs: 10000, roundTripMs: 100, contextTime: 20 };
  assert.throws(() => audioStartTime({ ...args, roundTripMs: 1700 }), /too late/);
  assert.throws(() => audioStartTime({ ...args, ackServerS: 51 }), /too late/);
  assert.throws(() => audioStartTime({ ...args, serverOnsetS: NaN }), /unavailable/);
});

function buffer(channels: number, frames: number, sampleRate: number, value = 0): AudioBuffer {
  const data = Array.from({ length: channels }, () => new Float32Array(frames).fill(value));
  return { length: frames, sampleRate, duration: frames / sampleRate, numberOfChannels: channels,
    getChannelData: (channel: number) => data[channel] } as AudioBuffer;
}

test('one audio timeline retains both phrases with exactly 180 ms of silence', () => {
  const context = { sampleRate: 1000, createBuffer: buffer };
  const joined = combineAudio(context, [buffer(1, 500, 1000, 0.25), buffer(1, 1500, 1000, 0.5)], scene);
  assert.equal(joined.duration, 2.18);
  assert.equal(joined.getChannelData(0)[499], 0.25);
  assert.ok(joined.getChannelData(0).slice(500, 680).every((value) => value === 0));
  assert.equal(joined.getChannelData(0)[680], 0.5);
  assert.equal(joined.getChannelData(0)[2179], 0.5);
  assert.throws(() => combineAudio(context, [buffer(1, 500, 1000)], scene), /missing/);
  assert.throws(() => combineAudio(context, [buffer(1, 800, 1000), buffer(1, 1500, 1000)], scene), /timing/);
});

test('audio ownership survives stale snapshots, cancels on newer overrides, and ignores late callbacks', async () => {
  const keys = ['window', 'document', 'Option', 'fetch', 'AudioContext', 'performance'];
  const originals = new Map(keys.map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  class Node {
    value = ''; textContent = ''; disabled = false; dataset: Record<string, string> = {};
    callbacks = new Map<string, () => void>();
    add(option: { value: string }): void { if (!this.value) this.value = option.value; }
    addEventListener(name: string, callback: () => void): void { this.callbacks.set(name, callback); }
    emit(name: string): void { this.callbacks.get(name)?.(); }
  }
  const nodes = new Map<string, Node>();
  const node = (id: string): Node => {
    if (!nodes.has(id)) nodes.set(id, new Node());
    return nodes.get(id)!;
  };
  const sources: Array<{ started: boolean; stopped: boolean; onended: (() => void) | null; start: () => void; stop: () => void; connect: () => void; disconnect: () => void }> = [];
  class Audio {
    sampleRate = 1000; currentTime = 100; state = 'running'; destination = {};
    addEventListener(): void {}
    async resume(): Promise<void> {}
    createBuffer = buffer;
    async decodeAudioData(bytes: ArrayBuffer): Promise<AudioBuffer> { return buffer(1, new Uint8Array(bytes)[0] * 100, 1000); }
    getOutputTimestamp(): { contextTime: number; performanceTime: number } { return { contextTime: 100, performanceTime: 100000 }; }
    createBufferSource(): typeof sources[number] {
      const source = { started: false, stopped: false, onended: null,
        start() { this.started = true; }, stop() { this.stopped = true; }, connect() {}, disconnect() {} };
      sources.push(source); return source;
    }
  }
  let fetchAudio: ((path: string) => Promise<unknown>) | undefined;
  const response = (path: string) => ({ ok: true, arrayBuffer: async () => new Uint8Array([path.endsWith('hello.wav') ? 5 : 15]).buffer });
  const globals = {
    window: { addEventListener() {}, setTimeout: () => 1, clearTimeout() {} },
    document: { getElementById: node },
    Option: class { constructor(public label: string, public value: string) {} },
    fetch: async (path: string) => path.endsWith('demo.json') ? { ok: true, json: async () => demo } : fetchAudio ? fetchAudio(path) : response(path),
    AudioContext: Audio, performance: { now: () => 100000 },
  };
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { configurable: true, value });
  const flush = async () => { for (let i = 0; i < 16; i++) await Promise.resolve(); };
  try {
    const commands: Array<Command & { commandId: string }> = [];
    const player = new PerformancePlayer((command) => { const commandId = `cmd-${commands.length}`; commands.push({ ...command, commandId }); return commandId; }, () => {});
    await flush(); player.setConnected(true);
    node('performance-play').emit('click'); await flush();
    const first = commands.find((command) => command.type === 'perform')!;
    assert.equal(first.type, 'perform');
    if (first.type !== 'perform') throw new Error('No performance');
    const accepted: PerformanceState = { id: first.performance_id, scene: 'greeting', emotion: 'Happy', intensity: 0.7,
      text: scene.text, start_at_s: 10.8, audio_duration_s: 2.18, phase: 'anticipating', phrase_index: null };
    const ack = { v: 1 as const, type: 'ack' as const, id: first.commandId, ok: true, server_time_s: 10, performance: accepted };
    player.receiveAck(ack, 10);
    assert.equal(sources.length, 1); assert.equal(sources[0].started, true);
    player.receiveState({ server_time_s: 9.99, performance: null } as State);
    assert.equal(sources[0].stopped, false, 'a pre-command snapshot must not cancel scheduled audio');
    player.receiveState({ server_time_s: 10.1, performance: accepted } as State);

    // A failed new preparation leaves the currently playing moment intact.
    fetchAudio = async () => { throw new Error('Audio missing'); };
    node('performance-play').emit('click'); await flush();
    assert.equal(sources[0].stopped, false);
    assert.equal(commands.filter((command) => command.type === 'perform').length, 1);
    fetchAudio = undefined;
    player.receiveState({ server_time_s: 10.2, performance: { ...accepted, id: 'another-tab' } } as State);
    assert.equal(sources[0].stopped, true, 'a newer remote replacement stops this tab audio');
    player.receiveAck(ack, 10);
    assert.equal(sources.length, 1, 'a repeated ack must not replay speech');

    // Stop during asynchronous decoding invalidates the generation before send.
    const resolvers: Array<() => void> = [];
    fetchAudio = (path) => new Promise((resolve) => { resolvers.push(() => resolve(response(path))); });
    node('performance-play').emit('click'); await flush();
    player.cancel('Stopped');
    for (const resolve of resolvers) resolve(); await flush();
    assert.equal(commands.filter((command) => command.type === 'perform').length, 1);

    // Stop after sending but before ack scopes cancellation to the abandoned ID.
    fetchAudio = undefined;
    node('performance-play').emit('click'); await flush();
    const next = commands.filter((command) => command.type === 'perform').at(-1)!;
    if (next.type !== 'perform') throw new Error('No replacement');
    player.cancel('Stopped');
    player.receiveAck({ ...ack, id: next.commandId, performance: { ...accepted, id: next.performance_id } }, 10);
    assert.equal(sources.length, 1);
    assert.ok(commands.some((command) => command.type === 'cancel_performance' && command.performance_id === next.performance_id));
    player.setConnected(false); player.setConnected(true);
    assert.equal(sources.length, 1, 'reconnect never replays audio');
  } finally {
    for (const [key, descriptor] of originals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else Reflect.deleteProperty(globalThis, key);
    }
  }
});
