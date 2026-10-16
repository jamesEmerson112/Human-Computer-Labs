import assert from 'node:assert/strict';
import test from 'node:test';
import { Connection } from '../src/connection.ts';

test('rejects offline commands, requires a fresh state, and never replays commands after reconnect', () => {
  const originals = new Map(['window', 'location', 'WebSocket', 'performance'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  let now = 0;
  const intervals: Array<() => void> = [], timeouts: Array<() => void> = [];
  const errors: string[] = [], statuses: Array<[string, boolean]> = [];
  class FakeSocket {
    static OPEN = 1;
    static instances: FakeSocket[] = [];
    readyState = 0;
    sent: string[] = [];
    callbacks = new Map<string, (event?: unknown) => void>();
    constructor(_url: string) { FakeSocket.instances.push(this); }
    addEventListener(name: string, callback: (event?: unknown) => void): void { this.callbacks.set(name, callback); }
    emit(name: string, message?: unknown): void { this.callbacks.get(name)?.(message); }
    message(message: unknown): void { this.emit('message', { data: JSON.stringify(message) }); }
    open(): void { this.readyState = 1; this.emit('open'); }
    send(value: string): void { this.sent.push(value); }
    close(): void { this.readyState = 3; this.emit('close'); }
  }
  const globals = {
    window: { setInterval: (callback: () => void) => intervals.push(callback), setTimeout: (callback: () => void) => timeouts.push(callback) },
    location: { protocol: 'http:', host: '127.0.0.1:8082' },
    WebSocket: FakeSocket,
    performance: { now: () => now },
  };
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { configurable: true, value });
  try {
    const connection = new Connection({ hello: () => {}, state: () => {}, ack: () => {}, reset: () => {}, error: (message) => errors.push(message), status: (message, ready) => statuses.push([message, ready]) });
    const first = FakeSocket.instances[0];
    connection.send({ type: 'stop' });
    assert.equal(first.sent.length, 0);
    first.open();
    first.message({ v: 1, type: 'hello' });
    assert.equal(connection.ready, false);
    first.message({ v: 1, type: 'state' });
    assert.equal(connection.ready, true);
    const commandId = connection.send({ type: 'play', name: 'nod' });
    assert.equal(first.sent.length, 1);
    assert.equal(JSON.parse(first.sent[0]).type, 'play');
    assert.equal(JSON.parse(first.sent[0]).id, commandId);
    now = 2500;
    intervals[0]();
    assert.equal(connection.ready, false);
    connection.send({ type: 'reset' });
    assert.equal(first.sent.length, 1);
    timeouts[0]();
    const second = FakeSocket.instances[1];
    second.open();
    second.message({ v: 1, type: 'state' });
    assert.equal(connection.ready, false);
    second.message({ v: 1, type: 'hello' });
    second.message({ v: 1, type: 'state' });
    assert.equal(connection.ready, true);
    assert.equal(second.sent.length, 0);
    assert.equal(errors.length, 2);
    assert.ok(statuses.some(([message, ready]) => message.includes('interrupted') && !ready));
  } finally {
    for (const [key, descriptor] of originals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else Reflect.deleteProperty(globalThis, key);
    }
  }
});
