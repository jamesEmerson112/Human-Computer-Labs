import assert from 'node:assert/strict';
import type { IncomingMessage } from 'node:http';
import test from 'node:test';
import type { ProxyOptions } from 'vite';
import config from '../vite.config.ts';

function check(path: string, host: string | undefined, origin?: string) {
  const proxy = config.server!.proxy![path] as ProxyOptions;
  const request = { headers: { host, origin } } as IncomingMessage;
  return { result: proxy.bypass!(request, undefined, proxy), headers: request.headers, proxy };
}

test('development proxy accepts only its own local browser origin before translating headers', () => {
  for (const host of ['127.0.0.1:5173', 'localhost:5173']) {
    const { result, headers, proxy } = check('/ws', host, `http://${host}`);
    assert.equal(result, undefined);
    assert.equal(headers.origin, 'http://127.0.0.1:8082');
    assert.equal(proxy.changeOrigin, true);
    assert.equal(proxy.ws, true);
  }
  assert.equal(config.server!.host, '127.0.0.1');
  assert.equal(config.server!.port, 5173);
  assert.equal(config.server!.strictPort, true);
});

test('development proxy rejects foreign, absent, mismatched, and opaque WebSocket origins without rewriting them', () => {
  for (const [host, origin] of [
    ['127.0.0.1:5173', 'https://example.com'],
    ['127.0.0.1:5173', undefined],
    ['127.0.0.1:5173', 'http://localhost:5173'],
    ['127.0.0.1:5173', 'null'],
    ['127.0.0.1:5174', 'http://127.0.0.1:5174'],
    ['attacker.example:5173', 'http://attacker.example:5173'],
    [undefined, 'http://127.0.0.1:5173'],
  ]) {
    const { result, headers } = check('/ws', host, origin);
    assert.equal(result, false);
    assert.equal(headers.origin, origin);
  }
});

test('local HTTP asset requests can omit Origin but cannot bypass a foreign Host or Origin', () => {
  for (const path of ['/api', '/assets']) {
    assert.equal(check(path, '127.0.0.1:5173').result, undefined);
    assert.equal(check(path, 'localhost:5173', 'http://localhost:5173').headers.origin, 'http://127.0.0.1:8082');
    assert.equal(check(path, 'localhost:5173', 'https://example.com').result, false);
    assert.equal(check(path, 'attacker.example:5173').result, false);
  }
});
