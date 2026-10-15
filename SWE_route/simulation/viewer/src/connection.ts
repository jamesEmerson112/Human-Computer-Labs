import type { Ack, Command, Hello, Message, State } from './protocol';

interface Callbacks {
  hello: (hello: Hello) => void;
  state: (state: State, receivedAtMs: number) => void;
  status: (status: string, ready: boolean) => void;
  ack: (ack: Ack, latencyMs: number) => void;
  error: (message: string) => void;
  reset: () => void;
}

/** No offline queue: commands require a current connection and a fresh state. */
export class Connection {
  private socket?: WebSocket;
  private counter = 0;
  private generation = 0;
  private receivedHello = false;
  private freshState = false;
  private lastStateMs = 0;
  private pending = new Map<string, number>();
  private retryMs = 500;

  constructor(private callbacks: Callbacks) {
    this.connect();
    window.setInterval(() => {
      if (this.freshState && performance.now() - this.lastStateMs > 2000) {
        this.freshState = false;
        this.callbacks.status('State interrupted · reconnecting', false);
        this.socket?.close();
      }
      for (const [id, sent] of this.pending) {
        if (performance.now() - sent > 5000) {
          this.pending.delete(id);
          this.callbacks.error('A command was not acknowledged. Check the connection before trying again.');
        }
      }
    }, 500);
  }

  get ready(): boolean { return this.socket?.readyState === WebSocket.OPEN && this.receivedHello && this.freshState; }

  send(command: Command): void {
    if (!this.ready) { this.callbacks.error('Waiting for a fresh connection. The command was not sent.'); return; }
    const id = `${this.generation}-${++this.counter}`;
    this.pending.set(id, performance.now());
    this.socket!.send(JSON.stringify({ v: 1, id, ...command }));
  }

  private connect(): void {
    this.generation += 1;
    this.receivedHello = false;
    this.freshState = false;
    this.pending.clear();
    this.callbacks.reset();
    this.callbacks.status('Connecting', false);
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
    this.socket = socket;
    socket.addEventListener('open', () => this.callbacks.status('Synchronizing', false));
    socket.addEventListener('message', (event: MessageEvent<string>) => {
      try {
        const message = JSON.parse(event.data) as Message;
        if (message.v !== 1) throw new Error('Unsupported server protocol.');
        if (message.type === 'hello') {
          this.receivedHello = true;
          this.callbacks.hello(message);
        } else if (message.type === 'state' && this.receivedHello) {
          const wasFresh = this.freshState;
          this.lastStateMs = performance.now();
          this.freshState = true;
          this.retryMs = 500;
          this.callbacks.state(message, this.lastStateMs);
          if (!wasFresh) this.callbacks.status('Connected', true);
        } else if (message.type === 'ack') {
          const sent = this.pending.get(message.id);
          this.pending.delete(message.id);
          if (sent !== undefined) this.callbacks.ack(message, performance.now() - sent);
          if (!message.ok) this.callbacks.error(message.error ?? 'The controller rejected that command.');
        }
      } catch (error) {
        this.callbacks.error(error instanceof Error ? error.message : 'Invalid server message.');
        socket.close();
      }
    });
    socket.addEventListener('error', () => socket.close());
    socket.addEventListener('close', () => {
      this.freshState = false;
      this.pending.clear();
      this.callbacks.status('Disconnected · retrying', false);
      window.setTimeout(() => this.connect(), this.retryMs);
      this.retryMs = Math.min(this.retryMs * 1.5, 5000);
    });
  }
}
