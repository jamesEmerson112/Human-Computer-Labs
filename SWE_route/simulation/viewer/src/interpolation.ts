import { Quaternion } from 'three';
import type { Pose, Quat, State, Vec3 } from './protocol';

export function interpolatePose(a: Pose, b: Pose, fraction: number): Pose {
  const t = Math.max(0, Math.min(1, fraction));
  return {
    position: a.position.map((value, index) => value + (b.position[index] - value) * t) as Vec3,
    quaternion: new Quaternion(...a.quaternion).slerp(new Quaternion(...b.quaternion), t).normalize().toArray() as Quat,
  };
}

/** Delay display by one control frame, interpolate actual poses, never predict motion. */
export class PoseBuffer {
  private frames: State[] = [];
  private receivedAtMs = 0;
  constructor(readonly delayMs = 1000 / 30) {}

  clear(): void { this.frames = []; this.receivedAtMs = 0; }

  push(state: State, nowMs: number): boolean {
    const latest = this.frames.at(-1);
    if (latest && (state.sequence <= latest.sequence || state.server_time_s < latest.server_time_s)) return false;
    this.frames.push(state);
    if (this.frames.length > 8) this.frames.shift();
    this.receivedAtMs = nowMs;
    return true;
  }

  sample(nowMs: number): Record<string, Pose> | undefined {
    const latest = this.frames.at(-1);
    if (!latest) return undefined;
    const target = latest.server_time_s + (Math.max(0, nowMs - this.receivedAtMs) - this.delayMs) / 1000;
    const oldest = this.frames[0];
    if (target <= oldest.server_time_s) return oldest.links;
    for (let i = 1; i < this.frames.length; i++) {
      const a = this.frames[i - 1], b = this.frames[i];
      if (target > b.server_time_s) continue;
      const duration = b.server_time_s - a.server_time_s;
      const t = duration > 0 ? (target - a.server_time_s) / duration : 1;
      return Object.fromEntries(Object.entries(b.links).map(([name, pose]) => [name,
        a.links[name] ? interpolatePose(a.links[name], pose, t) : pose]));
    }
    return latest.links;
  }
}
