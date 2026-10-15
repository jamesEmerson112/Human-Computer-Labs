export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];
export interface Pose { position: Vec3; quaternion: Quat }
export interface Limit { lower: number; upper: number; velocity: number }
export interface Gesture { name: string; label: string; description: string; group: string; revision: string }
export interface Hello {
  v: 1; type: 'hello'; gestures: Gesture[]; limits: Record<string, Limit>;
  physics_hz: number; control_hz: number;
}
export interface State {
  v: 1; type: 'state'; sequence: number; server_time_s: number;
  backend?: 'mujoco' | 'mujoco-source' | 'mujoco-kinematic';
  simulation_time_s: number; requested: Record<string, number>;
  actual: Record<string, number>; activity: string; speed: number;
  light: { enabled: boolean; brightness: number };
  links: Record<string, Pose>;
  diagnostics: Record<string, unknown>;
  timing: Record<string, unknown>;
}
export interface Ack { v: 1; type: 'ack'; id: string; ok: boolean; error?: string; server_time_s: number }
export type Message = Hello | State | Ack;
export type Command =
  | { type: 'play'; name: string }
  | { type: 'targets'; targets: Record<string, number> }
  | { type: 'pace'; speed: number }
  | { type: 'mode'; mode: 'mujoco' | 'mujoco-source' | 'mujoco-kinematic' }
  | { type: 'stop' | 'reset' | 'reload' }
  | { type: 'light'; enabled: boolean; brightness: number };
export interface Visual extends Pose {
  link: string; type: 'cylinder' | 'sphere' | 'box' | 'mesh';
  size?: number[]; mesh?: string; scale?: Vec3;
  rgba: [number, number, number, number]; material: string;
}
export interface SceneDescription {
  schema_version: 1; coordinate_system: 'z-up';
  limits: Record<string, Limit>; visuals: Visual[];
}
