import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import type { Pose, SceneDescription, State, Visual } from './protocol';
import { PoseBuffer } from './interpolation';

/**
 * Description: Optional settings for a Stage that is not the server viewer.
 * Fields:
 * - benchmark: the benchmark look; when undefined the page URL decides and the body class follows it.
 * - zoom: false lets the mouse wheel scroll the page instead of zooming.
 * - poseSource: link poses for a frame time in ms, used in place of the server pose buffer.
 */
export interface StageOptions {
  benchmark?: boolean;
  zoom?: boolean;
  poseSource?: (nowMs: number) => Record<string, Pose> | undefined;
}

export class Stage {
  readonly buffer = new PoseBuffer();
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(42, 1, 0.01, 30);
  readonly controls: OrbitControls;
  private links = new Map<string, THREE.Group>();
  private axes = new THREE.Group();
  private spotlight = new THREE.SpotLight(0xffe5a9, 0, 5, Math.PI / 6, 0.65, 2);
  private emitterMaterials: THREE.MeshStandardMaterial[] = [];
  private lightEnabled = true;
  private brightness = 1;
  private emitterPosition = new THREE.Vector3();
  private emitterDirection = new THREE.Vector3(-1, 0, 0);
  private frames = 0;
  private lastRateMs = performance.now();
  private fps = 0;
  private stopped = false;
  private frameRequest = 0;
  private resizeObserver: ResizeObserver;
  private benchmark: boolean;

  constructor(private element: HTMLElement, private onFrameRate: (fps: number) => void, private options: StageOptions = {}) {
    this.benchmark = options.benchmark ?? new URLSearchParams(location.search).has('benchmark');
    if (options.benchmark === undefined) document.body.classList.toggle('benchmark', this.benchmark);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'default' });
    this.renderer.setPixelRatio(this.benchmark ? 1 : Math.min(devicePixelRatio, 1.5));
    this.renderer.shadowMap.enabled = !this.benchmark;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.renderer.domElement.setAttribute('aria-label', options.zoom === false
      ? 'Interactive 3D lamp. Drag to orbit.'
      : 'Interactive 3D lamp. Drag to orbit, scroll to zoom.');
    this.element.append(this.renderer.domElement);
    this.scene.background = new THREE.Color('#151e22');
    this.scene.fog = new THREE.Fog('#151e22', 3.5, 10);
    this.camera.up.set(0, 0, 1);
    this.camera.fov = this.benchmark ? 75 : 42;
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.minDistance = 0.35;
    this.controls.maxDistance = 5;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    if (options.zoom === false) this.controls.enableZoom = false;
    this.resetView();
    this.setupEnvironment();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.element);
    this.resize();
    this.render(performance.now());
  }

  private setupEnvironment(): void {
    if (this.benchmark) {
      this.setupBenchmarkEnvironment();
      return;
    }
    const ambient = new THREE.HemisphereLight(0xdde9f2, 0x414538, 2.1);
    ambient.position.set(0, 0, 3);
    this.scene.add(ambient);
    const key = new THREE.DirectionalLight(0xf8eee0, 3.2);
    key.position.set(-1.5, -2, 3);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    Object.assign(key.shadow.camera, { left: -1.4, right: 1.4, top: 1.4, bottom: -1.4, near: 0.1, far: 7 });
    key.shadow.bias = -0.00015;
    key.shadow.normalBias = 0.01;
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x9fcbdc, 1.8);
    rim.position.set(1, 1.5, 2);
    this.scene.add(rim);
    const desk = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.7, 0.07), new THREE.MeshStandardMaterial({ color: '#69716c', roughness: 0.92 }));
    desk.position.set(-0.15, 0, -0.036);
    desk.receiveShadow = true;
    this.scene.add(desk);
    const wall = new THREE.Mesh(new THREE.BoxGeometry(0.025, 2, 1.5), new THREE.MeshStandardMaterial({ color: '#929c8a', roughness: 0.95 }));
    wall.position.set(-1.3, 0, 0.75);
    wall.receiveShadow = true;
    this.scene.add(wall);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: '#172025', roughness: 1 }));
    ground.position.z = -0.075;
    ground.receiveShadow = true;
    this.scene.add(ground);
    const grid = new THREE.GridHelper(2.2, 22, 0x7b8a7c, 0x7b8a7c);
    grid.scale.z = 1.6 / 2.2;
    grid.rotation.x = Math.PI / 2;
    grid.position.set(-0.15, 0, 0.0005);
    const gridMaterial = grid.material as THREE.Material;
    gridMaterial.transparent = true;
    gridMaterial.opacity = 0.16;
    gridMaterial.depthWrite = false;
    this.scene.add(grid);
    this.spotlight.castShadow = true;
    this.spotlight.shadow.mapSize.set(512, 512);
    this.spotlight.shadow.bias = -0.0001;
    this.spotlight.shadow.normalBias = 0.002;
    this.scene.add(this.spotlight, this.spotlight.target);
    this.axes.visible = false;
    this.scene.add(this.axes);
  }

  private setupBenchmarkEnvironment(): void {
    this.scene.fog = null;
    this.scene.background = new THREE.Color('#1f2937');
    this.renderer.toneMappingExposure = 1;
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const fill = new THREE.DirectionalLight(0xffffff, 1.4);
    fill.position.set(0, 0, 1).applyQuaternion(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.5, -0.5, 0, 'ZYX')));
    this.scene.add(fill);
    const material = (color: string) => new THREE.MeshStandardMaterial({ color, metalness: 0, roughness: 0.55 });
    const table = new THREE.Mesh(new THREE.BoxGeometry(2.6, 2, 0.04), material('#3f4959'));
    table.position.set(-0.2, 0, -0.02);
    const wall = new THREE.Mesh(new THREE.BoxGeometry(0.025, 2, 1.5), material('#8c93a0'));
    wall.position.set(-1.3, 0, 0.75);
    this.scene.add(table, wall);
    const minor: number[] = [], major: number[] = [];
    for (let i = -13; i <= 13; i++) {
      const target = i % 5 === 0 ? major : minor;
      target.push(i * 0.1 - 0.2, -1, 0.001, i * 0.1 - 0.2, 1, 0.001);
    }
    for (let i = -10; i <= 10; i++) {
      const target = i % 5 === 0 ? major : minor;
      target.push(-1.5, i * 0.1, 0.001, 1.1, i * 0.1, 0.001);
    }
    for (const [vertices, color] of [[minor, '#555f6d'], [major, '#6e7a8b']] as const) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
      this.scene.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color })));
    }
    this.spotlight.color.set('#ffe2a3');
    this.spotlight.angle = 24 * Math.PI / 180;
    this.spotlight.penumbra = 0.5;
    this.spotlight.distance = 4;
    this.spotlight.castShadow = false;
    this.axes.visible = false;
    this.scene.add(this.spotlight, this.spotlight.target, this.axes);
  }

  /** Build the lamp visuals; given mesh bytes keyed by scene mesh name, no asset is fetched. */
  async load(description: SceneDescription, meshData?: ReadonlyMap<string, ArrayBuffer>): Promise<void> {
    if (description.schema_version !== 1 || description.coordinate_system !== 'z-up') throw new Error('Unsupported scene description.');
    const loader = new STLLoader();
    const meshes = new Map<string, Promise<THREE.BufferGeometry>>();
    const createGeometry = (visual: Visual): Promise<THREE.BufferGeometry> => {
      const size = visual.size ?? [];
      if (visual.type === 'cylinder') {
        const geometry = new THREE.CylinderGeometry(size[0], size[0], size[1], 40);
        geometry.rotateX(Math.PI / 2); // Three cylinders use Y; the fixture uses Z.
        return Promise.resolve(geometry);
      }
      if (visual.type === 'sphere') return Promise.resolve(new THREE.SphereGeometry(size[0], 32, 20));
      if (visual.type === 'box') return Promise.resolve(new THREE.BoxGeometry(...size));
      if (!visual.mesh) throw new Error('A scene mesh has no asset path.');
      if (meshData) {
        const bytes = meshData.get(visual.mesh);
        if (!bytes) throw new Error(`No mesh data for ${visual.mesh}.`);
        if (!meshes.has(visual.mesh)) meshes.set(visual.mesh, Promise.resolve(loader.parse(bytes)));
      }
      if (!meshes.has(visual.mesh)) meshes.set(visual.mesh, loader.loadAsync(`/assets/${visual.mesh.split('/').map(encodeURIComponent).join('/')}`));
      return meshes.get(visual.mesh)!;
    };
    await Promise.all(description.visuals.map(async (visual) => {
      let link = this.links.get(visual.link);
      if (!link) {
        link = new THREE.Group();
        link.name = visual.link;
        this.links.set(visual.link, link);
        this.scene.add(link);
        const helper = new THREE.AxesHelper(0.09);
        helper.name = visual.link;
        this.axes.add(helper);
      }
      const chrome = visual.material.includes('chrome');
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color().setRGB(...visual.rgba.slice(0, 3) as [number, number, number], THREE.SRGBColorSpace),
        roughness: this.benchmark ? 0.55 : chrome ? 0.3 : 0.48, metalness: this.benchmark ? 0 : chrome ? 0.68 : 0.12,
        side: visual.type === 'mesh' ? THREE.DoubleSide : THREE.FrontSide,
      });
      if (visual.material.includes('light') && !this.benchmark) {
        material.emissive.set('#ffe1a0');
        this.emitterMaterials.push(material);
      }
      const mesh = new THREE.Mesh(await createGeometry(visual), material);
      mesh.position.fromArray(visual.position);
      mesh.quaternion.fromArray(visual.quaternion);
      if (visual.scale) mesh.scale.fromArray(visual.scale);
      mesh.castShadow = !visual.material.includes('light');
      mesh.receiveShadow = true;
      link.add(mesh);
    }));
  }

  receive(state: State, atMs: number): void {
    this.buffer.push(state, atMs);
    this.lightEnabled = state.light.enabled;
    this.brightness = state.light.brightness;
  }

  /** Set the lamp light without a server state, as the embedded lamp does. */
  setLight(enabled: boolean, brightness: number): void {
    this.lightEnabled = enabled;
    this.brightness = brightness;
  }

  /** Pause or resume the render loop; resuming twice never starts a second loop. */
  setRunning(running: boolean): void {
    if (running === !this.stopped) return;
    this.stopped = !running;
    if (!running) {
      cancelAnimationFrame(this.frameRequest);
      return;
    }
    this.frames = 0;
    this.lastRateMs = performance.now();
    this.render(this.lastRateMs);
  }

  /** Stop rendering for good and release the canvas, controls and GPU resources. */
  dispose(): void {
    this.setRunning(false);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  setAxes(enabled: boolean): void { this.axes.visible = enabled; }
  resetView(): void {
    if (this.benchmark) {
      this.camera.position.set(-0.72, -1.15, 0.85);
      this.controls.target.set(-0.08, 0, 0.38);
    } else {
      this.camera.position.set(-1.1, -1.55, 1.07);
      this.controls.target.set(-0.03, 0, 0.36);
    }
    this.controls.update();
  }
  lightView(): void {
    // A side view keeps the lamp and its pool of light visible together.
    this.camera.position.set(0.8, -1.7, 1.0);
    this.controls.target.set(-0.45, 0, 0.42);
    this.controls.update();
  }

  private applyPoses(poses: Record<string, Pose>): void {
    for (const [name, pose] of Object.entries(poses)) {
      const link = this.links.get(name);
      if (link) { link.position.fromArray(pose.position); link.quaternion.fromArray(pose.quaternion); }
      const helper = this.axes.getObjectByName(name);
      if (helper) { helper.position.fromArray(pose.position); helper.quaternion.fromArray(pose.quaternion); }
    }
    const emitter = poses.light_emitter_link;
    if (emitter) {
      this.emitterPosition.fromArray(emitter.position);
      this.emitterDirection.set(1, 0, 0).applyQuaternion(new THREE.Quaternion(...emitter.quaternion));
      this.spotlight.position.copy(this.emitterPosition).addScaledVector(this.emitterDirection, this.benchmark ? 0 : 0.004);
      this.spotlight.target.position.copy(this.emitterPosition).add(this.emitterDirection);
    }
    const intensity = this.lightEnabled ? this.brightness : 0;
    this.spotlight.intensity = this.benchmark ? intensity : intensity * 2.8;
    for (const material of this.emitterMaterials) material.emissiveIntensity = intensity * 0.34;
  }

  private resize(): void {
    const { width, height } = this.element.getBoundingClientRect();
    if (width <= 0 || height <= 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    // Resizing clears the canvas, so a paused stage draws one frame to stay visible.
    if (this.stopped) this.renderer.render(this.scene, this.camera);
  }

  private render = (now: number): void => {
    if (this.stopped) return;
    this.frameRequest = requestAnimationFrame(this.render);
    const poses = this.options.poseSource ? this.options.poseSource(now) : this.buffer.sample(now);
    if (poses) this.applyPoses(poses);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.frames += 1;
    if (now - this.lastRateMs >= 1000) {
      this.fps = this.frames * 1000 / (now - this.lastRateMs);
      this.frames = 0;
      this.lastRateMs = now;
      this.onFrameRate(this.fps);
    }
  };
}
