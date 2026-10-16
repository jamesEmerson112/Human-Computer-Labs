/**
 * Validates parsed gesture JSON the way lamp_sim/gestures.py does, so only inert, checked data reaches motion.
 *
 * APIs:
 * - GestureStep: one compiled step of a gesture
 * - Gesture: a validated gesture
 * - parseGesture: one parsed JSON object to a Gesture
 * - parseLibrary: a list of parsed JSON objects to a library keyed by name
 */

/**
 * Description: One validated step, as lamp_sim/gestures.py GestureStep holds it.
 * Fields:
 * - kind: 'targets', 'offsets', 'hold_seconds' or 'return_to_start'.
 * - angles: joint name and radians pairs in file order, empty for holds and returns.
 * - seconds: the step duration, 0.25 when the file gives none.
 * - onLimit: 'clamp' or 'reverse', 'clamp' when the file gives none.
 */
export interface GestureStep {
  kind: 'targets' | 'offsets' | 'hold_seconds' | 'return_to_start';
  angles: [string, number][];
  seconds: number;
  onLimit: 'clamp' | 'reverse';
}

/**
 * Description: A validated gesture, as lamp_sim/gestures.py Gesture holds it.
 * Fields:
 * - name, label: the unique name and display label.
 * - description: the description, empty when the file gives none.
 * - steps: the validated steps.
 * - group: the emotion family, empty for a utility clip.
 */
export interface Gesture {
  name: string;
  label: string;
  description: string;
  steps: GestureStep[];
  group: string;
}

/**
 * Description: Matches one character that Python's Unicode \s does not, which the schema's "\S" pattern needs.
 * Constraints: Python's \s adds the information separators U+001C to U+001F to Unicode White_Space.
 */
const NOT_SPACE = /[^\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/u;

/**
 * Description: The top-level keys gesture.schema.json allows.
 */
const GESTURE_KEYS = ['$schema', 'schema_version', 'name', 'label', 'description', 'group', 'units', 'steps'];

/**
 * Description: The step kinds in the order lamp_sim/gestures.py looks for them.
 */
const KINDS = ['targets', 'offsets', 'hold_seconds', 'return_to_start'] as const;

/**
 * Description: The keys each step kind allows.
 */
const STEP_KEYS: Record<GestureStep['kind'], string[]> = {
  targets: ['targets', 'duration_seconds'],
  offsets: ['offsets', 'duration_seconds', 'on_limit'],
  hold_seconds: ['hold_seconds'],
  return_to_start: ['return_to_start', 'duration_seconds'],
};

/**
 * Description: Tells whether a value is a plain JSON object.
 * Inputs:
 * - value: any parsed JSON value.
 * Outputs: true for an object that is not an array or null.
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Description: Rejects keys outside an allowed list.
 * Inputs:
 * - data: the object to check.
 * - keys: the allowed keys.
 * - location: a prefix for the error message.
 * Outputs: none.
 * Constraints: throws on the first unexpected key.
 */
function allowedKeys(data: Record<string, unknown>, keys: string[], location: string): void {
  const extra = Object.keys(data).find((key) => !keys.includes(key));
  if (extra !== undefined) throw new Error(`${location}: unexpected property ${extra}`);
}

/**
 * Description: Reads a string field with the schema's length and non-blank rules.
 * Inputs:
 * - data: the gesture object.
 * - key: the field name.
 * - required: whether the field must be present.
 * - max: the most code points allowed.
 * - nonblank: whether the text needs a character that is not whitespace.
 * Outputs: the text, or '' for an absent optional field.
 * Constraints: lengths count code points, as Python's len does.
 */
function textField(data: Record<string, unknown>, key: string, required: boolean, max: number, nonblank: boolean): string {
  if (!Object.hasOwn(data, key)) {
    if (required) throw new Error(`gesture: missing ${key}`);
    return '';
  }
  const value = data[key];
  if (typeof value !== 'string') throw new Error(`${key}: expected a string`);
  if ([...value].length > max || (nonblank && !NOT_SPACE.test(value))) throw new Error(`${key}: invalid length or blank text`);
  return value;
}

/**
 * Description: Reads a step duration with the schema's range.
 * Inputs:
 * - value: the duration field, or undefined when absent.
 * - location: a prefix for the error message.
 * Outputs: the seconds, 0.25 when absent.
 * Constraints: throws unless the value is a finite number above 0 and at most 60.
 */
function duration(value: unknown, location: string): number {
  if (value === undefined) return 0.25;
  if (typeof value !== 'number') throw new Error(`${location}: step duration must be a number`);
  if (!Number.isFinite(value) || !(value > 0 && value <= 60)) {
    throw new Error(`${location}: angles and durations must be finite numbers between 0 and 60 seconds`);
  }
  return value;
}

/**
 * Description: Validates one parsed gesture object, as parse_gesture in lamp_sim/gestures.py does with
 * gesture.schema.json.
 * Inputs:
 * - data: the parsed JSON value.
 * - jointNames: the joints the gesture may name.
 * Outputs: the validated Gesture.
 * Constraints: throws a readable error on the first problem. Duplicate JSON keys cannot be seen after
 * JSON.parse, so whoever parses the file text must reject them.
 */
export function parseGesture(data: unknown, jointNames: Iterable<string>): Gesture {
  const joints = new Set(jointNames);
  if (!isObject(data)) throw new Error('gesture: expected an object');
  allowedKeys(data, GESTURE_KEYS, 'gesture');
  if (Object.hasOwn(data, '$schema') && typeof data.$schema !== 'string') throw new Error('$schema: expected a string');
  if (data.schema_version !== 1) throw new Error('schema_version must be 1');
  if (data.units !== 'radians') throw new Error('units must be radians');
  const name = textField(data, 'name', true, 64, true);
  // Python's `$` also matches before one final newline.
  if (!/^[a-z][a-z0-9_]*$/.test(name.endsWith('\n') ? name.slice(0, -1) : name)) throw new Error('name must match ^[a-z][a-z0-9_]*$');
  const label = textField(data, 'label', true, 60, true);
  const description = textField(data, 'description', false, 500, false);
  const group = textField(data, 'group', false, 60, true);
  const values = data.steps;
  if (!Array.isArray(values)) throw new Error('steps must be an array');
  if (values.length < 1 || values.length > 128) throw new Error('steps must contain between 1 and 128 items');
  const steps = values.map((value: unknown, index): GestureStep => {
    const location = `steps.${index}`;
    if (!isObject(value)) throw new Error(`${location}: expected an object`);
    const kinds = KINDS.filter((key) => Object.hasOwn(value, key));
    if (kinds.length !== 1) throw new Error(`${location}: expected exactly one step kind`);
    const kind = kinds[0];
    allowedKeys(value, STEP_KEYS[kind], location);
    const angles: [string, number][] = [];
    if (kind === 'targets' || kind === 'offsets') {
      const entries = value[kind];
      if (!isObject(entries)) throw new Error(`${location}: expected an object`);
      if (Object.keys(entries).length === 0) throw new Error(`${location}: angles must not be empty`);
      for (const [joint, angle] of Object.entries(entries)) {
        if (typeof angle !== 'number') throw new Error(`${location}: angles must be numbers`);
        if (!joints.has(joint)) throw new Error(`${location}: unknown joint ${joint}`);
        if (!Number.isFinite(angle)) throw new Error(`${location}: angles and durations must be finite numbers`);
        angles.push([joint, angle]);
      }
    } else if (kind === 'return_to_start' && value.return_to_start !== true) {
      throw new Error(`${location}: return_to_start must be true`);
    }
    const seconds = duration(kind === 'hold_seconds' ? value.hold_seconds : value.duration_seconds, location);
    const onLimit = Object.hasOwn(value, 'on_limit') ? value.on_limit : 'clamp';
    if (onLimit !== 'clamp' && onLimit !== 'reverse') throw new Error(`${location}: on_limit must be clamp or reverse`);
    return { kind, angles, seconds, onLimit };
  });
  return { name, label, description, steps, group };
}

/**
 * Description: Validates a whole library, as load_gestures in lamp_sim/gestures.py does.
 * Inputs:
 * - items: parsed gesture objects, sorted by file name.
 * - jointNames: the joints the gestures may name.
 * Outputs: the gestures by name, in the given order.
 * Constraints: throws on any invalid gesture, a duplicate name or label, or an empty list. Error
 * messages name the item's position, because file names are not part of the data.
 */
export function parseLibrary(items: readonly unknown[], jointNames: Iterable<string>): Map<string, Gesture> {
  const joints = [...jointNames];
  const gestures = new Map<string, Gesture>();
  const labels = new Set<string>();
  items.forEach((item, index) => {
    let gesture: Gesture;
    try {
      gesture = parseGesture(item, joints);
    } catch (error) {
      throw new Error(`gesture ${index}: ${error instanceof Error ? error.message : String(error)}`);
    }
    if (gestures.has(gesture.name)) throw new Error(`gesture ${index}: duplicate gesture name '${gesture.name}'`);
    if (labels.has(gesture.label)) throw new Error(`gesture ${index}: duplicate gesture label '${gesture.label}'`);
    gestures.set(gesture.name, gesture);
    labels.add(gesture.label);
  });
  if (gestures.size === 0) throw new Error('No gestures were given.');
  return gestures;
}
