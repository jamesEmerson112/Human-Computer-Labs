/**
 * Reads the lamp URDF text into a joint chain and joint limits without a DOM parser, so it also runs in Node.
 *
 * APIs:
 * - UrdfJoint: one joint of the chain
 * - UrdfModel: the parsed links, joints and limits
 * - parseUrdf: URDF text to a UrdfModel
 */
import type { Limit, Vec3 } from '../protocol';

/**
 * Description: One revolute or fixed joint as the URDF states it.
 * Fields:
 * - name: the joint name.
 * - type: 'revolute' or 'fixed'.
 * - parent, child: the link names the joint connects.
 * - xyz, rpy: the joint origin in the parent link frame, in metres and radians.
 * - axis: the rotation axis in the joint frame, as written.
 */
export interface UrdfJoint {
  name: string;
  type: 'revolute' | 'fixed';
  parent: string;
  child: string;
  xyz: Vec3;
  rpy: Vec3;
  axis: Vec3;
}

/**
 * Description: The kinematic content of a URDF.
 * Fields:
 * - links: link names in file order.
 * - joints: joints in file order.
 * - limits: the hard limit intersected with the soft limit for each revolute joint, in file order.
 */
export interface UrdfModel {
  links: string[];
  joints: UrdfJoint[];
  limits: Record<string, Limit>;
}

/**
 * Description: A decimal number as Python's float() reads it, without the inf and nan spellings,
 * which the finite checks reject anyway.
 */
const DECIMAL = /^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$/;

/**
 * Description: Reads every attribute of one XML start tag.
 * Inputs:
 * - tag: the tag text between its name and closing angle bracket.
 * Outputs: the attributes by name.
 */
function attributes(tag: string): Record<string, string> {
  const found: Record<string, string> = {};
  for (const match of tag.matchAll(/([\w:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)) found[match[1]] = match[2] ?? match[3];
  return found;
}

/**
 * Description: Finds the first element of a name inside some XML text.
 * Inputs:
 * - body: the XML text to search.
 * - name: the element name.
 * Outputs: the element's attributes, or undefined when it is absent.
 */
function element(body: string, name: string): Record<string, string> | undefined {
  const match = new RegExp(`<${name}(?=[\\s/>])([^>]*)>`).exec(body);
  return match ? attributes(match[1]) : undefined;
}

/**
 * Description: Parses a finite decimal number.
 * Inputs:
 * - text: the attribute text.
 * - what: a name for the error message.
 * Outputs: the number.
 * Constraints: throws when the text is absent, empty, hexadecimal or not finite, where Number() would
 * quietly accept some of these.
 */
function number(text: string | undefined, what: string): number {
  const value = text !== undefined && DECIMAL.test(text) ? Number(text) : NaN;
  if (!Number.isFinite(value)) throw new Error(`Invalid ${what}: "${text ?? ''}".`);
  return value;
}

/**
 * Description: Parses three space-separated numbers.
 * Inputs:
 * - text: the attribute text, or undefined for the fallback.
 * - fallback: the value when the attribute is absent.
 * - what: a name for the error message.
 * Outputs: the three numbers.
 * Constraints: throws when the text does not hold exactly three finite numbers.
 */
function vector(text: string | undefined, fallback: Vec3, what: string): Vec3 {
  if (text === undefined) return fallback;
  const parts = text.trim().split(/\s+/);
  if (parts.length !== 3) throw new Error(`Invalid ${what}: "${text}".`);
  return parts.map((part) => number(part, what)) as Vec3;
}

/**
 * Description: Parses URDF text into links, joints and limits; the limits follow load_limits in
 * lamp_sim/motion.py.
 * Inputs:
 * - text: the URDF file text.
 * Outputs: the parsed model.
 * Constraints: XML comments are removed first. Throws on a joint type other than revolute or fixed, a
 * joint without a name, parent or child, a revolute joint without a limit, bad numbers, or a model with
 * no revolute joint. Entities in attribute values are not decoded.
 */
export function parseUrdf(text: string): UrdfModel {
  // The repeat counts keep the HTML comment opener out of the bundle text, which pages inline.
  const source = text.replace(/<!-{2}[\s\S]*?-{2}>/g, '');
  const links = [...source.matchAll(/<link(?=[\s/>])([^>]*)>/g)].map((match) => {
    const name = attributes(match[1]).name;
    if (!name) throw new Error('A link has no name.');
    return name;
  });
  const joints: UrdfJoint[] = [];
  const limits: Record<string, Limit> = {};
  for (const match of source.matchAll(/<joint(?=[\s/>])([^>]*?)(?:\/>|>([\s\S]*?)<\/joint>)/g)) {
    const head = attributes(match[1]);
    const body = match[2] ?? '';
    const name = head.name;
    if (head.type !== 'revolute' && head.type !== 'fixed') throw new Error(`Joint ${name} has unsupported type ${head.type}.`);
    const parent = element(body, 'parent')?.link;
    const child = element(body, 'child')?.link;
    if (!name || !parent || !child) throw new Error(`Joint ${name ?? '(unnamed)'} lacks a name, parent or child.`);
    const origin = element(body, 'origin') ?? {};
    joints.push({
      name, type: head.type, parent, child,
      xyz: vector(origin.xyz, [0, 0, 0], `${name} origin xyz`),
      rpy: vector(origin.rpy, [0, 0, 0], `${name} origin rpy`),
      axis: vector(element(body, 'axis')?.xyz, [1, 0, 0], `${name} axis`),
    });
    if (head.type !== 'revolute') continue;
    const hard = element(body, 'limit');
    if (!hard) throw new Error(`Missing limits for ${name}.`);
    let lower = number(hard.lower, `${name} lower limit`);
    let upper = number(hard.upper, `${name} upper limit`);
    const soft = element(body, 'safety_controller');
    if (soft) {
      lower = Math.max(lower, soft.soft_lower_limit === undefined ? lower : number(soft.soft_lower_limit, `${name} soft lower limit`));
      upper = Math.min(upper, soft.soft_upper_limit === undefined ? upper : number(soft.soft_upper_limit, `${name} soft upper limit`));
    }
    const velocity = number(hard.velocity, `${name} velocity`);
    if (lower > upper || velocity <= 0) throw new Error(`Invalid limits for ${name}.`);
    limits[name] = { lower, upper, velocity };
  }
  if (Object.keys(limits).length === 0) throw new Error('The model has no revolute joints.');
  return { links, joints, limits };
}
