/**
 * Maps a MELD emotion label to a lamp clip, picked at random from that emotion's family.
 *
 * APIs:
 * - Family: a core emotion family, or Calm for neutral
 * - Clip: the clip chosen for a MELD label
 * - MELD_FAMILIES: MELD's six non-neutral labels to their emotion wheel family
 * - CALM_POOL: the clips neutral picks from
 * - Chooser: picks clips without repeating the previous pick for a family
 */
import type { Gesture } from './gestures';

/**
 * Description: A core emotion family of the gesture library, or Calm for MELD's neutral.
 */
export type Family = 'Happy' | 'Sad' | 'Angry' | 'Fearful' | 'Surprised' | 'Disgusted' | 'Calm';

/**
 * Description: The clip chosen for a MELD label.
 * Fields:
 * - name: the gesture name.
 * - label: the gesture label, such as its emotion wheel path.
 * - family: the core emotion family, or Calm for neutral.
 * - calm: true when the clip came from the calm pool.
 */
export interface Clip {
  name: string;
  label: string;
  family: Family;
  calm: boolean;
}

/**
 * Description: MELD's six non-neutral labels to the emotion wheel family whose clips they play.
 */
export const MELD_FAMILIES: Readonly<Record<string, Exclude<Family, 'Calm'>>> = {
  joy: 'Happy',
  sadness: 'Sad',
  anger: 'Angry',
  fear: 'Fearful',
  surprise: 'Surprised',
  disgust: 'Disgusted',
};

/**
 * Description: The utility clips MELD's neutral picks from.
 */
export const CALM_POOL: readonly string[] = ['idle_hold', 'nod', 'curious', 'scan'];

/**
 * Description: Picks a clip for each MELD label, uniformly from its family's pool.
 * Constraints: a family's pool is every gesture whose group equals the family, the core clip
 * included; neutral uses the calm clips present in the library. A pick never repeats the previous
 * pick for the same family when the pool holds more than one clip.
 */
export class Chooser {
  private pools = new Map<Family, Gesture[]>();
  private last = new Map<Family, string>();

  /**
   * Description: Builds the pools from a library.
   * Inputs:
   * - gestures: the validated gestures, in library order.
   * - random: returns a number in [0, 1); injected so tests can seed it.
   * Outputs: the chooser.
   */
  constructor(gestures: Iterable<Gesture>, private random: () => number = Math.random) {
    const list = [...gestures];
    for (const family of new Set(Object.values(MELD_FAMILIES))) this.pools.set(family, list.filter((gesture) => gesture.group === family));
    this.pools.set('Calm', CALM_POOL.flatMap((name) => list.filter((gesture) => gesture.name === name)));
  }

  /**
   * Description: Lists the clip names a MELD label can pick.
   * Inputs:
   * - meldLabel: a MELD emotion label.
   * Outputs: the pool's gesture names.
   * Constraints: throws on an unknown label.
   */
  pool(meldLabel: string): string[] {
    return this.poolFor(this.family(meldLabel)).map((gesture) => gesture.name);
  }

  /**
   * Description: Picks a clip for a MELD label.
   * Inputs:
   * - meldLabel: one of neutral, joy, surprise, anger, sadness, disgust or fear; case and surrounding
   *   spaces are ignored.
   * Outputs: the clip, or null when the library has no clip for that family.
   * Constraints: throws on any other label.
   */
  choose(meldLabel: string): Clip | null {
    const family = this.family(meldLabel);
    const pool = this.poolFor(family);
    if (pool.length === 0) return null;
    const previous = this.last.get(family);
    const options = pool.length > 1 ? pool.filter((gesture) => gesture.name !== previous) : pool;
    const index = Math.min(options.length - 1, Math.floor(this.random() * options.length));
    const gesture = options[Math.max(0, index)];
    this.last.set(family, gesture.name);
    return { name: gesture.name, label: gesture.label, family, calm: family === 'Calm' };
  }

  /**
   * Description: Finds the family of a MELD label.
   * Inputs:
   * - meldLabel: the label.
   * Outputs: the family, Calm for neutral.
   * Constraints: throws on an unknown label.
   */
  private family(meldLabel: string): Family {
    const label = typeof meldLabel === 'string' ? meldLabel.trim().toLowerCase() : '';
    if (label === 'neutral') return 'Calm';
    const family = Object.hasOwn(MELD_FAMILIES, label) ? MELD_FAMILIES[label] : undefined;
    if (!family) throw new Error(`Unknown MELD emotion: ${String(meldLabel)}`);
    return family;
  }

  /**
   * Description: The pool of a family.
   * Inputs:
   * - family: the family.
   * Outputs: its gestures, possibly empty.
   */
  private poolFor(family: Family): Gesture[] {
    return this.pools.get(family) ?? [];
  }
}
