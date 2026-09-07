import { findHydrogenBonds, molecularOpacity, periodicMolecules, type HydrogenBond, type Vector3 } from './water-geometry';

/** Atom: clip x,y, depth, radius | opacity, kind (0 oxygen, 1 hydrogen), 0, 0. */
export const ATOM_STRIDE = 8;
/** Bond vertex: clip x,y, depth, dash phase | opacity, kind (0 covalent, 1 hydrogen bond), 0, 0. */
export const BOND_STRIDE = 8;

const IMAGES = 27;
/** Molecules fade out by this fraction of the box; see molecularOpacity. */
const WINDOW = .66;
const ATOM_RADIUS = [.44, .26, .26];
const YAW = .57, PITCH = .32;
const DASHES_PER_CLIP_UNIT = 34;
/** Geometric criteria admit at most a few acceptors per donor molecule. */
const HYDROGEN_BONDS_PER_MOLECULE = 6;

/** Maps the observation sphere onto the vertical clip range. */
export const viewScale = (box: number) => 1 / (WINDOW * box);

export interface SceneCounts { atomCount: number; bondVertices: number; hydrogenBonds: number }

export function createScene(particles: number) {
  const atomCapacity = particles * IMAGES * 3;
  const bondCapacity = particles * IMAGES * (2 + HYDROGEN_BONDS_PER_MOLECULE);
  const atoms = new Float32Array(atomCapacity * ATOM_STRIDE);
  const scratch = new Float32Array(atoms.length);
  const bonds = new Float32Array(bondCapacity * 2 * BOND_STRIDE);
  const order = new Int32Array(atomCapacity);
  const imageOfCell = new Int32Array(particles * IMAGES);
  const donors: HydrogenBond[][] = Array.from({ length: particles }, () => []);
  const sin = [Math.sin(YAW), Math.sin(PITCH)], cos = [Math.cos(YAW), Math.cos(PITCH)];
  const projected: Vector3 = [0, 0, 0];

  /** Rotates a box-centred position in angstrom into clip space with its depth. */
  function project(x: number, y: number, z: number, scale: number): Vector3 {
    const rx = x * cos[0] + z * sin[0], rz = -x * sin[0] + z * cos[0];
    projected[0] = rx * scale;
    projected[1] = (y * cos[1] - rz * sin[1]) * scale;
    projected[2] = (y * sin[1] + rz * cos[1]) * scale;
    return projected;
  }
  const cellOfShift = (shift: Vector3, box: number) =>
    (Math.round(shift[0] / box) + 1) * 9 + (Math.round(shift[1] / box) + 1) * 3 + Math.round(shift[2] / box) + 1;

  function writeBond(target: number, position: Vector3, phase: number, opacity: number, kind: number): void {
    bonds[target] = position[0];
    bonds[target + 1] = position[1];
    bonds[target + 2] = position[2];
    bonds[target + 3] = phase;
    bonds[target + 4] = opacity;
    bonds[target + 5] = kind;
  }

  /** The box arrives per update, so density and molecule-count changes rescale the view. */
  function update(sites: Float32Array, box: number): SceneCounts {
    const scale = viewScale(box);
    const images = periodicMolecules(sites, box);
    imageOfCell.fill(-1);
    for (let i = 0; i < images.length; i++) {
      imageOfCell[images[i].molecule * IMAGES + cellOfShift(images[i].shift, box)] = i;
    }
    let atomCount = 0;
    for (const image of images) {
      const base = image.molecule * 12;
      for (let atom = 0; atom < 3; atom++) {
        const clip = project(
          sites[base + atom * 4] + image.shift[0] - box / 2,
          sites[base + atom * 4 + 1] + image.shift[1] - box / 2,
          sites[base + atom * 4 + 2] + image.shift[2] - box / 2, scale);
        const target = atomCount * ATOM_STRIDE;
        scratch[target] = clip[0];
        scratch[target + 1] = clip[1];
        scratch[target + 2] = clip[2];
        scratch[target + 3] = ATOM_RADIUS[atom] * scale;
        scratch[target + 4] = image.opacity;
        scratch[target + 5] = atom === 0 ? 0 : 1;
        order[atomCount] = atomCount++;
      }
    }
    // Painter's order: distant atoms first, so nearer spheres blend over them.
    const sorted = order.subarray(0, atomCount);
    sorted.sort((a, b) => scratch[a * ATOM_STRIDE + 2] - scratch[b * ATOM_STRIDE + 2]);
    for (let i = 0; i < atomCount; i++) {
      atoms.set(scratch.subarray(sorted[i] * ATOM_STRIDE, (sorted[i] + 1) * ATOM_STRIDE), i * ATOM_STRIDE);
    }

    for (const list of donors) list.length = 0;
    let hydrogenBonds = 0;
    for (const bond of findHydrogenBonds(sites, box)) {
      if (bond.strength <= 0) continue;
      donors[bond.donor].push(bond);
      hydrogenBonds++;
    }
    let vertices = 0;
    const segment = (from: Vector3, to: Vector3, opacity: number, kind: number) => {
      if (vertices + 2 > bondCapacity * 2) return;
      const length = Math.hypot(to[0] - from[0], to[1] - from[1], to[2] - from[2]);
      writeBond(vertices++ * BOND_STRIDE, from, 0, opacity, kind);
      writeBond(vertices++ * BOND_STRIDE, to, kind === 1 ? length * DASHES_PER_CLIP_UNIT : 0, opacity, kind);
    };
    const site = (molecule: number, atom: number, shift: Vector3): Vector3 => {
      const base = molecule * 12 + atom * 4;
      return [...project(sites[base] + shift[0] - box / 2, sites[base + 1] + shift[1] - box / 2,
        sites[base + 2] + shift[2] - box / 2, scale)];
    };
    for (const image of images) {
      const oxygen = site(image.molecule, 0, image.shift);
      for (const hydrogen of [1, 2]) segment(oxygen, site(image.molecule, hydrogen, image.shift), image.opacity, 0);
      for (const bond of donors[image.molecule]) {
        const cells = [0, 1, 2].map(axis => Math.round((image.shift[axis] + bond.shift[axis]) / box));
        if (cells.some(index => Math.abs(index) > 1)) continue;
        const acceptor = imageOfCell[bond.acceptor * IMAGES + (cells[0] + 1) * 9 + (cells[1] + 1) * 3 + cells[2] + 1];
        if (acceptor < 0) continue;
        const opacity = Math.min(image.opacity, images[acceptor].opacity) * bond.strength;
        if (opacity < .02) continue;
        const shift = image.shift.map((value, axis) => value + bond.shift[axis]) as Vector3;
        segment(site(image.molecule, bond.hydrogen, image.shift), site(bond.acceptor, 0, shift), opacity, 1);
      }
    }
    return { atomCount, bondVertices: vertices, hydrogenBonds };
  }

  return { atoms, bonds, atomCapacity, bondCapacity, update };
}
