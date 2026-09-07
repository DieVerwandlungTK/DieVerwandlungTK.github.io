export type Vector3 = [number, number, number];

/** TIP4P-Ew (Horn et al. 2004) in angstrom, kJ/mol, e, amu, ps. */
export const OH_LENGTH = 0.9572;
export const HOH_ANGLE = 104.52 * Math.PI / 180;
export const M_OFFSET = 0.125;
export const Q_H = 0.52422;
export const Q_M = -1.04844;
export const SIGMA_O = 3.16435;
export const EPSILON_O = 0.680946;
export const MASS_O = 15.99943;
export const MASS_H = 1.007947;
export const MOLECULE_MASS = MASS_O + 2 * MASS_H;
export const COULOMB = 1389.35456;
export const BOLTZMANN = 0.0083144626;
/** Acceleration in A/ps^2 per (kJ/mol/A) of force and amu of mass. */
export const FORCE_TO_ACCELERATION = 100;
/** Conventional ice Ic cell holding eight molecules. */
export const ICE_CELL = 6.35;
export const MOLECULE_COUNTS = [64, 216, 512] as const;

/**
 * Sites in the body frame: x along the bisector towards the hydrogens, y in the
 * molecular plane, z out of plane, origin at the centre of mass.
 */
export function bodySites(): { oxygen: Vector3; hydrogen: [Vector3, Vector3]; charge: Vector3 } {
  const half = HOH_ANGLE / 2;
  const along = OH_LENGTH * Math.cos(half), across = OH_LENGTH * Math.sin(half);
  const shift = 2 * MASS_H * along / MOLECULE_MASS;
  return {
    oxygen: [-shift, 0, 0],
    hydrogen: [[along - shift, across, 0], [along - shift, -across, 0]],
    charge: [M_OFFSET - shift, 0, 0],
  };
}

/** Diagonal inertia tensor of the rigid molecule in amu A^2. */
export function principalMoments(): Vector3 {
  const { oxygen, hydrogen } = bodySites();
  const sites: [number, Vector3][] = [[MASS_O, oxygen], [MASS_H, hydrogen[0]], [MASS_H, hydrogen[1]]];
  const moment = (first: number, second: number) =>
    sites.reduce((sum, [mass, site]) => sum + mass * (site[first] ** 2 + site[second] ** 2), 0);
  return [moment(1, 2), moment(0, 2), moment(0, 1)];
}

export const cellsFor = (molecules: number): number => Math.round((molecules / 8) ** (1 / 3));

/** Ice density is eight molecules per cell; the ratio compresses that isotropically. */
export const boxLength = (molecules: number, densityRatio: number): number =>
  ICE_CELL * cellsFor(molecules) / densityRatio ** (1 / 3);

/** Stays under half the box so the minimum image convention holds. */
export const cutoffFor = (box: number): number => Math.min(9, 0.49 * box);
