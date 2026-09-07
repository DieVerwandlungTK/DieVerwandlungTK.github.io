"""Export the initial explicit-water frame as the no-WebGPU SVG fallback.

Mirrors the projection, observation window and bond criteria of src/scene.ts so the
static image matches the first rendered frame.
"""
import json
import math
from pathlib import Path

import numpy as np

WINDOW = .66
ATOM_RADIUS = (.44, .26, .26)
YAW, PITCH = .57, .32
PIXELS = 270
OO_MAX, OHA_MIN = 3.5, 150.0

root = Path(__file__).resolve().parents[1]
meta = json.loads((root / 'public/data/water.json').read_text())
molecules, box = meta['particles'], meta['box']
scale = 1 / (WINDOW * box)
raw = np.fromfile(root / 'public/data/water.bin', dtype='<f4', count=molecules * 9).reshape(molecules, 3, 3)
sites = raw.astype(float)


def project(point):
    x, y, z = point - box / 2
    rx = x * math.cos(YAW) + z * math.sin(YAW)
    rz = -x * math.sin(YAW) + z * math.cos(YAW)
    ry = y * math.cos(PITCH) - rz * math.sin(PITCH)
    depth = (y * math.sin(PITCH) + rz * math.cos(PITCH)) * scale
    return 300 + rx * scale * PIXELS, 270 - ry * scale * PIXELS, depth


def opacity_of(center):
    t = min(1, max(0, (np.linalg.norm(center) / box - .42) / .24))
    return 1 - t * t * (3 - 2 * t)


def fog(depth):
    return min(.5, max(.04, .25 - depth * .27))


images = {}
for molecule in range(molecules):
    for cell in np.ndindex(3, 3, 3):
        shift = (np.array(cell) - 1) * box
        opacity = opacity_of(sites[molecule, 0] + shift - box / 2)
        if opacity > 0:
            images[(molecule, cell)] = (shift, opacity)

hydrogen_bonds = []
for donor in range(molecules):
    for acceptor in range(molecules):
        if donor == acceptor:
            continue
        delta = sites[acceptor, 0] - sites[donor, 0]
        shift = -box * np.round(delta / box)
        delta = delta + shift
        distance = np.linalg.norm(delta)
        if not .1 < distance <= OO_MAX:
            continue
        for hydrogen in (1, 2):
            oh = sites[donor, hydrogen] - sites[donor, 0]
            ha = delta - oh
            cosine = -float(oh @ ha) / (np.linalg.norm(oh) * np.linalg.norm(ha))
            angle = math.degrees(math.acos(max(-1, min(1, cosine))))
            if angle < OHA_MIN:
                continue
            strength = max(0, min(1, (OO_MAX - distance) / .18, (angle - OHA_MIN) / 8))
            if strength > 0:
                hydrogen_bonds.append((donor, hydrogen, acceptor, shift, strength))

parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 540">',
         '<defs>'
         '<radialGradient id="oxygen" cx="30%" cy="25%"><stop stop-color="#dcecf3"/>'
         '<stop offset=".35" stop-color="#5a95ad"/><stop offset="1" stop-color="#215877"/></radialGradient>'
         '<radialGradient id="hydrogen" cx="30%" cy="25%"><stop stop-color="#ffffff"/>'
         '<stop offset=".45" stop-color="#c9d7dd"/><stop offset="1" stop-color="#94a9b3"/></radialGradient>'
         '</defs>']

lines = ['<g fill="none" stroke-linecap="round">']
for (molecule, cell), (shift, opacity) in images.items():
    oxygen = project(sites[molecule, 0] + shift)
    for hydrogen in (1, 2):
        end = project(sites[molecule, hydrogen] + shift)
        alpha = opacity * .72 * (1 - fog((oxygen[2] + end[2]) / 2))
        lines.append(f'<path d="M{oxygen[0]:.2f} {oxygen[1]:.2f}L{end[0]:.2f} {end[1]:.2f}" '
                     f'stroke="#4f7889" stroke-opacity="{alpha:.2f}" stroke-width="2.1"/>')
for donor, hydrogen, acceptor, bond_shift, strength in hydrogen_bonds:
    for (molecule, cell), (shift, opacity) in images.items():
        if molecule != donor:
            continue
        target_cell = tuple(int(round(value)) + 1 for value in (shift + bond_shift) / box)
        neighbour = images.get((acceptor, target_cell))
        if neighbour is None:
            continue
        alpha = min(opacity, neighbour[1]) * strength * .62
        if alpha < .02:
            continue
        start = project(sites[donor, hydrogen] + shift)
        end = project(sites[acceptor, 0] + shift + bond_shift)
        lines.append(f'<path d="M{start[0]:.2f} {start[1]:.2f}L{end[0]:.2f} {end[1]:.2f}" '
                     f'stroke="#4f8fa6" stroke-opacity="{alpha * (1 - fog((start[2] + end[2]) / 2)):.2f}" '
                     f'stroke-width="1.5" stroke-dasharray="3.6 2.9"/>')
lines.append('</g>')
parts += lines

spheres = []
for (molecule, cell), (shift, opacity) in images.items():
    for atom in range(3):
        x, y, depth = project(sites[molecule, atom] + shift)
        radius = ATOM_RADIUS[atom] * scale * (1 + depth * .16) * PIXELS
        alpha = opacity * (1 - fog(depth))
        fill = 'oxygen' if atom == 0 else 'hydrogen'
        spheres.append((depth, f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
                               f'fill="url(#{fill})" opacity="{alpha:.2f}"/>'))
parts += [circle for _, circle in sorted(spheres, key=lambda item: item[0])]
parts.append('</svg>')
(root / 'public/molecules.svg').write_text('\n'.join(parts) + '\n')
print(f'Generated public/molecules.svg: {len(images)} molecule images, {len(hydrogen_bonds)} hydrogen bonds')
