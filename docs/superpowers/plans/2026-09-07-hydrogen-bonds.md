# Explicit water and hydrogen-bond playback

User requests explicit hydrogen-bond breaking/reforming and more visible thermal
motion, following concern about particles teleporting across periodic boundaries.

## Design

Replace the displayed mW sample with an explicit rigid O-H-H water simulation using
OpenMM TIP4P-Ew (virtual charge site not drawn). Start from proton-disordered ice Ic
and heat at fixed periodic volume. Keep model/protocol limitations visible in the
background caption. Hydrogen bonds are a geometric analysis of the trajectory,
not extra chemical springs. Distinguish solid intramolecular O-H bonds from dashed
intermolecular hydrogen bonds. Use O-O <=3.5 Å and O-H···O >=150 degrees, with
minimum-image distances. Rigid O-H bonds do not break during heating.

Use 64 molecules for a legible enlarged molecular view. Sample approximately every
0.02 ps; store O,H1,H2 Cartesian coordinates in a little-endian float32 binary file.
The metadata format is version 2: particles means molecule count; box in Å;
atomOrder [O,H,H]; coordinateFile 'water.bin'; frames contain temperature, timePs,
and optional kineticTemperature/structural metrics. Frame-major, molecule-major,
O xyz,H1 xyz,H2 xyz. Wrap O into [0,box), keep its H sites whole with O (H can be
outside the box). Header has model, protocol, seed, timestepFs, citation.

Decoder attaches positions to frames for existing sampling interfaces, validates
exact length/finiteness. Interpolate O with minimum image and local H orientations
without tearing molecules across the box. Render 27 neighboring cell images only
within a spherical observation window, fade whole molecules to zero before culling.
This is a window into periodic bulk water, not a droplet or free surface.

Render the true trajectory at a constant chosen speed, with user-selectable 0.5x,
1x,2x,4x rates and visible simulation time. Increase default temporal throughput
relative to old 48s display; do not artificially scale velocities with temperature.

## Tasks
- [x] Offline: independent simulation implementer owns scripts/generate_explicit_water.py,
  scripts/test_explicit_water.py, public/data/water.json and water.bin, and a concise
  docs/explicit-water-simulation.md report. Run force/geometry/data checks; document
  thermal and hydrogen-bond changes. Do not replace outputs until simulation passes.
- [x] Frontend: parent owns decoder, molecular geometry, tests, vgpu shaders,
  playback controls, static export, caption, documentation and browser tests.
- [x] Integration: generate matching fallback, unit tests, production build,
  GPU screenshots at low/high temperatures, boundary continuity, pause/scrub/speed,
  reduced motion and asset failure tests. Review final code independently.

## Execution notes
The existing empty/uncommitted repository is the user's working preview. Work in
place and preserve all homepage content. Do not deploy without the remote details.
