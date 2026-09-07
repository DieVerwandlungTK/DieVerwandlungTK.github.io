# Speed distribution implementation plan

Spec: `docs/superpowers/specs/2026-09-08-speed-distribution-design.md`.

- [x] Add pure distribution functions using water-model constants and velocity xyz
  at offsets 8–10 of the 16-float molecular state. Test normalization by quadrature,
  analytical peaks, seeded velocity sampling, overflow and EMA response before wiring UI.
- [x] Add a theme-aware 340×64 inline SVG with 12 bars, fixed 0–18 Å/ps and
  0–0.26 ps/Å axes, a reference curve and an updated accessible label. Add the
  canonical-sampling caveat beside it; hide unavailable readouts.
- [x] Extend the existing stats tick to 200 ms and pair readStats/readState in the
  same synchronous request batch. Guard overlapping reads and invalidate pending
  samples on reset, count changes and disposal. Initialize EMA from the first sample
  so every shown histogram is normalized, including under reduced motion.
- [x] Verify Node tests, build and browser tests for geometry, measured-temperature
  labeling, heating response at 216 molecules, pause/reset/reduced motion, and mobile
  overflow. Inspect screenshots for panel overlap and fix layout if required.

Keep the current realtime simulation, GPU kernels and buffer layout unchanged.
Work in the existing user checkout; no deployment or commit is requested.

## Results

26 Node tests and 15 homepage/distribution browser tests passed; production build
and diff whitespace checks passed. Heating from 200 K to 500 K shifted the histogram
centre by more than 1.5 Å/ps and beyond its measured cold fluctuation at 216 molecules.
Desktop and mobile screenshots were inspected; mobile panel now flows below the
molecular view to avoid covering the introduction after adding the chart.

Review caught stale force diagnostics after a paused reset. A regression reproduced
an endless reset loop; clearing the existing forceTorque buffer in load() fixes it.
step() recomputes forces before integration, so this only resets old diagnostics;
the GPU kernels and state layout are unchanged. Scoped re-review approved the fix.
