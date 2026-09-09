# Brownian / SDE comparison implementation plan

**Goal:** A separate Playground experiment sharing temperature and number density between explicit collision dynamics and a calibrated overdamped SDE.
**Architecture:** A seeded, DOM-independent 2D ideal-gas bath/tracer ensemble supplies calibration data. After calibration, freeze D and compare fresh displacement origins against independently seeded Wiener increments. Canvas views and SVG charts consume bounded records.
**Tech stack:** TypeScript, Canvas 2D, SVG, existing Vite and Playwright; no new dependencies.
**Spec:** User-approved design in this conversation (2026-09-09).

## Constraints and model
- Preserve the water experiment, link both experiments, support relative deployment paths.
- Reduced units kB=1, bath mass=1, tracer mass=8; 64 noninteracting bath particles per replica, 64 independent replicas; periodic square box with L=sqrt(N/rho).
- Elastic bath/tracer collisions; bath particles may pass through each other. Fixed small timestep with overlap correction is an approximation, explicitly stated. Remove total momentum initially and normalize conserved kinetic energy to the selected temperature.
- Warm up for 20 time units; calibrate for 80; regress ensemble MSD over calibration times 40–80; D=slope/4. Reject nonpositive/nonfinite estimates. Show finite-window limitations, not a guaranteed asymptotic diffusion coefficient.
- Freeze D for a separate 60-unit comparison with reset origins. SDE dX=sqrt(2D)dW. Gamma=T/D is an effective inferred value, not an independent physical measurement.
- Reset all samples and calibration on temperature/density change. Bound histories; pause when hidden and under reduced motion; no WebGPU dependency.

## Tasks
- [x] Model: write unit tests for elastic momentum/energy conservation, thermal initialization, boundary-unwrapped displacement, seeded repeatability, independent SDE variance 2Dt per axis, regression slope/4 and invalid slopes. Implement in src/brownian-model.ts.
- [x] Experiment state: tests for warmup/calibration/comparison transitions, frozen calibration, reset and finite long runs. Implement src/brownian-experiment.ts.
- [x] Page: create brownian.html, src/brownian.ts, src/brownian.css and src/brownian-charts.ts. Link from playground.html and add Vite entry. Show shared controls, paired trajectories, MSD and x-displacement distributions, calibration status and equations/model limitations.
- [x] Verify: npm test, npm run build, targeted browser tests covering navigation, reset, pause/resume, complete calibration, parameter invalidation, reduced motion, no WebGPU and narrow-screen overflow. Inspect screenshot and review numerical/data lifecycle logic.

## Commands
`npx tsx --test tests/brownian*.test.ts`
`npm test`
`npm run build`
`npx playwright test tests/browser/brownian.spec.ts`

## Verification results
- `npm test`: 34 passed, including conservation, Wiener variance, unwrapped displacement, fitted D and experiment lifecycle.
- `npm run build`: TypeScript and production build passed.
- Browser tests for Brownian, existing Playground and homepage: 15 passed; mobile, no WebGPU, downloaded JSON contents and project-path deployment covered.
- Manual numerical sweep: (T, rho) = (0.5, 0.2), (0.5, 1), (3, 0.2), (3, 1), (1, 0.6) completed with finite positive fitted D. This is a stability check, not a convergence or physical validation claim.
- Desktop/mobile screenshots inspected. Independent code review found no important defects; its data-export test suggestion was implemented.
- Work is in the current workspace on `feat/brownian-sde-comparison`; no deployment performed.
