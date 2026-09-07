# Speed distribution readout

Add a small chart to the simulation panel showing the molecules' translational speed
distribution as a histogram, with the Maxwell-Boltzmann curve for the measured kinetic
temperature drawn over it. Moving the temperature slider then visibly widens and
flattens the distribution.

## What this does and does not show

In any classical system at equilibrium the momentum distribution is Maxwell-Boltzmann
regardless of the interactions, and this simulation's Langevin thermostat imposes that
distribution directly in its O step. The agreement is therefore not evidence about
water; it is a check that the sampling is canonical, and a way to see the temperature
rather than only read it. The caption says so.

The potential-energy distribution would carry information about the interactions, but
it has no analytic reference curve, so it is out of scope here.

## The chart

Translational speed only. The histogram holds 12 bins of probability density over
0 to 18 A/ps, and the curve is

    f(v) = (4 / sqrt(pi)) a^{3/2} v^2 exp(-a v^2),   a = m / (2 kB T)

with `m = 18.015324` amu, `kB = 0.0083144626` kJ/mol/K and the project's
`FORCE_TO_ACCELERATION = 100` conversion, evaluated at the measured translational
kinetic temperature from `readStats()`.

**Both axes are fixed.** The horizontal axis runs 0 to 18 A/ps; the vertical axis is
fixed to the theoretical peak at 150 K, 0.2231 ps/A. Auto-scaling either axis would
keep the shape constant as the temperature changes and hide the one thing the chart
exists to show. With fixed axes the area stays constant while the distribution moves:
the most probable speed is 3.72 A/ps at 150 K, 4.08 at the 180 K default, 5.26 at
300 K and 6.79 at 500 K, and the peak density falls from 0.2231 to 0.1222 across that
range. The 18 A/ps upper bound holds 99.7% of the distribution at 500 K and all of it
below 300 K; speeds beyond it are counted in the last bin rather than dropped, so the
histogram always integrates to one.

Bins update by an exponential moving average with a weight of 1/3 per sample. Sampled
every 200 ms, that is a time constant of about half a second: fast enough to follow the
slider, slow enough that the sqrt(N) fluctuation reads as texture rather than noise. At
64 molecules a bin holds about 5 molecules and visibly jitters; at 512 it is smooth.

## Data flow

`src/main.ts` already polls `readStats()` on a timer. Raise that timer from 500 ms to
200 ms and read `readState()` on the same tick; its 16 floats per molecule already carry
the velocities, so no shader or buffer change is needed and the extra readback is 32 KB
at 512 molecules. No new timer, no second readback path.

## Modules

- `src/speed-distribution.ts` (new, pure functions, no GPU and no DOM):
  `SPEED_BINS = 12`, `SPEED_AXIS_MAX = 18`, `DENSITY_AXIS_MAX = 0.2231`,
  `SMOOTHING = 1 / 3`, `maxwellBoltzmann(speed, temperature)`,
  `mostProbableSpeed(temperature)`, `speedHistogram(state, molecules, out)` writing
  probability density per bin, and `smoothHistogram(current, sample)` applying the
  moving average in place.
- `src/distribution-chart.ts` (new): owns the inline SVG's bar rectangles and the curve
  path, and exposes `update(bins, temperature)`, which rewrites their geometry and the
  container's `aria-label`. SVG rather than canvas: it inherits the page's theme, and a
  browser test can assert on the DOM.
- `index.html` and `src/style.css`: a 340x64 chart under the metrics row, on the same
  translucent backing `.simulation-details[open]` already uses, since the panel is
  transparent and sits over the molecules.
- `src/main.ts`: the polling change and a few lines handing the state to the chart.

## Accessibility and reduced motion

The chart carries `role="img"` and an `aria-label` naming the measured temperature and
the most probable speed, refreshed with the chart. Under `prefers-reduced-motion` the
simulation does not step, so the chart draws the initial configuration once and stays
still.

## Verification

Node tests for `src/speed-distribution.ts`:

- the curve integrates to 1 within 1e-3 by numerical quadrature at 150, 300 and 500 K;
- its maximum sits at `sqrt(2 kB T / m)` and its peak value matches the closed form;
- a histogram of seeded Maxwell-Boltzmann velocities matches the curve within 5%
  relative per bin for a large sample, which cross-checks the two functions against
  each other;
- speeds above `SPEED_AXIS_MAX` land in the last bin and the histogram sums to one;
- `smoothHistogram` reaches 1 - 1/e of a step change in three samples.

Browser tests:

- twelve bars exist with finite heights, and the curve path has a non-empty geometry;
- the histogram's centre of mass at a 500 K set point is higher than at 200 K, by more
  than the fluctuation at 216 molecules;
- the `aria-label` reports the measured temperature;
- at 375 px the page still has no horizontal overflow.
