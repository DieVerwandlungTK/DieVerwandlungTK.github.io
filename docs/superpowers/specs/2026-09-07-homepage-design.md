# Research homepage design

The user selected vgpu and precomputed trajectories, requested placeholder content,
and has no trajectory data. Build a static single-page research homepage with About,
Research, Publications, Presentations, and Links. Use Japanese body copy and English
section labels. No invented personal achievements or live placeholder contact links.

Use vgpu 0.4.0 to render the background. Generate a reproducible mW coarse-grained
water trajectory offline, starting with ice Ic and heating at fixed volume. Each
particle represents a molecule; no explicit hydrogen atoms or bonds. Describe the
model and heating protocol accurately, without claiming equilibrium melting-point
measurement. Store physical metadata and trajectory alongside the site. Validate
forces against finite differences and check structural disorder before describing
the observed trajectory as melting.

Use a pale blue canvas, dark navy typography, restrained orange highlights, generous
space, and a molecular illustration on the right of the first screen. Text panels
remain readable over the background. Playback supports pause, replay, scrubbing,
reduced motion, background-tab suspension, and static fallback when GPU or loading
fails. Play forward only; never present a reversed trajectory as freezing.

Vite builds with relative asset paths for GitHub Pages project subdirectories.
Include a Pages workflow and local/deployment instructions; no remote is configured,
so publishing requires the user's GitHub repository details.
