// Decorative diffusion over the real text. This is not model inference.
const vocabulary = Array.from('言葉知識空波光夢世界問いアイウエオカキクケコabcdefghijklmnopqrstuvwxyz');
const duration = 1600;

export function initDiffusion() {
  const main = document.querySelector('main');
  if (!main) return;
  const mode = Math.random() < 0.5 ? 'mask' : 'uniform';
  main.dataset.diffusion = mode;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let cancel = () => {};
  let manuallyStarted = false;
  const replay = document.getElementById('diffusion-replay');
  if (replay) {
    replay.hidden = false;
    replay.addEventListener('click', () => {
      manuallyStarted = true;
      play();
    });
  }

  function play() {
    if (!main) return;
    cancel();

    const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        return node.textContent?.trim() && !node.parentElement?.closest('[aria-hidden="true"], script, style, .sr-only, button')
          ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      },
    });
    const originals: Text[] = [];
    while (walker.nextNode()) originals.push(walker.currentNode as Text);
    const pending: { element: HTMLElement; overlay: HTMLElement; text: string; at: number }[] = [];
    const runs: { element: HTMLElement; original: Text }[] = [];
    for (const original of originals) {
      const fragment = document.createElement('diffusion-run');
      runs.push({ element: fragment, original });
      for (const text of original.data.match(/\s+|[a-zA-Z0-9]+|[^\s]/gu) ?? []) {
        if (!text.trim()) {
          fragment.append(text);
          continue;
        }
        const element = document.createElement('diffusion-token');
        element.className = 'text-diffusion-token';
        element.textContent = text;
        const overlay = document.createElement('diffusion-noise');
        overlay.className = 'text-diffusion-noise';
        overlay.setAttribute('aria-hidden', 'true');
        element.append(overlay);
        fragment.append(element);
        // Spread reveal times even when a random source produces repeated values.
        const at = 0.15 + ((pending.length * 0.61803398875 + Math.random() * 0.25) % 1) * 0.85;
        pending.push({ element, overlay, text, at });
      }
      original.replaceWith(fragment);
    }

    let elapsed = 0;
    let previous: number | undefined;
    let frame = 0;
    let lastStep = -1;
    function finish() {
      cancelAnimationFrame(frame);
      for (const run of runs) run.element.replaceWith(run.original);
      pending.length = 0;
      reducedMotion.removeEventListener('change', onMotionChange);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    }
    function onMotionChange() { if (reducedMotion.matches) finish(); }
    function onVisibilityChange() {
      cancelAnimationFrame(frame);
      previous = undefined;
      if (!document.hidden) frame = requestAnimationFrame(render);
    }
    function render(now: number) {
      if (document.hidden) return;
      if (previous !== undefined) elapsed += now - previous;
      previous = now;
      const progress = Math.min(1, elapsed / duration);
      if (progress >= 1) { finish(); return; }
      const step = Math.floor(progress * 16);
      if (step !== lastStep) {
        for (let i = pending.length - 1; i >= 0; i--) {
          const token = pending[i];
          if (progress >= token.at) {
            token.element.replaceWith(token.text);
            pending.splice(i, 1);
          } else if (!reducedMotion.matches || !token.overlay.dataset.noise) {
            // Reduced motion keeps the noise still until each token resolves.
            const noise = mode === 'mask' ? '▰' : Array.from(token.text, () => vocabulary[Math.floor(Math.random() * vocabulary.length)]).join('');
            token.element.dataset.noise = noise;
            token.overlay.dataset.noise = noise;
          }
        }
        lastStep = step;
      }
      frame = requestAnimationFrame(render);
    }
    reducedMotion.addEventListener('change', onMotionChange);
    document.addEventListener('visibilitychange', onVisibilityChange);
    cancel = finish;
    render(performance.now());
    // The initial noise must reach a painted frame before its clock starts.
    previous = undefined;
  }
  function startWhenReady() {
    void document.fonts.ready.then(() => {
      if (!manuallyStarted) play();
    });
  }
  if (document.readyState === 'complete') startWhenReady();
  else window.addEventListener('load', startWhenReady, { once: true });
}
