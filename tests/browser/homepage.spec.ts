import { test, expect } from '@playwright/test';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';

const ready = async (page: import('@playwright/test').Page) => {
  await expect(page.locator('.molecular-stage')).toHaveClass(/ready/, { timeout: 20000 });
};

test('heating the sample raises the measured temperature and advances simulation time', async ({ page }) => {
  test.setTimeout(120000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto('/');
  await ready(page);
  await page.screenshot({ path: 'test-results/cold.png' });
  await page.getByLabel('温度').fill('500');
  await expect(page.locator('#temperature')).toHaveText('500');
  await expect.poll(async () => Number((await page.locator('#kinetic-temperature').textContent())?.replace(/\D/g, '')),
    { timeout: 30000 }).toBeGreaterThan(400);
  await expect.poll(async () => Number((await page.locator('#sim-time').textContent())?.replace(/[^\d.]/g, '')),
    { timeout: 10000 }).toBeGreaterThan(2);
  await page.screenshot({ path: 'test-results/hot.png' });
  expect(errors).toEqual([]);
});

test('pausing holds the simulation and reset returns to the ice lattice', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  await page.getByRole('button', { name: '計算を一時停止' }).click();
  const held = await page.locator('#sim-time').textContent();
  await page.waitForTimeout(600);
  await expect(page.locator('#sim-time')).toHaveText(held!);
  await page.getByRole('button', { name: '氷から再開' }).click();
  await expect(page.locator('#sim-time')).toHaveText('0.0 ps');
  await expect(page.getByRole('button', { name: '計算を再開' })).toBeEnabled();
});

test('molecule count and density change the cell and stay stable', async ({ page }) => {
  test.setTimeout(90000);
  await page.goto('/');
  await ready(page);
  await page.getByLabel('分子数').selectOption('512');
  await expect(page.locator('#box-length')).toHaveText('25.4 Å');
  await page.getByLabel('密度').fill('140');
  await expect(page.locator('#density-ratio')).toHaveText('1.40');
  await expect.poll(async () => Number((await page.locator('#box-length').textContent())?.replace(/[^\d.]/g, '')))
    .toBeLessThan(25.4);
  await expect.poll(async () => Number((await page.locator('#bond-count').textContent())?.replace(/\D/g, '')),
    { timeout: 20000 }).toBeGreaterThan(100);
  // Switching size and squeezing the box must not deform any molecule.
  const worstBond = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    const sites = await simulation.readSites();
    let worst = 0;
    for (let molecule = 0; molecule < simulation.molecules; molecule++) {
      const oxygen = [0, 1, 2].map(axis => sites[molecule * 12 + axis]);
      for (const hydrogen of [1, 2]) {
        const bond = [0, 1, 2].map(axis => sites[molecule * 12 + hydrogen * 4 + axis] - oxygen[axis]);
        worst = Math.max(worst, Math.abs(Math.hypot(...bond) - 0.9572));   // OH_LENGTH
      }
    }
    return worst;
  });
  expect(worstBond).toBeLessThan(0.001);
});

test('reduced motion shows the ice lattice without stepping', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await ready(page);
  await expect(page.locator('#sim-time')).toHaveText('0.0 ps');
  await expect(page.locator('#playback-status')).toHaveText('一時停止中');
  await page.getByRole('button', { name: '計算を再開' }).click();
  await expect(page.locator('#playback-status')).toHaveText('計算中');
});

test('no WebGPU shows a real static molecule image and readable sections', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
  await page.goto('/');
  await expect(page.locator('#playback-status')).toHaveText('静止画を表示中');
  await expect(page.getByLabel('温度')).toBeDisabled();
  expect(await page.locator('#static-molecules').evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0)).toBe(true);
  await page.screenshot({ path: 'test-results/static-fallback.png' });
  for (const name of ['About', 'Research', 'Publications', 'Presentations', 'Elsewhere']) {
    await expect(page.getByRole('heading', { name: new RegExp(name) })).toBeVisible();
  }
});

test('a failed ice configuration fetch degrades to the static image', async ({ page }) => {
  await page.route('**/data/ice-216.bin', route => route.abort());
  await page.goto('/');
  await expect(page.locator('#playback-status')).toHaveText('静止画を表示中');
  await expect(page.locator('.molecular-stage')).not.toHaveClass(/ready/);
  await expect(page.getByLabel('密度')).toBeDisabled();
});

test('the model approximations are stated without opening any disclosure', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  const caveats = page.locator('.simulation-caveats');
  await expect(caveats).toBeVisible();
  const text = await caveats.innerText();
  // The four approximations the design requires the page to *state*, not hide behind a summary
  // the visitor has to find and open: reaction-field cutoff (not PME), f32 single precision, the
  // Langevin thermostat's friction, and that cooling does not refreeze the sample.
  expect(text).toContain('PME');
  expect(text).toContain('f32');
  expect(text).toContain('5 ps⁻¹');
  expect(text).toContain('戻りません');
  // The disclosure itself must genuinely be closed and irrelevant to the assertions above.
  await expect(page.locator('.simulation-details')).not.toHaveJSProperty('open', true);
  const details = page.locator('.simulation-details p').first();
  await expect(details).toBeHidden();
});

test('the model approximations stay visible under prefers-reduced-motion and on a mobile viewport', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('/');
  await ready(page);
  const caveats = page.locator('.simulation-caveats');
  await expect(caveats).toBeVisible();
  const text = await caveats.innerText();
  expect(text).toContain('PME');
  expect(text).toContain('f32');
  expect(text).toContain('5 ps⁻¹');
  expect(text).toContain('戻りません');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('small mobile screens retain navigation and avoid horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await ready(page);
  // Narrow viewports default to the smallest sample.
  await expect(page.getByLabel('分子数')).toHaveValue('64');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('navigation').getByRole('link', { name: 'Publications' }).click();
  await expect(page).toHaveURL(/#publications$/);
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true });
});

test('production assets and the simulation work under a GitHub Pages project path', async ({ page }) => {
  const root = resolve('dist');
  const mime: Record<string, string> = { '.html':'text/html', '.js':'text/javascript', '.css':'text/css',
    '.json':'application/json', '.svg':'image/svg+xml', '.bin':'application/octet-stream' };
  const server = createServer(async (req, res) => {
    const url = new URL(req.url!, 'http://localhost');
    if (!url.pathname.startsWith('/research-homepage/')) { res.writeHead(404).end(); return; }
    const filename = resolve(root, url.pathname.slice('/research-homepage/'.length) || 'index.html');
    if (!filename.startsWith(root + '/')) { res.writeHead(403).end(); return; }
    try {
      const bytes = await readFile(filename);
      res.writeHead(200, { 'Content-Type': mime[extname(filename)] || 'application/octet-stream' }).end(bytes);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const address = server.address() as { port: number };
    await page.goto(`http://127.0.0.1:${address.port}/research-homepage/`);
    await ready(page);
    await expect(page.getByRole('button', { name: '計算を一時停止' })).toBeEnabled();
    expect(await page.locator('#static-molecules').evaluate((img: HTMLImageElement) => img.naturalWidth > 0)).toBe(true);
  } finally {
    await page.goto('about:blank');
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
});
