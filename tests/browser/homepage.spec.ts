import { test, expect } from '@playwright/test';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';

test('GPU playback can pause, scrub, reach the endpoint, and replay', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto('/');
  await expect(page.locator('.molecular-stage')).toHaveClass(/ready/, { timeout: 20000 });
  await page.getByRole('button', { name: '軌跡を一時停止' }).click();
  const timeline = page.getByRole('slider');
  const paused = await timeline.inputValue();
  await page.waitForTimeout(150);
  await expect(timeline).toHaveValue(paused);
  await page.screenshot({ path: 'test-results/cold.png' });
  await timeline.fill('1000');
  await expect(page.locator('#temperature')).toHaveText('450');
  await expect(page.locator('#sim-time')).toHaveText('40.0 ps');
  await expect(page.locator('#playback-status')).toHaveText('再生終了');
  await page.screenshot({ path: 'test-results/hot.png' });
  await page.getByRole('button', { name: '軌跡を最初から再生' }).click();
  await expect(page.locator('#temperature')).toHaveText('180');
  await expect(page.locator('#sim-time')).toHaveText('0.0 ps');
  await expect(page.getByRole('button', { name: '軌跡を一時停止' })).toBeEnabled();
  expect(errors).toEqual([]);
  await page.screenshot({ path: 'test-results/desktop.png', fullPage: true });
});

test('faster playback rates advance simulation time from the same position', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.molecular-stage')).toHaveClass(/ready/, { timeout: 20000 });
  await page.getByRole('button', { name: '軌跡を一時停止' }).click();
  await page.getByRole('slider').fill('0');
  await page.getByLabel('再生速度').selectOption('4');
  await page.getByRole('button', { name: '軌跡を再生', exact: true }).click();
  await page.waitForTimeout(1000);
  // A quarter of the 30 s base duration; a second is well past 5% of it.
  expect(Number(await page.getByRole('slider').inputValue())).toBeGreaterThan(50);
  await expect(page.locator('#sim-time')).not.toHaveText('0.0 ps');
});

test('reduced motion starts paused and still permits explicit playback', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await expect(page.locator('.molecular-stage')).toHaveClass(/ready/, { timeout: 20000 });
  await expect(page.getByRole('slider')).toHaveValue('0');
  await expect(page.locator('#playback-status')).toHaveText('一時停止中');
  await page.getByRole('button', { name: '軌跡を再生', exact: true }).click();
  await expect(page.locator('#playback-status')).toHaveText('事前計算軌跡を再生中');
});

test('no WebGPU shows a real static molecule image and readable sections', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
  await page.goto('/');
  await expect(page.locator('#playback-status')).toHaveText('静止画を表示中');
  await expect(page.getByRole('slider')).toBeDisabled();
  expect(await page.locator('#static-molecules').evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0)).toBe(true);
  await page.screenshot({ path: 'test-results/static-fallback.png' });
  for (const name of ['About', 'Research', 'Publications', 'Presentations', 'Elsewhere']) {
    await expect(page.getByRole('heading', { name: new RegExp(name) })).toBeVisible();
  }
});

for (const asset of ['water.json', 'water.bin']) {
  test(`a failed ${asset} fetch degrades to the static image`, async ({ page }) => {
  await page.route(`**/data/${asset}`, route => route.abort());
  await page.goto('/');
  await expect(page.locator('#playback-status')).toHaveText('静止画を表示中');
  await expect(page.locator('.molecular-stage')).not.toHaveClass(/ready/);
  await expect(page.getByRole('slider')).toBeDisabled();
  await expect(page.getByLabel('再生速度')).toBeDisabled();
  });
}

test('small mobile screens retain navigation and avoid horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await expect(page.locator('.molecular-stage')).toHaveClass(/ready/, { timeout: 20000 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('navigation').getByRole('link', { name: 'Publications' }).click();
  await expect(page).toHaveURL(/#publications$/);
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true });
});

test('production assets and trajectory work under a GitHub Pages project path', async ({ page }) => {
  const root = resolve('dist');
  const mime: Record<string, string> = { '.html':'text/html', '.js':'text/javascript', '.css':'text/css', '.json':'application/json', '.svg':'image/svg+xml', '.bin':'application/octet-stream' };
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
    await expect(page.locator('.molecular-stage')).toHaveClass(/ready/, { timeout: 20000 });
    await expect(page.getByRole('button', { name: '軌跡を一時停止' })).toBeEnabled();
    expect(await page.locator('#static-molecules').evaluate((img: HTMLImageElement) => img.naturalWidth > 0)).toBe(true);
  } finally {
    await page.goto('about:blank');
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
});
