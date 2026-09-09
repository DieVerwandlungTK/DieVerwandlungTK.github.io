import { test, expect } from '@playwright/test';

for (const [random, mode] of [[0.2, 'mask'], [0.8, 'uniform']] as const) {
  test(`${mode} reveals the actual page text within two seconds`, async ({ page }) => {
    await page.addInitScript(value => { Math.random = () => value; }, random);
    await page.clock.install();
    await page.clock.pauseAt(new Date());
    await page.goto('/');
    await expect(page.locator('main')).toHaveAttribute('data-diffusion', mode);
    await expect(page.getByRole('heading', { name: 'Takato Kinoshita' })).toBeVisible();
    const pending = page.locator('.text-diffusion-token');
    expect(await pending.count()).toBeGreaterThan(0);
    if (mode === 'mask') await expect(pending.first()).toHaveAttribute('data-noise', '▰');
    else await expect(pending.first()).not.toHaveAttribute('data-noise', '▰');
    const before = await page.locator('#hero-title').boundingBox();
    const count = await pending.count();
    await page.clock.runFor(800);
    expect(await pending.count()).toBeGreaterThan(0);
    expect(await pending.count()).toBeLessThan(count);
    await page.clock.runFor(1000);
    await expect(pending).toHaveCount(0);
    await expect(page.locator('#hero-title')).toHaveText('Takato Kinoshita');
    expect(await page.locator('#hero-title').boundingBox()).toEqual(before);
    await page.getByRole('link', { name: '研究について' }).click();
    await expect(page).toHaveURL(/#research$/);
  });
}

test('reduced motion still reveals text automatically on load and reload', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.clock.install();
  await page.clock.pauseAt(new Date());
  for (const reload of [false, true]) {
    if (reload) await page.reload();
    else await page.goto('/');
    expect(await page.locator('.text-diffusion-token').count()).toBeGreaterThan(0);
    await page.clock.runFor(1800);
    await expect(page.locator('.text-diffusion-token')).toHaveCount(0);
    await expect(page.locator('#hero-title')).toHaveText('Takato Kinoshita');
  }
});

test('opening a background tab preserves the animation until it becomes visible', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => true });
  });
  await page.clock.install();
  await page.clock.pauseAt(new Date());
  await page.goto('/');
  await page.clock.runFor(2500);
  await page.evaluate(() => {
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => false });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  expect(await page.locator('.text-diffusion-token').count()).toBeGreaterThan(0);
  await page.clock.runFor(1800);
  await expect(page.locator('.text-diffusion-token')).toHaveCount(0);
});

test('explicit replay works with reduced motion and repeated clicks', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.clock.install();
  await page.clock.pauseAt(new Date());
  await page.goto('/');
  await page.getByRole('button', { name: '演出を再生' }).click({ timeout: 3000 });
  expect(await page.locator('.text-diffusion-token').count()).toBeGreaterThan(0);
  await page.clock.runFor(400);
  await page.getByRole('button', { name: '演出を再生' }).click({ timeout: 3000 });
  await page.clock.runFor(1800);
  await expect(page.locator('.text-diffusion-token')).toHaveCount(0);
  await expect(page.locator('#hero-title')).toHaveText('Takato Kinoshita');
  await expect(page.locator('diffusion-run')).toHaveCount(0);
});

test('slow page resources do not consume the entrance animation', async ({ page }) => {
  let release!: () => void;
  const ready = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/delayed-entrance.svg', async route => {
    await ready;
    await route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>' });
  });
  await page.addInitScript(() => {
    document.addEventListener('DOMContentLoaded', () => {
      const img = document.createElement('img');
      img.src = '/delayed-entrance.svg';
      img.hidden = true;
      document.body.append(img);
    });
  });
  await page.clock.install();
  await page.clock.pauseAt(new Date());
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await page.clock.runFor(2500);
  release();
  await page.waitForLoadState('load');
  expect(await page.locator('.text-diffusion-token').count()).toBeGreaterThan(0);
  await page.clock.runFor(1800);
  await expect(page.locator('.text-diffusion-token')).toHaveCount(0);
});
