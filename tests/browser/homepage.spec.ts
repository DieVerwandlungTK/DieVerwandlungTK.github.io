import { test, expect } from '@playwright/test';

test('homepage presents research without loading the simulation', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => requests.push(request.url()));
  await page.goto('/');
  for (const name of ['About', 'Research', 'Publications', 'Presentations', 'Elsewhere']) {
    await expect(page.getByRole('heading', { name: new RegExp(name) })).toBeVisible();
  }
  await expect(page.locator('.hero-description')).toContainText('拡散言語モデル');
  await expect(page.locator('canvas')).toHaveCount(0);
  expect(await page.evaluate(() => 'waterSimulation' in window)).toBe(false);
  expect(requests.filter(url => /ice-.*\.bin|playground.*\.js|background.*\.js/.test(url))).toEqual([]);
  await page.getByRole('navigation').getByRole('link', { name: 'Playground' }).click();
  await expect(page).toHaveURL(/\/playground.html$/);
  await expect(page.getByRole('heading', { name: 'Water simulation' })).toBeVisible();
  await page.getByRole('link', { name: 'ホームページへ戻る' }).click();
  await expect(page.locator('.hero-description')).toBeVisible();
});

test('mobile homepage keeps all navigation links accessible', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('/');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await expect(page.getByRole('navigation').getByRole('link', { name: 'Playground' })).toBeInViewport();
  await page.getByRole('navigation').getByRole('link', { name: 'Publications' }).click();
  await expect(page).toHaveURL(/#publications$/);
  await page.screenshot({ path: 'test-results/homepage-mobile.png', fullPage: true });
});
