import { test, expect, type Page } from '@playwright/test';

const ready = async (page: Page) => {
  await page.goto('/');
  await expect(page.locator('#speed-distribution')).toBeVisible({ timeout: 20000 });
};
const center = (page: Page) => page.locator('.distribution-bar').evaluateAll(bars => {
  const weights = bars.map(bar => Number(bar.getAttribute('height')));
  return weights.reduce((sum,weight,i)=>sum+weight*(i+.5)*1.5,0)/weights.reduce((a,b)=>a+b,0);
});

test('twelve finite bars and a reference curve report the measured temperature', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await ready(page);
  await expect(page.locator('.distribution-bar')).toHaveCount(12);
  const heights = await page.locator('.distribution-bar').evaluateAll(bars=>bars.map(bar=>Number(bar.getAttribute('height'))));
  expect(heights.every(height=>Number.isFinite(height)&&height>=0&&height<=45)).toBe(true);
  expect(heights.some(height=>height>0)).toBe(true);
  await expect(page.locator('.distribution-curve')).toHaveAttribute('d', /^M.+L/);
  const measured = await page.locator('#kinetic-temperature').textContent();
  await expect(page.locator('#speed-distribution')).toHaveAttribute('role','img');
  await expect(page.locator('#speed-distribution')).toHaveAttribute('aria-label',new RegExp(`実測温度 ${measured}`));
  // Changing the thermostat while paused must not relabel the measured distribution.
  const before = await page.locator('#speed-distribution').innerHTML();
  const description = await page.locator('#speed-distribution').getAttribute('aria-label');
  await page.getByLabel('温度', { exact:true }).fill('500');
  await page.waitForTimeout(650);
  expect(await page.locator('#speed-distribution').innerHTML()).toBe(before);
  await expect(page.locator('#speed-distribution')).toHaveAttribute('aria-label',description!);
  await expect(page.locator('#sim-time')).toHaveText('0.0 ps');
  await page.screenshot({ path:'test-results/distribution-desktop.png' });
});

test('at 216 molecules heating to 500 K shifts the distribution beyond its cold fluctuation', async ({ page }) => {
  test.setTimeout(90000);
  await ready(page);
  await expect(page.getByLabel('分子数')).toHaveValue('216');
  await page.getByLabel('温度', { exact:true }).fill('200');
  await expect.poll(async()=>parseFloat((await page.locator('#sim-time').textContent())!),{timeout:30000}).toBeGreaterThan(2);
  const collect = async () => {
    const result:number[]=[];
    for(let i=0;i<7;i++){await page.waitForTimeout(225);result.push(await center(page));}
    return result;
  };
  const cold = await collect();
  await page.getByLabel('温度', { exact:true }).fill('500');
  await expect.poll(async()=>parseFloat((await page.locator('#kinetic-temperature').textContent())!),{timeout:30000}).toBeGreaterThan(400);
  await expect.poll(()=>center(page),{timeout:30000}).toBeGreaterThan(6.6);
  const hot = await collect();
  const mean = (values:number[])=>values.reduce((a,b)=>a+b,0)/values.length;
  expect(mean(hot)-mean(cold)).toBeGreaterThan(1.5);
  expect(mean(hot)-mean(cold)).toBeGreaterThan(Math.max(...cold)-Math.min(...cold));
  await page.screenshot({ path:'test-results/distribution-hot.png' });
});

test('a paused count change and reset replace the histogram instead of mixing old samples', async ({ page }) => {
  await page.emulateMedia({ reducedMotion:'reduce' });
  await ready(page);
  await page.getByLabel('温度', { exact:true }).fill('500');
  await page.getByLabel('分子数').selectOption('64');
  await expect(page.locator('#speed-distribution')).toBeVisible();
  await expect(page.locator('#speed-distribution')).toHaveAttribute('aria-label',/実測温度 500 K/);
  await page.getByLabel('温度', { exact:true }).fill('200');
  await page.getByRole('button',{name:'氷から再開'}).click();
  await expect(page.locator('#speed-distribution')).toHaveAttribute('aria-label',/実測温度 200 K/);
  await expect(page.locator('#sim-time')).toHaveText('0.0 ps');
});

test('the chart fits a 375 px viewport without horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width:375,height:812 });
  await page.emulateMedia({ reducedMotion:'reduce' });
  await ready(page);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  const bounds = await page.locator('#speed-distribution svg').boundingBox();
  expect(bounds!.width).toBeGreaterThan(200);
  expect(bounds!.x+bounds!.width).toBeLessThanOrEqual(375);
  const copy = await page.locator('.hero-copy').boundingBox();
  const panel = await page.locator('.simulation-panel').boundingBox();
  expect(panel!.y).toBeGreaterThan(copy!.y+copy!.height);
  await page.screenshot({ path:'test-results/distribution-mobile.png',fullPage:true });
});

test('resetting a divergent paused sample restores a stable histogram without stepping', async ({ page }) => {
  await page.emulateMedia({ reducedMotion:'reduce' });
  await ready(page);
  const diverged = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setDensity(15);
    simulation.step(20);
    const stats = await simulation.readStats();
    simulation.setDensity(1);
    return stats;
  });
  expect(diverged.nonFinite).toBe(true);
  await page.getByRole('button',{name:'氷から再開'}).click();
  await expect(page.locator('#speed-distribution')).toBeVisible();
  await expect(page.locator('#speed-distribution')).toHaveAttribute('aria-label',/実測温度 180 K/);
  const geometry = await page.locator('#speed-distribution').innerHTML();
  await page.waitForTimeout(450);
  expect(await page.locator('#speed-distribution').innerHTML()).toBe(geometry);
  await expect(page.locator('#sim-time')).toHaveText('0.0 ps');
});
