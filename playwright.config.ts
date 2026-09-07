import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests/browser',
  use: {
    channel: 'chromium',
    baseURL: 'http://127.0.0.1:4173',
    viewport: { width: 1440, height: 1000 },
    launchOptions: { args: ['--enable-unsafe-webgpu'] },
  },
  webServer: { command: 'npm run preview', url: 'http://127.0.0.1:4173', reuseExistingServer: !process.env.CI },
});
