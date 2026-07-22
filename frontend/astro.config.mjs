import { defineConfig } from 'astro/config';
import tailwind from "@astrojs/tailwind";
import node from '@astrojs/node'; // For SSR

import react from '@astrojs/react';

export default defineConfig({
  output: "server", // 👈 Change to SSR
  adapter: node({ mode: 'standalone' }), // Required for SSR
  integrations: [tailwind(), react()],
});