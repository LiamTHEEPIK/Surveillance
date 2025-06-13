import {defineConfig} from 'vite';
import {ElementPlusResolver} from 'unplugin-vue-components/resolvers';

import Vue from '@vitejs/plugin-vue';
import AutoImport from 'unplugin-auto-import/vite';
import Components from 'unplugin-vue-components/vite';
import Icons from 'unplugin-icons/vite';
import IconsResolver from 'unplugin-icons/resolver';

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5500,
  },
  plugins: [
    Vue(),
    Icons({
      autoInstall: true,
    }),
    AutoImport({
      imports: [
        'vue',
        'vue-router',
      ],
      resolvers: [
        ElementPlusResolver(),
        IconsResolver(),
      ],
    }),
    Components({
      resolvers: [
        ElementPlusResolver(),
        IconsResolver({
          prefix: 'icon',
          enabledCollections: ['ep'],
        }),
      ],
    }),
  ],
});
