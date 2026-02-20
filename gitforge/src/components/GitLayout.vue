<template>
  <div class="layout-5col neumorphism" :class="`theme-${currentTheme}`">
    <section class="panel">
      <h3>Files</h3>
      <ul>
        <li v-for="file in fileStatus" :key="file.path">{{ file.path }} · {{ file.status }}</li>
      </ul>
    </section>

    <section class="panel">
      <h3>Monaco</h3>
      <p>{{ activeFile || 'Выберите файл слева' }}</p>
    </section>

    <section class="panel">
      <TerminalPanel />
    </section>

    <section class="panel">
      <h3>PR</h3>
      <ul>
        <li v-for="pr in pullRequests" :key="pr.id">#{{ pr.id }} {{ pr.title }}</li>
      </ul>
    </section>

    <section class="panel">
      <h3>Browser</h3>
      <p>Embedded webview placeholder</p>
    </section>

    <ThemePanel v-model="currentTheme" />
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import TerminalPanel from './TerminalPanel.vue'
import ThemePanel from './ThemePanel.vue'

const currentTheme = ref(localStorage.getItem('gitforge-theme') || 'professional')
const fileStatus = ref([
  { path: 'src-tauri/src/mcp/server.rs', status: 'WT_MODIFIED' },
  { path: 'src/components/GitLayout.vue', status: 'WT_NEW' },
])
const activeFile = ref('src/components/GitLayout.vue')
const pullRequests = ref([
  { id: 1, title: 'Bootstrap MCP server skeleton' },
  { id: 2, title: 'Build 5-column Neumorphism layout' },
])

watch(currentTheme, (value) => {
  localStorage.setItem('gitforge-theme', value)
})
</script>

<style>
.layout-5col {
  display: grid;
  grid-template-columns: 280px 1fr 320px 280px 400px;
  gap: 1rem;
  height: 100vh;
  padding: 1rem;
  background: var(--bg);
  color: var(--text);
}
.neumorphism {
  --bg: #16161e;
  --surface: #212134;
  --text: #e2e8f0;
  --accent: #4f46e5;
}
.theme-light { --bg: #f7f9fc; --surface: #ffffff; --text: #0f172a; }
.theme-warm { --bg: #1f1512; --surface: #2d1d18; --text: #f5e5d8; }
.theme-cool { --bg: #0b1220; --surface: #121d33; --text: #dbeafe; }
.theme-minimal { --bg: #111827; --surface: #1f2937; --text: #e5e7eb; }
.panel {
  background: var(--surface);
  border-radius: 16px;
  padding: 1rem;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
  overflow: auto;
}
</style>
