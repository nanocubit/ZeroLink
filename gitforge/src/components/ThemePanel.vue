<template>
  <div class="theme-panel neumorphism-card">
    <h4>Theme</h4>
    <div class="themes">
      <button
        v-for="theme in themes"
        :key="theme"
        :class="['theme-chip', { active: modelValue === theme }]"
        @click="$emit('update:modelValue', theme)"
      >
        {{ theme }}
      </button>
    </div>
  </div>
</template>

<script setup>
defineProps({
  modelValue: {
    type: String,
    required: true,
  },
})

defineEmits(['update:modelValue'])

const themes = ['professional', 'light', 'warm', 'cool', 'minimal']
</script>

<style scoped>
.theme-panel {
  position: fixed;
  right: 24px;
  bottom: 24px;
  padding: 12px;
  background: var(--surface);
  color: var(--text, #e2e8f0);
}
.themes {
  display: grid;
  gap: 8px;
  margin-top: 8px;
}
.theme-chip {
  border: none;
  border-radius: 10px;
  padding: 6px 10px;
  cursor: pointer;
}
.theme-chip.active {
  outline: 2px solid var(--accent, #4f46e5);
}
</style>
