<template>
  <div class="terminal-container">
    <div class="terminal-header">
      <select v-model="activeAgent">
        <option value="local">Local Shell</option>
        <option value="claude">Claude</option>
        <option value="cursor">Cursor</option>
        <option value="bgpt">BPGT Local</option>
      </select>
      <button class="voice-btn" :class="{ listening: isListening }" @click="toggleVoice">
        🎙️ {{ isListening ? 'Слушаю...' : 'Голос' }}
      </button>
    </div>

    <pre class="terminal-output">{{ output.join('\n') }}</pre>

    <div class="terminal-input">
      <span>$</span>
      <input v-model="inputLine" @keyup.enter="sendCommand" placeholder="git status, git commit..." />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const isListening = ref(false)
const activeAgent = ref('local')
const inputLine = ref('')
const output = ref(['GitForge Terminal Terminator ready'])

const append = (line) => {
  output.value = [...output.value.slice(-120), line]
}

const toggleVoice = () => {
  isListening.value = !isListening.value
  append(isListening.value ? '🔴 Voice capture enabled' : '🔇 Voice capture disabled')
}

const sendCommand = () => {
  const cmd = inputLine.value.trim()
  if (!cmd) {
    return
  }

  append(`$ ${cmd}`)
  if (cmd.startsWith('git ')) {
    append(`⚡ libgit2 route queued via MCP (${activeAgent.value})`)
  } else {
    append(`🤖 agent(${activeAgent.value}) route queued via MCP`)
  }

  inputLine.value = ''
}
</script>

<style scoped>
.terminal-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 320px;
}
.terminal-header,
.terminal-input {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
}
.voice-btn.listening {
  background: #ef4444;
  color: white;
}
.terminal-output {
  flex: 1;
  margin: 0;
  padding: 12px;
  overflow: auto;
  font-size: 13px;
  line-height: 1.4;
}
.terminal-input input {
  width: 100%;
  border: none;
  background: transparent;
  color: inherit;
}
</style>
