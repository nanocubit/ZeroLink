# GitForge

> Forge your Git workflow — MCP-native desktop IDE with local AI agent and terminal-first flow.

GitForge is a compact desktop Git workspace that combines editor, terminal, PR management, browser panel, and MCP server in one app. The goal is to reduce context switching between VS Code, terminal, Git hosting UI, and AI assistant tools.

## Repository Description (for GitHub)

**Short description:**

`MCP-native Git IDE (Tauri + Vue) with libgit2 terminal workflow, local voice agent, and 5-panel workspace.`

## Core Features

- **MCP JSON-RPC server** for Claude/Cursor/GPT integrations (`tools/list`, `git_status`, `git_commit`, PR/worktree methods).
- **Terminal Terminator direction**: git operations routed through `libgit2` backend APIs.
- **5-panel workspace UI**: Files, Editor, Terminal, PR panel, Browser panel.
- **Local BPGT agent memory** backed by `redb`.
- **SQLite metadata layer** for PRs and worktree tracking.

## Tech Stack

- **Desktop shell:** Tauri (Rust)
- **Backend:** Rust + git2 + rusqlite + redb + tokio/tungstenite
- **Frontend:** Vue 3
- **Protocol:** MCP-style JSON-RPC over WebSocket

## Current Status

This repository is at **MVP foundation** stage:

- backend MCP methods are implemented for core git/pr/worktree flows;
- UI is implemented as an interactive MVP shell;
- production hardening (security, full E2E, packaging/release pipeline) is still in progress.

## Local Development

### Prerequisites

- Rust toolchain (stable)
- Node.js 20+
- npm

### Quick start

```bash
# frontend deps
cd gitforge
npm install

# rust format check
cargo fmt --all -- --check

# tests (when crates.io access is available)
cargo test
```

## High-priority Next Steps

1. Wire Vue panels to real Tauri `invoke` calls (remove mock data paths).
2. Add MCP contract/integration tests for all methods and edge-cases.
3. Implement secure command execution policy for terminal routing.
4. Add release pipeline with signed builds and staged rollout.

## License

TBD (set before public release).
