use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub method: String,
    pub params: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub result: Option<serde_json::Value>,
    pub error: Option<McpError>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct McpError {
    pub code: i32,
    pub message: String,
}

pub struct GitForgeMcp {
    repo_path: Arc<String>,
    db: Arc<Mutex<rusqlite::Connection>>,
}

impl GitForgeMcp {
    pub fn new(repo_path: String) -> Result<Self, String> {
        let db_path = format!("{repo_path}/gitforge.db");
        let db = rusqlite::Connection::open(&db_path)
            .map_err(|e| format!("failed to open sqlite db: {e}"))?;

        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS prs (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                from_branch TEXT,
                to_branch TEXT,
                state TEXT DEFAULT 'open',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS worktrees (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                path TEXT,
                branch TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );",
        )
        .map_err(|e| format!("failed to initialize db: {e}"))?;

        Ok(Self {
            repo_path: Arc::new(repo_path),
            db: Arc::new(Mutex::new(db)),
        })
    }

    pub async fn serve(self: Arc<Self>, host: String) -> Result<String, String> {
        let listener = TcpListener::bind(&host)
            .await
            .map_err(|e| format!("failed to bind MCP server: {e}"))?;

        println!("🤖 MCP Server listening on {host}");

        while let Ok((stream, addr)) = listener.accept().await {
            println!("MCP client connected: {addr}");
            let server = Arc::clone(&self);
            tokio::spawn(async move {
                if let Err(e) = server.handle_connection(stream).await {
                    eprintln!("MCP connection error: {e}");
                }
            });
        }

        Ok("MCP server stopped".to_string())
    }

    async fn handle_connection(&self, stream: tokio::net::TcpStream) -> Result<(), String> {
        let ws = accept_async(stream)
            .await
            .map_err(|e| format!("websocket handshake failed: {e}"))?;

        let (mut write, mut read) = ws.split();

        while let Some(msg) = read.next().await {
            let msg = msg.map_err(|e| format!("websocket read error: {e}"))?;
            if let Message::Text(text) = msg {
                let response = match serde_json::from_str::<McpRequest>(&text) {
                    Ok(req) => self.execute_mcp(&req).await,
                    Err(e) => McpResponse {
                        jsonrpc: "2.0".to_string(),
                        id: serde_json::Value::Null,
                        result: None,
                        error: Some(McpError {
                            code: -32700,
                            message: format!("parse error: {e}"),
                        }),
                    },
                };

                let response_text = serde_json::to_string(&response)
                    .map_err(|e| format!("response serialization error: {e}"))?;

                write
                    .send(Message::Text(response_text))
                    .await
                    .map_err(|e| format!("websocket send error: {e}"))?;
            }
        }

        Ok(())
    }

    async fn execute_mcp(&self, req: &McpRequest) -> McpResponse {
        let result = match req.method.as_str() {
            "tools/list" => self.tools_list(),
            "git_status" => self.git_status(),
            "git_commit" => self.git_commit(&req.params),
            "git_create_pr" => self.git_create_pr(&req.params),
            "prs_list" => self.prs_list(),
            "git_worktree_create" => self.git_worktree_create(&req.params),
            "git_worktree_list" => self.git_worktree_list(),
            _ => Err(McpError {
                code: -32601,
                message: format!("method '{}' not found", req.method),
            }),
        };

        match result {
            Ok(result) => McpResponse {
                jsonrpc: "2.0".to_string(),
                id: req.id.clone(),
                result: Some(result),
                error: None,
            },
            Err(error) => McpResponse {
                jsonrpc: "2.0".to_string(),
                id: req.id.clone(),
                result: None,
                error: Some(error),
            },
        }
    }

    pub async fn execute_mcp_for_tauri(&self, req: &McpRequest) -> McpResponse {
        self.execute_mcp(req).await
    }

    fn tools_list(&self) -> Result<serde_json::Value, McpError> {
        Ok(serde_json::json!([
            {
                "name": "git_status",
                "description": "Show git repository status",
                "inputSchema": {}
            },
            {
                "name": "git_commit",
                "description": "Create commit from current index",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "git_create_pr",
                "description": "Create pull request metadata record",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "from": {"type": "string"},
                        "to": {"type": "string"}
                    },
                    "required": ["title", "from", "to"]
                }
            },
            {
                "name": "git_worktree_create",
                "description": "Create git worktree and register in sqlite",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "path": {"type": "string"},
                        "branch": {"type": "string"}
                    },
                    "required": ["name", "path", "branch"]
                }
            }
        ]))
    }

    fn open_repo(&self) -> Result<git2::Repository, McpError> {
        git2::Repository::open(self.repo_path.as_str()).map_err(|_| McpError {
            code: -32000,
            message: "repository not found".to_string(),
        })
    }

    fn git_status(&self) -> Result<serde_json::Value, McpError> {
        let repo = self.open_repo()?;
        let mut status_opts = git2::StatusOptions::new();
        status_opts.include_untracked(true).recurse_untracked_dirs(true);

        let statuses = repo
            .statuses(Some(&mut status_opts))
            .map_err(|e| McpError {
                code: -32001,
                message: e.to_string(),
            })?;

        let files: Vec<_> = statuses
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "path": entry.path().unwrap_or(""),
                    "status": format!("{:?}", entry.status())
                })
            })
            .collect();

        Ok(serde_json::json!({
            "success": true,
            "count": files.len(),
            "files": files
        }))
    }

    fn git_commit(&self, params: &serde_json::Value) -> Result<serde_json::Value, McpError> {
        let message = params
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("MCP commit")
            .to_string();

        let repo = self.open_repo()?;
        let mut index = repo.index().map_err(|e| McpError {
            code: -32002,
            message: format!("failed to open index: {e}"),
        })?;

        index.write().map_err(|e| McpError {
            code: -32003,
            message: format!("failed to write index: {e}"),
        })?;

        let tree_id = index.write_tree().map_err(|e| McpError {
            code: -32004,
            message: format!("failed to write tree: {e}"),
        })?;

        let tree = repo.find_tree(tree_id).map_err(|e| McpError {
            code: -32005,
            message: format!("failed to find tree: {e}"),
        })?;

        let signature = repo
            .signature()
            .or_else(|_| git2::Signature::now("GitForge MCP", "mcp@gitforge.dev"))
            .map_err(|e| McpError {
                code: -32006,
                message: format!("failed to create signature: {e}"),
            })?;

        let parent_commit = repo
            .head()
            .ok()
            .and_then(|h| h.target())
            .and_then(|oid| repo.find_commit(oid).ok());

        let commit_id = if let Some(parent) = parent_commit.as_ref() {
            repo.commit(Some("HEAD"), &signature, &signature, &message, &tree, &[parent])
        } else {
            repo.commit(Some("HEAD"), &signature, &signature, &message, &tree, &[])
        }
        .map_err(|e| McpError {
            code: -32007,
            message: format!("failed to commit: {e}"),
        })?;

        Ok(serde_json::json!({
            "success": true,
            "message": message,
            "commit": commit_id.to_string()
        }))
    }

    fn git_create_pr(&self, params: &serde_json::Value) -> Result<serde_json::Value, McpError> {
        let title = params
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or(McpError {
                code: -32602,
                message: "missing 'title'".to_string(),
            })?;

        let from = params
            .get("from")
            .and_then(|v| v.as_str())
            .unwrap_or("feature");
        let to = params
            .get("to")
            .and_then(|v| v.as_str())
            .unwrap_or("main");

        let db = self.db.lock().map_err(|_| McpError {
            code: -32010,
            message: "db lock poisoned".to_string(),
        })?;

        db.execute(
            "INSERT INTO prs (title, from_branch, to_branch) VALUES (?1, ?2, ?3)",
            rusqlite::params![title, from, to],
        )
        .map_err(|e| McpError {
            code: -32011,
            message: format!("failed to save PR: {e}"),
        })?;

        Ok(serde_json::json!({
            "success": true,
            "title": title,
            "from": from,
            "to": to,
            "id": db.last_insert_rowid()
        }))
    }

    fn prs_list(&self) -> Result<serde_json::Value, McpError> {
        let db = self.db.lock().map_err(|_| McpError {
            code: -32010,
            message: "db lock poisoned".to_string(),
        })?;

        let mut stmt = db
            .prepare(
                "SELECT id, title, from_branch, to_branch, state, created_at FROM prs ORDER BY id DESC",
            )
            .map_err(|e| McpError {
                code: -32012,
                message: format!("failed to prepare query: {e}"),
            })?;

        let rows = stmt
            .query_map([], |row| {
                Ok(serde_json::json!({
                    "id": row.get::<_, i64>(0)?,
                    "title": row.get::<_, String>(1)?,
                    "from": row.get::<_, String>(2)?,
                    "to": row.get::<_, String>(3)?,
                    "state": row.get::<_, String>(4)?,
                    "created_at": row.get::<_, String>(5)?
                }))
            })
            .map_err(|e| McpError {
                code: -32013,
                message: format!("failed to list PRs: {e}"),
            })?;

        let mut items = Vec::new();
        for row in rows {
            items.push(row.map_err(|e| McpError {
                code: -32014,
                message: format!("failed to parse PR row: {e}"),
            })?);
        }

        Ok(serde_json::json!({ "items": items }))
    }

    fn git_worktree_create(&self, params: &serde_json::Value) -> Result<serde_json::Value, McpError> {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or(McpError {
                code: -32602,
                message: "missing 'name'".to_string(),
            })?;
        let path = params
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or(McpError {
                code: -32602,
                message: "missing 'path'".to_string(),
            })?;
        let branch = params
            .get("branch")
            .and_then(|v| v.as_str())
            .ok_or(McpError {
                code: -32602,
                message: "missing 'branch'".to_string(),
            })?;

        let repo = self.open_repo()?;
        if !Path::new(path).exists() {
            std::fs::create_dir_all(path).map_err(|e| McpError {
                code: -32015,
                message: format!("failed to create worktree path: {e}"),
            })?;
        }

        let mut refname = format!("refs/heads/{branch}");
        if repo.find_reference(&refname).is_err() {
            let head_commit = repo
                .head()
                .ok()
                .and_then(|h| h.target())
                .and_then(|oid| repo.find_commit(oid).ok())
                .ok_or(McpError {
                    code: -32016,
                    message: "unable to derive HEAD commit for new branch".to_string(),
                })?;

            repo.branch(branch, &head_commit, false).map_err(|e| McpError {
                code: -32017,
                message: format!("failed to create branch: {e}"),
            })?;
            refname = format!("refs/heads/{branch}");
        }

        repo.worktree(name, Path::new(path), None)
            .map_err(|e| McpError {
                code: -32018,
                message: format!("failed to create worktree: {e}"),
            })?;

        let db = self.db.lock().map_err(|_| McpError {
            code: -32010,
            message: "db lock poisoned".to_string(),
        })?;

        db.execute(
            "INSERT OR REPLACE INTO worktrees (name, path, branch) VALUES (?1, ?2, ?3)",
            rusqlite::params![name, path, branch],
        )
        .map_err(|e| McpError {
            code: -32019,
            message: format!("failed to register worktree: {e}"),
        })?;

        Ok(serde_json::json!({
            "success": true,
            "name": name,
            "path": path,
            "branch": branch,
            "ref": refname
        }))
    }

    fn git_worktree_list(&self) -> Result<serde_json::Value, McpError> {
        let db = self.db.lock().map_err(|_| McpError {
            code: -32010,
            message: "db lock poisoned".to_string(),
        })?;

        let mut stmt = db
            .prepare("SELECT name, path, branch, created_at FROM worktrees ORDER BY id DESC")
            .map_err(|e| McpError {
                code: -32020,
                message: format!("failed to prepare query: {e}"),
            })?;

        let rows = stmt
            .query_map([], |row| {
                Ok(serde_json::json!({
                    "name": row.get::<_, String>(0)?,
                    "path": row.get::<_, String>(1)?,
                    "branch": row.get::<_, String>(2)?,
                    "created_at": row.get::<_, String>(3)?
                }))
            })
            .map_err(|e| McpError {
                code: -32021,
                message: format!("failed to list worktrees: {e}"),
            })?;

        let mut items = Vec::new();
        for row in rows {
            items.push(row.map_err(|e| McpError {
                code: -32022,
                message: format!("failed to parse worktree row: {e}"),
            })?);
        }

        Ok(serde_json::json!({ "items": items }))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("gitforge-{label}-{nanos}"));
        dir.to_string_lossy().to_string()
    }

    fn init_repo_with_file(repo_dir: &str) {
        fs::create_dir_all(repo_dir).expect("create repo dir");
        let repo = git2::Repository::init(repo_dir).expect("init repo");
        let file_path = Path::new(repo_dir).join("README.md");
        fs::write(&file_path, "hello gitforge
").expect("write file");

        let mut index = repo.index().expect("repo index");
        index.add_path(Path::new("README.md")).expect("stage readme");
        index.write().expect("write index");
    }

    #[tokio::test]
    async fn mcp_tools_list_returns_expected_entries() {
        let repo_dir = temp_path("tools-list");
        init_repo_with_file(&repo_dir);

        let server = GitForgeMcp::new(repo_dir.clone()).expect("create mcp server");
        let req = McpRequest {
            jsonrpc: "2.0".into(),
            id: serde_json::json!(1),
            method: "tools/list".into(),
            params: serde_json::json!({}),
        };

        let resp = server.execute_mcp_for_tauri(&req).await;
        assert!(resp.error.is_none());
        let tools = resp.result.expect("tools result");
        assert!(tools.is_array());
        assert!(tools
            .as_array()
            .expect("tools array")
            .iter()
            .any(|tool| tool.get("name") == Some(&serde_json::json!("git_status"))));
    }

    #[tokio::test]
    async fn mcp_git_create_pr_and_list_roundtrip() {
        let repo_dir = temp_path("pr-roundtrip");
        init_repo_with_file(&repo_dir);

        let server = GitForgeMcp::new(repo_dir.clone()).expect("create mcp server");

        let create = McpRequest {
            jsonrpc: "2.0".into(),
            id: serde_json::json!(2),
            method: "git_create_pr".into(),
            params: serde_json::json!({
                "title": "Test PR",
                "from": "feature/test",
                "to": "main"
            }),
        };

        let create_resp = server.execute_mcp_for_tauri(&create).await;
        assert!(create_resp.error.is_none(), "{:?}", create_resp.error.map(|e| e.message));

        let list = McpRequest {
            jsonrpc: "2.0".into(),
            id: serde_json::json!(3),
            method: "prs_list".into(),
            params: serde_json::json!({}),
        };

        let list_resp = server.execute_mcp_for_tauri(&list).await;
        assert!(list_resp.error.is_none());
        let items = list_resp
            .result
            .expect("list result")
            .get("items")
            .expect("items key")
            .as_array()
            .expect("items array")
            .clone();

        assert!(!items.is_empty());
        assert_eq!(items[0].get("title"), Some(&serde_json::json!("Test PR")));
    }

    #[tokio::test]
    async fn mcp_git_worktree_create_and_list_roundtrip() {
        let repo_dir = temp_path("worktree-roundtrip");
        init_repo_with_file(&repo_dir);

        let server = GitForgeMcp::new(repo_dir.clone()).expect("create mcp server");

        let wt_path = Path::new(&repo_dir).join(".worktrees").join("feature-x");
        let req = McpRequest {
            jsonrpc: "2.0".into(),
            id: serde_json::json!(4),
            method: "git_worktree_create".into(),
            params: serde_json::json!({
                "name": "feature-x",
                "path": wt_path.to_string_lossy(),
                "branch": "feature/x"
            }),
        };

        let create_resp = server.execute_mcp_for_tauri(&req).await;
        assert!(create_resp.error.is_none(), "{:?}", create_resp.error.map(|e| e.message));

        let list_req = McpRequest {
            jsonrpc: "2.0".into(),
            id: serde_json::json!(5),
            method: "git_worktree_list".into(),
            params: serde_json::json!({}),
        };

        let list_resp = server.execute_mcp_for_tauri(&list_req).await;
        assert!(list_resp.error.is_none());
        let items = list_resp
            .result
            .expect("worktree list result")
            .get("items")
            .expect("items key")
            .as_array()
            .expect("items array");

        assert!(items.iter().any(|i| i.get("name") == Some(&serde_json::json!("feature-x"))));
    }
}
