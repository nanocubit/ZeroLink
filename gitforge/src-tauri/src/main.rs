use std::sync::Arc;

mod agent;
mod mcp {
    pub mod server;
}

use agent::BpgtAgent;
use mcp::server::GitForgeMcp;

#[tauri::command]
async fn mcp_call(method: String, params: serde_json::Value, repo_path: String) -> Result<serde_json::Value, String> {
    let server = GitForgeMcp::new(repo_path)?;
    let request = mcp::server::McpRequest {
        jsonrpc: "2.0".to_string(),
        id: serde_json::json!(1),
        method,
        params,
    };

    let response = {
        let server = Arc::new(server);
        server.execute_mcp_for_tauri(&request).await
    };

    match response.error {
        Some(err) => Err(err.message),
        None => Ok(response.result.unwrap_or_default()),
    }
}

#[tauri::command]
async fn voice_process(text: String, db_path: String) -> Result<String, String> {
    let agent = BpgtAgent::new(&db_path);
    agent.process_voice(&text).await
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![mcp_call, voice_process])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
