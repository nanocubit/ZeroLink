use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

const MEMORY_TABLE: TableDefinition<&str, &str> = TableDefinition::new("agent_memory");

pub struct BpgtAgent {
    db: Database,
    model: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct AgentMemory {
    pub context: String,
    pub last_commands: Vec<String>,
    pub repo_state: String,
}

#[derive(Debug, Clone, Copy)]
enum VoiceIntent {
    Status,
    Commit,
    CreatePr,
    ToolsList,
}

impl BpgtAgent {
    pub fn new(db_path: &str) -> Self {
        let db = Database::create(db_path).expect("failed to create redb db");
        {
            let write_txn = db.begin_write().expect("failed to begin write transaction");
            {
                let _ = write_txn
                    .open_table(MEMORY_TABLE)
                    .expect("failed to open memory table");
            }
            write_txn.commit().expect("failed to commit memory table creation");
        }

        Self { db, model: None }
    }

    pub async fn process_voice(&self, text: &str) -> Result<String, String> {
        let _active_model = self.model.as_deref().unwrap_or("bgpt-fallback");
        let intent = self.parse_intent(text);
        let method = match intent {
            VoiceIntent::Status => "git_status",
            VoiceIntent::Commit => "git_commit",
            VoiceIntent::CreatePr => "git_create_pr",
            VoiceIntent::ToolsList => "tools/list",
        };

        self.persist_last_command(method)?;
        Ok(method.to_string())
    }

    fn parse_intent(&self, text: &str) -> VoiceIntent {
        let lowered = text.to_lowercase();
        if lowered.contains("статус") || lowered.contains("status") {
            VoiceIntent::Status
        } else if lowered.contains("коммит") || lowered.contains("commit") {
            VoiceIntent::Commit
        } else if lowered.contains("пулреквест") || lowered.contains("pr") {
            VoiceIntent::CreatePr
        } else {
            VoiceIntent::ToolsList
        }
    }

    fn persist_last_command(&self, command: &str) -> Result<(), String> {
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| format!("failed to begin write transaction: {e}"))?;
        {
            let mut table = write_txn
                .open_table(MEMORY_TABLE)
                .map_err(|e| format!("failed to open memory table: {e}"))?;
            table
                .insert("last_command", command)
                .map_err(|e| format!("failed to save memory: {e}"))?;
        }
        write_txn
            .commit()
            .map_err(|e| format!("failed to commit memory: {e}"))
    }

    pub fn last_command(&self) -> Result<Option<String>, String> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| format!("failed to begin read transaction: {e}"))?;
        let table = read_txn
            .open_table(MEMORY_TABLE)
            .map_err(|e| format!("failed to open memory table: {e}"))?;

        let value = table
            .get("last_command")
            .map_err(|e| format!("failed to read memory: {e}"))?
            .map(|v| v.value().to_string());

        Ok(value)
    }
}
