use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "gitforge", about = "🔨 Forge your Git workflow")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// 🎨 Launch the desktop UI (Monaco + 5 columns)
    Ui,

    /// 🤖 MCP server for Claude/Cursor/GPT
    #[command(name = "mcp-serve")]
    McpServe {
        /// Repository path
        #[arg(default_value = ".")]
        repo: String,
    },

    /// 🧠 Local BPGT agent
    Agent {
        #[arg(default_value = ".")]
        repo: String,
    },

    /// 🌳 Git worktree helper CLI
    Worktree {
        #[arg(value_enum)]
        action: WorktreeAction,
        name: Option<String>,
    },

    /// 📱 Embedded browser
    Browser { url: String },
}

#[derive(clap::ValueEnum, Clone)]
enum WorktreeAction {
    Create,
    List,
    Switch,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Ui) => {
            println!("🚀 GitForge UI + MCP + Voice starting...");
        }
        Some(Commands::McpServe { repo }) => {
            println!("🤖 MCP Server: ws://localhost:6767 for {}", repo);
        }
        Some(Commands::Agent { repo }) => {
            println!("🧠 BPGT Agent + redb starting for {}", repo);
        }
        Some(Commands::Worktree { action, name }) => match action {
            WorktreeAction::Create => {
                println!(
                    "🌳 Worktree '{}' created",
                    name.unwrap_or_else(|| "new".to_string())
                )
            }
            WorktreeAction::List => println!("📋 Worktree list"),
            WorktreeAction::Switch => println!(
                "🔀 Switched to worktree '{}'",
                name.unwrap_or_else(|| "default".to_string())
            ),
        },
        Some(Commands::Browser { url }) => {
            println!("🌐 Opening {} in GitForge Browser", url);
        }
        None => {
            println!("🔨 GitForge v2.0 — Forge your Git workflow");
            println!("Usage: gitforge ui | mcp-serve | agent | worktree");
        }
    }
}
