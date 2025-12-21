use std::env;

fn get_env_var(key: &str, default: &str) -> String {
    env::var(key).unwrap_or(default.to_string())
}
fn get_env_var_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .unwrap_or(default.to_string())
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("Could not parse {}", key))
}

#[derive(Clone, Debug)]
pub struct ApplicationConfig {
    pub host: String,
    pub port: String,
    pub server_workers: usize,

    pub batcher_config: BatcherConfig,
    pub runner_config: RunnerConfig,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BatcherConfig {
    pub max_batch_size: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        BatcherConfig {
            max_batch_size: 200,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub threads: usize,
    pub parallel_games: usize,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        RunnerConfig {
            threads: 6,
            parallel_games: 200,
        }
    }
}

impl ApplicationConfig {
    pub fn load() -> Self {
        let host = get_env_var("HOST", "127.0.0.1");
        let port = get_env_var("PORT", "8080");
        let server_workers = get_env_var_usize("WORKERS", 4);

        ApplicationConfig {
            host,
            port,
            server_workers,
            batcher_config: create_batcher_config(),
            runner_config: create_runner_config(),
        }
    }
}

fn create_batcher_config() -> BatcherConfig {
    let max_batch_size = get_env_var_usize("MAX_BATCH_SIZE", 200);
    BatcherConfig { max_batch_size }
}

fn create_runner_config() -> RunnerConfig {
    let threads = get_env_var_usize("PLAY_CORES", 6);
    let parallel_games = get_env_var_usize("PARALLEL_GAMES", 200);
    RunnerConfig {
        threads,
        parallel_games,
    }
}
